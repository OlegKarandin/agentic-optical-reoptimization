# src/storm_reoptimizer/eval/runner.py
"""Episode rollout (eval design spec, "The episode model").

Timing cannot be scored as a single call -- "act now or wait?" is only
meaningful across a sequence -- so a scenario is a rollout over the track's
hours and all three decisions happen inside it.

Two modelling choices, stated so nobody has to infer them:

  * A commit lands IMMEDIATELY in the model; lead time is bookkeeping. An
    action at hour t on a lever with lead time L records
    effective_at_index = index(t) + L, and Task 10 marks the service dead if
    a realized cut hits it earlier than that. Delaying the commit instead
    would leave the next hour's menu blind to a pending action -- a different
    and less useful fiction than the one we chose.
  * snapshot_create() is called after inject_failure, never snapshot_branch:
    branch returns a FROZEN id naming the pre-mutation state, so passing it
    to evaluate_objective would silently score the unmodified network.
  * The service's representative point is re-derived at the TOP OF EVERY
    HOUR, from the server's current working path. Exposure is therefore a
    property of where the service actually rides now, not of where it rode
    when the episode started -- a protective reroute makes p_cut fall, which
    is the whole point of committing one.

commit_plan is in the rollout but never in the decision set -- the harness is
the approval gate. reconcile is never called: with the MCP layer's default
actuator it can only ever answer in_sync=True.

Every server interaction goes through mcp_client.call_tool_json against a
real subprocess, per CLAUDE.md's hard seam."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

from mcp.client import Client

from ..events.filters import get_filter
from ..events.geo import damage_footprint
from ..geo_mapper import Edge, load_edges, map_geo_event_to_assets
from ..mcp_client import call_tool_json
from .cone import (
    Segment, expected_capacity_at_risk_gbps, nearest_span_offset_km,
    p_cut_region,
)
from .decisions import ConstraintDecision, Decider, candidate_index
from .ledger import SpareLedger, spares_needed
from .observation import build_observation, latest_issuance, lead_time_hours_for
from .plans import PlanTranslationError, build_topology_index, plan_from_candidate
from .replay import restore_after_cuts
from .scenario_file import ConeAtHorizon, Issuance, ScenarioFile

MAX_ITERATIONS = 5

# The one event type this harness runs. It is deliberately NOT a scenario-file
# key: the schema is strict and every episode is a storm, so a key nobody ever
# varies would be one more thing to typo. Both the risk-group builder and the
# exposure span filter read it, and assertions.py imports it for the
# realized-cuts and claimant-exposure invariants -- one name, so the filter
# that decides which fibres a storm can cut and the filter that decides which
# spans score exposure can never disagree.
EVENT_TYPE = "storm"


@dataclass(frozen=True)
class Action:
    hour: str
    hour_index: int
    lever: str
    effective_at_index: int
    # site -> transponders charged there by this action. Empty for an
    # ip_reroute. Renamed from `pairs: int` -- a pair spans two sites and
    # cannot be scoped to one (exposure-and-depot design, §4.1).
    spares: dict
    service_id: str
    # The avoid set this candidate was routed under -- the one fact that
    # makes "I already routed around that risk group" checkable by the
    # decider an hour later. Defaulted so the positional constructions in
    # tests/eval/test_scoring.py keep working; run_episode always supplies
    # it. A dict field makes Action unhashable in practice; nothing in the
    # harness hashes one.
    avoid: dict = field(default_factory=dict)
    # Which side of the harness spent the spares: the decider's own choice
    # ("decider", the default -- every action run_episode records today) or
    # the harness's own deterministic restoration logic ("harness", added by
    # a later post-cut-restoration task). Last field so existing positional
    # constructions keep working.
    origin: str = "decider"


def action_payloads(actions, hours=()) -> tuple[dict, ...]:
    """What the decider is told about its OWN committed actions this episode.
    Plain dicts, so observation.py never has to import this module (the
    dependency runs one way) and the whole Observation stays
    JSON-serializable for the trace and the prompt.

    `effective_at_index` is a raw index into `scenario.hours` -- useless, and
    actively misleading, for an episode whose hours are non-positional labels
    (T2a/T2b/T3a/T3b all declare `hours: [t0, t1, t2, t6]`, where index 3 is
    the hour LABELLED t6, not "t3"). `effective_at_hour` resolves the index
    against the caller's `hours` sequence so the decider is told the label,
    not just the position. `hours` defaults to `()` so existing positional
    callers (tests/eval/test_scoring.py) that never pass it keep working;
    every index is then out of range and resolves to `None`, same as an
    index that runs past the end of a real `hours` list -- e.g. acting on the
    episode's last hour with a lead time of 1."""
    return tuple({"hour": a.hour, "lever": a.lever, "spares": a.spares,
                  "avoid": a.avoid, "origin": a.origin,
                  "service_id": a.service_id,
                  "effective_at_index": a.effective_at_index,
                  "effective_at_hour": (hours[a.effective_at_index]
                                        if a.effective_at_index < len(hours)
                                        else None)}
                 for a in actions)


def unconstrained_menu_projection(menu: dict,
                                  oms_nodes: dict | None = None) -> dict:
    """What decision 2 is shown of the menu it is about to reshape.

    decisions.py states the mechanic: decision 2's `avoid` is what
    build_layered_graph forbids, so it changes which candidates EXIST, while
    decision 3 only reorders. Until now the agent made that decision with no
    view of what it was about to delete -- and one real run avoided a risk
    group containing its own current corridor, which removed every stay-put
    candidate from the menu before the graded step ever saw one
    (remediation spec, F3).

    Label, lever and spare cost only. The cost vector is DELIBERATELY absent:
    weighing it is decision 3's job, and showing it here would collapse the
    two steps into one and make the objective decision a rubber stamp.

    `oms_nodes` resolves each candidate's `new_lightpaths` to endpoint SITES
    (ledger.spares_needed) -- the static optical adjacency, unchanged across
    an episode's hours, so callers read it once from
    `ServiceGeometry.oms_nodes` rather than re-fetching it."""
    oms_nodes = oms_nodes or {}
    return {"status": menu.get("status"),
            "candidates": [{"candidate_label": f"candidate_{i}",
                            "lever": candidate["lever"],
                            "spares_needed": spares_needed(candidate,
                                                          oms_nodes)}
                           for i, candidate in enumerate(
                               menu.get("candidates") or [])]}


def menu_for_prompt(menu: dict, oms_nodes: dict | None = None) -> dict:
    """The routing menu with two derived fields per candidate: the label the
    objective decision must answer with, and `spares_needed` -- the
    candidate's OWN spare cost PER SITE (ledger.spares_needed), which
    cost_vector["transponders"] is not.

    Lives here rather than in agent.py because BOTH readers need it: the model
    reads it at decision 3, and the trace records it so a rollout can be read
    afterwards. One function so the two can never disagree (run-viewer design,
    §5.1). Contrast unconstrained_menu_projection above, which strips the cost
    vector on purpose -- that one is shown to decision 2, this one to decision
    3.

    `oms_nodes` -- see unconstrained_menu_projection above."""
    oms_nodes = oms_nodes or {}
    projected = dict(menu)
    projected["candidates"] = [
        {**candidate, "candidate_label": f"candidate_{i}",
         "spares_needed": spares_needed(candidate, oms_nodes)}
        for i, candidate in enumerate(menu.get("candidates") or [])]
    return projected


def _candidate_oms(candidate: dict, oms_sequences: dict[str, tuple[str, ...]]
                   ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The OMS ids a candidate would put the service on -- the union of its
    reused lightpaths' known sequences and its new lightpaths' own sequences
    -- and separately, any reused lightpath id `oms_sequences` cannot
    resolve. Deduped (order-preserving): a candidate's own two legs never
    collide, but a reused leg and a new leg both touching the same OMS is
    possible and must not double-count."""
    oms_ids: list[str] = []
    unresolved: list[str] = []
    for lp_id in candidate.get("reused_lightpaths") or []:
        seq = oms_sequences.get(lp_id)
        if seq is None:
            unresolved.append(lp_id)
            continue
        oms_ids.extend(seq)
    for run in candidate.get("new_lightpaths") or []:
        oms_ids.extend(run["oms_sequence"])
    return tuple(dict.fromkeys(oms_ids)), tuple(unresolved)


def _path_delta(candidate_oms: tuple[str, ...], working_oms: tuple[str, ...],
                cuttable_span_by_oms: dict[str, Segment]) -> dict:
    """Whether a candidate moves the service AT ALL, stated in the harness's
    own OMS vocabulary rather than left for the decider to infer from cost
    figures or from p_cut similarity -- both of which the T1 inert-reroute
    finding (2026-08-31) shows are unreliable proxies for this categorical
    fact. `oms_retained_cuttable` is the one that names the finding's own
    defect directly: these storm-cuttable spans you are on now, you would
    still be on."""
    candidate_set, working_set = set(candidate_oms), set(working_oms)
    return {
        "changes_working_path": candidate_set != working_set,
        "oms_added": sorted(candidate_set - working_set),
        "oms_removed": sorted(working_set - candidate_set),
        "oms_retained_cuttable": sorted(
            (candidate_set & working_set) & cuttable_span_by_oms.keys()),
    }


def _protection_collision(candidate_oms: tuple[str, ...],
                          protection_oms: tuple[str, ...]) -> dict:
    """Whether committing this candidate would ride the service's OWN
    protection corridor -- invisible to `exposure` (working-path-only by
    construction) and to `path_delta` above (which only compares against the
    working path). The automatic 1:1 protection switchover that saved
    storm-svc-1 in the finding's T1b rollout depends on protection staying a
    DIFFERENT corridor from working; a candidate that collapses the two
    trades one invisible no-op for a different invisible cost."""
    shared = sorted(set(candidate_oms) & set(protection_oms))
    return {"collides": bool(shared), "oms_shared_with_protection": shared}


def _residual_exposure(candidate_oms: tuple[str, ...],
                       cuttable_span_by_oms: dict[str, Segment],
                       issuance: Issuance, damage_radius_km: float,
                       demand_gbps: float) -> dict[str, dict]:
    """Per horizon of the current issuance, the SAME p_cut_region call
    observation.py's own exposure row makes (build_observation, "the single
    quantity every gold rationale is arithmetic over"), but over the spans a
    CANDIDATE would ride rather than the spans the service rides now. This is
    the continuous half of the finding's fix; `_path_delta` above is the
    categorical half -- neither substitutes for the other, since a candidate
    can score near-equal p_cut to the status quo at a completely different
    corridor, and the same unchanged corridor's p_cut shifts every hour as
    the cone advances (both observed directly in the finding's own trace)."""
    spans = tuple(cuttable_span_by_oms[oms_id] for oms_id in candidate_oms
                  if oms_id in cuttable_span_by_oms)
    per_horizon: dict[str, dict] = {}
    for horizon, cone in issuance.horizons.items():
        lat, lon = cone.center["lat"], cone.center["lon"]
        p_cut = round(p_cut_region(spans, lat, lon, cone.width_km,
                                   damage_radius_km), 4)
        per_horizon[horizon] = {
            "offset_km": (round(nearest_span_offset_km(spans, lat, lon), 1)
                         if spans else None),
            "p_cut": p_cut,
            "ecar_gbps": round(
                expected_capacity_at_risk_gbps(p_cut, demand_gbps), 2),
        }
    return per_horizon


def menu_with_path_facts(menu: dict, geometry: ServiceGeometry,
                         service_id: str, *, issuance: Issuance,
                         damage_radius_km: float, demand_gbps: float) -> dict:
    """The routing menu with three deterministic, per-candidate facts about
    what committing it would actually do to `service_id`'s OWN path -- the
    fix for the T1 inert-reroute finding (2026-08-31-t1-inert-reroute-
    finding.md). Before this, the menu carried only cost figures, so a
    zero-cost candidate that leaves the service on the exact fibre the
    episode is about protecting strictly dominated a genuinely different,
    slightly costlier one -- not because the model misjudged, but because
    nothing in the payload could distinguish the two cases.

    `path_delta` and `collides_with_protection` are categorical (does this
    move the service, does it eat its own standby); `residual_exposure` is
    continuous (how exposed is where it lands), computed with the identical
    model and inputs observation.py's own exposure row uses so the two are
    directly comparable. Deliberately three separate fields rather than one
    blended score -- blending two question-shapes into one ranking is
    exactly what produced the 277-vs-278 collapse the finding documents.

    Called once, on the raw menu route_service returns, BEFORE both
    decider.objective() and the trace's own menu_for_prompt recording -- so
    both readers see it without either needing a signature change.

    A `reused_lightpaths` id with no resolvable OMS sequence is reported in
    `residual_exposure_unresolved` rather than silently contributing zero
    exposure, mirroring ServiceGeometry.unmapped_nodes' policy: a gap is
    shown, never quietly absorbed.

    Builds a new dict; `menu` itself is never mutated, matching
    menu_for_prompt's own contract."""
    working_oms = geometry.path_oms.get(service_id, {}).get("working", ())
    protection_oms = geometry.path_oms.get(service_id, {}).get(
        "protection", ())
    annotated = dict(menu)
    candidates = []
    for candidate in menu.get("candidates") or []:
        candidate_oms, unresolved = _candidate_oms(
            candidate, geometry.oms_sequences)
        facts = {
            "residual_exposure": _residual_exposure(
                candidate_oms, geometry.cuttable_span_by_oms, issuance,
                damage_radius_km, demand_gbps),
            "path_delta": _path_delta(
                candidate_oms, working_oms, geometry.cuttable_span_by_oms),
            "collides_with_protection": _protection_collision(
                candidate_oms, protection_oms),
        }
        if unresolved:
            facts["residual_exposure_unresolved"] = list(unresolved)
        candidates.append({**candidate, **facts})
    annotated["candidates"] = candidates
    return annotated


def _visible_services(obs) -> list[str]:
    """Services with a nonzero cut probability at any horizon of this hour's
    issuance, plus the actionable service unconditionally.

    Deliberately WIDER than the agent's p_cut_threshold projection: the viewer
    must be able to show a service the agent was not shown, because the gap
    between what was true and what was projected is the subject of the
    eval-fairness design. The typical hour has two such services against 571
    at max_p_cut 0.0, so the trimmed record stays small (run-viewer design,
    §5.1). The actionable service is kept even at p_cut == 0.0, mirroring
    project_observation's `keep` set (agent.py) -- a successful reroute is
    exactly what drives its p_cut to 0, and the viewer's job is to show that
    settle, not drop the service the whole episode is about the moment it
    recovers (final-review fix, 2026-08-29)."""
    visible = {
        svc for svc, per_horizon in obs.exposure.items()
        if any(float(entry["p_cut"]) > 0.0
               for entry in per_horizon.values())}
    if obs.service_under_test in obs.exposure:
        visible.add(obs.service_under_test)
    return sorted(visible)


def observation_record(obs, geometry: ServiceGeometry) -> dict:
    """The hour's ground truth: what was actually the case, as against
    `projected`, which is what the decider was shown. Store BOTH -- a trace
    with only the projection cannot show the gap, and one with only the truth
    cannot show what the agent read (run-viewer design, "Principle").

    `horizon_totals` and `restorable_groups` are kept WHOLE while the rows are
    trimmed: both are already computed over the full roster, and a total (or
    a group list) silently covering only the visible rows "would be worse
    than no total at all" (observation.py's `_horizon_totals`, docstring).
    Neither key is touched below -- `payload = obs.to_dict()` already carries
    both, and trimming only reaches `exposure`/`services`."""
    visible = _visible_services(obs)
    payload = obs.to_dict()
    payload["exposure"] = {s: payload["exposure"][s] for s in visible}
    payload["services"] = [row for row in payload["services"]
                           if row["id"] in visible]
    return {
        "observation": payload,
        "service_points": {s: list(geometry.points[s]) for s in visible
                           if s in geometry.points},
        "service_paths": {s: geometry.paths[s] for s in visible
                          if s in geometry.paths},
        "unmapped_nodes": {s: geometry.unmapped_nodes[s] for s in visible
                           if s in geometry.unmapped_nodes},
        # What build_observation actually scored exposure over -- the storm-
        # cuttable subset of the working path, not the whole polyline. The
        # viewer draws this to show exactly which fibre put a service at
        # risk, alongside `service_paths`' full route.
        "cuttable_spans": {s: [[list(point) for point in span]
                               for span in geometry.cuttable_spans[s]]
                          for s in visible if s in geometry.cuttable_spans},
    }


@dataclass(frozen=True)
class EpisodeTrace:
    scenario_id: str
    decider_name: str
    run_index: int
    hours: tuple[dict, ...]
    actions: tuple[Action, ...]
    ledger_debits: tuple[dict, ...]
    spares_remaining: int
    terminal_status: str          # converged | declared_infeasible | hit_cap
    final_routing: dict
    affected_by_hour: dict[str, tuple[str, ...]]
    tool_calls: int
    wall_clock_s: float
    # oms_id -> [src_node_id, dst_node_id], the static optical adjacency the
    # viewer needs to draw a candidate's route from its `new_lightpaths[]
    # .oms_sequence`. Once per episode, from the get_topology(layer="optical")
    # call service_geometry already makes. Defaulted so the positional
    # constructions in tests/eval/test_scoring.py keep working.
    oms_nodes: dict[str, list[str]] = field(default_factory=dict)
    # One record per service `replay.restore_after_cuts` attempted this
    # episode, across every hour with a realized cut -- the same records
    # already folded into each hour's own `record["restorations"]`,
    # flattened here so a reader does not have to walk every hour to see
    # what the deterministic replay did. Defaulted for the same reason as
    # `oms_nodes` above.
    restorations: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {**asdict(self),
                "actions": [asdict(a) for a in self.actions]}


class _CountingClient:
    """Wraps a Client so the trace can report tool_calls without every call
    site threading a counter."""

    def __init__(self, client: Client) -> None:
        self.client = client
        self.calls = 0

    async def call(self, name: str, arguments: dict | None = None, **kw):
        self.calls += 1
        return await call_tool_json(self.client, name, arguments, **kw)


@dataclass(frozen=True)
class ServiceGeometry:
    """One hour's read of where every service physically rides.

    `cuttable_spans` is what build_observation now consumes -- the event
    filter's admitted subset of the working path's real spans, as against
    `points`, the averaged midpoint the exposure arithmetic used to reduce
    each service to (Task 3 of the exposure-and-depot plan). `paths` is the
    full polyline either was derived from, and it can disagree with both in
    ways that matter: D1's SUT sits 58.4 km off a 7.5 km half-width cone --
    outside the polygon -- at a midpoint-model p_cut of 0.976. A viewer
    drawing only paths would make D1 look like a harness bug; one drawing
    only the scored spans would hide why it isn't (run-viewer design, "Draw
    the path, mark the point").

    `points` is kept for the viewer's own trace (`observation_record`'s
    `service_points`) and for `service_points()`'s existing callers
    (derived.py no longer among them; tools/probe_episode.py still is) -- it
    is no longer the exposure input.

    `unmapped_nodes` exists because runner.py's coordinate lookup drops any
    node the LOCAL topology lacks. All 180 edges resolve today, so it is
    normally empty -- but a service whose path is partly unmappable would
    otherwise get a midpoint computed from the mappable half, silently."""
    points: dict[str, tuple[float, float]]
    # service_id -> {"working": [node_id, ...], "protection": [node_id, ...]}
    paths: dict[str, dict[str, list[str]]]
    # oms_id -> [src_node_id, dst_node_id]
    oms_nodes: dict[str, list[str]]
    unmapped_nodes: dict[str, list[str]]
    # service_id -> the WORKING path's spans that this event's filter admits,
    # as ((lat, lon), (lat, lon)) pairs. This is what build_observation scores
    # exposure over. Working path only, matching what `points` has always
    # averaged: storm-svc-1 reads 1 aerial span of 2 here, and 2 of 4 if the
    # protection path were included.
    cuttable_spans: dict[str, tuple[Segment, ...]]
    # service_id -> (src_site, dst_site). The two sites a new lightpath for
    # this service would charge a transponder to, and the key the
    # co-terminating grouping in observation.py buckets on.
    endpoint_sites: dict[str, tuple[str, str]]
    # lightpath_id -> its OMS sequence. oms_seq_by_lp, computed and discarded
    # every hour before this field existed -- exposed so menu_with_path_facts
    # can resolve a candidate's `reused_lightpaths` ids to the OMS it would
    # actually ride, which is what the T1 inert-reroute finding
    # (2026-08-31) shows the menu never told the decider. Defaulted so the
    # hand-built ServiceGeometry in test_runner.py's totals test keeps
    # constructing.
    oms_sequences: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # oms_id -> its span, for every OMS the event's own filter admits --
    # the SAME edge_mount/filter_fn walk `cuttable` above already makes,
    # keyed by OMS instead of by service, so a span this scores and a span
    # `cuttable_spans` scores for a service can never disagree. What
    # menu_with_path_facts sums a candidate's residual exposure over.
    cuttable_span_by_oms: dict[str, Segment] = field(default_factory=dict)
    # service_id -> {"working": (oms_id, ...), "protection": (oms_id, ...)},
    # ordered and NOT deduped (unlike `paths`, which is node ids for the
    # viewer's midpoint) -- the OMS-id walk menu_with_path_facts diffs a
    # candidate's own OMS set against, to say whether a commit actually
    # moves the service or collides with its own protection corridor.
    path_oms: dict[str, dict[str, tuple[str, ...]]] = field(
        default_factory=dict)
    # service_id -> the PROTECTION path's spans that this event's filter
    # admits, keyed only for services WITH a protection path (value may be
    # `()` when that path has no cuttable span). This is what makes a
    # protected service's storm exposure a JOINT both-legs-cut probability
    # (cone.p_cut_service) rather than the working leg's alone -- the
    # protection leg's own geometry was invisible to the model before this
    # field existed.
    protection_cuttable_spans: dict[str, tuple[Segment, ...]] = field(
        default_factory=dict)


async def service_geometry(client: Client, topology_path: str | Path, *,
                           call=None,
                           edges: list[Edge] | None = None) -> ServiceGeometry:
    """Everything the four per-hour reads already know about where services
    ride. `service_points` is this function's `points` field and nothing more;
    the rest was computed and discarded before (run-viewer design, §5.1).

    No new server calls: the same four, in the same order, so a trace's
    `tool_calls` figure stays comparable with every run recorded before this
    change.

    `edges` is an optional pre-loaded `load_edges(topology_path)` result --
    run_episode already loads it once per episode to define risk groups, and
    this function is called once per HOUR, so accepting it here avoids
    re-parsing the topology JSON and rebuilding every Shapely LineString on
    every single hour for no reason. Callers with no edges handy (every
    existing one) leave it None and it is loaded exactly as before."""
    invoke = call if call is not None else partial(call_tool_json, client)
    raw = json.loads(Path(topology_path).read_text(encoding="utf-8-sig"))
    coords = {n["id"]: (n["lat"], n["lon"]) for n in raw["graph"]["nodes"]}
    ip = await invoke("get_topology", {"layer": "ip"})
    optical = await invoke("get_topology", {"layer": "optical"})
    lightpaths = await invoke("get_lightpaths", expect_list=True)
    services = await invoke("get_services")

    oms_by_id = {o["id"]: o for o in optical["oms"]}
    oms_seq_by_lp = {lp["id"]: lp["oms_sequence"] for lp in lightpaths}
    lp_by_link = {lnk["id"]: lnk.get("lightpath_id")
                  for lnk in ip["ip_links"]}
    oms_nodes = {o["id"]: [o["src_node_id"], o["dst_node_id"]]
                 for o in optical["oms"]}

    # The storm's own vulnerability filter, keyed on the LOCAL topology's
    # edges (mount_type lives there, not on the server's OMS records) -- the
    # same table _define_horizon_risk_groups uses, so a fibre this module
    # scores exposure over and a fibre the risk group would name can never
    # disagree.
    filter_fn = get_filter(EVENT_TYPE)
    if edges is None:
        edges = load_edges(topology_path)
    edge_mount: dict[tuple[str, str], Edge] = {}
    for edge in edges:
        edge_mount[(edge.src, edge.dst)] = edge
        edge_mount[(edge.dst, edge.src)] = edge

    def walk(link_ids) -> list[str]:
        # Dedupe before returning: a node shared between two adjacent OMS legs
        # (any interior junction on a multi-hop path) would otherwise be
        # counted once per adjacent leg and skew the representative point
        # toward it -- see task-9-report.md's fix-report section.
        nodes: list[str] = []
        for link_id in link_ids or ():
            lp_id = lp_by_link.get(link_id)
            for oms_id in oms_seq_by_lp.get(lp_id, []):
                oms = oms_by_id[oms_id]
                nodes.extend([oms["src_node_id"], oms["dst_node_id"]])
        return list(dict.fromkeys(nodes))

    def walk_legs(link_ids) -> list[tuple[str, str]]:
        """The ordered OMS legs a path rides, as (src_node, dst_node). The
        node walker above DEDUPES, which is right for a representative point
        and destroys the span structure -- so this is a second pass over the
        same data rather than a projection of the first."""
        legs: list[tuple[str, str]] = []
        for link_id in link_ids or ():
            lp_id = lp_by_link.get(link_id)
            for oms_id in oms_seq_by_lp.get(lp_id, []):
                oms = oms_by_id[oms_id]
                legs.append((oms["src_node_id"], oms["dst_node_id"]))
        return legs

    def walk_oms(link_ids) -> tuple[str, ...]:
        """The ordered OMS ids a path rides -- what path_oms stores, and what
        menu_with_path_facts diffs a candidate's own OMS set against. NOT
        deduped: two legs of one path never share an OMS, unlike the nodes
        `walk` collapses at a shared junction."""
        oms_ids: list[str] = []
        for link_id in link_ids or ():
            lp_id = lp_by_link.get(link_id)
            oms_ids.extend(oms_seq_by_lp.get(lp_id, []))
        return tuple(oms_ids)

    # Every OMS the event's own filter admits, keyed by id -- the SAME
    # edge_mount/filter_fn condition `cuttable` below applies per service, so
    # a span menu_with_path_facts scores for a CANDIDATE and a span scored
    # here for the SERVICE it would replace can never disagree.
    cuttable_span_by_oms: dict[str, Segment] = {}
    for oms_id, oms in oms_by_id.items():
        a, b = oms["src_node_id"], oms["dst_node_id"]
        edge = edge_mount.get((a, b))
        if edge is None or not filter_fn(edge) or a not in coords or b not in coords:
            continue
        cuttable_span_by_oms[oms_id] = (coords[a], coords[b])

    points: dict[str, tuple[float, float]] = {}
    paths: dict[str, dict[str, list[str]]] = {}
    path_oms: dict[str, dict[str, tuple[str, ...]]] = {}
    unmapped: dict[str, list[str]] = {}
    cuttable: dict[str, tuple[Segment, ...]] = {}
    endpoints: dict[str, tuple[str, str]] = {}
    for svc in services["services"]:
        working = walk(svc["working_path"])
        protection = walk(svc.get("protection_path"))
        paths[svc["id"]] = {"working": working, "protection": protection}
        path_oms[svc["id"]] = {
            "working": walk_oms(svc["working_path"]),
            "protection": walk_oms(svc.get("protection_path"))}
        missing = [n for n in dict.fromkeys(working + protection)
                   if n not in coords]
        if missing:
            unmapped[svc["id"]] = missing
        known = [coords[n] for n in working if n in coords]
        if known:
            points[svc["id"]] = (sum(p[0] for p in known) / len(known),
                                 sum(p[1] for p in known) / len(known))

        legs = walk_legs(svc["working_path"])
        spans = tuple(
            (coords[a], coords[b]) for a, b in legs
            if a in coords and b in coords
            and (edge := edge_mount.get((a, b))) is not None
            and filter_fn(edge))
        cuttable[svc["id"]] = spans
        endpoints[svc["id"]] = (svc["src_router"].removeprefix("router_"),
                                svc["dst_router"].removeprefix("router_"))
    # Same OMS-keyed cuttable-span index `cuttable_span_by_oms` above builds,
    # just resolved through each service's PROTECTION leg instead of its
    # working one -- so a span this scores for a service's protection and a
    # span the service-keyed exposure walk scores for its working leg can
    # never disagree. Keyed only for services with a protection path at all.
    protection_cuttable = {
        svc_id: tuple(cuttable_span_by_oms[o]
                      for o in path_oms[svc_id]["protection"]
                      if o in cuttable_span_by_oms)
        for svc_id in path_oms if path_oms[svc_id].get("protection")}

    return ServiceGeometry(points=points, paths=paths, oms_nodes=oms_nodes,
                           unmapped_nodes=unmapped, cuttable_spans=cuttable,
                           endpoint_sites=endpoints,
                           oms_sequences={lp_id: tuple(seq) for lp_id, seq
                                         in oms_seq_by_lp.items()},
                           cuttable_span_by_oms=cuttable_span_by_oms,
                           path_oms=path_oms,
                           protection_cuttable_spans=protection_cuttable)


async def service_points(client: Client, topology_path: str | Path, *,
                         call=None) -> dict[str, tuple[float, float]]:
    """A representative (lat, lon) per service: the midpoint of its working
    path's node coordinates. This is what turns a cone centre into a per-
    service offset, and it is derived from the SERVER's view of the service
    plus the LOCAL topology's coordinates -- the same split the geo mapper
    already uses.

    `call` is an optional async `(name, arguments=None, **kw)` replacement for
    the four tool calls. run_episode passes its `_CountingClient.call` so the
    per-hour re-read shows up in the trace's `tool_calls` instead of happening
    off the books; every other caller (derived.py, tools/probe_episode.py)
    keeps the plain-client behaviour by leaving it None.

    Now a thin projection of `service_geometry`, which keeps the polyline this
    function averages away. The signature and return type are deliberately
    unchanged: derived.py and tools/probe_episode.py call it and neither wants
    the geometry."""
    return (await service_geometry(client, topology_path, call=call)).points


def horizon_risk_group_asset_ids(cone: ConeAtHorizon, damage_radius_km: float,
                                 *, edges: list[Edge], oms: list[dict],
                                 filter_fn) -> list[str]:
    """The fiber asset ids one horizon's risk group names: every `fiber_*`
    element of every OMS whose two endpoints are a local-topology edge that
    the DAMAGE FOOTPRINT (`events.geo.damage_footprint` of this cone's
    centre, width and `damage_radius_km`) touches AND the event's own filter
    admits.

    Extracted from `_define_horizon_risk_groups` so `assertions.
    assert_risk_group_covers_measurable_exposure` can ask what a horizon's
    group WOULD contain without defining it on a server -- which is what
    makes the pre-flight check and the real rollout structurally unable to
    disagree about the answer, the same discipline
    `assert_pair_derived_geometry_is_equal` states for derived geometry.

    `oms` is `get_topology(layer="optical")["oms"]`, passed in rather than
    fetched so the caller can read it once and reuse it across every episode
    sharing a state file.

    Both node orders are matched: the local topology's edge direction and the
    server's OMS direction are independent facts and do agree only by
    accident."""
    # NOT `cone.cone`. That polygon is the track-containment circle -- where
    # the storm CENTRE probably goes -- and intersecting it alone is the
    # 2026-08-31 seam defect: it produced an EMPTY group at every horizon of
    # both T1a issuances while the probability side, which buffers by
    # damage_radius_km in `cone._region_in_sigmas`, read p_cut 0.13 on the
    # same spans. `ConeAtHorizon.cone` stays untouched -- it is scenario-file
    # data and still the right thing for the viewer to draw as the cone.
    footprint = damage_footprint(cone.center["lat"], cone.center["lon"],
                                 cone.width_km, damage_radius_km)
    exposed = map_geo_event_to_assets(footprint, edges, filter_fn)
    pairs = ({(e.src, e.dst) for e in exposed}
             | {(e.dst, e.src) for e in exposed})
    return [a for o in oms
            if (o["src_node_id"], o["dst_node_id"]) in pairs
            for a in o["elements"] if a.startswith("fiber_")]


async def _define_horizon_risk_groups(
    counting: _CountingClient, scenario: ScenarioFile, issuance, *,
    edges: list[Edge], defined: set[str],
) -> dict[str, str]:
    """One risk group per horizon of the current issuance -- the asset lists
    Decision 2 chooses among. Pure GIS on this side (map_geo_event_to_assets),
    define_risk_group on the server's, exactly the seam CLAUDE.md draws.

    define_risk_group rejects a duplicate rg_id, so ids are minted per
    (scenario, issuance, horizon) and re-definition is skipped."""
    filter_fn = get_filter(EVENT_TYPE)
    topo = await counting.call("get_topology", {"layer": "optical"})
    rg_ids: dict[str, str] = {}
    for horizon, cone in issuance.horizons.items():
        rg_id = f"rg_{scenario.id}_{issuance.issued_at}_{horizon}"
        rg_ids[horizon] = rg_id
        if rg_id in defined:
            continue
        fiber_ids = horizon_risk_group_asset_ids(
            cone, scenario.damage_radius_km, edges=edges, oms=topo["oms"],
            filter_fn=filter_fn)
        await counting.call("define_risk_group", {
            "rg_id": rg_id, "asset_ids": fiber_ids,
            "metadata": {"event_type": EVENT_TYPE, "scenario": scenario.id,
                         "issued_at": issuance.issued_at, "horizon": horizon}})
        defined.add(rg_id)
    return rg_ids


def _only_transient_findings_about(violations, service_id: str) -> bool:
    """Every reported violation is `transient` AND names `service_id` itself.

    `transient` is the server's own word for "present in an INTERMEDIATE
    state of the plan and gone by its final state" (multilayer_optical_
    network's `validate.Violation.transient`). A violation of that shape,
    about the very service being restored, is not damage the plan causes: it
    is the outage the plan is repairing, still visible in the window between
    the plan's `provision_lightpath` op and its closing `reroute_service`
    op."""
    return bool(violations) and all(
        v.get("transient") and v.get("asset_id") == service_id
        for v in violations)


async def try_commit(counting, index, candidate: dict, service_id: str, *,
                     prefix: str, basis: str, level: str,
                     split_transient_outage: bool = False,
                     ) -> tuple[dict | None, dict | None]:
    """validate_plan + commit_plan for one candidate under baseline="standing".
    Returns (commit_result, None) on success or (None, rejection) -- the same
    rejection dicts run_episode has always recorded.

    `split_transient_outage` (default False -- the hourly decision loop's
    behaviour is unchanged) commits the plan as TWO plans, provisioning
    first and the closing `reroute_service` second, when and only when the
    single combined plan is refused solely for `_only_transient_findings_
    about(…, service_id)`.

    **Why this exists** (found live, 2026-09-05, Task 15). `baseline=
    "standing"` moves a violation the standing state already carries into
    `pre_existing` -- but only a NON-transient one (`validate.py`: `if not
    v.transient and (v.type, v.asset_id) in standing_keys`). A plan that
    restores an ALREADY-DROPPED service is always at least two ops
    (`plans.plan_from_candidate`: one or more `provision_lightpath`, then
    `reroute_service`), so the outage is still present after op 0 and gone
    at the final state -- i.e. tagged transient, kept out of `pre_existing`,
    and `ValidationReport.ok` is `not self.violations`, so the plan is
    refused. `commit_plan` re-validates and refuses for the same reason, so
    there is no "commit anyway". The consequence is structural, not
    scenario-specific: WITHOUT this split, `replay.restore_after_cuts` can
    never restore anything, in any episode -- confirmed against the real
    server, where the T1 pair's two gold rollouts came back at a byte-equal
    7800.0 Gbps-h and the enumerator could not separate spend from hold at
    all. Splitting the same ops into two commits validates and commits
    cleanly (also confirmed live): part 1 leaves the outage present at its
    own final state, so it IS non-transient there and IS `pre_existing`;
    part 2 is a single op after which the service is up, so it reports
    nothing.

    The split is not free of risk and is deliberately opt-in: if part 1
    commits and part 2 is then refused, the network keeps a provisioned but
    unused lightpath (spectrum consumed) and the caller sees a rejection
    with no ledger debit. That is reported as an ordinary rejection."""
    try:
        plan = plan_from_candidate(index, candidate, service_id, prefix=prefix)
    except PlanTranslationError as exc:
        return None, {"type": "plan_translation_failed", "message": str(exc)}

    async def _validate_and_commit(one_plan: dict) -> tuple[dict | None, dict | None]:
        report = await counting.call("validate_plan", {
            "plan": one_plan, "basis": basis, "level": level,
            "baseline": "standing"})
        if not report["ok"]:
            return None, {"type": "validation_violations",
                          "violations": report["violations"]}
        commit = await counting.call("commit_plan", {
            "plan": one_plan, "dry_run": False, "confirm": True,
            "basis": basis, "level": level, "baseline": "standing"})
        if commit["status"] != "committed":
            return None, {"type": "commit_" + commit["status"],
                          "validation": commit.get("validation")}
        return commit, None

    commit, rejection = await _validate_and_commit(plan)
    if rejection is None:
        return commit, None
    if not (split_transient_outage
            and rejection["type"] == "validation_violations"
            and len(plan["ops"]) > 1
            and _only_transient_findings_about(rejection["violations"],
                                               service_id)):
        return None, rejection

    _head, head_rejection = await _validate_and_commit({"ops": plan["ops"][:-1]})
    if head_rejection is not None:
        return None, head_rejection
    return await _validate_and_commit({"ops": plan["ops"][-1:]})


async def run_episode(
    client: Client, scenario: ScenarioFile, decider: Decider, *,
    topology_path: str | Path, run_index: int = 0,
    trace_path: str | Path | None = None,
) -> EpisodeTrace:
    """One rollout of one episode by one decider. The server must already be
    connected against `scenario.state_file`."""
    started = time.monotonic()
    counting = _CountingClient(client)
    edges = load_edges(topology_path)
    # oms_nodes is filled in from the first hour's geometry below (it is the
    # static optical adjacency, unchanged across the episode) -- constructing
    # it here would cost an extra, uncounted-or-miscounted server call before
    # the per-hour loop even starts.
    ledger = SpareLedger(inventory=dict(scenario.spare_inventory),
                         depot_site=scenario.depot_site, oms_nodes={})
    defined_rgs: set[str] = set()

    hours: list[dict] = []
    actions: list[Action] = []
    affected_by_hour: dict[str, tuple[str, ...]] = {}
    all_restorations: list[dict] = []
    terminal_status = "converged"

    for hour_index, hour in enumerate(scenario.hours):
        record: dict = {"hour": hour, "iterations": [], "rejections": []}
        # Re-read EVERY hour, not once per episode. plans.py's only op is
        # reroute_service, so a commit really does move the working path on
        # the server -- and a point computed before the loop describes a
        # corridor the service may no longer ride. That is what told one real
        # run that the lightpath it had committed an hour earlier, precisely
        # to escape a cone, was still inside it (remediation spec, F4).
        # Costs four tool calls an hour, counted through `counting`.
        geometry = await service_geometry(client, topology_path,
                                          call=counting.call, edges=edges)
        # Static across the episode; re-assigning it every hour is a no-op
        # in practice and keeps the ledger's oms_nodes from ever going stale
        # without adding a server call of its own.
        ledger.oms_nodes = geometry.oms_nodes
        # ClaudeDecider needs the same static OMS adjacency to resolve a
        # candidate's new_lightpaths to endpoint sites in _menu_for_prompt
        # (see agent.py's ClaudeDecider.__init__) -- suite.build_deciders()
        # constructs it with oms_nodes={} and nothing else ever sets it,
        # which crashes the first real costed candidate with a ValueError
        # (whole-branch final review, finding 1). Baselines have no such
        # attribute, hence the guard.
        if hasattr(decider, "oms_nodes"):
            decider.oms_nodes = geometry.oms_nodes
        services = tuple((await counting.call("get_services"))["services"])
        # For menu_with_path_facts' residual_exposure -- the SUT's own
        # demand, already on this hour's roster fetch, so no extra call.
        sut_demand_gbps = next(
            s["demand_gbps"] for s in services
            if s["id"] == scenario.service_under_test)
        # The roster scoring reads to know who could have survived -- the
        # final simulate_ip_routing reports links, not services.
        record["services"] = [s["id"] for s in services]
        # scoring.gbps_hours_lost's demand input -- the SAME roster fetch
        # above, so a service's charged demand and the one scoring reads can
        # never disagree. record["services"] only ever carried ids; nothing
        # before this needed the per-service Gbps figure at hour-record
        # granularity.
        record["demands"] = {s["id"]: s["demand_gbps"] for s in services}
        obs_kwargs = dict(service_spans=geometry.cuttable_spans,
                          services=services,
                          spares_on_hand=ledger.on_hand,
                          actions_taken=action_payloads(
                              actions, hours=scenario.hours),
                          spares_spent=ledger.spent,
                          endpoint_sites=geometry.endpoint_sites,
                          depot_site=scenario.depot_site,
                          protection_spans=geometry.protection_cuttable_spans)

        issuance = latest_issuance(scenario, hour)
        rg_ids = await _define_horizon_risk_groups(
            counting, scenario, issuance, edges=edges, defined=defined_rgs)

        obs = build_observation(scenario, hour, risk_group_ids=rg_ids,
                                **obs_kwargs)
        record.update(observation_record(obs, geometry))
        timing = decider.timing(obs)
        record["timing"] = timing.to_dict()
        # What the DECIDER was shown, as against the ground truth above. Read
        # off the decider rather than recomputed: the projection is
        # decider-owned and a recomputation would silently diverge the day it
        # changes. Baselines have none and honestly record null.
        record["projected"] = getattr(decider, "last_projection", None)

        if timing.action == "act":
            index = await build_topology_index(client)
            # What EXISTS before decision 2 narrows it. One call per acting
            # hour, reused across every iteration -- it does not depend on
            # the decider's answer. Derived from ConstraintDecision's own
            # defaults (avoid={}, the working posture) rather than hand-
            # copied as a literal, so a future change to those defaults
            # cannot silently desync this probe from the posture the agent
            # actually uses (whole-branch review finding I5 was exactly this
            # trap, fixed in decisions.py; this call site had reintroduced
            # it). Counted, unlike build_topology_index above it, so the
            # trace's tool_calls stays honest (remediation spec, W3.3).
            probe_args = ConstraintDecision(
                avoid={}, reasoning="unconstrained probe"
            ).route_service_args(scenario.service_under_test)
            probe = unconstrained_menu_projection(
                await counting.call("route_service", probe_args),
                geometry.oms_nodes)
            record["unconstrained_menu"] = probe
            last_rejection: dict | None = None
            committed = False
            for iteration in range(MAX_ITERATIONS):
                obs = build_observation(
                    scenario, hour, risk_group_ids=rg_ids, iteration=iteration,
                    last_rejection=last_rejection,
                    **{**obs_kwargs, "spares_on_hand": ledger.on_hand,
                       "spares_spent": ledger.spent,
                       "actions_taken": action_payloads(
                           actions, hours=scenario.hours)})
                constraints = decider.constraints(obs, probe)
                menu = await counting.call(
                    "route_service",
                    constraints.route_service_args(scenario.service_under_test))
                # T1 inert-reroute finding (2026-08-31): annotate BEFORE
                # decider.objective() sees it, so both the decider's own
                # prompt (via its internal menu_for_prompt) and the trace's
                # recorded menu below pick up the same facts for free.
                menu = menu_with_path_facts(
                    menu, geometry, scenario.service_under_test,
                    issuance=issuance, damage_radius_km=scenario.damage_radius_km,
                    demand_gbps=sut_demand_gbps)
                choice = decider.objective(obs, menu)
                step = {"iteration": iteration,
                        "constraints": constraints.to_dict(),
                        "menu_status": menu.get("status"),
                        "menu_size": len(menu.get("candidates") or []),
                        # The full candidate list decision 3 weighed -- cost
                        # vector, restored/shortfall, and the OMS sequences
                        # the viewer draws a route from. Recorded through the
                        # SAME function the prompt uses (menu_for_prompt), so
                        # the two cannot disagree.
                        "menu": menu_for_prompt(menu, geometry.oms_nodes),
                        "projected": getattr(decider, "last_projection",
                                             None),
                        "objective": choice.to_dict()}
                record["iterations"].append(step)

                idx = candidate_index(choice.choice)
                if idx is None:
                    # NOT terminal. "My constraints left me nothing
                    # acceptable" is a correctable mistake of exactly the
                    # shape last_rejection exists for: decisions.py says
                    # decision 2 changes which candidates EXIST, so the fix
                    # is to loosen `avoid` and look again. Ending the hour
                    # here left no path from the declaration back to the
                    # correction (remediation spec, W2.3).
                    last_rejection = {
                        "type": "declared_infeasible",
                        "menu_size": len(menu.get("candidates") or [])}
                    record["rejections"].append(last_rejection)
                    step["outcome"] = "declared_infeasible"
                    continue
                candidates = menu.get("candidates") or []
                if idx >= len(candidates):
                    last_rejection = {"type": "invalid_choice",
                                      "choice": choice.choice,
                                      "menu_size": len(candidates)}
                    record["rejections"].append(last_rejection)
                    step["outcome"] = "invalid_choice"
                    continue
                candidate = candidates[idx]

                if not ledger.can_afford(candidate):
                    last_rejection = ledger.rejection(candidate)
                    record["rejections"].append(last_rejection)
                    step["outcome"] = "insufficient_spares"
                    continue

                commit, rejection = await try_commit(
                    counting, index, candidate, scenario.service_under_test,
                    prefix=f"eval-{hour}-{iteration}",
                    basis=constraints.basis, level=constraints.level)
                if rejection is not None:
                    last_rejection = rejection
                    record["rejections"].append(last_rejection)
                    if rejection["type"] == "validation_violations":
                        step["outcome"] = "rejected"
                    elif rejection["type"].startswith("commit_"):
                        step["outcome"] = rejection["type"][len("commit_"):]
                    else:
                        step["outcome"] = rejection["type"]
                    continue

                spares = ledger.debit(candidate, hour=hour,
                                     service_id=scenario.service_under_test)
                lead = lead_time_hours_for(candidate["lever"],
                                           scenario.lead_time_hours)
                actions.append(Action(
                    hour=hour, hour_index=hour_index,
                    lever=candidate["lever"],
                    effective_at_index=hour_index + lead, spares=spares,
                    service_id=scenario.service_under_test,
                    avoid=dict(constraints.avoid)))
                step["outcome"] = "committed"
                # scoring.decision_label's chosen_lever rule reads this.
                step["lever"] = candidate["lever"]
                step["intended_snapshot_id"] = commit["intended_snapshot_id"]
                # T1 inert-commit finding: a candidate that neither moves the
                # working path nor spends a spare is a free no-op the harness
                # was previously indistinguishable from a real action --
                # scoring's timing_effective (below) and inert_commits both
                # read this.
                step["inert"] = (
                    not candidate["path_delta"]["changes_working_path"]
                    and not spares)
                committed = True
                break
            else:
                # The cap is only "declared_infeasible" when the LAST word was
                # a declaration; a loop that spun on invalid choices or failed
                # validations is still hit_cap.
                last_outcome = (record["iterations"][-1].get("outcome")
                                if record["iterations"] else None)
                terminal_status = ("declared_infeasible"
                                   if last_outcome == "declared_infeasible"
                                   else "hit_cap")
            record["committed"] = committed

        # Every hour, act or wait: "act" iff a NON-inert commit actually
        # happened. A wait hour's `iterations` is always [], so this reads
        # "wait" for free without a separate branch -- and an act hour whose
        # only commit was inert (T1's free reuse-of-own-path case) is scored
        # as though it had waited, which is the whole point of this field.
        record["timing_effective"] = "act" if any(
            s.get("outcome") == "committed" and not s.get("inert")
            for s in record["iterations"]) else "wait"

        cuts = scenario.realized.get(hour, ())
        if cuts:
            await counting.call("inject_failure", {"asset_ids": list(cuts)})
            affected: set[str] = set()
            for asset_id in cuts:
                result = await counting.call("get_affected_services",
                                             {"asset_id": asset_id})
                affected.update(result["services"])
            affected_by_hour[hour] = tuple(sorted(affected))
            # snapshot_create, never snapshot_branch -- a branch id names the
            # PRE-mutation state and would silently score an unfailed network.
            record["snapshot_id"] = (
                await counting.call("snapshot_create"))["id"]
            # What the cut actually dropped, read from the SAME tool the
            # episode's own final scoring call uses (simulate_ip_routing),
            # right after inject_failure and before the replay below -- so
            # "attempted only if actually dropped" (replay.py) is answered
            # from ground truth, not from get_affected_services' asset-
            # adjacency guess.
            routing = await counting.call("simulate_ip_routing")
            record["dropped_after_cut"] = sorted(
                d["service_id"] for d in routing["dropped"]["services"])
            # The deterministic post-cut restoration replay (T1 spend-or-
            # hold redesign spec, 5.1): spend whatever spares remain on the
            # services this cut actually dropped, in the DECIDER's own
            # `claim_priority` order (last-stated `timing` this hour --
            # `record["timing"]` is always set above, act or wait).
            priority = tuple(record["timing"].get("claim_priority", ()))
            restore_actions, restore_records = await restore_after_cuts(
                counting, scenario=scenario, hour=hour, hour_index=hour_index,
                affected=record["dropped_after_cut"], priority=priority,
                ledger=ledger, geometry=geometry, issuance=issuance,
                rg_for_cut_hour=rg_ids.get(hour),
                index_factory=lambda: build_topology_index(client))
            actions.extend(restore_actions)
            record["restorations"] = restore_records
            all_restorations.extend(restore_records)
        hours.append(record)

    final_routing = await counting.call("simulate_ip_routing")
    trace = EpisodeTrace(
        scenario_id=scenario.id, decider_name=decider.name,
        run_index=run_index, hours=tuple(hours), actions=tuple(actions),
        ledger_debits=tuple(ledger.debits), spares_remaining=ledger.on_hand,
        terminal_status=terminal_status, final_routing=final_routing,
        affected_by_hour=affected_by_hour, tool_calls=counting.calls,
        wall_clock_s=round(time.monotonic() - started, 3),
        oms_nodes=geometry.oms_nodes, restorations=tuple(all_restorations))

    if trace_path is not None:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace.to_dict(), indent=2, default=str),
                        encoding="utf-8")
    return trace
