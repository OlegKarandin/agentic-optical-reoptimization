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
from ..geo_mapper import Edge, load_edges, map_geo_event_to_assets
from ..mcp_client import call_tool_json
from .decisions import Decider, candidate_index
from .ledger import SpareLedger
from .observation import build_observation, latest_issuance, lead_time_hours_for
from .plans import PlanTranslationError, build_topology_index, plan_from_candidate
from .scenario_file import ScenarioFile

MAX_ITERATIONS = 5


@dataclass(frozen=True)
class Action:
    hour: str
    hour_index: int
    lever: str
    effective_at_index: int
    pairs: int
    service_id: str
    # The avoid set this candidate was routed under -- the one fact that
    # makes "I already routed around that risk group" checkable by the
    # decider an hour later. Defaulted so the positional constructions in
    # tests/eval/test_scoring.py keep working; run_episode always supplies
    # it. A dict field makes Action unhashable in practice; nothing in the
    # harness hashes one.
    avoid: dict = field(default_factory=dict)


def action_payloads(actions) -> tuple[dict, ...]:
    """What the decider is told about its OWN committed actions this episode.
    Plain dicts, so observation.py never has to import this module (the
    dependency runs one way) and the whole Observation stays
    JSON-serializable for the trace and the prompt."""
    return tuple({"hour": a.hour, "lever": a.lever, "pairs": a.pairs,
                  "avoid": a.avoid,
                  "effective_at_index": a.effective_at_index}
                 for a in actions)


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
    keeps the plain-client behaviour by leaving it None."""
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

    points: dict[str, tuple[float, float]] = {}
    for svc in services["services"]:
        nodes: list[str] = []
        for link_id in svc["working_path"]:
            lp_id = lp_by_link.get(link_id)
            for oms_id in oms_seq_by_lp.get(lp_id, []):
                oms = oms_by_id[oms_id]
                nodes.extend([oms["src_node_id"], oms["dst_node_id"]])
        # Dedupe before averaging: a node shared between two adjacent OMS
        # legs (any interior junction on a multi-hop path) would otherwise
        # be counted once per adjacent leg and skew the representative
        # point toward it -- see task-9-report.md's fix-report section.
        nodes = list(dict.fromkeys(nodes))
        known = [coords[n] for n in nodes if n in coords]
        if not known:
            continue
        points[svc["id"]] = (sum(p[0] for p in known) / len(known),
                             sum(p[1] for p in known) / len(known))
    return points


async def _define_horizon_risk_groups(
    counting: _CountingClient, scenario: ScenarioFile, issuance, *,
    edges: list[Edge], defined: set[str],
) -> dict[str, str]:
    """One risk group per horizon of the current issuance -- the asset lists
    Decision 2 chooses among. Pure GIS on this side (map_geo_event_to_assets),
    define_risk_group on the server's, exactly the seam CLAUDE.md draws.

    define_risk_group rejects a duplicate rg_id, so ids are minted per
    (scenario, issuance, horizon) and re-definition is skipped."""
    filter_fn = get_filter("storm")
    topo = await counting.call("get_topology", {"layer": "optical"})
    rg_ids: dict[str, str] = {}
    for horizon, cone in issuance.horizons.items():
        rg_id = f"rg_{scenario.id}_{issuance.issued_at}_{horizon}"
        rg_ids[horizon] = rg_id
        if rg_id in defined:
            continue
        exposed = map_geo_event_to_assets(cone.cone, edges, filter_fn)
        pairs = ({(e.src, e.dst) for e in exposed}
                 | {(e.dst, e.src) for e in exposed})
        fiber_ids = [a for oms in topo["oms"]
                     if (oms["src_node_id"], oms["dst_node_id"]) in pairs
                     for a in oms["elements"] if a.startswith("fiber_")]
        await counting.call("define_risk_group", {
            "rg_id": rg_id, "asset_ids": fiber_ids,
            "metadata": {"event_type": "storm", "scenario": scenario.id,
                         "issued_at": issuance.issued_at, "horizon": horizon}})
        defined.add(rg_id)
    return rg_ids


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
    ledger = SpareLedger(on_hand=scenario.spares_on_hand)
    defined_rgs: set[str] = set()

    hours: list[dict] = []
    actions: list[Action] = []
    affected_by_hour: dict[str, tuple[str, ...]] = {}
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
        points = await service_points(client, topology_path,
                                      call=counting.call)
        services = tuple((await counting.call("get_services"))["services"])
        # The roster scoring reads to know who could have survived -- the
        # final simulate_ip_routing reports links, not services.
        record["services"] = [s["id"] for s in services]
        obs_kwargs = dict(service_points=points, services=services,
                          spares_on_hand=ledger.on_hand,
                          actions_taken=action_payloads(actions),
                          spares_spent=ledger.spent)

        issuance = latest_issuance(scenario, hour)
        rg_ids = await _define_horizon_risk_groups(
            counting, scenario, issuance, edges=edges, defined=defined_rgs)

        obs = build_observation(scenario, hour, risk_group_ids=rg_ids,
                                **obs_kwargs)
        timing = decider.timing(obs)
        record["timing"] = timing.to_dict()

        if timing.action == "act":
            index = await build_topology_index(client)
            last_rejection: dict | None = None
            committed = False
            for iteration in range(MAX_ITERATIONS):
                obs = build_observation(
                    scenario, hour, risk_group_ids=rg_ids, iteration=iteration,
                    last_rejection=last_rejection,
                    **{**obs_kwargs, "spares_on_hand": ledger.on_hand,
                       "spares_spent": ledger.spent,
                       "actions_taken": action_payloads(actions)})
                constraints = decider.constraints(obs)
                menu = await counting.call(
                    "route_service",
                    constraints.route_service_args(scenario.service_under_test))
                choice = decider.objective(obs, menu)
                step = {"iteration": iteration,
                        "constraints": constraints.to_dict(),
                        "menu_status": menu.get("status"),
                        "menu_size": len(menu.get("candidates") or []),
                        "objective": choice.to_dict()}
                record["iterations"].append(step)

                idx = candidate_index(choice.choice)
                if idx is None:
                    terminal_status = "declared_infeasible"
                    step["outcome"] = "declared_infeasible"
                    break
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

                try:
                    plan = plan_from_candidate(
                        index, candidate, scenario.service_under_test,
                        prefix=f"eval-{hour}-{iteration}")
                except PlanTranslationError as exc:
                    last_rejection = {"type": "plan_translation_failed",
                                      "message": str(exc)}
                    record["rejections"].append(last_rejection)
                    step["outcome"] = "plan_translation_failed"
                    continue

                report = await counting.call("validate_plan", {
                    "plan": plan, "basis": constraints.basis,
                    "level": constraints.level})
                if not report["ok"]:
                    last_rejection = {"type": "validation_violations",
                                      "violations": report["violations"]}
                    record["rejections"].append(last_rejection)
                    step["outcome"] = "rejected"
                    continue

                commit = await counting.call("commit_plan", {
                    "plan": plan, "dry_run": False, "confirm": True,
                    "basis": constraints.basis, "level": constraints.level})
                if commit["status"] != "committed":
                    last_rejection = {"type": "commit_" + commit["status"],
                                      "validation": commit.get("validation")}
                    record["rejections"].append(last_rejection)
                    step["outcome"] = commit["status"]
                    continue

                pairs = ledger.debit(candidate, hour=hour,
                                     service_id=scenario.service_under_test)
                lead = lead_time_hours_for(candidate["lever"],
                                           scenario.lead_time_hours)
                actions.append(Action(
                    hour=hour, hour_index=hour_index,
                    lever=candidate["lever"],
                    effective_at_index=hour_index + lead, pairs=pairs,
                    service_id=scenario.service_under_test,
                    avoid=dict(constraints.avoid)))
                step["outcome"] = "committed"
                # scoring.decision_label's chosen_lever rule reads this.
                step["lever"] = candidate["lever"]
                step["intended_snapshot_id"] = commit["intended_snapshot_id"]
                committed = True
                break
            else:
                terminal_status = "hit_cap"
            record["committed"] = committed

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
        hours.append(record)

    final_routing = await counting.call("simulate_ip_routing")
    trace = EpisodeTrace(
        scenario_id=scenario.id, decider_name=decider.name,
        run_index=run_index, hours=tuple(hours), actions=tuple(actions),
        ledger_debits=tuple(ledger.debits), spares_remaining=ledger.on_hand,
        terminal_status=terminal_status, final_routing=final_routing,
        affected_by_hour=affected_by_hour, tool_calls=counting.calls,
        wall_clock_s=round(time.monotonic() - started, 3))

    if trace_path is not None:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace.to_dict(), indent=2, default=str),
                        encoding="utf-8")
    return trace
