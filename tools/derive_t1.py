# tools/derive_t1.py
"""Task 14 (T1 spend-or-hold redesign plan): the tool that actually WRITES
the T1 pair's two scenario YAMLs (T1a.yaml -- "conserve"/"hold" -- and
T1b.yaml -- "spend") once Task 15 has chosen a site. Parametrised end to
end (SUT, depot, claimants, escape node, hours, bearings/radii) so
re-deriving with a different site or a retuned bearing is one command, not
a hand edit -- the same reason `tools/derive_episodes.py` exists for the
older exposure-and-depot plan's three pairs.

**Generalised (T2/T3 probe redesign plan, Task 9) into a parametrised pair
deriver any pair can use** -- SUT posture (protected/unprotected), per-half
cut specs (`--half-a-cuts`/`--half-b-cuts`, tokens `claimants`/
`claimant:<id>`/`sut`), alt-span damage-footprint containment assertions
(`--alt-span`), a bearing bisection for a two-corridor equality
(`--pcut-match-claimant`), and the derived-geometry/flip-scalar/probe-flip
gates now run IN-TOOL rather than only by the later test suite. `tools/
derive_t2.py` and `tools/derive_t3.py` are thin default-supplying wrappers
around this same `derive()`/`_build_parser()`. Every T1-specific behaviour
below (posture=protected, half A cuts the claimants, half B cuts the SUT's
first hops, no alt-span, no probe-flip declaration) is exactly this
generaliser's DEFAULT shape, so running this tool with T1's own arguments is
unchanged.

**The construction, per the redesign spec**
(docs/superpowers/specs/2026-09-05-t1-spend-or-hold-redesign.md, S4):

  * `origin` is the DEPOT SITE's own coordinates -- NOT a service's averaged
    path point (`runner.ServiceGeometry.points`) the way the older
    `derive_episodes.py` used for its three pairs. Both of the SUT's
    legs, and the claimant corridor, all physically LEAVE the depot, so the
    depot's own point is the one frame every bearing/radius below is
    measured from.
  * `t0` cone: one horizon (`decision_hour + lead_time` hours out, e.g. t3
    at decision_hour=t1/lead_time=2), placed at `(t0_bearing, t0_radius)`
    from the origin. BYTE-IDENTICAL in both halves -- the "baseline tie" at
    the first issuance.
  * Half A ("conserve"/"hold"): the t1-issuance revision resolves TOWARD the
    claimant corridor, at `(toward_bearing, toward_radius)` -- CHOSEN, not
    solved.
  * Half B ("spend"): the t1-issuance revision resolves AWAY from the
    claimant corridor, at `away_bearing` (or, when `--pcut-match-claimant`
    is given, a bearing SOLVED over a range so a second corridor's own
    p_cut matches the first's exactly), with its RADIUS SOLVED (via
    `derive_episodes.solve_radius_for_target_pcut`, generalised this same
    task to accept a `pcut_fn` callable) so the SUT's own JOINT
    `cone.p_cut_service` (working AND protection when protected, working
    alone when not) matches half A's to below 1e-9 -- comfortably inside
    `derived.DERIVED_TOLERANCE` (1e-3). Only the claimant corridor's own
    exposure is left to differ.

**Realized cuts, resolved live, never typed** (`get_topology(layer=
"optical")`'s own `oms[].elements`, filtered to the `fiber_*` prefix,
matching `runner.horizon_risk_group_asset_ids`' own convention), now
expanded from each half's `--half-a-cuts`/`--half-b-cuts` cut spec by
`_realized_for` rather than hardcoded:

  * T1's own default half A: every claimant service's own WORKING-path OMS
    sequence (`ServiceGeometry.path_oms[claimant]["working"]`) -- both
    claimant ids given on `--claimants`, so both directions of the corridor
    go down.
  * T1's own default half B: the SUT's own working- and protection-path
    FIRST HOP (`path_oms[sut]["working"][0]` / `["protection"][0]` -- the
    span leaving the depot, matching S4.1's "leaves its home site on aerial
    spans" framing) -- cutting only the near span is enough to take the
    whole leg down and is what makes this the minimal, honest realization
    of "both legs cut".

**Gold is a placeholder.** `tools/compute_gold.py` is the real enumerator
(S4.6); this tool only needs a syntactically valid `Gold` so the file
round-trips through `scenario_file.load_scenario` before that task exists.
`gold_spare_action` (A=conserve, B=spend) is likewise provisional -- the
qualitative expectation the construction is aimed at, not a computed
answer.

Runs in THIS repo's env (storm-reoptimizer) against a LIVE
multilayer-optical-mcp server, same seam every tool in this directory
observes (imports nothing from multilayer_optical_network; talks to the
model only through mcp_client's stdio Client).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# tools/ is not a package (find_sut.py's own module docstring) -- this repo's
# convention for cross-tool reuse is a plain sibling import, which resolves
# because Python puts the RUNNING SCRIPT's own directory at sys.path[0].
# tests/eval/test_derive_t1.py inserts the `tools/` directory by hand before
# `import derive_t1`, matching test_find_sut.py's own precedent.
import derive_episodes as de  # noqa: E402

from storm_reoptimizer.eval.assertions import (          # noqa: E402
    _derived_mismatches, probe_flip_mismatches,
)
from storm_reoptimizer.eval.cone import (               # noqa: E402
    Segment, nearest_span_offset_km, p_cut_region, p_cut_service,
)
from storm_reoptimizer.eval.decisions import ConstraintDecision  # noqa: E402
from storm_reoptimizer.eval.derived import (             # noqa: E402
    FLIP_VARS, derived_geometry_from_spans, flip_scalars_from_spans,
)
from storm_reoptimizer.eval.observation import build_observation  # noqa: E402
from storm_reoptimizer.eval.probe import answer_probe    # noqa: E402
from storm_reoptimizer.eval.runner import (              # noqa: E402
    EVENT_TYPE, horizon_risk_group_asset_ids, service_geometry,
)
from storm_reoptimizer.eval.scenario_file import (       # noqa: E402
    ConeAtHorizon, Gold, Issuance, ScenarioFile, dump_scenario,
)
from storm_reoptimizer.events.filters import get_filter  # noqa: E402
from storm_reoptimizer.events.geo import (               # noqa: E402
    damage_footprint, damage_footprint_radius_km,
)
from storm_reoptimizer.geo_mapper import (               # noqa: E402
    load_edges, map_geo_event_to_assets,
)
from storm_reoptimizer.mcp_client import call_tool_json, connect_server  # noqa: E402

DEFAULT_TOPOLOGY = de.DEFAULT_TOPOLOGY
DEFAULT_STATE = de.DEFAULT_STATE
DEFAULT_SERVER_COMMAND = de.DEFAULT_SERVER_COMMAND

# Placeholder gold (mirrors derive_episodes.py's own `_DUMMY_GOLD` -- same
# shape, same "not gold" intent), pending the real enumerator
# (tools/compute_gold.py, spec S4.6). `decision_at_t0="wait"` is the one
# non-arbitrary part of the placeholder: the spec states plainly (S4.4) that
# at t0 the revision has not arrived yet, so waiting is correct BY
# INSPECTION in both halves, independent of what the enumerator later finds
# for t1.
_PLACEHOLDER_GOLD = Gold(
    survived=(), max_spares_wasted=0, decision_at_t0="wait", label="n/a",
    rationale="PROVISIONAL -- run tools/compute_gold.py --write")

# First-cut citation tokens for `scoring.cites_flip_variable_frac` (matches
# rationale using words for the NEW spend/hold decision this pair grades,
# analogous in spirit to the older pairs' own `flip_variable` lists) --
# provisional, like `gold` above: nothing specifies the final vocabulary,
# and a later task (or a human, alongside the real gold rationale) is free
# to retune it. `escape` was dropped from the SHIPPED T1a.yaml/T1b.yaml
# (transponder-pairing spec, 2026-09-21 -- it never appears anywhere in
# the observation payload the model reads) but is deliberately left here
# too, dropped, so re-running this generator does not silently reintroduce it.
FLIP_VARIABLE_TOKENS = ("spend", "hold", "claimant")


def _node_coords(topology_path: str | Path) -> dict[str, tuple[float, float]]:
    """node_id -> (lat, lon). Same two-line convention runner.service_
    geometry's own `coords` local and find_sut.py's own `_node_coords` use;
    kept local here for the same reason find_sut.py keeps its own copy --
    tools/ is not a package, so nothing here is imported from there."""
    data = json.loads(Path(topology_path).read_text(encoding="utf-8-sig"))
    return {n["id"]: (n["lat"], n["lon"]) for n in data["graph"]["nodes"]}


def _vulnerable_pairs(edges, filter_fn) -> set[tuple[str, str]]:
    """Every (node, node) pair, BOTH orders, whose local-topology edge the
    event's own vulnerability filter admits -- the same both-orders matching
    `runner.horizon_risk_group_asset_ids` does, because an OMS's direction
    and the topology edge's direction are independent facts."""
    admitted = [e for e in edges if filter_fn(e)]
    return ({(e.src, e.dst) for e in admitted}
            | {(e.dst, e.src) for e in admitted})


def _fiber_ids(oms_by_id: dict, oms_id: str, *,
               vulnerable_pairs: set[tuple[str, str]] | None = None) -> list[str]:
    """Every `fiber_*` element of one OMS -- the SAME filter `runner.
    horizon_risk_group_asset_ids` applies, just keyed by a single OMS id
    instead of walking every OMS a hazard footprint touches.

    `vulnerable_pairs` (from `_vulnerable_pairs`) additionally drops an OMS
    the EVENT FILTER does not admit at all, returning `[]` for it. Without
    it this function will happily name a BURIED fibre as a storm's realized
    cut whenever a service's optical route happens to run over one -- which
    `assertions.assert_realized_cuts_pass_the_event_filter` (invariant 1)
    rejects outright, and which is physically wrong besides. Found for real
    on 2026-09-05 (Task 15): the T1 claimant pair's own optical route is
    `jalgaon -> dhulia -> nasik -> mumbai -> nasik -> dhulia`, of which only
    the first hop is aerial; the unfiltered version of this function emitted
    all five OMS's fibres as the storm's cut. Defaulted to None (no
    filtering) so nothing else that calls this changes behaviour."""
    if (vulnerable_pairs is not None
            and (oms_by_id[oms_id]["src_node_id"],
                 oms_by_id[oms_id]["dst_node_id"]) not in vulnerable_pairs):
        return []
    return sorted(e for e in oms_by_id[oms_id]["elements"]
                 if e.startswith("fiber_"))


def _elapsed_hours(hours: tuple[str, ...], a: str, b: str) -> int:
    return hours.index(b) - hours.index(a)


def _cut_tokens(spec: str) -> list[str]:
    return [t.strip() for t in spec.split(",") if t.strip()]


def _parse_alt_span(text: str) -> tuple[str, str, str, str]:
    """`A:B:A_EXPECT:B_EXPECT`, expectations `in`/`out` -- whether the edge
    A<->B must be inside each half's t3 DAMAGE FOOTPRINT (spec 4.1: T2's
    `khandwa-dhar` is out in A and in in B; that containment IS the flip)."""
    parts = text.split(":")
    if len(parts) != 4 or parts[2] not in ("in", "out") or parts[3] not in ("in", "out"):
        raise SystemExit(f"--alt-span {text!r}: expected A:B:in|out:in|out")
    return parts[0], parts[1], parts[2], parts[3]


def _flip_tie_vars(spec: str | None) -> tuple[str, ...]:
    """Resolve `--require-flip-tie`'s CLI value into the `FLIP_VARS` names
    the two halves must tie bit-identically. `None` (flag omitted
    entirely) -> `()`, no requirement at all -- T1/T2's own default, and
    the flag's original behaviour for every caller that never sets it. The
    sentinel `"__all__"` (the flag given BARE, no value -- see `_build_
    parser`'s `const`) -> all of `FLIP_VARS`, preserving the flag's
    original all-or-nothing meaning. Anything else is a comma-separated
    list of specific names (T3's own use: only `largest_restorable_group_
    ecar_gbps` need tie, not the four network-wide sums that cannot -- see
    `tools/derive_t3.py`'s own docstring)."""
    if spec is None:
        return ()
    if spec == "__all__":
        return FLIP_VARS
    names = tuple(v.strip() for v in spec.split(",") if v.strip())
    unknown = [v for v in names if v not in FLIP_VARS]
    if unknown:
        raise SystemExit(
            f"--require-flip-tie: unknown FLIP_VARS name(s) {unknown!r}; "
            f"choices are {list(FLIP_VARS)}")
    return names


def _flip_tie_mismatches(flip_values: dict[str, dict[str, float]],
                         tie_vars: tuple[str, ...]) -> list[str]:
    """Every named var that is NOT bit-identical across the two halves --
    the actual tie check, extracted so a test can exercise it directly
    against a synthetic `flip_values` without going through argparse or a
    live server."""
    return [var for var in tie_vars
           if flip_values["A"][var] != flip_values["B"][var]]


def _parse_probe_flip(text: str) -> dict[str, dict]:
    """`A:kind:claimant:expected,B:kind:claimant:expected` -> per-half
    `metadata.probe_flip` dicts (assertions.probe_flip_mismatches' shape)."""
    out: dict[str, dict] = {}
    for item in _cut_tokens(text):
        parts = item.split(":")
        if len(parts) != 4 or parts[0] not in ("A", "B"):
            raise SystemExit(f"--probe-flip item {item!r}: expected "
                             f"A|B:kind:claimant:expected")
        out[parts[0]] = {"kind": parts[1], "claimant": parts[2],
                         "expected": parts[3]}
    if set(out) != {"A", "B"}:
        raise SystemExit("--probe-flip must declare both A and B")
    return out


def _edge_in_footprint(cone: ConeAtHorizon, damage_radius_km: float, edges,
                       filter_fn, a: str, b: str) -> bool:
    """The SAME containment rule runner.horizon_risk_group_asset_ids applies
    when it mints a horizon's risk group: the damage footprint of the cone,
    intersected with the topology edges the event filter admits."""
    footprint = damage_footprint(cone.center["lat"], cone.center["lon"],
                                 cone.width_km, damage_radius_km)
    exposed = map_geo_event_to_assets(footprint, edges, filter_fn)
    return any({e.src, e.dst} == {a, b} for e in exposed)


def _realized_for(tokens, *, claimants, sut, path_oms, oms_by_id,
                  vulnerable_pairs, protected: bool) -> list[str]:
    """Expand a half's cut spec into fiber ids, through the event filter.
    `claimants`: every claimant's whole working path; `claimant:<id>`: one
    claimant's; `sut`: the SUT's working first hop, plus its protection
    first hop when protected (the span(s) leaving the depot)."""
    ids: set[str] = set()
    for token in tokens:
        if token == "claimants":
            targets = list(claimants)
        elif token.startswith("claimant:"):
            targets = [token.split(":", 1)[1]]
            if targets[0] not in claimants:
                raise SystemExit(f"cut spec names {targets[0]!r}, not one of "
                                 f"--claimants {list(claimants)}")
        elif token == "sut":
            legs = path_oms.get(sut, {})
            hops = [legs["working"][0]]
            if protected:
                hops.append(legs["protection"][0])
            for oms_id in hops:
                ids |= set(_fiber_ids(oms_by_id, oms_id,
                                      vulnerable_pairs=vulnerable_pairs))
            continue
        else:
            raise SystemExit(f"unknown cut token {token!r}")
        for cid in targets:
            for oms_id in path_oms.get(cid, {}).get("working", ()):
                ids |= set(_fiber_ids(oms_by_id, oms_id,
                                      vulnerable_pairs=vulnerable_pairs))
    return sorted(ids)


async def derive(args: argparse.Namespace) -> dict:
    """Do every step the brief describes, printing every number, and return
    a small dict of the two written paths plus the key diagnostics (so a
    caller -- or this module's own __main__ -- can decide what to do with a
    run that comes back with a WARNING, without re-parsing stdout).

    `--scan-away-bearings` is a diagnostic short-circuit: it returns
    `{"scanned": True}` before any risk group is defined or any file is
    written."""
    server_command = (json.loads(args.server_command) if args.server_command
                      else DEFAULT_SERVER_COMMAND)
    hours = tuple(args.hours.split(","))
    claimants = tuple(args.claimants.split(","))
    id_a, id_b = f"{args.pair}a", f"{args.pair}b"
    protected = args.sut_posture == "protected"
    if args.decision_hour not in hours:
        raise SystemExit(
            f"--decision-hour {args.decision_hour!r} not in --hours {hours}")
    d_idx = hours.index(args.decision_hour)
    exposure_idx = d_idx + args.lead_time
    if exposure_idx >= len(hours):
        raise SystemExit(
            f"decision_hour {args.decision_hour!r} (index {d_idx}) + "
            f"lead_time {args.lead_time} = index {exposure_idx}, past the "
            f"last hour of {hours} -- the exposure horizon must be one of "
            f"the declared hours")
    exposure_horizon = hours[exposure_idx]
    first_issued = hours[0]
    print(f"pair={args.pair!r} sut_posture={args.sut_posture!r}  "
         f"hours={hours}  decision_hour={args.decision_hour!r}  "
         f"lead_time={args.lead_time}  -> exposure_horizon="
         f"{exposure_horizon!r} (index {exposure_idx})")

    edges = load_edges(args.topology)
    node_coords = _node_coords(args.topology)
    if args.depot not in node_coords:
        raise SystemExit(f"--depot {args.depot!r} not a node in {args.topology}")
    origin_lat, origin_lon = node_coords[args.depot]
    print(f"origin (depot {args.depot!r}): ({origin_lat}, {origin_lon})")

    if args.escape_node not in node_coords:
        print(f"WARNING: --escape-node {args.escape_node!r} is not a node "
             f"in {args.topology} at all")
    elif args.escape_node not in {e.dst if e.src == args.depot else e.src
                                  for e in edges
                                  if args.depot in (e.src, e.dst)}:
        print(f"WARNING: --escape-node {args.escape_node!r} is not a direct "
             f"topology neighbour of --depot {args.depot!r} -- recorded "
             f"anyway (metadata.escape_route_node is descriptive, not "
             f"itself routed), but double-check the site")

    async with connect_server(
            args.topology, server_command=server_command, env=dict(os.environ),
            extra_args=["--state", args.state]) as client:
        geometry = await service_geometry(client, args.topology, edges=edges)
        services_resp = await call_tool_json(client, "get_services")
        services = services_resp["services"]
        demands = {s["id"]: float(s["demand_gbps"]) for s in services}
        optical = await call_tool_json(client, "get_topology", {"layer": "optical"})
        oms_by_id = {o["id"]: o for o in optical["oms"]}
        filter_fn = get_filter(EVENT_TYPE)
        vulnerable_pairs = _vulnerable_pairs(edges, filter_fn)

        if args.sut not in geometry.cuttable_spans:
            raise SystemExit(f"--sut {args.sut!r} not a known service")
        working_spans: tuple[Segment, ...] = geometry.cuttable_spans.get(args.sut, ())
        protection_spans: tuple[Segment, ...] | None = (
            geometry.protection_cuttable_spans.get(args.sut, ())
            if protected else None)
        print(f"SUT {args.sut!r} working spans ({len(working_spans)}): "
             f"{working_spans}")
        if protected:
            print(f"SUT {args.sut!r} protection spans ({len(protection_spans)}): "
                 f"{protection_spans}")
            if not protection_spans:
                raise SystemExit(
                    f"--sut-posture protected but SUT {args.sut!r} has no "
                    f"protection cuttable spans -- this pair's whole "
                    f"premise (both legs aerial and inside the cone) "
                    f"cannot hold; pick a different --sut/--depot or "
                    f"--sut-posture unprotected")
        else:
            server_protection = geometry.protection_cuttable_spans.get(args.sut, ())
            if server_protection:
                print(f"WARNING: --sut-posture unprotected but the server "
                     f"reports protection cuttable spans for {args.sut!r}: "
                     f"{server_protection} -- double-check --sut-posture")
        if not working_spans:
            print(f"WARNING: SUT {args.sut!r} has an empty working "
                 f"cuttable-span set -- this pair's whole premise cannot "
                 f"hold; pick a different --sut/--depot")

        # ---- SUT legs, read back live (spec 4.1's "MUST be read back live --
        # before freezing"), as node lists, not just OMS ids.
        sut_working_oms = geometry.path_oms.get(args.sut, {}).get("working", ())
        sut_protection_oms = geometry.path_oms.get(args.sut, {}).get(
            "protection", ())
        print(f"SUT {args.sut!r} live WORKING leg: "
             f"{[geometry.oms_nodes[o] for o in sut_working_oms]}")
        if protected:
            print(f"SUT {args.sut!r} live PROTECTION leg: "
                 f"{[geometry.oms_nodes[o] for o in sut_protection_oms]}")

        def joint_pcut(lat: float, lon: float) -> float:
            return p_cut_service(working_spans, protection_spans, lat, lon,
                                 args.width_km, args.damage_radius_km)

        # ---- claimant corridor check (hard): every claimant must leave the --
        # depot on an aerial span, or a survivor groom / pin-order change
        # moved it out from under this construction without anyone noticing.
        #
        # Checks whichever END of the working OMS sequence actually touches
        # the depot, not always legs[0]: a bidirectional claimant PAIR (T1's
        # own `t1-claimant-jalgaon-dhulia-fwd`/`-rev`) has one direction
        # whose working path LEAVES the depot (first hop) and one whose
        # working path ARRIVES at it (last hop) -- confirmed live, the
        # naive "always legs[0]" reading raises on the `-rev` service, which
        # is a real T1 claimant, not a construction error. Either end is the
        # SAME physical span, so `vulnerable_pairs` (both node orders) reads
        # identically regardless of which end is checked.
        for cid in claimants:
            legs = geometry.path_oms.get(cid, {}).get("working", ())
            if not legs:
                raise SystemExit(
                    f"{cid} has no working OMS path at all -- is it a known "
                    f"service?")
            first, last = legs[0], legs[-1]
            if args.depot in geometry.oms_nodes.get(first, []):
                hop = first
            elif args.depot in geometry.oms_nodes.get(last, []):
                hop = last
            else:
                hop = None
            nodes = geometry.oms_nodes.get(hop, []) if hop is not None else []
            if hop is None or tuple(nodes) not in vulnerable_pairs:
                raise SystemExit(
                    f"{cid} does not leave {args.depot} on an aerial "
                    f"corridor (working path {[geometry.oms_nodes.get(o) for o in legs]}); "
                    f"a survivor groom or pin order moved it")

        # ---- t0 (shared) and half A ("toward") cones -----------------
        t0_cone = de.resolve_cone(
            de.ConeSpec(width_km=args.width_km, bearing_deg=args.t0_bearing,
                       radius_km=args.t0_radius),
            (origin_lat, origin_lon))
        print(f"t0 cone: bearing={args.t0_bearing} radius={args.t0_radius} "
             f"width={args.width_km} -> centre={t0_cone.center}")

        toward_cone = de.resolve_cone(
            de.ConeSpec(width_km=args.width_km, bearing_deg=args.toward_bearing,
                       radius_km=args.toward_radius),
            (origin_lat, origin_lon))
        print(f"half A (toward) cone: bearing={args.toward_bearing} "
             f"radius={args.toward_radius} width={args.width_km} -> "
             f"centre={toward_cone.center}")

        target_pcut = joint_pcut(toward_cone.center["lat"], toward_cone.center["lon"])
        print(f"half A SUT joint p_cut (target for half B) = {target_pcut!r}")

        # ---- --scan-away-bearings: diagnostic, writes nothing ---------------
        if args.scan_away_bearings is not None:
            lo_s, hi_s, step_s = args.scan_away_bearings.split(":")
            lo, hi, step = float(lo_s), float(hi_s), float(step_s)
            print(f"\n--scan-away-bearings {lo}:{hi}:{step} -- diagnostic "
                 f"only, writes nothing")
            n_steps = int(round((hi - lo) / step))
            for i in range(n_steps + 1):
                bearing = lo + i * step
                try:
                    radius = de.solve_radius_for_target_pcut(
                        origin_lat, origin_lon, bearing, target_pcut,
                        spans=(), width_km=args.width_km,
                        damage_radius_km=args.damage_radius_km,
                        pcut_fn=joint_pcut)
                except ValueError:
                    print(f"  bearing={bearing:.3f}: no bracket")
                    continue
                cone = de.resolve_cone(
                    de.ConeSpec(width_km=args.width_km, bearing_deg=bearing,
                               radius_km=radius),
                    (origin_lat, origin_lon))
                achieved = joint_pcut(cone.center["lat"], cone.center["lon"])
                joint_delta = abs(achieved - target_pcut)
                w_a = p_cut_region(working_spans, toward_cone.center["lat"],
                                   toward_cone.center["lon"], args.width_km,
                                   args.damage_radius_km)
                w_b = p_cut_region(working_spans, cone.center["lat"],
                                   cone.center["lon"], args.width_km,
                                   args.damage_radius_km)
                claimant_row = {}
                for cid in claimants:
                    spans_c = geometry.cuttable_spans.get(cid, ())
                    pa = p_cut_region(spans_c, toward_cone.center["lat"],
                                      toward_cone.center["lon"], args.width_km,
                                      args.damage_radius_km)
                    pb = p_cut_region(spans_c, cone.center["lat"],
                                      cone.center["lon"], args.width_km,
                                      args.damage_radius_km)
                    claimant_row[cid] = (pa, pb)
                print(f"  bearing={bearing:.3f} radius={radius!r} "
                     f"joint(A={target_pcut!r}, B={achieved!r}, "
                     f"delta={joint_delta:.3e}) working(A={w_a!r}, "
                     f"B={w_b!r}, delta={abs(w_a - w_b):.3e}) "
                     f"claimants={claimant_row}")
            return {"scanned": True}

        # ---- half B bearing: either given directly, or solved so a second --
        # corridor's own p_cut matches the first's (--pcut-match-claimant).
        if args.pcut_match_claimant is not None and args.away_bearing is not None:
            raise SystemExit(
                "--pcut-match-claimant and --away-bearing are mutually "
                "exclusive -- pick one")
        if args.pcut_match_claimant is None and args.away_bearing is None:
            raise SystemExit(
                "one of --away-bearing or --pcut-match-claimant is required")

        if args.pcut_match_claimant is not None:
            a_id, b_id = args.pcut_match_claimant.split(":")
            if args.away_bearing_range is None:
                raise SystemExit(
                    "--pcut-match-claimant requires --away-bearing-range LO:HI")
            lo_s, hi_s = args.away_bearing_range.split(":")
            lo, hi = float(lo_s), float(hi_s)
            step = args.bearing_step
            a_spans = geometry.cuttable_spans.get(a_id, ())
            b_spans = geometry.cuttable_spans.get(b_id, ())
            a_pcut_at_toward = p_cut_region(
                a_spans, toward_cone.center["lat"], toward_cone.center["lon"],
                args.width_km, args.damage_radius_km)
            print(f"\n--pcut-match-claimant {a_id}:{b_id} -- {a_id}'s own "
                 f"p_cut at half A's cone = {a_pcut_at_toward!r}; scanning "
                 f"bearings {lo}:{hi} step {step} for {b_id}'s matching "
                 f"p_cut")
            n_steps = int(round((hi - lo) / step))
            rows: list[tuple[float, float, float, float]] = []
            zero_gap: list[tuple[float, float]] = []
            for i in range(n_steps + 1):
                bearing = lo + i * step
                try:
                    radius = de.solve_radius_for_target_pcut(
                        origin_lat, origin_lon, bearing, target_pcut,
                        spans=(), width_km=args.width_km,
                        damage_radius_km=args.damage_radius_km,
                        pcut_fn=joint_pcut)
                except ValueError:
                    print(f"  bearing={bearing:.3f}: no bracket")
                    continue
                cone_b = de.resolve_cone(
                    de.ConeSpec(width_km=args.width_km, bearing_deg=bearing,
                               radius_km=radius),
                    (origin_lat, origin_lon))
                b_pcut = p_cut_region(b_spans, cone_b.center["lat"],
                                      cone_b.center["lon"], args.width_km,
                                      args.damage_radius_km)
                gap = b_pcut - a_pcut_at_toward
                rows.append((bearing, radius, b_pcut, gap))
                print(f"  bearing={bearing:.3f} radius={radius!r} "
                     f"{b_id}_pcut={b_pcut!r} gap={gap!r}")
                if gap == 0.0:
                    zero_gap.append((bearing, radius))
            if not zero_gap:
                raise SystemExit(
                    "--pcut-match-claimant found no bearing with an exact "
                    "zero p_cut gap:\n  " + "\n  ".join(
                        f"bearing={b:.3f} radius={r!r} {b_id}_pcut={pc!r} "
                        f"gap={g!r}" for b, r, pc, g in rows))
            runs: list[list[tuple[float, float]]] = [[zero_gap[0]]]
            for prev, curr in zip(zero_gap, zero_gap[1:]):
                if abs(curr[0] - prev[0] - step) < 1e-9:
                    runs[-1].append(curr)
                else:
                    runs.append([curr])
            longest = max(runs, key=len)
            away_bearing, away_radius = longest[len(longest) // 2]
            print(f"chosen away bearing (middle of the longest zero-gap "
                 f"run, length {len(longest)}) = {away_bearing!r}, "
                 f"radius={away_radius!r}")
        else:
            away_bearing = args.away_bearing
            away_radius = de.solve_radius_for_target_pcut(
                origin_lat, origin_lon, away_bearing, target_pcut,
                spans=(), width_km=args.width_km,
                damage_radius_km=args.damage_radius_km, pcut_fn=joint_pcut)

        away_cone = de.resolve_cone(
            de.ConeSpec(width_km=args.width_km, bearing_deg=away_bearing,
                       radius_km=away_radius),
            (origin_lat, origin_lon))
        achieved_pcut = joint_pcut(away_cone.center["lat"], away_cone.center["lon"])
        delta_pcut = abs(achieved_pcut - target_pcut)
        print(f"half B (away) cone: bearing={away_bearing} "
             f"radius SOLVED={away_radius!r} width={args.width_km} -> "
             f"centre={away_cone.center}")
        print(f"half B SUT joint p_cut achieved = {achieved_pcut!r}  "
             f"|delta vs half A| = {delta_pcut:.3e}")
        if delta_pcut >= 1e-9:
            raise SystemExit(
                f"solve_radius_for_target_pcut did not converge: "
                f"|{achieved_pcut} - {target_pcut}| = {delta_pcut:.3e} >= "
                f"1e-9 -- widen the search range or pick a different "
                f"--away-bearing")

        # ---- diagnostic 0: per-leg (single-leg) p_cut at each half's own --
        # t3 cone -- the spec's `assert_both_legs_exposed` invariant (S4.7):
        # each leg must carry non-trivial exposure on ITS OWN, not just
        # jointly, at the exposure horizon in BOTH halves. Only the working
        # leg exists for an unprotected posture.
        print("\nper-leg p_cut at each half's own t3 cone (assert_both_legs_"
             "exposed evidence):")
        for label, cone in (("A", toward_cone), ("B", away_cone)):
            legs = [("working", working_spans)]
            if protected:
                legs.append(("protection", protection_spans))
            for leg_name, spans in legs:
                pc = p_cut_region(spans, cone.center["lat"], cone.center["lon"],
                                  args.width_km, args.damage_radius_km)
                print(f"  half {label} SUT {leg_name} leg: p_cut={pc!r}")

        # ---- diagnostic 1: baseline tie -- both legs inside the SHARED --
        # t0 damage footprint (both halves share this exact cone, so this
        # is computed once, not per half).
        footprint_radius = damage_footprint_radius_km(
            args.width_km, args.damage_radius_km)
        print(f"\nt0 baseline tie check (damage footprint radius = "
             f"{footprint_radius:.3f} km):")
        legs = [("working", working_spans)]
        if protected:
            legs.append(("protection", protection_spans))
        for leg_name, spans in legs:
            offset = nearest_span_offset_km(spans, t0_cone.center["lat"],
                                            t0_cone.center["lon"])
            inside = offset <= footprint_radius
            print(f"  SUT {leg_name} leg: offset={offset:.3f} km -> "
                 f"{'INSIDE' if inside else 'OUTSIDE'} "
                 f"{'(PASS)' if inside else '(WARN: not a baseline tie)'}")

        # ---- diagnostic 2: claimant p_cut separation between halves ------
        print("\nclaimant p_cut per half (single-leg -- claimants are "
             "unprotected):")
        claimant_pcut = {"A": {}, "B": {}}
        for label, cone in (("A", toward_cone), ("B", away_cone)):
            for cid in claimants:
                spans = geometry.cuttable_spans.get(cid, ())
                pc = p_cut_region(spans, cone.center["lat"], cone.center["lon"],
                                  args.width_km, args.damage_radius_km)
                claimant_pcut[label][cid] = pc
                print(f"  half {label} {cid!r}: p_cut={pc!r}")
        rep = claimants[0]
        claimant_delta = abs(claimant_pcut["A"][rep] - claimant_pcut["B"][rep])
        print(f"  claimant p_cut delta (half A vs B, {rep!r}) = "
             f"{claimant_delta:.6f}  "
             f"{'(PASS, >= 0.2)' if claimant_delta >= 0.2 else '(WARN: < 0.2)'}")

        # ---- realized cuts, resolved live, through each half's own cut spec
        realized_a = _realized_for(
            _cut_tokens(args.half_a_cuts), claimants=claimants, sut=args.sut,
            path_oms=geometry.path_oms, oms_by_id=oms_by_id,
            vulnerable_pairs=vulnerable_pairs, protected=protected)
        realized_b = _realized_for(
            _cut_tokens(args.half_b_cuts), claimants=claimants, sut=args.sut,
            path_oms=geometry.path_oms, oms_by_id=oms_by_id,
            vulnerable_pairs=vulnerable_pairs, protected=protected)
        print(f"\nhalf A realized cuts ({args.half_a_cuts!r}): {realized_a}")
        if not realized_a:
            print("WARNING: no fiber ids resolved for half A -- its "
                 "realized cut would be empty")
        print(f"half B realized cuts ({args.half_b_cuts!r}): {realized_b}")
        if not realized_b:
            print("WARNING: no fiber ids resolved for half B -- its "
                 "realized cut would be empty")

        # ---- alt-span containment: an edge that must sit differently --
        # inside/outside each half's own damage footprint.
        if args.alt_span:
            print("\nalt-span containment checks:")
            for spec in args.alt_span:
                a, b, a_expect, b_expect = _parse_alt_span(spec)
                in_a = _edge_in_footprint(toward_cone, args.damage_radius_km,
                                          edges, filter_fn, a, b)
                in_b = _edge_in_footprint(away_cone, args.damage_radius_km,
                                          edges, filter_fn, a, b)
                print(f"  {a}<->{b}: half A "
                     f"{'IN' if in_a else 'OUT'} (expect {a_expect!r}), "
                     f"half B {'IN' if in_b else 'OUT'} "
                     f"(expect {b_expect!r})")
                if in_a != (a_expect == "in") or in_b != (b_expect == "in"):
                    raise SystemExit(
                        f"--alt-span {spec}: half A "
                        f"{'IN' if in_a else 'OUT'} (expected {a_expect!r}), "
                        f"half B {'IN' if in_b else 'OUT'} "
                        f"(expected {b_expect!r})")

        # ---- widest_avoid_feasible, measured live, per half --------------
        async def widest_avoid_feasible(label: str, cone: ConeAtHorizon
                                        ) -> tuple[bool, str]:
            rg_id = f"rg_derive_{args.pair}_{args.sut}_{label}_widest"
            fiber_ids = horizon_risk_group_asset_ids(
                cone, args.damage_radius_km, edges=edges, oms=optical["oms"],
                filter_fn=filter_fn)
            await call_tool_json(client, "define_risk_group", {
                "rg_id": rg_id, "asset_ids": fiber_ids,
                "metadata": {"tool": "derive_t1", "pair": args.pair,
                            "half": label}})
            # basis="risk_group"/level="risk_group" is now DERIVED from a
            # risk_groups avoid, not a free choice: T2a/T3a/T3b's gold
            # constraint decision only validates under this basis/level pair
            # (T2a.yaml's own widest_avoid_feasible narration -- "route_
            # service(avoid={risk_groups:[...]}) returns ZERO candidates
            # under EITHER basis" -- checked both), and ConstraintDecision
            # derives exactly this pair whenever `risk_groups` is named.
            constraints = ConstraintDecision(
                avoid={"risk_groups": [rg_id]},
                reasoning="derive_t1 widest-avoid probe")
            menu = await call_tool_json(
                client, "route_service", constraints.route_service_args(args.sut))
            candidates = menu.get("candidates") or []
            feasible = len(candidates) > 0
            print(f"  half {label}: rg={rg_id!r} assets={len(fiber_ids)} "
                 f"status={menu.get('status')!r} candidates={len(candidates)} "
                 f"-> widest_avoid_feasible={feasible}")
            return feasible, rg_id

        print("\nwidest_avoid_feasible (live route_service under this "
             "half's own t3 cone as a risk-group avoid):")
        widest_feasible: dict[str, bool] = {}
        rg_ids: dict[str, str] = {}
        for label, cone in (("A", toward_cone), ("B", away_cone)):
            feasible, rg_id = await widest_avoid_feasible(label, cone)
            widest_feasible[label] = feasible
            rg_ids[label] = rg_id

        # ---- shared scalars, computed once, stamped into both files ------
        cone_motion_km = de.radial_offset_km(
            t0_cone.center["lat"], t0_cone.center["lon"],
            toward_cone.center["lat"], toward_cone.center["lon"])
        elapsed = _elapsed_hours(hours, first_issued, args.decision_hour)
        cone_motion_kmh = cone_motion_km / elapsed if elapsed else cone_motion_km
        print(f"\ncone_motion_kmh = offset(t0 centre, half A t3 centre) "
             f"({cone_motion_km:.3f} km) / {elapsed} h = {cone_motion_kmh!r} "
             f"-- STAMPED IDENTICALLY into both halves")

        metadata_common = {
            "label_rule": "spare_action_by_deadline",
            "exposure_legs": "both" if protected else "working",
            "escape_route_node": args.escape_node,
            "claimant_services": list(claimants),
            "cone_width_km": args.width_km,
            "cone_motion_kmh": round(cone_motion_kmh, 6),
            "n_future_claimants": len(claimants),
            "exposure_horizon_hours": args.lead_time,
            "spares_on_hand": 1,
        }

        def make_scenario(id_: str, gold_action: str, cone_t3: ConeAtHorizon,
                          realized_ids: list[str], widest: bool) -> ScenarioFile:
            forecast = {
                first_issued: Issuance(issued_at=first_issued,
                                       horizons={exposure_horizon: t0_cone}),
                args.decision_hour: Issuance(
                    issued_at=args.decision_hour,
                    horizons={exposure_horizon: cone_t3}),
            }
            metadata = {**metadata_common, "gold_spare_action": gold_action,
                       "widest_avoid_feasible": widest}
            return ScenarioFile(
                id=id_, pair=args.pair, seed=args.seed, state_file=args.state,
                service_under_test=args.sut, track=args.track, hours=hours,
                decision_hour=args.decision_hour, lead_time_hours=args.lead_time,
                spares_on_hand=1, depot_site=args.depot,
                spare_inventory={args.depot: 1},
                damage_radius_km=args.damage_radius_km,
                track_revision_km_per_hour_ahead=30.0, reference_avoid={},
                forecast=forecast,
                realized={exposure_horizon: tuple(realized_ids)},
                gold=_PLACEHOLDER_GOLD, flip_variable=FLIP_VARIABLE_TOKENS,
                metadata=metadata)

        scenario_a = make_scenario(id_a, "conserve", toward_cone,
                                   realized_a, widest_feasible["A"])
        scenario_b = make_scenario(id_b, "spend", away_cone,
                                   realized_b, widest_feasible["B"])

        # ---- derived-geometry gate, IN-TOOL: the same pure comparison --
        # assertions.assert_pair_derived_geometry_is_equal makes, run here so
        # a bad geometry never reaches disk in the first place.
        mismatches = _derived_mismatches(
            scenario_a, derived_geometry_from_spans(
                scenario_a, working_spans, protection_spans),
            scenario_b, derived_geometry_from_spans(
                scenario_b, working_spans, protection_spans))
        if mismatches:
            raise SystemExit(
                "derived geometry differs across the halves:\n  "
                + "\n  ".join(mismatches))
        print("\nderived-geometry gate: PASSED (every derived scalar equal "
             "across the halves)")

        # ---- flip scalars, IN-TOOL: printed, and (opt-in) required tied --
        flip_values: dict[str, dict[str, float]] = {}
        for label, scenario in (("A", scenario_a), ("B", scenario_b)):
            flip_values[label] = flip_scalars_from_spans(
                scenario, spans=geometry.cuttable_spans, demands_gbps=demands,
                endpoint_sites=geometry.endpoint_sites,
                protection_spans=geometry.protection_cuttable_spans).values()
        print("\nflip scalars (FLIP_VARS), per half:")
        for var in FLIP_VARS:
            print(f"  {var:45s} A={flip_values['A'][var]!r:>20s}  "
                 f"B={flip_values['B'][var]!r}")
        tie_vars = _flip_tie_vars(args.require_flip_tie)
        if tie_vars:
            not_tied = _flip_tie_mismatches(flip_values, tie_vars)
            if not_tied:
                raise SystemExit(
                    "--require-flip-tie: these FLIP_VARS are not "
                    "bit-identical across the halves:\n  " + "\n  ".join(
                        f"{var}: {flip_values['A'][var]!r} (A) vs "
                        f"{flip_values['B'][var]!r} (B)" for var in not_tied))

        # ---- claimed_competing_ecar_gbps/_at, from observation's own -----
        # AGENT-FACING figure (`_restorable_groups`, via build_observation),
        # not a separately-rounded figure of this tool's own -- so the
        # scenario's declared claim matches, bit for bit, what a real
        # rollout would actually show the agent (T1a.yaml/T2a.yaml's own
        # precedent, "two readers, two roundings, one quantity"). Restricted
        # to the groups whose OWN `members` intersect `--claimants` -- the
        # SAME filter `assertions.assert_claim_is_one_lightpath` applies
        # (`claimant_ids & set(g.get("members") or ())`) -- not the
        # network-wide largest group at this horizon, which can name a
        # service outside this pair's own claim once more than one
        # depot-eligible group exists at the horizon.
        for label, scenario in (("A", scenario_a), ("B", scenario_b)):
            obs = build_observation(
                scenario, args.decision_hour,
                service_spans=geometry.cuttable_spans, services=services,
                spares_on_hand=1, endpoint_sites=geometry.endpoint_sites,
                depot_site=args.depot,
                protection_spans=geometry.protection_cuttable_spans)
            groups = obs.restorable_groups.get(exposure_horizon, ())
            if not groups:
                raise SystemExit(
                    f"half {label}: observation._restorable_groups found NO "
                    f"depot-eligible group at {exposure_horizon!r} -- the "
                    f"claimants are not reading as exposed/depot-eligible; "
                    f"check --claimants and --depot")
            claimant_ids = set(claimants)
            own_groups = [g for g in groups
                         if claimant_ids & set(g.get("members") or ())]
            if not own_groups:
                raise SystemExit(
                    f"half {label}: none of observation._restorable_groups' "
                    f"{len(groups)} group(s) at {exposure_horizon!r} have "
                    f"members intersecting --claimants {sorted(claimant_ids)!r} "
                    f"-- the claimants are not reading as exposed/"
                    f"depot-eligible; check --claimants and --depot")
            largest_group = max(own_groups, key=lambda g: g["ecar_gbps"])
            claimed = largest_group["ecar_gbps"]
            print(f"half {label} claimed_competing_ecar_gbps "
                 f"(observation._restorable_groups, largest of the claim's "
                 f"OWN groups at {exposure_horizon!r}) = {claimed!r}  "
                 f"members={largest_group['members']}")
            new_metadata = {**scenario.metadata,
                            "claimed_competing_ecar_gbps": claimed,
                            "claimed_competing_ecar_at": exposure_horizon}
            if label == "A":
                scenario_a = dataclasses.replace(scenario, metadata=new_metadata)
            else:
                scenario_b = dataclasses.replace(scenario, metadata=new_metadata)

        # ---- probe flip, live: the SAME question a decider may ask, --
        # answered here for every declared claimant under each half's own
        # widest-avoid group (already defined above), and gated in-tool --
        # exactly `assert_probe_flips`' own check, run against this tool's
        # own freshly-derived halves instead of a later server round trip.
        call = functools.partial(call_tool_json, client)
        probe_answers: dict[str, dict[str, dict]] = {"A": {}, "B": {}}
        print("\nprobe flip (live, per half's own widest-avoid group):")
        for label, cone, rg_id in (("A", toward_cone, rg_ids["A"]),
                                   ("B", away_cone, rg_ids["B"])):
            issuance = Issuance(issued_at=args.decision_hour,
                                horizons={exposure_horizon: cone})
            for cid in claimants:
                answer = await answer_probe(
                    call, service_id=cid, risk_group_id=rg_id, geometry=geometry,
                    issuance=issuance, damage_radius_km=args.damage_radius_km,
                    demands=demands)
                probe_answers[label][cid] = answer.to_dict()
                print(f"  half {label} {cid!r} under {rg_id!r}: "
                     f"{probe_answers[label][cid]}")

        declared_probe_flip = (_parse_probe_flip(args.probe_flip)
                              if args.probe_flip else None)
        if declared_probe_flip is not None:
            scenario_a = dataclasses.replace(
                scenario_a, metadata={**scenario_a.metadata,
                                      "probe_flip": declared_probe_flip["A"]})
            scenario_b = dataclasses.replace(
                scenario_b, metadata={**scenario_b.metadata,
                                      "probe_flip": declared_probe_flip["B"]})

        probe_problems = probe_flip_mismatches(
            scenario_a, probe_answers["A"], scenario_b, probe_answers["B"])
        if probe_problems:
            raise SystemExit(
                "probe flip does not match the declared/expected shape:\n  "
                + "\n  ".join(probe_problems))
        print("probe flip gate: PASSED")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for scenario in (scenario_a, scenario_b):
        path = out_dir / f"{scenario.id}.yaml"
        path.write_text(dump_scenario(scenario), encoding="utf-8")
        written[scenario.id] = str(path)
        print(f"\nwrote {path}")

    return {
        "written": written,
        "exposure_horizon": exposure_horizon,
        "target_pcut": target_pcut,
        "achieved_pcut": achieved_pcut,
        "delta_pcut": delta_pcut,
        "claimant_pcut_delta": claimant_delta,
        "widest_avoid_feasible": widest_feasible,
        "flip_values": flip_values,
        "probe_answers": probe_answers,
        "away_bearing": away_bearing,
    }


def _build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    """Extracted from `main()` so `tests/eval/test_derive_t1.py` can exercise
    the REAL parser (flag names, requiredness, defaults) without invoking
    `main()` itself, which would go on to `asyncio.run(derive(...))` against
    a real server.

    `defaults`, when given (Task 9, T2/T3 probe redesign): every flag that
    is `required=True` with no default here becomes OPTIONAL when its name
    is a key of `defaults`, and `parser.set_defaults(**defaults)` supplies
    the value -- exactly how `tools/derive_t2.py`/`tools/derive_t3.py` turn
    this same parser into a pair-specific one with no flags of their own."""
    defaults = defaults or {}
    parser = argparse.ArgumentParser(prog="derive_t1")
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--pair", default="T1")
    parser.add_argument("--sut", required=("sut" not in defaults))
    parser.add_argument("--sut-posture", choices=("protected", "unprotected"),
                        default="protected")
    parser.add_argument("--depot", required=("depot" not in defaults))
    parser.add_argument("--claimants", required=("claimants" not in defaults),
                        help="comma-separated service ids, e.g. "
                             "claimant-x-y-fwd,claimant-x-y-rev")
    parser.add_argument("--escape-node",
                        required=("escape_node" not in defaults),
                        help="the node the SUT's escape route reroutes "
                             "through -- recorded as metadata."
                             "escape_route_node (descriptive)")
    parser.add_argument("--hours", default="t0,t1,t2,t3,t4,t5")
    parser.add_argument("--decision-hour", default="t1")
    parser.add_argument("--lead-time", type=int, default=2)
    parser.add_argument("--width-km", type=float, default=90.0)
    parser.add_argument("--t0-radius", type=float,
                        required=("t0_radius" not in defaults))
    parser.add_argument("--t0-bearing", type=float,
                        required=("t0_bearing" not in defaults))
    parser.add_argument("--toward-bearing", type=float,
                        required=("toward_bearing" not in defaults))
    parser.add_argument("--toward-radius", type=float,
                        required=("toward_radius" not in defaults))
    parser.add_argument("--half-a-cuts", default="claimants",
                        help="comma tokens: claimants | claimant:<id> | sut")
    parser.add_argument("--half-b-cuts", default="sut")
    parser.add_argument("--alt-span", action="append", default=None,
                        metavar="A:B:in|out:in|out",
                        help="assert edge A<->B's damage-footprint containment per half")
    parser.add_argument("--away-bearing", type=float, default=None)   # was required
    parser.add_argument("--pcut-match-claimant", default=None, metavar="A_ID:B_ID",
                        help="solve the away bearing so B's claimant p_cut EXACTLY "
                             "equals A's (the whole-suite tie needs the same float)")
    parser.add_argument("--away-bearing-range", default=None, metavar="LO:HI")
    parser.add_argument("--bearing-step", type=float, default=1.0)
    parser.add_argument("--scan-away-bearings", default=None, metavar="LO:HI:STEP",
                        help="diagnostic: print per-bearing solved radius, working-leg "
                             "and joint deltas, claimant p_cuts; write nothing")
    parser.add_argument("--probe-flip", default=None,
                        metavar="A:kind:claimant:expected,B:kind:claimant:expected")
    parser.add_argument("--require-flip-tie", nargs="?", const="__all__",
                        default=None, metavar="VAR1,VAR2,...",
                        help="require bit-identical FLIP_VARS across the "
                             "halves; bare flag (no value) means ALL of "
                             "FLIP_VARS (T1's own original all-or-nothing "
                             "meaning), a comma-separated list means only "
                             "those named vars (T3's own use, since a full "
                             "tie is unreachable/incompatible with "
                             "assert_flip_dominates for a two-cone pair), "
                             "omitted entirely means no requirement")
    parser.add_argument("--damage-radius-km", type=float, default=74.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--track", default="hudhud")
    parser.add_argument("--out-dir",
                        default="src/storm_reoptimizer/eval/scenarios")
    parser.add_argument("--server-command", default=None,
                        help="JSON list; defaults to this workspace's "
                             "multilayer-optical-mcp conda-env workaround")
    parser.set_defaults(**defaults)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(derive(args))


if __name__ == "__main__":
    main()
