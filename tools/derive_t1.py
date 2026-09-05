# tools/derive_t1.py
"""Task 14 (T1 spend-or-hold redesign plan): the tool that actually WRITES
the T1 pair's two scenario YAMLs (T1a.yaml -- "conserve"/"hold" -- and
T1b.yaml -- "spend") once Task 15 has chosen a site. Parametrised end to
end (SUT, depot, claimants, escape node, hours, bearings/radii) so
re-deriving with a different site or a retuned bearing is one command, not
a hand edit -- the same reason `tools/derive_episodes.py` exists for the
older exposure-and-depot plan's three pairs.

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
    claimant corridor, at `away_bearing`, with its RADIUS SOLVED (via
    `derive_episodes.solve_radius_for_target_pcut`, generalised this same
    task to accept a `pcut_fn` callable) so the SUT's own JOINT
    `cone.p_cut_service` (working AND protection -- the whole point of this
    redesign: both legs are aerial and inside the cone, so switchover does
    not save the SUT the way it did before) matches half A's to below
    1e-9 -- comfortably inside `derived.DERIVED_TOLERANCE` (1e-6). Only the
    claimant corridor's own exposure is left to differ.

**Realized cuts, resolved live, never typed** (`get_topology(layer=
"optical")`'s own `oms[].elements`, filtered to the `fiber_*` prefix,
matching `runner.horizon_risk_group_asset_ids`' own convention):

  * Half A: every claimant service's own WORKING-path OMS sequence
    (`ServiceGeometry.path_oms[claimant]["working"]`) -- both claimant ids
    given on `--claimants`, so both directions of the corridor go down.
  * Half B: the SUT's own working- and protection-path FIRST HOP
    (`path_oms[sut]["working"][0]` / `["protection"][0]` -- the span
    leaving the depot, matching S4.1's "leaves its home site on aerial
    spans" framing) -- cutting only the near span is enough to take the
    whole leg down and is what makes this the minimal, honest realization
    of "both legs cut".

**Gold is a placeholder.** `tools/compute_gold.py` (a later task) is the
real enumerator (S4.6); this tool only needs a syntactically valid `Gold`
so the file round-trips through `scenario_file.load_scenario` before that
task exists. `gold_spare_action` (A=conserve, B=spend) is likewise
provisional -- the qualitative expectation the construction is aimed at,
not a computed answer.

Runs in THIS repo's env (storm-reoptimizer) against a LIVE
multilayer-optical-mcp server, same seam every tool in this directory
observes (imports nothing from multilayer_optical_network; talks to the
model only through mcp_client's stdio Client).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
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

from storm_reoptimizer.eval.cone import (               # noqa: E402
    Segment, nearest_span_offset_km, p_cut_region, p_cut_service,
)
from storm_reoptimizer.eval.decisions import ConstraintDecision  # noqa: E402
from storm_reoptimizer.eval.observation import build_observation  # noqa: E402
from storm_reoptimizer.eval.runner import (              # noqa: E402
    EVENT_TYPE, horizon_risk_group_asset_ids, service_geometry,
)
from storm_reoptimizer.eval.scenario_file import (       # noqa: E402
    ConeAtHorizon, Gold, Issuance, ScenarioFile, dump_scenario,
)
from storm_reoptimizer.events.filters import get_filter  # noqa: E402
from storm_reoptimizer.events.geo import damage_footprint_radius_km  # noqa: E402
from storm_reoptimizer.geo_mapper import load_edges       # noqa: E402
from storm_reoptimizer.mcp_client import call_tool_json, connect_server  # noqa: E402

DEFAULT_TOPOLOGY = de.DEFAULT_TOPOLOGY
DEFAULT_STATE = de.DEFAULT_STATE
DEFAULT_SERVER_COMMAND = de.DEFAULT_SERVER_COMMAND

# Placeholder gold (mirrors derive_episodes.py's own `_DUMMY_GOLD` -- same
# shape, same "not gold" intent), pending Task 15's real enumerator
# (tools/compute_gold.py, spec S4.6). `decision_at_t0="wait"` is the one
# non-arbitrary part of the placeholder: the spec states plainly (S4.4) that
# at t0 the revision has not arrived yet, so waiting is correct BY
# INSPECTION in both halves, independent of what the enumerator later finds
# for t1.
_PLACEHOLDER_GOLD = Gold(
    survived=(), max_spares_wasted=0, decision_at_t0="wait", label="n/a",
    rationale="PROVISIONAL -- run tools/compute_gold.py --write")

# First-cut citation tokens for `scoring.cites_flip_variable` (matches
# rationale using words for the NEW spend/hold decision this pair grades,
# analogous in spirit to the older pairs' own `flip_variable` lists) --
# provisional, like `gold` above: nothing in this task's brief specifies the
# final vocabulary, and a later task (or a human, alongside the real gold
# rationale) is free to retune it.
FLIP_VARIABLE_TOKENS = ("spend", "hold", "claimant", "escape")


def _node_coords(topology_path: str | Path) -> dict[str, tuple[float, float]]:
    """node_id -> (lat, lon). Same two-line convention runner.service_
    geometry's own `coords` local and find_sut.py's own `_node_coords` use;
    kept local here for the same reason find_sut.py keeps its own copy --
    tools/ is not a package, so nothing here is imported from there."""
    data = json.loads(Path(topology_path).read_text(encoding="utf-8-sig"))
    return {n["id"]: (n["lat"], n["lon"]) for n in data["graph"]["nodes"]}


def _fiber_ids(oms_by_id: dict, oms_id: str) -> list[str]:
    """Every `fiber_*` element of one OMS -- the SAME filter `runner.
    horizon_risk_group_asset_ids` applies, just keyed by a single OMS id
    instead of walking every OMS a hazard footprint touches."""
    return sorted(e for e in oms_by_id[oms_id]["elements"]
                 if e.startswith("fiber_"))


def _elapsed_hours(hours: tuple[str, ...], a: str, b: str) -> int:
    return hours.index(b) - hours.index(a)


async def derive(args: argparse.Namespace) -> dict:
    """Do every step the brief describes, printing every number, and return
    a small dict of the two written paths plus the key diagnostics (so a
    caller -- or this module's own __main__ -- can decide what to do with a
    run that comes back with a WARNING, without re-parsing stdout)."""
    server_command = (json.loads(args.server_command) if args.server_command
                      else DEFAULT_SERVER_COMMAND)
    hours = tuple(args.hours.split(","))
    claimants = tuple(args.claimants.split(","))
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
    print(f"hours={hours}  decision_hour={args.decision_hour!r}  "
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
        optical = await call_tool_json(client, "get_topology", {"layer": "optical"})
        oms_by_id = {o["id"]: o for o in optical["oms"]}

        if args.sut not in geometry.cuttable_spans:
            raise SystemExit(f"--sut {args.sut!r} not a known service")
        working_spans: tuple[Segment, ...] = geometry.cuttable_spans.get(args.sut, ())
        protection_spans: tuple[Segment, ...] = geometry.protection_cuttable_spans.get(
            args.sut, ())
        print(f"SUT {args.sut!r} working spans ({len(working_spans)}): "
             f"{working_spans}")
        print(f"SUT {args.sut!r} protection spans ({len(protection_spans)}): "
             f"{protection_spans}")
        if not working_spans or not protection_spans:
            print(f"WARNING: SUT {args.sut!r} has an empty working or "
                 f"protection cuttable-span set -- this pair's whole premise "
                 f"(both legs aerial and inside the cone) cannot hold; "
                 f"pick a different --sut/--depot")

        def joint_pcut(lat: float, lon: float) -> float:
            return p_cut_service(working_spans, protection_spans, lat, lon,
                                 args.width_km, args.damage_radius_km)

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

        # ---- half B ("away") cone: radius SOLVED against the joint p_cut --
        away_radius = de.solve_radius_for_target_pcut(
            origin_lat, origin_lon, args.away_bearing, target_pcut,
            spans=(), width_km=args.width_km,
            damage_radius_km=args.damage_radius_km, pcut_fn=joint_pcut)
        away_cone = de.resolve_cone(
            de.ConeSpec(width_km=args.width_km, bearing_deg=args.away_bearing,
                       radius_km=away_radius),
            (origin_lat, origin_lon))
        achieved_pcut = joint_pcut(away_cone.center["lat"], away_cone.center["lon"])
        delta_pcut = abs(achieved_pcut - target_pcut)
        print(f"half B (away) cone: bearing={args.away_bearing} "
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
        # jointly, at the exposure horizon in BOTH halves.
        print("\nper-leg p_cut at each half's own t3 cone (assert_both_legs_"
             "exposed evidence):")
        for label, cone in (("A", toward_cone), ("B", away_cone)):
            for leg_name, spans in (("working", working_spans),
                                    ("protection", protection_spans)):
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
        for leg_name, spans in (("working", working_spans),
                                ("protection", protection_spans)):
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

        # ---- realized cuts, resolved live -------------------------------
        claimant_fiber_ids = sorted({
            fid for cid in claimants
            for oms_id in geometry.path_oms.get(cid, {}).get("working", ())
            for fid in _fiber_ids(oms_by_id, oms_id)})
        print(f"\nhalf A realized cuts (claimant corridor, both directions): "
             f"{claimant_fiber_ids}")
        if not claimant_fiber_ids:
            print("WARNING: no fiber ids resolved for the claimant corridor "
                 "-- half A's realized cut would be empty")

        sut_working_oms = geometry.path_oms.get(args.sut, {}).get("working", ())
        sut_protection_oms = geometry.path_oms.get(args.sut, {}).get(
            "protection", ())
        if not sut_working_oms or not sut_protection_oms:
            raise SystemExit(
                f"SUT {args.sut!r} has no working+protection OMS sequence "
                f"({sut_working_oms!r}/{sut_protection_oms!r}) -- is it "
                f"really a protected service?")
        working_first_hop, protection_first_hop = sut_working_oms[0], sut_protection_oms[0]
        for leg_name, oms_id in (("working", working_first_hop),
                                 ("protection", protection_first_hop)):
            nodes = geometry.oms_nodes.get(oms_id, [])
            if args.depot not in nodes:
                print(f"WARNING: SUT {leg_name} first-hop OMS {oms_id!r} "
                     f"nodes {nodes} do not include depot {args.depot!r} -- "
                     f"'first hop' may not be the depot-leaving span")
        sut_fiber_ids = sorted(set(_fiber_ids(oms_by_id, working_first_hop))
                               | set(_fiber_ids(oms_by_id, protection_first_hop)))
        print(f"half B realized cuts (SUT working+protection first hop): "
             f"{sut_fiber_ids}")
        if not sut_fiber_ids:
            print("WARNING: no fiber ids resolved for the SUT's first-hop "
                 "legs -- half B's realized cut would be empty")

        # ---- widest_avoid_feasible, measured live, per half --------------
        filter_fn = get_filter(EVENT_TYPE)

        async def widest_avoid_feasible(label: str, cone: ConeAtHorizon) -> bool:
            rg_id = f"rg_derive_t1_{args.sut}_{label}_widest"
            fiber_ids = horizon_risk_group_asset_ids(
                cone, args.damage_radius_km, edges=edges, oms=optical["oms"],
                filter_fn=filter_fn)
            await call_tool_json(client, "define_risk_group", {
                "rg_id": rg_id, "asset_ids": fiber_ids,
                "metadata": {"tool": "derive_t1", "half": label}})
            # basis="risk_group"/level="risk_group", NOT the physical/link
            # default: `decisions.py`'s own `_BASES` comment states this is
            # "load-bearing, not decorative" for exactly this avoid
            # mechanic (T2a/T3a/T3b's gold constraint decision only
            # validates under this basis/level pair), and T2a.yaml's own
            # widest_avoid_feasible narration ("route_service(avoid=
            # {risk_groups:[...]}) returns ZERO candidates under EITHER
            # basis" -- i.e. it checked both, and risk_group is the one
            # this field is named for) confirms this is the established
            # convention for this exact scalar, not a free choice.
            constraints = ConstraintDecision(
                avoid={"risk_groups": [rg_id]}, basis="risk_group",
                level="risk_group",
                reasoning="derive_t1 widest-avoid probe")
            menu = await call_tool_json(
                client, "route_service", constraints.route_service_args(args.sut))
            candidates = menu.get("candidates") or []
            feasible = len(candidates) > 0
            print(f"  half {label}: rg={rg_id!r} assets={len(fiber_ids)} "
                 f"status={menu.get('status')!r} candidates={len(candidates)} "
                 f"-> widest_avoid_feasible={feasible}")
            return feasible

        print("\nwidest_avoid_feasible (live route_service under this "
             "half's own t3 cone as a risk-group avoid):")
        widest_feasible = {
            "A": await widest_avoid_feasible("A", toward_cone),
            "B": await widest_avoid_feasible("B", away_cone),
        }

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
            "exposure_legs": "both",
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
                id=id_, pair="T1", seed=args.seed, state_file=args.state,
                service_under_test=args.sut, track=args.track, hours=hours,
                decision_hour=args.decision_hour, lead_time_hours=args.lead_time,
                spares_on_hand=1, depot_site=args.depot,
                spare_inventory={args.depot: 1},
                damage_radius_km=args.damage_radius_km, reference_avoid={},
                forecast=forecast,
                realized={exposure_horizon: tuple(realized_ids)},
                gold=_PLACEHOLDER_GOLD, flip_variable=FLIP_VARIABLE_TOKENS,
                metadata=metadata)

        scenario_a = make_scenario("T1a", "conserve", toward_cone,
                                   claimant_fiber_ids, widest_feasible["A"])
        scenario_b = make_scenario("T1b", "spend", away_cone,
                                   sut_fiber_ids, widest_feasible["B"])

        # ---- claimed_competing_ecar_gbps/_at, from observation's own -----
        # AGENT-FACING figure (`_restorable_groups`, via build_observation),
        # not a separately-rounded figure of this tool's own -- so the
        # scenario's declared claim matches, bit for bit, what a real
        # rollout would actually show the agent (T1a.yaml/T2a.yaml's own
        # precedent, "two readers, two roundings, one quantity").
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
            claimed = groups[0]["ecar_gbps"]
            print(f"half {label} claimed_competing_ecar_gbps "
                 f"(observation._restorable_groups, largest group at "
                 f"{exposure_horizon!r}) = {claimed!r}  members="
                 f"{groups[0]['members']}")
            new_metadata = {**scenario.metadata,
                            "claimed_competing_ecar_gbps": claimed,
                            "claimed_competing_ecar_at": exposure_horizon}
            if label == "A":
                scenario_a = dataclasses.replace(scenario, metadata=new_metadata)
            else:
                scenario_b = dataclasses.replace(scenario, metadata=new_metadata)

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
    }


def _build_parser() -> argparse.ArgumentParser:
    """Extracted from `main()` so `tests/eval/test_derive_t1.py` can exercise
    the REAL parser (flag names, requiredness, defaults) without invoking
    `main()` itself, which would go on to `asyncio.run(derive(...))` against
    a real server."""
    parser = argparse.ArgumentParser(prog="derive_t1")
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--sut", required=True)
    parser.add_argument("--depot", required=True)
    parser.add_argument("--claimants", required=True,
                        help="comma-separated service ids, e.g. "
                             "claimant-x-y-fwd,claimant-x-y-rev")
    parser.add_argument("--escape-node", required=True,
                        help="the node the SUT's escape route reroutes "
                             "through -- recorded as metadata."
                             "escape_route_node (descriptive)")
    parser.add_argument("--hours", default="t0,t1,t2,t3,t4,t5")
    parser.add_argument("--decision-hour", default="t1")
    parser.add_argument("--lead-time", type=int, default=2)
    parser.add_argument("--width-km", type=float, default=90.0)
    parser.add_argument("--t0-radius", type=float, required=True)
    parser.add_argument("--t0-bearing", type=float, required=True)
    parser.add_argument("--toward-bearing", type=float, required=True)
    parser.add_argument("--toward-radius", type=float, required=True)
    parser.add_argument("--away-bearing", type=float, required=True)
    parser.add_argument("--damage-radius-km", type=float, default=74.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--track", default="hudhud")
    parser.add_argument("--out-dir",
                        default="src/storm_reoptimizer/eval/scenarios")
    parser.add_argument("--server-command", default=None,
                        help="JSON list; defaults to this workspace's "
                             "multilayer-optical-mcp conda-env workaround")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(derive(args))


if __name__ == "__main__":
    main()
