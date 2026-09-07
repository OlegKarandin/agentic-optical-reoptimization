# tools/probe_episode.py
"""Episode-authoring probe (eval design spec, "Rehearsal: author every
episode by walking it").

Episode YAML cannot be written blind: "storm-svc-1 at the cone's edge (~25%
containment) while svc-b is dead-centre (~90%)" is a claim about real
coordinates in the real loaded state, and "D1's menu contains no ip_reroute"
is a claim about a real menu. This prints those numbers so the author writes
down what is true rather than what sounds right.

Two more probes, both gated on an ALREADY-WRITTEN scenario YAML (`--scenario
PATH`) rather than raw lat/lon/width -- the T1 spend-or-hold redesign spec's
own tooling section (2026-09-05, §7):

  * `--dump-prompt HOUR` prints the exact projected payload the agent would
    receive at that hour, with no API key: the episode runs against a decider
    that always answers "wait", so decision 2/3 never fire and no lever is
    ever costed or committed. `project_observation` is the SAME function
    agent.py's ClaudeDecider calls before every real prompt, so what this
    prints is not a reconstruction of what the agent sees -- it is a
    byte-identical read of it.
  * `--post-commit CANDIDATE_LABEL` answers "if I actually commit this menu
    entry, what happens to the SUT's and every claimant's real exposure" --
    as against `menu_with_path_facts`'s `residual_exposure`, which is the
    harness's PROJECTION of that answer, computed before anything is
    committed. Branches off a fresh snapshot, commits for real, re-reads
    `service_geometry` from the live (mutated) server, prints the honest
    numbers, then restores -- so running this probe never leaves a mark on
    the state file.

Runs in THIS repo's env over MCP -- it is NOT the offline-build exception and
imports nothing from multilayer_optical_network."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from storm_reoptimizer.eval.agent import project_observation
from storm_reoptimizer.eval.cone import (
    cone_polygon, p_cut_point, p_cut_service, radial_offset_km,
)
from storm_reoptimizer.eval.decisions import TimingDecision
from storm_reoptimizer.eval.ledger import spares_needed
from storm_reoptimizer.eval.observation import latest_issuance
from storm_reoptimizer.eval.plans import build_topology_index
from storm_reoptimizer.eval.runner import (
    _CountingClient, menu_for_prompt, run_episode, service_geometry,
    service_points, try_commit,
)
from storm_reoptimizer.eval.scenario_file import load_scenario
from storm_reoptimizer.events.filters import get_filter
from storm_reoptimizer.geo_mapper import load_edges, map_geo_event_to_assets
from storm_reoptimizer.mcp_client import call_tool_json, connect_server

# tools/ -> repo root. scenario_file.py records `state_file` relative to
# here (every scenario YAML's "eval/states/..." value; see suite.py's own
# REPO_ROOT for the same resolution rule), not relative to the caller's cwd.
REPO_ROOT = Path(__file__).parent.parent


class _WaitsAlways:
    """The trivial decider `--dump-prompt` runs the episode against: every
    hour is a "wait", so decision 2/3 (`constraints`/`objective`) never fire
    and no server-mutating call is ever made. This is what lets the probe
    need no API key and leave the loaded state untouched -- there is nothing
    to branch or restore because nothing was ever committed."""

    name = "probe:dump-prompt"

    async def timing(self, obs) -> TimingDecision:
        return TimingDecision(action="wait", reasoning="probe_episode "
                              "--dump-prompt: never acts",
                              contested_claim=None, claim_priority=())

    async def constraints(self, obs, unconstrained_menu=None):
        raise NotImplementedError("timing() always waits; never reached")

    async def objective(self, obs, menu):
        raise NotImplementedError("timing() always waits; never reached")


class _RecordingDecider:
    """Copied from tests/eval/test_runner.py:397-417 (`_RecordingDecider`),
    verbatim, per the plan's own instruction to reuse rather than
    reimplement it. Wraps a real decider and keeps every observation the
    harness handed it. Lets a caller read what the rollout SHOWED the
    decider, which the trace does not record -- only the decisions come back
    in EpisodeTrace."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.observations = []          # one per hour: the timing observation
        self.probes = []                # one per constraints() call

    async def timing(self, obs):
        self.observations.append(obs)
        return await self.inner.timing(obs)

    async def constraints(self, obs, unconstrained_menu=None):
        self.probes.append(unconstrained_menu)
        return await self.inner.constraints(obs, unconstrained_menu)

    async def objective(self, obs, menu):
        return await self.inner.objective(obs, menu)


async def probe(topology: Path, state: Path, service: str, lat: float,
                lon: float, width_km: float, damage_radius_km: float,
                avoid: dict, server_command: list[str] | None,
                env: dict | None) -> None:
    async with connect_server(topology, server_command=server_command,
                              env=env,
                              extra_args=["--state", str(state)]) as client:
        points = await service_points(client, topology)
        services = (await call_tool_json(client, "get_services"))["services"]

        print(f"\n=== exposure at cone ({lat}, {lon}) width {width_km} km ===")
        print(f"{'service':<24}{'demand':>8}{'offset_km':>11}{'p_cut':>8}")
        for svc in sorted(services, key=lambda s: s["id"]):
            point = points.get(svc["id"])
            if point is None:
                continue
            offset = radial_offset_km(lat, lon, *point)
            p_cut = p_cut_point(offset, width_km, damage_radius_km)
            print(f"{svc['id']:<24}{svc['demand_gbps']:>8.0f}"
                  f"{offset:>11.1f}{p_cut:>8.2f}")

        edges = load_edges(topology)
        exposed = map_geo_event_to_assets(
            cone_polygon(lat, lon, width_km), edges, get_filter("storm"))
        print(f"\n=== exposed aerial edges ({len(exposed)}) ===")
        for edge in exposed:
            print(f"  {edge.src} <-> {edge.dst}")

        # oms_id -> [src_node_id, dst_node_id] -- same shape
        # runner.service_geometry/assertions._oms_nodes build, needed to
        # resolve a candidate's new_lightpaths to endpoint SITES.
        optical = await call_tool_json(client, "get_topology",
                                       {"layer": "optical"})
        oms_nodes = {o["id"]: [o["src_node_id"], o["dst_node_id"]]
                    for o in optical["oms"]}

        menu = await call_tool_json(client, "route_service", {
            "service_id": service, "protected": False, "basis": "physical",
            "level": "link", "best_effort": False, "avoid": avoid})
        print(f"\n=== route_service({service}, avoid={avoid}) -> "
              f"{menu['status']} ===")
        for i, cand in enumerate(menu["candidates"]):
            print(f"  candidate_{i}: lever={cand['lever']:<16} "
                  f"restored={cand['restored_gbps']:>6.0f}G "
                  f"shortfall={cand['shortfall_gbps']:>6.0f}G "
                  f"spares={spares_needed(cand, oms_nodes)}")
            print(f"      {json.dumps(cand['cost_vector'])}")
        if not menu["candidates"]:
            print("  (empty menu)")


async def dump_prompt(topology: Path, scenario_path: Path, hour: str,
                      server_command: list[str] | None,
                      env: dict | None) -> None:
    """Run `scenario_path` end to end against `_WaitsAlways` (no API key,
    nothing ever committed) and print the exact projected payload
    `agent.ClaudeDecider` would have sent the model at `hour`.

    A full rollout, not a single-hour reconstruction: `build_observation`
    needs the running `actions_taken`/`spares_spent`/`ledger` state and the
    current issuance, all of which only exist by walking every earlier hour
    first -- exactly what `run_episode` already does."""
    scenario = load_scenario(scenario_path)
    if hour not in scenario.hours:
        raise SystemExit(f"{hour!r} is not one of this episode's hours "
                         f"{scenario.hours}")
    state = REPO_ROOT / scenario.state_file
    decider = _RecordingDecider(_WaitsAlways())
    async with connect_server(topology, server_command=server_command,
                              env=env,
                              extra_args=["--state", str(state)]) as client:
        await run_episode(client, scenario, decider, topology_path=topology)

    obs = next((o for o in decider.observations if o.hour == hour), None)
    if obs is None:
        raise SystemExit(f"no observation was recorded at {hour!r} (rollout "
                         f"ended early?); recorded hours were "
                         f"{[o.hour for o in decider.observations]}")
    print(json.dumps(project_observation(obs), indent=2))


async def post_commit(topology: Path, scenario_path: Path,
                      candidate_label: str, avoid: dict,
                      server_command: list[str] | None,
                      env: dict | None) -> None:
    """Branch off the loaded state, commit `candidate_label` from
    `route_service`'s own (unconstrained, `avoid`-only) menu for the
    scenario's service under test, re-read `service_geometry` from the now-
    mutated server, print the SUT's and every declared claimant's REAL p_cut
    at each horizon of the issuance in force at `scenario.decision_hour`,
    then restore -- so the state file this reads is exactly as it found it.

    This is the honest counterpart to `menu_with_path_facts`'s
    `residual_exposure`: that field is the harness's PROJECTION of this same
    answer, computed from the candidate's own `new_lightpaths` before
    anything is committed. Both should agree; this is how you'd catch it if
    they didn't."""
    scenario = load_scenario(scenario_path)
    state = REPO_ROOT / scenario.state_file
    claimants = tuple(scenario.metadata.get("claimant_services", ()))
    issuance = latest_issuance(scenario, scenario.decision_hour)

    async with connect_server(topology, server_command=server_command,
                              env=env,
                              extra_args=["--state", str(state)]) as client:
        base_id = (await call_tool_json(client, "snapshot_create"))["id"]
        await call_tool_json(client, "snapshot_branch", {"parent_id": base_id})
        try:
            # menu_for_prompt's spares_needed resolves each candidate's
            # lightpaths to endpoint SITES through this map -- same shape and
            # same source as probe()'s own oms_nodes above.
            optical = await call_tool_json(client, "get_topology",
                                           {"layer": "optical"})
            oms_nodes = {o["id"]: [o["src_node_id"], o["dst_node_id"]]
                        for o in optical["oms"]}
            menu = await call_tool_json(client, "route_service", {
                "service_id": scenario.service_under_test, "protected": False,
                "basis": "physical", "level": "link", "best_effort": False,
                "avoid": avoid})
            labeled = menu_for_prompt(menu, oms_nodes)
            candidates = {c["candidate_label"]: c
                         for c in labeled["candidates"]}
            if candidate_label not in candidates:
                print(f"no candidate {candidate_label!r} in this menu "
                      f"(status={menu.get('status')}); have "
                      f"{sorted(candidates)}")
                return
            candidate = candidates[candidate_label]

            index = await build_topology_index(client)
            counting = _CountingClient(client)
            commit, rejection = await try_commit(
                counting, index, candidate, scenario.service_under_test,
                prefix=f"probe-post-commit-{candidate_label}",
                basis="physical", level="link")
            if rejection is not None:
                print(f"commit of {candidate_label} "
                      f"({candidate['lever']}) REJECTED: {rejection}")
                return
            print(f"committed {candidate_label} ({candidate['lever']}): "
                 f"{commit}")

            geometry = await service_geometry(client, topology)
            print(f"\n=== post-commit exposure at {scenario.decision_hour} "
                 f"({scenario.service_under_test} + {len(claimants)} "
                 f"claimant(s)) ===")
            print(f"{'service':<36}{'role':>10}{'horizon':>8}{'p_cut':>8}")
            for svc_id in (scenario.service_under_test, *claimants):
                working = geometry.cuttable_spans.get(svc_id, ())
                protection = geometry.protection_cuttable_spans.get(svc_id)
                role = ("SUT" if svc_id == scenario.service_under_test
                       else "claimant")
                for horizon, cone in issuance.horizons.items():
                    p_cut = p_cut_service(
                        working, protection, cone.center["lat"],
                        cone.center["lon"], cone.width_km,
                        scenario.damage_radius_km)
                    print(f"{svc_id:<36}{role:>10}{horizon:>8}{p_cut:>8.3f}")
        finally:
            await call_tool_json(client, "snapshot_restore",
                                 {"snapshot_id": base_id})


def main() -> None:
    p = argparse.ArgumentParser(prog="probe_episode")
    p.add_argument("--topology", required=True, type=Path)
    # Required for the original lat/lon/width probe; --dump-prompt and
    # --post-commit instead read the state file named by --scenario's own
    # `state_file`, so neither requires this.
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--service", default="storm-svc-1")
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--width-km", type=float, default=None)
    p.add_argument("--damage-radius-km", default=74.0, type=float)
    p.add_argument("--avoid", default="{}",
                   help='JSON, e.g. \'{"risk_groups": ["rg_t3"]}\'')
    p.add_argument("--server-command", default=None,
                   help="JSON list; see mcp_client.DEFAULT_SERVER_COMMAND")
    p.add_argument("--scenario", type=Path, default=None,
                   help="Episode YAML (scenario_file.load_scenario); "
                        "required by --dump-prompt/--post-commit.")
    p.add_argument("--dump-prompt", default=None, metavar="HOUR",
                   help="Run --scenario against a wait-only decider (no API "
                        "key) and print project_observation(obs) for HOUR.")
    p.add_argument("--post-commit", default=None, metavar="CANDIDATE_LABEL",
                   help="Branch, commit this route_service menu label for "
                        "--scenario's SUT, print its and every claimant's "
                        "real post-commit p_cut per horizon, then restore.")
    args = p.parse_args()
    server_command = (json.loads(args.server_command)
                      if args.server_command else None)

    if args.dump_prompt is not None:
        if args.scenario is None:
            p.error("--dump-prompt requires --scenario")
        asyncio.run(dump_prompt(args.topology, args.scenario,
                                args.dump_prompt, server_command, None))
        return
    if args.post_commit is not None:
        if args.scenario is None:
            p.error("--post-commit requires --scenario")
        asyncio.run(post_commit(args.topology, args.scenario,
                                args.post_commit, json.loads(args.avoid),
                                server_command, None))
        return

    if args.state is None or args.lat is None or args.lon is None \
            or args.width_km is None:
        p.error("--state, --lat, --lon and --width-km are required unless "
                "--dump-prompt or --post-commit is given")
    asyncio.run(probe(
        args.topology, args.state, args.service, args.lat, args.lon,
        args.width_km, args.damage_radius_km, json.loads(args.avoid),
        server_command, None))


if __name__ == "__main__":
    main()
