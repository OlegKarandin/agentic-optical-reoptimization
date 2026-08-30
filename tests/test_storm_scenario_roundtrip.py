"""Full step-4 pipeline, end to end against a real server -- the storm
scenario's acceptance proof, the same evidentiary bar
test_step1_roundtrip.py and the geo-mapper's Task 6 already set. Nothing
here is hardcoded to a specific outcome except the demands' endpoints
(satna/allahabad, mumbai/pune -- seeded offline by tools/build_storm_state.py,
see the design spec's 2026-08-18 addendum), chosen for their real, verified
topology structure -- see
docs/superpowers/specs/2026-08-07-storm-scenario-design.md."""
import asyncio
from pathlib import Path

from storm_reoptimizer.events.filters import get_filter
from storm_reoptimizer.events.hudhud_track import hudhud_track
from storm_reoptimizer.geo_mapper import load_edges
from storm_reoptimizer.mcp_client import connect_server
from storm_reoptimizer.scenario_storm import run_storm_scenario

TOPOLOGY_PATH = (
    Path(__file__).parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)


async def _run(state_path, local_server_command, local_server_env):
    async with connect_server(
        TOPOLOGY_PATH, server_command=local_server_command, env=local_server_env,
        extra_args=["--state", str(state_path)],
    ) as client:
        # storm-svc-1 (satna<->allahabad) and unexposed-svc (mumbai<->pune,
        # far southwest, nowhere near the Hudhud corridor -- a real negative
        # case for exposed_services' filter) are already seeded by the
        # offline builder; no live solve_allocation call needed here.
        edges = load_edges(TOPOLOGY_PATH)
        events = hudhud_track(interval_hours=1.0)
        result = await run_storm_scenario(
            client, events, edges, get_filter("storm"), "storm_rg",
        )
        return result


def test_full_storm_pipeline_finds_the_forced_service_but_cannot_replan_it(
    storm_state_path, local_server_command, local_server_env,
):
    result = asyncio.run(
        _run(storm_state_path, local_server_command, local_server_env)
    )

    # The real multi-hour exposure from Task 2's numbers shows up here too
    # (order matches load_edges' fixed topology-JSON edge order -- verified
    # directly against src/storm_reoptimizer/data/toy_india_topology.json's
    # graph.edges array, which lists satna->rewa before satna->jhansi; the
    # brief's inline comment claiming the reverse was a documentation slip,
    # not a code issue -- see task-7-report.md). satna->jabalpur trails the
    # other two, matching that same fixed edge order -- it's the third
    # aerial direction added in the exposure-and-depot design (§3.2,
    # Option B) and it sits inside the same storm cone this hour.
    assert result.exposed_by_hour["2014-10-13T23:30:00+00:00"] == [
        ("satna", "rewa"), ("satna", "jhansi"), ("satna", "jabalpur"),
    ]

    exposed_ids = {s["service"]["id"] for s in result.exposed_services}
    assert "storm-svc-1" in exposed_ids

    # satna has degree 3 and all three of its spans are now aerial (§3.2,
    # Option B); this hour's cone -- run over the FULL forecast track, per
    # run_storm_scenario's "reroute clear of the full forecast cone" design
    # (CLAUDE.md, scenario 1) -- covers all three simultaneously. A service
    # terminating at satna therefore has no egress left outside the risk
    # group at all: "avoid the risk group" and "reach satna" are mutually
    # exclusive, so no_solution is the correct, physically honest outcome,
    # not a solver defect. The old buried-jabalpur topology masked this real
    # vulnerability -- a storm wide enough to reach every aerial direction
    # out of a hub really does strand it. (This is the real historical
    # Hudhud track against the real topology, not a synthetic fixture --
    # unlike tests/eval/test_runner.py's EXPOSURE_SMOKE, there is no cone to
    # retune here without falsifying the scenario.)
    replan = result.replans["storm-svc-1"]
    assert replan["status"] == "no_solution"
    assert replan["path_a"] is None and replan["path_b"] is None

    # No candidate route exists, so nothing was QoT-verified for this
    # service -- verify_qot is never called for it (see run_storm_scenario's
    # `if replan.get("path_a") is not None` gate).
    assert "storm-svc-1" not in result.qot_checks

    assert isinstance(result.validation["violations"], list)


def test_the_unexposed_service_is_excluded_from_exposed_services(
    storm_state_path, local_server_command, local_server_env,
):
    result = asyncio.run(
        _run(storm_state_path, local_server_command, local_server_env)
    )
    exposed_ids = {s["service"]["id"] for s in result.exposed_services}
    assert "storm-svc-1" in exposed_ids
    assert "unexposed-svc" not in exposed_ids
    # It was still audited (get_exposure is called for every service), just
    # not both_intersect -- proves exposed_services is a real filter over a
    # real full audit, not "return everything that was ever checked."
    audited_ids = {a["service"]["id"] for a in result.audited_services}
    assert "unexposed-svc" in audited_ids
