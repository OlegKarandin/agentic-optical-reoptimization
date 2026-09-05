"""tools/derive_t1.py's `derive()` is live end to end (shells out to a real
MCP server, per this repo's own seam) and is exercised manually instead of
by an automated test -- see the Task 14 report for the real smoke-test run
and its output, matching test_find_sut.py's own precedent for its `evaluate`
function. This file tests the PURE helper functions instead: no server, no
solver, real and fast."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import derive_t1  # noqa: E402

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)


def test_node_coords_reads_the_real_topology():
    """Pure JSON parse against the real, checked-in toy topology -- no
    server. satna is the depot every T1 construction so far has used."""
    coords = derive_t1._node_coords(TOPOLOGY_PATH)
    assert coords["satna"] == (24.58333, 80.83333)
    assert len(coords) > 100   # the toy topology has many nodes


def test_fiber_ids_filters_to_fiber_prefixed_elements_only():
    """Same filter runner.horizon_risk_group_asset_ids applies (fiber_*
    only), just keyed by a single OMS id instead of walking a hazard
    footprint's whole edge set."""
    oms_by_id = {
        "oms_ab": {"elements": ["fiber_a_b_0", "fiber_a_b_1", "edfa_a_b_0"]},
        "oms_cd": {"elements": ["fiber_c_d_0"]},
    }
    assert derive_t1._fiber_ids(oms_by_id, "oms_ab") == [
        "fiber_a_b_0", "fiber_a_b_1"]
    assert derive_t1._fiber_ids(oms_by_id, "oms_cd") == ["fiber_c_d_0"]


def test_elapsed_hours_is_the_index_difference():
    hours = ("t0", "t1", "t2", "t3", "t4", "t5")
    assert derive_t1._elapsed_hours(hours, "t0", "t1") == 1
    assert derive_t1._elapsed_hours(hours, "t0", "t3") == 3
    assert derive_t1._elapsed_hours(hours, "t1", "t0") == -1


def test_cli_parses_every_flag_the_brief_names():
    """Exercises the REAL parser `main()` uses (`_build_parser`, extracted
    for exactly this reason) -- no server, no `derive()` call. Confirms
    every flag name in the task brief's own CLI interface line is wired
    with the expected default/required shape, so a typo in a flag name
    would fail here rather than only at live-run time."""
    args = derive_t1._build_parser().parse_args([
        "--sut", "storm-svc-1", "--depot", "satna",
        "--claimants", "claimant-a,claimant-b",
        "--escape-node", "jabalpur",
        "--t0-radius", "40", "--t0-bearing", "210",
        "--toward-bearing", "210", "--toward-radius", "74",
        "--away-bearing", "0",
    ])
    assert args.hours == "t0,t1,t2,t3,t4,t5"
    assert args.decision_hour == "t1"
    assert args.lead_time == 2
    assert args.width_km == 90.0
    assert args.claimants == "claimant-a,claimant-b"
    assert args.topology == derive_t1.DEFAULT_TOPOLOGY
    assert args.state == derive_t1.DEFAULT_STATE
    assert args.out_dir == "src/storm_reoptimizer/eval/scenarios"
