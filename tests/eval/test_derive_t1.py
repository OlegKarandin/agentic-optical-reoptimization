"""tools/derive_t1.py's `derive()` is live end to end (shells out to a real
MCP server, per this repo's own seam) and is exercised manually instead of
by an automated test -- see the Task 14 report for the real smoke-test run
and its output, matching test_find_sut.py's own precedent for its `evaluate`
function. This file tests the PURE helper functions instead: no server, no
solver, real and fast."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import derive_t1  # noqa: E402
import derive_t2  # noqa: E402
import derive_t3  # noqa: E402

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


def test_cut_tokens_and_span_and_probe_flip_parsers():
    assert derive_t1._cut_tokens("claimants, sut") == ["claimants", "sut"]
    assert derive_t1._cut_tokens("claimant:c-x") == ["claimant:c-x"]
    assert derive_t1._parse_alt_span("khandwa:dhar:out:in") == (
        "khandwa", "dhar", "out", "in")
    with pytest.raises(SystemExit):
        derive_t1._parse_alt_span("khandwa:dhar:maybe:in")
    flips = derive_t1._parse_probe_flip(
        "A:restorable:c1:restorable,B:restorable:c1:not_restorable")
    assert flips == {
        "A": {"kind": "restorable", "claimant": "c1", "expected": "restorable"},
        "B": {"kind": "restorable", "claimant": "c1",
              "expected": "not_restorable"}}


def test_realized_for_expands_claimant_and_sut_tokens_through_the_event_filter():
    oms_by_id = {
        "oms_jk": {"src_node_id": "jalgaon", "dst_node_id": "khandwa",
                   "elements": ["fiber_jalgaon_khandwa_0", "edfa_x"]},
        "oms_jb": {"src_node_id": "jalgaon", "dst_node_id": "buldhana",
                   "elements": ["fiber_jalgaon_buldhana_0"]},
        "oms_jd": {"src_node_id": "jalgaon", "dst_node_id": "dhulia",
                   "elements": ["fiber_jalgaon_dhulia_0"]},
        "oms_js": {"src_node_id": "jalgaon", "dst_node_id": "surat",
                   "elements": ["fiber_jalgaon_surat_0"]},   # buried: never cut
    }
    path_oms = {"s": {"working": ("oms_jb",), "protection": ("oms_jd",)},
                "ck": {"working": ("oms_jk",), "protection": ()},
                "cs": {"working": ("oms_js",), "protection": ()}}
    aerial = {("jalgaon", "khandwa"), ("khandwa", "jalgaon"),
              ("jalgaon", "buldhana"), ("buldhana", "jalgaon"),
              ("jalgaon", "dhulia"), ("dhulia", "jalgaon")}
    kw = dict(claimants=("ck", "cs"), sut="s", path_oms=path_oms,
              oms_by_id=oms_by_id, vulnerable_pairs=aerial)
    assert derive_t1._realized_for(["claimants"], protected=True, **kw) == [
        "fiber_jalgaon_khandwa_0"]
    assert derive_t1._realized_for(["claimant:ck", "sut"], protected=True, **kw) == [
        "fiber_jalgaon_buldhana_0", "fiber_jalgaon_dhulia_0",
        "fiber_jalgaon_khandwa_0"]
    assert derive_t1._realized_for(["sut"], protected=False, **kw) == [
        "fiber_jalgaon_buldhana_0"]
    with pytest.raises(SystemExit):
        derive_t1._realized_for(["claimant:nope"], protected=True, **kw)


def test_the_t1_parser_defaults_are_unchanged_and_the_new_flags_default_to_t1():
    args = derive_t1._build_parser().parse_args([
        "--sut", "s", "--depot", "d", "--claimants", "c", "--escape-node", "e",
        "--t0-radius", "40", "--t0-bearing", "210",
        "--toward-bearing", "210", "--toward-radius", "74"])
    assert (args.pair, args.sut_posture, args.half_a_cuts, args.half_b_cuts) == (
        "T1", "protected", "claimants", "sut")
    assert args.alt_span is None and args.probe_flip is None
    assert args.away_bearing is None and args.pcut_match_claimant is None
    assert args.require_flip_tie is False


def test_derive_t2_and_t3_defaults_are_the_specs():
    """T3's shape here is the FALLBACK design (Task 8,
    docs/superpowers/plans/notes/2026-09-06-t2-t3-authoring.md): the
    originally-drafted survivor-groom variant was checked live and the
    zero-spare groom was never offered, so T3 shipped as the two-corridor
    restorability variant instead (SUT + two claimants, no survivor pins;
    half A cuts one claimant, half B cuts the OTHER claimant AND the SUT;
    `probe_flip` kind `restorable`, not the primary design's
    `spares_needed`).

    The two corridors are `khandwa` and `buldhana`, not spec 4.2's
    `khandwa`/`dhulia`: the live solve gives the unprotected SUT a working
    leg whose only near-depot aerial span IS `jalgaon<->dhulia`, so a dhulia
    claimant is perfectly correlated with the SUT and its realized cut IS
    the SUT's own first-hop cut (2026-09-06, Task 11; see
    `tools/derive_t3.py`'s docstring and `T3_PINS`). `--require-flip-tie` is
    deliberately OFF for the same task's reason: four of the five FLIP_VARS
    are a network-wide sum that two different cones cannot tie bit for bit,
    and a full tie would fail `assert_flip_dominates` anyway."""
    t2 = derive_t2.build_parser().parse_args(
        ["--t0-radius", "1", "--t0-bearing", "0", "--toward-bearing", "350",
         "--toward-radius", "60", "--away-bearing", "358"])
    assert (t2.pair, t2.width_km, t2.damage_radius_km) == ("T2", 60.0, 50.0)
    assert t2.hours == "t0,t1,t2,t3,t4,t5,t6,t7"
    assert t2.sut == "t2-svc-jalgaon-nagpur" and t2.sut_posture == "protected"
    assert t2.claimants == "t2-claimant-jalgaon-khandwa"
    assert t2.state == "eval/states/t2-jalgaon-s17.json"
    assert (t2.half_a_cuts, t2.half_b_cuts) == ("claimants", "claimants,sut")
    assert t2.alt_span == ["khandwa:dhar:out:in"]
    assert derive_t1._parse_probe_flip(t2.probe_flip)["B"]["expected"] == "not_restorable"
    t3 = derive_t3.build_parser().parse_args(
        ["--t0-radius", "1", "--t0-bearing", "0", "--toward-bearing", "20",
         "--toward-radius", "60", "--away-bearing-range", "250:300"])
    assert (t3.pair, t3.sut_posture) == ("T3", "unprotected")
    assert t3.claimants == (
        "t3-claimant-jalgaon-khandwa,t3-claimant-jalgaon-buldhana")
    assert (t3.half_a_cuts, t3.half_b_cuts) == (
        "claimant:t3-claimant-jalgaon-khandwa",
        "claimant:t3-claimant-jalgaon-buldhana,sut")
    assert t3.alt_span == ["khandwa:dhar:out:out", "buldhana:amravati:in:in"]
    assert t3.pcut_match_claimant == (
        "t3-claimant-jalgaon-khandwa:t3-claimant-jalgaon-buldhana")
    assert t3.require_flip_tie is False
    assert t3.state == "eval/states/t3-jalgaon-s17.json"
    assert derive_t1._parse_probe_flip(t3.probe_flip) == {
        "A": {"kind": "restorable", "claimant": "t3-claimant-jalgaon-khandwa",
              "expected": "restorable"},
        "B": {"kind": "restorable", "claimant": "t3-claimant-jalgaon-buldhana",
              "expected": "not_restorable"}}
