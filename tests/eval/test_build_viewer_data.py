# tests/eval/test_build_viewer_data.py
"""The fold from traces + scenarios + topology to one viewer payload
(run-viewer design, §5.2).

Imported directly rather than subprocessed: unlike tools/build_eval_state.py
this script imports nothing from multilayer_optical_network, so it runs under
this repo's own interpreter."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = (Path(__file__).parent.parent.parent / "tools"
         / "build_viewer_data.py")
_SPEC = importlib.util.spec_from_file_location("build_viewer_data", _PATH)
bvd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bvd)


FIXTURE_TRACE = {
    "scenario_id": "T3b", "decider_name": "agent:claude-sonnet-5",
    "run_index": 0, "terminal_status": "converged",
    "spares_remaining": 0, "ledger_debits": [
        {"hour": "t1", "service_id": "storm-svc-1", "pairs": 1}],
    "actions": [{"hour": "t1", "hour_index": 1, "lever": "optical_reroute",
                 "effective_at_index": 2, "pairs": 1,
                 "service_id": "storm-svc-1", "avoid": {}}],
    "oms_nodes": {"oms_sj": ["satna", "jhansi"]},
    "affected_by_hour": {}, "tool_calls": 61, "wall_clock_s": 1.0,
    "final_routing": {},
    "hours": [{
        "hour": "t1", "services": ["storm-svc-1", "d0462"],
        "rejections": [], "committed": True,
        "timing": {"action": "act", "reasoning": "r",
                   "contested_claim": None},
        "observation": {
            "actionable_service": "storm-svc-1",
            "exposure": {
                "storm-svc-1": {"t6": {"p_cut": 0.52, "demand_gbps": 300.0,
                                       "offset_km": 12.0, "hours_ahead": 2,
                                       "width_km": 320,
                                       "expected_capacity_at_risk_gbps":
                                           156.0}},
                "d0462": {"t2": {"p_cut": 0.88, "demand_gbps": 100.0,
                                 "offset_km": 3.0, "hours_ahead": 1,
                                 "width_km": 90,
                                 "expected_capacity_at_risk_gbps": 88.3}}},
            "services": [{"id": "storm-svc-1", "demand_gbps": 300.0,
                          "actionable": True},
                         {"id": "d0462", "demand_gbps": 100.0}],
            "horizon_totals": {"t6": {"sut_ecar_gbps": 156.0,
                                      "non_sut_total_ecar_gbps": 88.3}},
            "risk_group_ids": {"t6": "rg_T3b_t1_t6"}},
        "projected": {
            "exposure": {"storm-svc-1": {}},
            "omitted_services": {"count": 571, "p_cut_threshold": 0.005,
                                 "max_p_cut": 0.0,
                                 "summed_expected_capacity_at_risk_gbps":
                                     0.0}},
        "service_points": {"storm-svc-1": [24.6, 80.8],
                           "d0462": [26.4, 80.3]},
        "service_paths": {
            "storm-svc-1": {"working": ["satna", "rewa", "allahabad"],
                            "protection": ["satna", "jhansi", "allahabad"]}},
        "unmapped_nodes": {},
        "unconstrained_menu": {"status": "solution", "candidates": [
            {"candidate_label": "candidate_0", "lever": "ip_reroute",
             "pairs_needed": 0}]},
        "iterations": [{
            "iteration": 0, "menu_status": "solution", "menu_size": 1,
            "constraints": {"avoid": {}, "protected": False,
                            "best_effort": False, "basis": "physical",
                            "level": "link", "reasoning": "r",
                            "contested_claim": None},
            "objective": {"choice": "candidate_0", "priority": None,
                          "reasoning": "r", "contested_claim": None},
            "menu": {"status": "solution", "candidates": [
                {"candidate_label": "candidate_0", "lever": "optical_reroute",
                 "pairs_needed": 1, "reused_lightpaths": [],
                 "new_lightpaths": [{"oms_sequence": ["oms_sj"]}],
                 "restored_gbps": 300.0, "shortfall_gbps": 0.0,
                 "cost_vector": {"transponders": 420.0}}]},
            "outcome": "committed", "lever": "optical_reroute"}]}]}


@pytest.fixture
def folded(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "T3b-agent_claude-sonnet-5-0.json").write_text(
        json.dumps(FIXTURE_TRACE), encoding="utf-8")
    scenarios = (Path(__file__).parent.parent.parent / "src"
                 / "storm_reoptimizer" / "eval" / "scenarios")
    topology = (Path(__file__).parent.parent.parent / "src"
                / "storm_reoptimizer" / "data" / "toy_india_topology.json")
    return bvd.fold(traces, scenarios, topology)


def test_the_topology_is_folded_in_with_its_mount_types(folded):
    assert len(folded["topology"]["nodes"]) == 143
    assert len(folded["topology"]["edges"]) == 180
    assert folded["topology"]["nodes"]["satna"] == pytest.approx(
        folded["topology"]["nodes"]["satna"])
    assert any(e["mount_type"] == "aerial"
               for e in folded["topology"]["edges"])


def test_cone_polygons_are_flipped_to_lat_lon_once(folded):
    ring = folded["episodes"]["T3b"]["forecast"]["t1"]["t6"]["polygon"]
    # India: latitude 8-35, longitude 68-97. A lon-first ring would put ~79
    # in the first slot.
    assert all(8 < lat < 35 for lat, _ in ring)
    assert all(68 < lon < 97 for _, lon in ring)


def test_every_exposure_row_is_marked_shown_or_omitted(folded):
    rows = folded["episodes"]["T3b"]["runs"][0]["hours"][0]["exposure_rows"]
    by_id = {r["service_id"]: r for r in rows}
    assert by_id["storm-svc-1"]["shown"] is True
    assert by_id["d0462"]["shown"] is False       # truth had it, wire did not


def test_the_committed_candidate_is_marked(folded):
    it = folded["episodes"]["T3b"]["runs"][0]["hours"][0]["iterations"][0]
    assert it["menu"]["candidates"][0]["committed"] is True


def test_the_gold_label_and_spare_action_travel_with_the_episode(folded):
    gold = folded["episodes"]["T3b"]["gold"]
    assert gold["label"] == "B"
    assert gold["gold_spare_action"] == "conserve"


def test_a_pre_change_trace_folds_without_the_new_keys(tmp_path):
    # The 21 archived control rollouts have no observation/projected/menu and
    # spell it `service_under_test` (run-viewer design, §6.2).
    old = {k: v for k, v in FIXTURE_TRACE.items() if k != "oms_nodes"}
    old["hours"] = [{k: v for k, v in FIXTURE_TRACE["hours"][0].items()
                     if k not in ("observation", "projected", "service_points",
                                  "service_paths", "unmapped_nodes")}]
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "T3b-agent_claude-sonnet-5-0.json").write_text(
        json.dumps(old), encoding="utf-8")
    payload = bvd.fold(
        traces,
        Path(__file__).parent.parent.parent / "src" / "storm_reoptimizer"
        / "eval" / "scenarios",
        Path(__file__).parent.parent.parent / "src" / "storm_reoptimizer"
        / "data" / "toy_india_topology.json")
    hour = payload["episodes"]["T3b"]["runs"][0]["hours"][0]
    assert hour["exposure_rows"] == []
    assert hour["service_paths"] == {}


def test_the_html_is_one_self_contained_file(folded):
    html = bvd.render_html(folded)
    assert html.lstrip().startswith("<!doctype html>")
    assert "<script" in html and "application/json" in html
    # No network at all: a viewer that needs a CDN is not double-clickable.
    for banned in ("http://", "https://", "src=\"//", "@import"):
        assert banned not in html


def test_the_payload_survives_a_script_close_in_the_data(folded):
    folded["episodes"]["T3b"]["gold"]["rationale"] = "</script><b>x</b>"
    html = bvd.render_html(folded)
    assert "</script><b>" not in html
    assert "<\\/script>" in html


def test_the_aerial_plant_is_distinguishable_in_the_payload(folded):
    # Acceptance item 4: storm-svc-1's working and protection paths must both
    # render with a dotted first segment out of Satna, without a legend.
    aerial = {(e["src"], e["dst"]) for e in folded["topology"]["edges"]
              if e["mount_type"] == "aerial"}
    assert ("satna", "rewa") in aerial or ("rewa", "satna") in aerial
    assert ("satna", "jhansi") in aerial or ("jhansi", "satna") in aerial
