# tests/eval/test_build_viewer_data.py
"""The fold from traces + scenarios + topology to one viewer payload
(run-viewer design, §5.2).

Imported directly rather than subprocessed: unlike tools/build_eval_state.py
this script imports nothing from multilayer_optical_network, so it runs under
this repo's own interpreter."""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

_PATH = (Path(__file__).parent.parent.parent / "tools"
         / "build_viewer_data.py")
_SPEC = importlib.util.spec_from_file_location("build_viewer_data", _PATH)
bvd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bvd)


# Field names and nesting here are checked against a REAL trace: this is not
# hand-guessed. `spares_needed`/`Action.spares`/`SpareLedger.debits`'s dict
# shape, the 3-key `horizon_totals`, and the 2-bucket `omitted_services` were
# all confirmed by running a scripted, always-acting episode against the
# live server (SMOKE, exposure-and-depot's own smoke fixture in
# tests/eval/test_runner.py) and inspecting the resulting EpisodeTrace.to_dict()
# JSON directly -- see the final-review fix wave's report for the exact
# command. The narrative values below (which service, which numbers) are
# still curated by hand for readability, as the ORIGINAL fixture's were; only
# the shape was stale (whole-branch final review, finding 2), and that is
# what this fixture now certifies.
FIXTURE_TRACE = {
    "scenario_id": "T3b", "decider_name": "agent:claude-sonnet-5",
    "run_index": 0, "terminal_status": "converged",
    "spares_remaining": 0, "ledger_debits": [
        {"hour": "t1", "service_id": "storm-svc-1",
         "spares": {"satna": 1, "allahabad": 1}}],
    "actions": [{"hour": "t1", "hour_index": 1, "lever": "optical_reroute",
                 "effective_at_index": 2,
                 "spares": {"satna": 1, "allahabad": 1},
                 "service_id": "storm-svc-1", "avoid": {}}],
    "oms_nodes": {"oms_sj": ["satna", "jhansi"]},
    # EpisodeTrace.restorations (Task 7): the deterministic post-cut replay's
    # own records, flattened across the episode -- replay._empty_record's
    # shape plus the "restored" outcome's real lever/spares/effective_at_hour
    # (replay.py line ~202).
    "restorations": [
        {"hour": "t1", "service_id": "d0462", "outcome": "restored",
         "lever": "optical_reroute", "spares": {"satna": 1},
         "effective_at_hour": "t2", "rejection": None}],
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
                                           156.0,
                                       # revision.py's forecast-revision band
                                       # (2026-09-10, spec 5.1) -- carried
                                       # through load_run untouched, same as
                                       # every other exposure-row field.
                                       "p_cut_if_track_revised": {
                                           "revision_radius_km": 30.0,
                                           "min": 0.41, "max": 0.63,
                                           "mean": 0.52}}},
                "d0462": {"t2": {"p_cut": 0.88, "demand_gbps": 100.0,
                                 "offset_km": 3.0, "hours_ahead": 1,
                                 "width_km": 90,
                                 "expected_capacity_at_risk_gbps": 88.3}}},
            "services": [{"id": "storm-svc-1", "demand_gbps": 300.0,
                          "actionable": True},
                         {"id": "d0462", "demand_gbps": 100.0}],
            # observation.py's real 3-key shape (Task 10):
            # sut_ecar_gbps / largest_restorable_group_ecar_gbps /
            # non_sut_ineligible_ecar_gbps. The old fixture's
            # `non_sut_total_ecar_gbps` never existed in this shape.
            "horizon_totals": {"t6": {"sut_ecar_gbps": 156.0,
                                      "largest_restorable_group_ecar_gbps":
                                          88.3,
                                      "non_sut_ineligible_ecar_gbps": 88.3}},
            "risk_group_ids": {"t6": "rg_T3b_t1_t6"}},
        "projected": {
            "exposure": {"storm-svc-1": {}},
            # agent.py's real 2-bucket shape: below_threshold (quiet) and
            # ineligible_for_depot (badly exposed but nowhere the depot can
            # reach), each its own {count, max_p_cut,
            # summed_expected_capacity_at_risk_gbps} rollup. The old
            # fixture's single flat bucket never existed in this shape.
            "omitted_services": {
                "p_cut_threshold": 0.005,
                "below_threshold": {"count": 569, "max_p_cut": 0.0,
                                    "summed_expected_capacity_at_risk_gbps":
                                        0.0},
                "ineligible_for_depot": {"count": 2, "max_p_cut": 0.9,
                                         "summed_expected_capacity_at_risk_gbps":
                                             90.0}}},
        "service_points": {"storm-svc-1": [24.6, 80.8],
                           "d0462": [26.4, 80.3]},
        "service_paths": {
            "storm-svc-1": {"working": ["satna", "rewa", "allahabad"],
                            "protection": ["satna", "jhansi", "allahabad"]}},
        "unmapped_nodes": {},
        "unconstrained_menu": {"status": "solution", "candidates": [
            {"candidate_label": "candidate_0", "lever": "ip_reroute",
             "spares_needed": {}}]},
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
                 "spares_needed": {"satna": 1, "allahabad": 1},
                 "reused_lightpaths": [],
                 "new_lightpaths": [{"oms_sequence": ["oms_sj"]}],
                 "restored_gbps": 300.0, "shortfall_gbps": 0.0,
                 "cost_vector": {"transponders": 420.0}}]},
            "outcome": "committed", "lever": "optical_reroute"}]},
        # Decidable-hours rule (2026-09-10, spec 7.1): a skipped hour still
        # carries a full hour record, just with no projection and no
        # iterations. `load_run` copies hour records straight through, so
        # this is a regression guard against a future trim swallowing the
        # `timing.skipped` flag.
        {"hour": "t2", "services": ["storm-svc-1", "d0462"],
         "rejections": [], "committed": False,
         "timing": {"action": "wait",
                    "reasoning": "skipped: nothing decidable",
                    "contested_claim": None, "claim_priority": [],
                    "skipped": True},
         "projected": None, "iterations": [], "observation": {},
         "unmapped_nodes": {}}]}


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
    # `t3`, not the retired 2026-08-31 T3's `t6`: the re-authored T3 (plan
    # 2026-09-06, Task 11) runs on consecutive hour labels t0..t7 with the
    # exposure horizon at t3, like T1 and T2.
    ring = folded["episodes"]["T3b"]["forecast"]["t1"]["t3"]["polygon"]
    # India: latitude 8-35, longitude 68-97. A lon-first ring would put ~79
    # in the first slot.
    assert all(8 < lat < 35 for lat, _ in ring)
    assert all(68 < lon < 97 for _, lon in ring)


def test_every_exposure_row_is_marked_shown_or_omitted(folded):
    rows = folded["episodes"]["T3b"]["runs"][0]["hours"][0]["exposure_rows"]
    by_id = {r["service_id"]: r for r in rows}
    assert by_id["storm-svc-1"]["shown"] is True
    assert by_id["d0462"]["shown"] is False       # truth had it, wire did not


def test_a_skipped_hour_survives_load_run_untouched(folded):
    # Decidable-hours rule (2026-09-10, spec 7.1). `load_run` copies hour
    # records straight through -- this is a regression guard against a
    # future trim silently swallowing the `timing.skipped` flag the viewer
    # keys its skip rendering off of.
    hour = folded["episodes"]["T3b"]["runs"][0]["hours"][1]
    assert hour["hour"] == "t2"
    assert hour["timing"]["skipped"] is True
    assert hour["projected"] is None
    assert hour["iterations"] == []


def test_a_forecast_revision_band_survives_load_run_untouched(folded):
    # revision.py's `p_cut_if_track_revised` (2026-09-10, spec 5.1) is on the
    # observation entry `_exposure_rows` spreads via `**entry` -- assert it
    # is not dropped on the way into the folded payload's exposure_rows.
    rows = folded["episodes"]["T3b"]["runs"][0]["hours"][0]["exposure_rows"]
    by_id = {r["service_id"]: r for r in rows}
    assert by_id["storm-svc-1"]["p_cut_if_track_revised"] == {
        "revision_radius_km": 30.0, "min": 0.41, "max": 0.63, "mean": 0.52}


def test_the_committed_candidate_is_marked(folded):
    it = folded["episodes"]["T3b"]["runs"][0]["hours"][0]["iterations"][0]
    assert it["menu"]["candidates"][0]["committed"] is True


def test_restorations_survive_load_run(folded):
    # EpisodeTrace.restorations (Task 7) -- the replay's own records, passed
    # through by build_viewer_data.load_run untouched.
    run = folded["episodes"]["T3b"]["runs"][0]
    assert run["restorations"] == FIXTURE_TRACE["restorations"]


def test_the_gold_label_and_spare_action_travel_with_the_episode(folded):
    gold = folded["episodes"]["T3b"]["gold"]
    # The re-authored T3 (plan 2026-09-06, Task 11) grades the spend/hold
    # axis like T1 and T2, so its labels are `hold`/`spend` rather than the
    # retired episode's `A`/`B` avoid-width labels.
    assert gold["label"] == "spend"
    assert gold["gold_spare_action"] == "spend"


def test_a_pre_change_trace_folds_without_the_new_keys(tmp_path):
    # The 21 archived control rollouts have no observation/projected/menu and
    # spell it `service_under_test` (run-viewer design, §6.2).
    old = {k: v for k, v in FIXTURE_TRACE.items()
          if k not in ("oms_nodes", "restorations")}
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
    run = payload["episodes"]["T3b"]["runs"][0]
    hour = run["hours"][0]
    assert hour["exposure_rows"] == []
    assert hour["service_paths"] == {}
    assert run["restorations"] == []


def test_the_html_is_one_self_contained_file(folded):
    html = bvd.render_html(folded)
    assert html.lstrip().startswith("<!doctype html>")
    assert "<script" in html and "application/json" in html
    # No network at all: a viewer that needs a CDN is not double-clickable.
    for banned in ("http://", "https://", "src=\"//", "@import"):
        assert banned not in html


def test_a_back_link_is_opt_in_and_escaped(folded):
    assert 'id="back-link"' not in bvd.render_html(folded)
    html = bvd.render_html(folded, back_link='https://x.test/r?a=1&b="2"')
    assert ('<a id="back-link" href="https://x.test/r?a=1&amp;b=&quot;2&quot;">'
            in html)
    assert "__BACKLINK__" not in html


def test_quiet_hours_are_unclickable_and_skipped_by_arrow_keys(folded):
    html = bvd.render_html(folded)
    assert "' quiet'" in html and ".hcell.quiet" in html
    assert "if (!quiet) {" in html  # no click handler on a quiet cell
    assert "nextLiveHour(episode, run, state.hourIndex, step)" in html


def test_uncalled_hours_render_no_observation_or_ranking(folded):
    html = bvd.render_html(folded)
    assert "if (!agentCalled(hour)) {" in html
    assert "No decision this hour: the model was not called." in html


def test_the_joint_cut_table_reads_the_wire_payload(folded):
    html = bvd.render_html(folded)
    assert "function jointCutTable(hour)" in html
    assert "obs.cut_outcomes" in html and "obs.hours_down_if_cut" in html


def test_probe_validity_is_judged_against_the_risk_group_in_force(folded):
    # A probe answer holds only under the risk group it was asked under; a
    # later issuance replaces the group and makes it stale. Judged against
    # the viewed hour, never a later one.
    html = bvd.render_html(folded)
    assert "valid now?" in html and "re-shown later" not in html
    assert "riskGroupsInForce(run, state.hourIndex)" in html
    assert "stale: risk group revised at" in html


def test_the_joint_cut_table_explains_itself(folded):
    html = bvd.render_html(folded)
    assert "Storm outcomes at" in html
    assert "mutually exclusive and sum to 1." in html
    assert "none (everything survives)" in html


def test_no_inversion_warning_on_the_ranking(folded):
    assert "inverted: SUT ranks" not in bvd.render_html(folded)


def test_the_scrubber_marks_events_not_a_spend_hold_verdict(folded):
    # 389346a replaced renderScrubber's per-hour gold-vs-agent color strip
    # with a text line naming the events that drive a decision (a forecast
    # revision, a realized cut). The strip judged the decider's own spend
    # against gold -- a judgment the scoreboard already makes per run -- so
    # nothing per-hour may render a match/mismatch verdict any more. This is
    # JS embedded in the generated HTML, so it is checked as a string
    # assertion on the rendered output rather than executed.
    html = bvd.render_html(folded)
    assert "hcell-events" in html
    assert "parts.push('forecast')" in html
    assert "parts.push('cut')" in html
    assert "'strip ' + cls" not in html


def test_the_payload_survives_a_script_close_in_the_data(folded):
    folded["episodes"]["T3b"]["gold"]["rationale"] = "</script><b>x</b>"
    html = bvd.render_html(folded)
    assert "</script><b>" not in html
    assert "<\\/script>" in html


def test_a_stale_duplicate_run_key_is_dropped_for_the_newer_file(tmp_path):
    # Real incident (eval/traces/T1a-agent_claude-sonnet-5-{0,rerun}.json,
    # 2026-09-09): two trace files can independently claim the same
    # (scenario_id, decider_name, run_index). populateRunDropdown labels a
    # run only `${decider_name} #${run_index}` (the JS template above), so
    # without dedup the run picker shows two entries captioned identically
    # and there is no way to tell which one is current. Keep the file with
    # the newer mtime; drop the other before load_run ever sees it.
    traces = tmp_path / "traces"
    traces.mkdir()
    stale = dict(FIXTURE_TRACE, terminal_status="stale-marker")
    fresh = dict(FIXTURE_TRACE, terminal_status="fresh-marker")
    # Filenames sort the stale one first ("-0" < "-rerun"), so a dedup that
    # accidentally keyed off file order instead of mtime would still pass a
    # test that only ever wrote the fresh file second -- write stale second
    # and rely solely on the explicit utime below to prove it's mtime-driven.
    fresh_path = traces / "T3b-agent_claude-sonnet-5-rerun.json"
    stale_path = traces / "T3b-agent_claude-sonnet-5-0.json"
    fresh_path.write_text(json.dumps(fresh), encoding="utf-8")
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    old_time = time.time() - 3600
    new_time = time.time()
    os.utime(stale_path, (old_time, old_time))
    os.utime(fresh_path, (new_time, new_time))
    scenarios = (Path(__file__).parent.parent.parent / "src"
                 / "storm_reoptimizer" / "eval" / "scenarios")
    topology = (Path(__file__).parent.parent.parent / "src"
                / "storm_reoptimizer" / "data" / "toy_india_topology.json")
    payload = bvd.fold(traces, scenarios, topology)
    runs = payload["episodes"]["T3b"]["runs"]
    assert len(runs) == 1
    assert runs[0]["terminal_status"] == "fresh-marker"


def test_the_aerial_plant_is_distinguishable_in_the_payload(folded):
    # Acceptance item 4: storm-svc-1's working and protection paths must both
    # render with a dotted first segment out of Satna, without a legend.
    aerial = {(e["src"], e["dst"]) for e in folded["topology"]["edges"]
              if e["mount_type"] == "aerial"}
    assert ("satna", "rewa") in aerial or ("rewa", "satna") in aerial
    assert ("satna", "jhansi") in aerial or ("jhansi", "satna") in aerial


def test_a_metrics_sidecar_is_not_mistaken_for_a_trace(tmp_path):
    """An episode_metrics dict carries a `scenario_id`, so an unfiltered
    *.json glob folds it in as a run with decider_name None -- captioned
    `null #null` in the dropdown, beside the real one."""
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "T3b-agent_claude-sonnet-5-0.json").write_text(
        json.dumps(FIXTURE_TRACE), encoding="utf-8")
    (traces / "T3b-agent_claude-sonnet-5-0-metrics.json").write_text(
        json.dumps({"scenario_id": "T3b", "decider": "agent:claude-sonnet-5",
                    "decision_label": "act", "label_correct": True,
                    "regret_gbps_h": 0.0, "acted_too_late": False,
                    "inert_commits": 0}), encoding="utf-8")
    scenarios = (Path(__file__).parent.parent.parent / "src"
                 / "storm_reoptimizer" / "eval" / "scenarios")
    topology = (Path(__file__).parent.parent.parent / "src"
                / "storm_reoptimizer" / "data" / "toy_india_topology.json")
    folded = bvd.fold(traces, scenarios, topology)
    runs = folded["episodes"]["T3b"]["runs"]
    assert len(runs) == 1
    assert runs[0]["decider_name"] == "agent:claude-sonnet-5"


def test_the_metrics_sidecar_rides_onto_the_run(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "T3b-agent_claude-sonnet-5-0.json").write_text(
        json.dumps(FIXTURE_TRACE), encoding="utf-8")
    metrics = {"scenario_id": "T3b", "decider": "agent:claude-sonnet-5",
               "decision_label": "spend", "label_correct": False,
               "regret_gbps_h": 250.0, "acted_too_late": False,
               "inert_commits": 2}
    (traces / "T3b-agent_claude-sonnet-5-0-metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8")
    scenarios = (Path(__file__).parent.parent.parent / "src"
                 / "storm_reoptimizer" / "eval" / "scenarios")
    topology = (Path(__file__).parent.parent.parent / "src"
                / "storm_reoptimizer" / "data" / "toy_india_topology.json")
    run = bvd.fold(traces, scenarios, topology)["episodes"]["T3b"]["runs"][0]
    assert run["metrics"] == metrics


def test_a_run_with_no_sidecar_says_so_rather_than_showing_zeros(folded):
    """"Trace predates this field" is not "field is empty" -- the viewer's own
    standing lesson (harness explainer, §13). `None`, not `{}`."""
    assert folded["episodes"]["T3b"]["runs"][0]["metrics"] is None


def test_the_land_backdrop_is_folded_in_as_lat_lon_rings(folded):
    # Decorative, but it must cover the plant: every node of the toy
    # topology sits inside the clip box the coastline was cut to.
    rings = folded["land"]
    assert rings and all(len(r) >= 4 for r in rings)
    lats = [p[0] for r in rings for p in r]
    lons = [p[1] for r in rings for p in r]
    for lat, lon in folded["topology"]["nodes"].values():
        assert min(lats) <= lat <= max(lats)
        assert min(lons) <= lon <= max(lons)


def test_probes_asked_at_a_decision_carry_no_validity_column(folded):
    # Asked at this very decision, so always "current": the column only
    # means something in the whole-run ledger, which mixes hours.
    html = bvd.render_html(folded)
    assert "wrap.appendChild(probeTable(probes, false));" in html
    assert "el.appendChild(probeTable(rows));" in html


def test_the_depot_and_forecast_state_is_a_table(folded):
    html = bvd.render_html(folded)
    assert "depot and forecast state" in html
    assert "const misc = document.createElement('pre');" not in html


def test_the_page_says_correct_not_gold(folded):
    html = bvd.render_html(folded)
    assert "vs gold" not in html and "gold outcome (label" not in html
    assert "(correct: ${gold.label})" in html
    assert "Gbps·h less lost than" in html


def test_a_landing_action_is_named_restored_or_rerouted(folded):
    html = bvd.render_html(folded)
    assert "parts.push('lands')" not in html
    assert "parts.push('restored')" in html
    assert "parts.push('rerouted')" in html
