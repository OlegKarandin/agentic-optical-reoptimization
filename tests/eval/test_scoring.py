"""Episode and cross-twin scoring (eval design spec, "Scoring"). Pure
functions over a ScenarioFile and an EpisodeTrace -- no server."""
import dataclasses
import textwrap

import pytest

from storm_reoptimizer.eval.runner import Action, EpisodeTrace
from storm_reoptimizer.eval.scenario_file import load_scenario
from storm_reoptimizer.eval.scoring import (
    cites_flip_variable, cross_twin_metrics, decision_label, episode_metrics,
)

BASE = textwrap.dedent("""
    id: {id}
    pair: P
    seed: 17
    state_file: eval/states/loaded-s17.json
    service_under_test: storm-svc-1
    track: hudhud
    hours: [t0, t1, t2, t3]
    decision_hour: t1
    lead_time_hours: 1
    spares_on_hand: 1
    damage_radius_km: 74
    reference_avoid: {{}}
    forecast:
      t0:
        t3: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: 25.0, lon: 81.0}}}}
    realized:
      t3: [fiber_004]
    gold:
      survived: [storm-svc-1, svc-b]
      max_spares_wasted: 0
      decision_at_t0: wait
      label: {label}
      rationale: test fixture
    flip_variable: [svc-b, centre]
    metadata:
      label_rule: timing_at_decision_hour
      cone_width_km: 90
      cone_motion_kmh: 20
      n_future_claimants: 1
      exposure_horizon_hours: 2
      spares_on_hand: 1
      widest_avoid_feasible: true
""")


def _scenario(tmp_path, sid, label):
    path = tmp_path / f"{sid}.yaml"
    path.write_text(BASE.format(id=sid, label=label), encoding="utf-8")
    return load_scenario(path)


def _trace(*, timing_at_t1="wait", actions=(), dropped=(), affected=None,
           rejections_at=(), committed_at=(), reasoning="the cone is centred "
           "on svc-b, so the spare is worth more held"):
    hours = []
    for hour in ("t0", "t1", "t2", "t3"):
        hours.append({
            "hour": hour,
            "services": ["storm-svc-1", "svc-b"],
            "timing": {"action": timing_at_t1 if hour == "t1" else "wait",
                       "reasoning": reasoning},
            "iterations": [],
            "rejections": [{"type": "validation_violations"}]
            if hour in rejections_at else [],
            "committed": hour in committed_at,
        })
    return EpisodeTrace(
        scenario_id="X", decider_name="test", run_index=0,
        hours=tuple(hours), actions=tuple(actions), ledger_debits=(),
        spares_remaining=1, terminal_status="converged",
        final_routing={"dropped": {"services": [
            {"service_id": s, "reason": "down", "on_link": None}
            for s in dropped]}},
        affected_by_hour=affected or {}, tool_calls=7, wall_clock_s=1.0)


def test_timing_label_reads_the_decision_hour_not_t0(tmp_path):
    s = _scenario(tmp_path, "Pa", "act")
    assert decision_label(s, _trace(timing_at_t1="act")) == "act"
    assert decision_label(s, _trace(timing_at_t1="wait")) == "wait"


def test_services_survived_excludes_dropped_services(tmp_path):
    s = _scenario(tmp_path, "Pa", "wait")
    m = episode_metrics(s, _trace(dropped=("svc-b",)))
    assert "svc-b" not in m["services_survived"]


def test_an_action_landing_after_the_cut_is_too_late_and_kills_the_service(
    tmp_path,
):
    s = _scenario(tmp_path, "Pa", "act")
    late = Action(hour="t3", hour_index=3, lever="optical_reroute",
                  effective_at_index=4, spares={"satna": 1, "allahabad": 1},
                  service_id="storm-svc-1")
    m = episode_metrics(s, _trace(
        actions=(late,), affected={"t3": ("storm-svc-1",)}))
    assert m["acted_too_late"] is True
    assert "storm-svc-1" not in m["services_survived"]


def test_an_ip_reroute_at_the_cut_hour_lands_in_time(tmp_path):
    s = _scenario(tmp_path, "Pa", "act")
    prompt = Action(hour="t3", hour_index=3, lever="ip_reroute",
                    effective_at_index=3, spares={}, service_id="storm-svc-1")
    m = episode_metrics(s, _trace(
        actions=(prompt,), affected={"t3": ("storm-svc-1",)}))
    assert m["acted_too_late"] is False
    assert "storm-svc-1" in m["services_survived"]


def test_spares_spent_on_a_service_never_cut_are_wasted(tmp_path):
    s = _scenario(tmp_path, "Pa", "act")
    trace = dataclasses.replace(
        _trace(actions=(Action("t1", 1, "optical_reroute", 2,
                               {"satna": 1}, "storm-svc-1"),), affected={}),
        ledger_debits=({"hour": "t1", "service_id": "storm-svc-1",
                        "spares": {"satna": 1}},))
    assert episode_metrics(s, trace)["spares_wasted"] == 1


def test_reexposure_is_a_restored_service_cut_by_a_later_hour(tmp_path):
    s = _scenario(tmp_path, "Pa", "act")
    m = episode_metrics(s, _trace(
        actions=(Action("t1", 1, "optical_reroute", 2, {"satna": 1},
                       "storm-svc-1"),),
        affected={"t3": ("storm-svc-1",)}))
    assert m["reexposed"] is True


def test_recovered_from_rejection_needs_both_a_rejection_and_a_commit(tmp_path):
    s = _scenario(tmp_path, "Pa", "act")
    assert episode_metrics(s, _trace(
        rejections_at=("t1",), committed_at=("t1",)))["recovered_from_rejection"]
    assert not episode_metrics(s, _trace(
        rejections_at=("t1",)))["recovered_from_rejection"]


def test_flip_variable_citation_matches_entities_in_the_reasoning(tmp_path):
    s = _scenario(tmp_path, "Pa", "wait")
    assert cites_flip_variable(_trace(), s.flip_variable)
    assert not cites_flip_variable(
        _trace(reasoning="I flipped a coin"), s.flip_variable)


def test_pair_solved_needs_both_halves_right(tmp_path):
    a, b = _scenario(tmp_path, "Pa", "wait"), _scenario(tmp_path, "Pb", "act")
    right = cross_twin_metrics(a, _trace(timing_at_t1="wait"),
                               b, _trace(timing_at_t1="act"))
    assert right["pair_solved"] is True
    half = cross_twin_metrics(a, _trace(timing_at_t1="wait"),
                              b, _trace(timing_at_t1="wait"))
    assert half["pair_solved"] is False
    assert half["halves_correct"] == 1
