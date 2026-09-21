"""Episode and cross-twin scoring (eval design spec, "Scoring"). Pure
functions over a ScenarioFile and an EpisodeTrace -- no server."""
import dataclasses
import textwrap

import pytest

from storm_reoptimizer.eval.runner import Action, EpisodeTrace
from storm_reoptimizer.eval.scenario_file import load_scenario
from storm_reoptimizer.eval.scoring import (
    cites_flip_variable, cross_twin_metrics, decision_label, episode_metrics,
    gbps_hours_lost, reexposed,
)

BASE = textwrap.dedent("""
    id: {id}
    pair: P
    seed: 17
    state_file: eval/states/loaded-s17.json
    service_under_test: storm-svc-1
    track: hudhud
    hours: {hours}
    decision_hour: t1
    lead_time_hours: 1
    spares_on_hand: 1
    depot_site: satna
    spare_inventory: {{satna: 1}}
    damage_radius_km: 74
    track_revision_km_per_hour_ahead: 30
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
      {gold_extra}
    flip_variable: [svc-b, centre]
    metadata:
      label_rule: {label_rule}
      cone_width_km: 90
      cone_motion_kmh: 20
      n_future_claimants: 1
      exposure_horizon_hours: 2
      spares_on_hand: 1
      widest_avoid_feasible: true
      claimant_services: []
""")


def _scenario(tmp_path, sid, label, *, label_rule="timing_at_decision_hour",
             hours="[t0, t1, t2, t3]", outcome_gbps_h=None):
    gold_extra = f"outcome_gbps_h: {outcome_gbps_h}" if outcome_gbps_h else ""
    path = tmp_path / f"{sid}.yaml"
    path.write_text(BASE.format(id=sid, label=label, hours=hours,
                                label_rule=label_rule, gold_extra=gold_extra),
                    encoding="utf-8")
    return load_scenario(path)


def _trace(*, timing_at_t1="wait", raw_at_t0="wait", effective_at_t0="wait",
           effective_at_t1=None,
           actions=(), dropped=(), affected=None,
           rejections_at=(), committed_at=(), reasoning="the cone is centred "
           "on svc-b, so the spare is worth more held",
           hours=("t0", "t1", "t2", "t3"), demands=None, debits=(),
           sut_p_cut_by_hour=None):
    # `dropped` grew a second shape (Task 8): a dict {hour: [service_id,
    # ...]} populates each hour's own `dropped_after_cut` (what the loss
    # metrics read), while the original flat tuple/list still only feeds
    # `final_routing["dropped"]["services"]` (what `services_survived`
    # reads) -- existing callers pass the flat shape and are unaffected.
    dropped_after_cut = dropped if isinstance(dropped, dict) else {}
    flat_dropped = (sorted({s for svcs in dropped.values() for s in svcs})
                   if isinstance(dropped, dict) else dropped)
    records = []
    for hour in hours:
        if hour == "t0":
            action = raw_at_t0
        elif hour == "t1":
            action = timing_at_t1
        else:
            action = "wait"
        # timing_effective defaults to the raw timing action -- t0 and t1 can
        # each be forced to disagree (effective_at_t0/effective_at_t1),
        # matching what the runner actually produces when a declared "act"
        # commits nothing at all, or commits inertly (Task 5:
        # timing_at_decision_hour then reads the disagreement too, not just
        # first_shot_correct at t0).
        if hour == "t0":
            effective = effective_at_t0
        elif hour == "t1" and effective_at_t1 is not None:
            effective = effective_at_t1
        else:
            effective = action
        record = {
            "hour": hour,
            "services": ["storm-svc-1", "svc-b"],
            "timing": {"action": action, "reasoning": reasoning},
            "timing_effective": effective,
            "iterations": [],
            "rejections": [{"type": "validation_violations"}]
            if hour in rejections_at else [],
            "committed": hour in committed_at,
            "dropped_after_cut": list(dropped_after_cut.get(hour, [])),
            "demands": dict(demands) if demands else {},
        }
        if sut_p_cut_by_hour and hour in sut_p_cut_by_hour:
            record["observation"] = {"exposure": {"storm-svc-1": {
                "h": {"p_cut": sut_p_cut_by_hour[hour]}}}}
        records.append(record)
    return EpisodeTrace(
        scenario_id="X", decider_name="test", run_index=0,
        hours=tuple(records), actions=tuple(actions),
        ledger_debits=tuple(debits),
        spares_remaining=1, terminal_status="converged",
        final_routing={"dropped": {"services": [
            {"service_id": s, "reason": "down", "on_link": None}
            for s in flat_dropped]}},
        affected_by_hour=affected or {}, tool_calls=7, wall_clock_s=1.0)


def test_timing_label_reads_the_decision_hour_not_t0(tmp_path):
    s = _scenario(tmp_path, "Pa", "act")
    assert decision_label(s, _trace(timing_at_t1="act")) == "act"
    assert decision_label(s, _trace(timing_at_t1="wait")) == "wait"


def test_timing_label_reads_effective_not_declared_at_the_decision_hour(
    tmp_path,
):
    """Task 5: an act that commits nothing (or commits inertly) is a wait.
    `timing_at_decision_hour` must read the decision hour's own
    `timing_effective`, not the raw declared `timing.action` -- a declared
    "act" that turned out to commit nothing (or commit inertly) grades as
    "wait", matching `first_shot_correct`'s own fallback expression."""
    s = _scenario(tmp_path, "Pa", "act")
    trace = _trace(timing_at_t1="act", effective_at_t1="wait")
    assert decision_label(s, trace) == "wait"


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


def test_reexposed_is_false_when_the_reroute_lands_clear(tmp_path):
    s = _scenario(tmp_path, "Pa", "act")
    m = episode_metrics(s, _trace(
        actions=(Action("t1", 1, "optical_reroute", 2, {"satna": 1},
                       "storm-svc-1"),),
        sut_p_cut_by_hour={"t3": 0.0}))
    assert m["reexposed"] is False


def test_reexposed_is_true_when_the_new_path_reenters_a_later_cone(tmp_path):
    s = _scenario(tmp_path, "Pa", "act")
    m = episode_metrics(s, _trace(
        actions=(Action("t1", 1, "optical_reroute", 2, {"satna": 1},
                       "storm-svc-1"),),
        sut_p_cut_by_hour={"t3": 0.35}))
    assert m["reexposed"] is True


def test_reexposed_is_keyed_off_effective_not_hour_index(tmp_path):
    # Calls reexposed() directly (not through episode_metrics) -- it is a
    # standalone, independently-testable function, not only a dict key.
    # Between commit and effectiveness the service is still on its OLD
    # path by construction -- exposure at the hour immediately after the
    # ACTION's hour_index (but before effective_at_index) must not count.
    s = _scenario(tmp_path, "Pa", "act", hours="[t0, t1, t2, t3, t4]")
    action = Action("t1", 1, "optical_reroute", 3, {"satna": 1},
                    "storm-svc-1")    # effective at index 3 (t3)
    trace = _trace(
        hours=("t0", "t1", "t2", "t3", "t4"), actions=(action,),
        sut_p_cut_by_hour={"t2": 0.9, "t4": 0.0})
    # t2 (index 2) is BEFORE effective_at_index 3 -- its exposure must not
    # be read even though it is > hour_index (1); only t4 (index 4, after
    # effective_at_index) is in scope, and it reads 0.0.
    assert reexposed(s, trace) is False


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


def test_first_shot_reads_the_effective_timing(tmp_path):
    s = _scenario(tmp_path, "X", "wait")
    trace = _trace(effective_at_t0="wait", raw_at_t0="act")
    assert episode_metrics(s, trace)["first_shot_correct"] is True


def test_inert_commits_are_counted(tmp_path):
    s = _scenario(tmp_path, "X", "wait")
    trace = _trace()
    hours = list(trace.hours)
    hours[1] = {**hours[1], "iterations": [{"outcome": "committed", "inert": True}]}
    trace = dataclasses.replace(trace, hours=tuple(hours))
    assert episode_metrics(s, trace)["inert_commits"] == 1


def test_pair_solved_needs_both_halves_right(tmp_path):
    a, b = _scenario(tmp_path, "Pa", "wait"), _scenario(tmp_path, "Pb", "act")
    right = cross_twin_metrics(a, _trace(timing_at_t1="wait"),
                               b, _trace(timing_at_t1="act"))
    assert right["pair_solved"] is True
    half = cross_twin_metrics(a, _trace(timing_at_t1="wait"),
                              b, _trace(timing_at_t1="wait"))
    assert half["pair_solved"] is False
    assert half["halves_correct"] == 1


# T1 spend-or-hold redesign (Task 8): `spare_action_by_deadline` reads
# whether the DECIDER (not the harness) spent a spare on the SUT before it
# needed to be cut, and `gbps_hours_lost`/`regret_gbps_h` turn the episode
# into one comparable outcome number, graded against the oracle's own two
# candidate outcomes (`Gold.outcome_gbps_h`).
def test_spare_action_spend_when_decider_debit_lands_before_cut(tmp_path):
    s = _scenario(tmp_path, "X", "spend", label_rule="spare_action_by_deadline")
    t = _trace(actions=(Action("t1", 1, "optical_reroute", 3, {"d": 1}, "storm-svc-1"),),
               debits=({"hour": "t1", "service_id": "storm-svc-1", "spares": {"d": 1},
                        "origin": "decider"},), dropped={"t3": ["svc-b"]})
    assert decision_label(s, t) == "spend"


def test_harness_debit_never_counts_as_spend(tmp_path):
    s = _scenario(tmp_path, "X", "hold", label_rule="spare_action_by_deadline")
    t = _trace(actions=(Action("t3", 3, "optical_reroute", 5, {"d": 1}, "storm-svc-1",
                               origin="harness"),),
               debits=({"hour": "t3", "service_id": "storm-svc-1", "spares": {"d": 1},
                        "origin": "harness"},), dropped={"t3": ["storm-svc-1"]})
    assert decision_label(s, t) == "hold"


def test_a_held_iteration_with_no_debits_labels_hold(tmp_path):
    # The `hold` exit at the objective step (decisions.py, HOLD_CHOICE): an
    # hour whose timing action was "act" but whose only iteration ended
    # `held` -- no ledger debit was ever made -- must not be mistaken for a
    # spend just because the hour's raw timing action was "act".
    s = _scenario(tmp_path, "X", "hold", label_rule="spare_action_by_deadline")
    trace = _trace(timing_at_t1="act")
    hours = list(trace.hours)
    idx = [h["hour"] for h in hours].index("t1")
    hours[idx] = {**hours[idx], "iterations": [{"outcome": "held"}]}
    trace = dataclasses.replace(trace, hours=tuple(hours))
    assert trace.ledger_debits == ()
    assert decision_label(s, trace) == "hold"


def test_gbps_hours_lost_counts_until_restoration_lands(tmp_path):
    # hours t0..t5; svc-b (100G) dropped at t3 (index 3); harness restoration
    # effective at index 5 -> down for t3 and t4 = 2 h x 100 G.
    s = _scenario(tmp_path, "X", "hold", label_rule="spare_action_by_deadline",
                  hours="[t0, t1, t2, t3, t4, t5]")
    t = _trace(hours=("t0", "t1", "t2", "t3", "t4", "t5"),
               demands={"storm-svc-1": 300.0, "svc-b": 100.0},
               dropped={"t3": ["svc-b"]},
               actions=(Action("t3", 3, "optical_reroute", 5, {"d": 1}, "svc-b",
                               origin="harness"),))
    assert gbps_hours_lost(s, t) == {"svc-b": 200.0}


def test_regret_is_total_minus_best_gold_outcome(tmp_path):
    s = _scenario(tmp_path, "X", "hold", label_rule="spare_action_by_deadline",
                  outcome_gbps_h="{spend: 600.0, hold: 400.0}")
    t = _trace(demands={"storm-svc-1": 300.0, "svc-b": 100.0},
               dropped={"t3": ["svc-b"]})          # never restored: 100 G x 1 h
    m = episode_metrics(s, t)
    assert m["gbps_hours_lost_total"] == 100.0
    assert m["regret_gbps_h"] == pytest.approx(100.0 - 400.0)


def test_sut_acted_too_late_against_a_realized_cut_counts_as_cut_at_c_realized(
    tmp_path,
):
    # The SUT is NEVER in any hour's dropped_after_cut (the server never
    # called it a drop -- protection likely covered it in the model), but a
    # REAL realized cut affected it at t2 (index 2, via affected_by_hour) and
    # the decider's own spare-spending action landed too late (effective at
    # index 3, after c_realized). cut_index[SUT] must be set to c_realized
    # (2) anyway -- the "acted too late" special case -- so gbps_hours_lost
    # scores the SUT as down starting at hour 2, not as never cut.
    s = _scenario(tmp_path, "X", "hold", label_rule="spare_action_by_deadline")
    t = _trace(demands={"storm-svc-1": 300.0},
               affected={"t2": ("storm-svc-1",)},
               actions=(Action("t2", 2, "optical_reroute", 3, {"d": 1},
                               "storm-svc-1"),))
    assert gbps_hours_lost(s, t) == {"storm-svc-1": 300.0}


def test_only_the_two_live_label_rules_remain():
    # Spec 7 (T2/T3 probe redesign): avoid_horizon_at_decision_hour and
    # chosen_lever_at_decision_hour were retired with the pairs that used
    # them; timing_at_decision_hour stays for D1.
    from storm_reoptimizer.eval.scoring import LABEL_RULES
    assert LABEL_RULES == ("timing_at_decision_hour", "spare_action_by_deadline")
