"""The forecast-blind fixed policy (eval design spec, "Baseline"). Both
variants are deterministic: same observation, same answer, every time."""
import pytest

from storm_reoptimizer.eval.baseline import (
    BASELINE_VARIANTS, ForecastBlindBaseline, ScriptedDecider, service_class,
)
from storm_reoptimizer.eval.decisions import TimingDecision
from storm_reoptimizer.eval.observation import Observation
from storm_reoptimizer.eval.scenario_file import ConeAtHorizon, Issuance

MENU = {
    "status": "solution",
    "candidates": [
        {"lever": "optical_reroute", "reused_lightpaths": [],
         "new_lightpaths": [{"oms_sequence": ["oms_1"], "lam": 0,
                             "mode_id": "300G@4.8dB", "gsnr_db": 12.0,
                             "bitrate_gbps": 300.0}],
         "restored_gbps": 300.0, "shortfall_gbps": 0.0,
         "cost_vector": {"dropped_traffic": 0.0, "transponders": 40.0,
                         "services_at_risk": 0, "total_margin": 5.0,
                         "added_latency": 9.0, "spectrum_used": 20,
                         "max_util": 0.7}},
        {"lever": "ip_reroute", "reused_lightpaths": ["lp-x"],
         "new_lightpaths": [], "restored_gbps": 250.0, "shortfall_gbps": 50.0,
         "cost_vector": {"dropped_traffic": 50.0, "transponders": 38.0,
                         "services_at_risk": 0, "total_margin": 5.0,
                         "added_latency": 4.0, "spectrum_used": 18,
                         "max_util": 0.8}},
    ],
    "pairs": [],
}


def _obs(*, offset_km, width_km, hours_ahead, spares=1):
    horizon = "t3"
    return Observation(
        scenario_id="X", service_under_test="storm-svc-1", hour="t1",
        hour_index=1, hours_remaining=3,
        issuance=Issuance(issued_at="t1", horizons={
            horizon: ConeAtHorizon(cone={"type": "Polygon", "coordinates": []},
                                   width_km=width_km,
                                   center={"lat": 25.0, "lon": 81.0})}),
        exposure={"storm-svc-1": {horizon: {
            "hours_ahead": hours_ahead, "offset_km": offset_km,
            "width_km": width_km, "p_cut": 0.5, "demand_gbps": 300.0}}},
        services=({"id": "storm-svc-1", "demand_gbps": 300.0,
                   "src_router": "router_satna",
                   "dst_router": "router_allahabad",
                   "working_path": [], "protection_path": []},),
        spares_on_hand=spares, lead_time_hours=1,
        risk_group_ids={horizon: "rg_X_t1_t3"})


def test_there_are_exactly_two_variants():
    assert BASELINE_VARIANTS == ("immediate", "at_deadline")


def test_immediate_acts_as_soon_as_the_service_is_inside_the_cone():
    b = ForecastBlindBaseline("immediate")
    assert b.timing(_obs(offset_km=10.0, width_km=90.0, hours_ahead=3)).action == "act"
    assert b.timing(_obs(offset_km=200.0, width_km=90.0, hours_ahead=3)).action == "wait"


def test_at_deadline_waits_until_lead_time_forces_the_issue():
    b = ForecastBlindBaseline("at_deadline")
    inside = dict(offset_km=10.0, width_km=90.0)
    assert b.timing(_obs(**inside, hours_ahead=3)).action == "wait"
    assert b.timing(_obs(**inside, hours_ahead=1)).action == "act"


def test_both_variants_are_deterministic():
    for variant in BASELINE_VARIANTS:
        b = ForecastBlindBaseline(variant)
        obs = _obs(offset_km=10.0, width_km=90.0, hours_ahead=1)
        assert [b.timing(obs).to_dict() for _ in range(5)] == [
            b.timing(obs).to_dict()] * 5


def test_constraints_avoid_only_the_nearest_exposed_horizon_and_pin_posture():
    d = ForecastBlindBaseline("immediate").constraints(
        _obs(offset_km=10.0, width_km=90.0, hours_ahead=3))
    assert d.avoid == {"risk_groups": ["rg_X_t1_t3"]}
    assert (d.protected, d.best_effort, d.basis, d.level) == (
        True, False, "srlg", "srlg")


def test_objective_is_a_fixed_service_class_priority_table():
    assert service_class(300.0) == "premium"
    assert service_class(100.0) == "standard"
    d = ForecastBlindBaseline("immediate").objective(
        _obs(offset_km=10.0, width_km=90.0, hours_ahead=1), MENU)
    # premium leads on dropped_traffic, so the full-restoration candidate wins.
    assert d.choice == "candidate_0"
    assert d.priority is not None


def test_an_empty_menu_yields_infeasible():
    d = ForecastBlindBaseline("immediate").objective(
        _obs(offset_km=10.0, width_km=90.0, hours_ahead=1),
        {"status": "no_solution", "candidates": [], "pairs": []})
    assert d.choice == "infeasible"


def test_scripted_decider_replays_what_it_was_given():
    s = ScriptedDecider(
        "alt", timing_by_hour={"t1": TimingDecision("act", "scripted")},
        default_timing=TimingDecision("wait", "scripted default"))
    assert s.timing(_obs(offset_km=10.0, width_km=90.0, hours_ahead=1)).action == "act"
