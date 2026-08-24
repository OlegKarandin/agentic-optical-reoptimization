"""The LLM decider (agent decider design spec, 2026-08-24). Unit tests only:
a fake Anthropic client, no network, no cost, and no `anthropic` install."""
import json

import pytest

from storm_reoptimizer.eval.agent import (
    P_CUT_ENUMERATION_THRESHOLD, project_observation,
)
from storm_reoptimizer.eval.observation import Observation
from storm_reoptimizer.eval.scenario_file import ConeAtHorizon, Issuance

HORIZON = "t3"


def _obs(*, others=(), sut_p_cut=0.3410, iteration=0, last_rejection=None):
    """T3a's shape at t1: the service under test centred in the far cone,
    plus whichever other claimants the test wants to place. `others` is a
    sequence of (service_id, p_cut, demand_gbps)."""
    exposure = {"storm-svc-1": {HORIZON: {
        "hours_ahead": 2, "offset_km": 0.0, "width_km": 320.0,
        "p_cut": sut_p_cut, "demand_gbps": 300.0}}}
    services = [{"id": "storm-svc-1", "demand_gbps": 300.0,
                 "src_router": "router_satna",
                 "dst_router": "router_allahabad",
                 "working_path": ["ipl-cand-storm-svc-1-0"],
                 "protection_path": ["ipl-prot-storm-svc-1-0"]}]
    for svc_id, p_cut, demand in others:
        exposure[svc_id] = {HORIZON: {
            "hours_ahead": 2, "offset_km": 129.3, "width_km": 320.0,
            "p_cut": p_cut, "demand_gbps": demand}}
        services.append({"id": svc_id, "demand_gbps": demand,
                         "src_router": "router_a", "dst_router": "router_b",
                         "working_path": [], "protection_path": []})
    return Observation(
        scenario_id="T3a", service_under_test="storm-svc-1", hour="t1",
        hour_index=1, hours_remaining=2,
        issuance=Issuance(issued_at="t1", horizons={
            HORIZON: ConeAtHorizon(
                cone={"type": "Polygon", "coordinates": []}, width_km=320.0,
                center={"lat": 24.855553, "lon": 81.327777})}),
        exposure=exposure, services=tuple(services), spares_on_hand=1,
        lead_time_hours=1, risk_group_ids={HORIZON: "rg_T3a_t1_t3"},
        iteration=iteration, last_rejection=last_rejection)


# The three claimants T3a's authoring note enumerates, plus two the same note
# excludes ("the only non-SUT services above p_cut 0.005 ... in EITHER half").
CLAIMANTS = [("d0363", 0.2639, 100.0), ("d0462", 0.1207, 100.0),
             ("d0212", 0.1207, 100.0)]
BELOW_THRESHOLD = [("d0001", 0.004, 100.0), ("d0002", 0.002, 300.0)]


def test_the_threshold_is_the_one_the_episode_authors_enumerated_against():
    # scenarios/T3a.yaml names 0.005 as the cut-off it used to decide which
    # non-SUT services were worth naming at all. Reusing it keeps the
    # projection's notion of "relevant" the same as the gold rationale's.
    assert P_CUT_ENUMERATION_THRESHOLD == 0.005


def test_a_claimant_above_the_threshold_survives_projection():
    payload = project_observation(_obs(others=CLAIMANTS + BELOW_THRESHOLD))
    assert set(payload["exposure"]) == {"storm-svc-1", "d0363", "d0462",
                                        "d0212"}
    assert [s["id"] for s in payload["services"]] == [
        "storm-svc-1", "d0363", "d0462", "d0212"]


def test_the_service_under_test_survives_even_when_its_own_p_cut_is_zero():
    payload = project_observation(_obs(sut_p_cut=0.0, others=CLAIMANTS))
    assert "storm-svc-1" in payload["exposure"]
    assert "storm-svc-1" in {s["id"] for s in payload["services"]}


def test_projection_keys_on_cut_probability_not_cone_containment():
    # D1's lesson: offset 58.4 km against a 15 km-wide cone (half-width
    # 7.5 km) is OUTSIDE the polygon, yet p_cut is 0.976. A containment
    # filter would drop the most exposed service in that episode.
    obs = _obs(sut_p_cut=0.0)
    obs.exposure["d0001"] = {HORIZON: {
        "hours_ahead": 2, "offset_km": 58.4, "width_km": 15.0,
        "p_cut": 0.976, "demand_gbps": 100.0}}
    payload = project_observation(obs)
    assert "d0001" in payload["exposure"]


def test_the_omission_summary_accounts_for_every_service_it_dropped():
    payload = project_observation(_obs(others=CLAIMANTS + BELOW_THRESHOLD))
    omitted = payload["omitted_services"]
    assert omitted["count"] == 2
    assert omitted["p_cut_threshold"] == 0.005
    assert omitted["max_p_cut"] == 0.004
    # 0.004*100 + 0.002*300 = 0.4 + 0.6
    assert omitted["summed_expected_capacity_at_risk_gbps"] == pytest.approx(
        1.0)
    assert payload["n_services_total"] == 6


def test_nothing_is_omitted_when_every_service_clears_the_threshold():
    payload = project_observation(_obs(others=CLAIMANTS))
    assert payload["omitted_services"] == {
        "count": 0, "p_cut_threshold": 0.005, "max_p_cut": 0.0,
        "summed_expected_capacity_at_risk_gbps": 0.0}


def test_projection_preserves_every_field_the_decision_points_read():
    obs = _obs(others=CLAIMANTS, iteration=2,
               last_rejection={"type": "insufficient_spares", "needed": 2,
                               "on_hand": 1})
    payload = project_observation(obs)
    raw = obs.to_dict()
    for key in ("scenario_id", "service_under_test", "hour",
                "hours_remaining", "issued_at", "cones", "spares_on_hand",
                "lead_time_hours", "risk_group_ids", "iteration",
                "last_rejection"):
        assert payload[key] == raw[key]


def test_projection_bounds_the_prompt_against_a_full_573_service_roster():
    # eval/states/loaded-s17.json carries 573 services (97,654 JSON chars for
    # the roster alone, plus one exposure entry per service per horizon).
    # Sending that raw would be ~50-70K tokens on every one of the up-to-11
    # calls an acting hour makes. This test is the guard on that.
    obs = _obs(others=[(f"d{i:04d}", 0.0, 100.0) for i in range(571)])
    assert len(json.dumps(obs.to_dict())) > 100_000
    payload = project_observation(obs)
    assert payload["n_services_total"] == 572
    assert payload["omitted_services"]["count"] == 571
    assert len(json.dumps(payload)) < 5_000
