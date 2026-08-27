"""Observation building, and above all issuance isolation (eval design spec,
"Testing": "the observation built for hour t contains issued_t and no later
issuance. This is the one leak that would silently invalidate every timing
result, so it is asserted rather than assumed")."""
import json

import pytest

from storm_reoptimizer.eval.observation import (
    build_observation, latest_issuance, lead_time_hours_for,
)
from storm_reoptimizer.eval.scenario_file import load_scenario

SERVICES = (
    {"id": "storm-svc-1", "src_router": "router_satna",
     "dst_router": "router_allahabad", "demand_gbps": 300.0,
     "working_path": ["ipl-a"], "protection_path": ["ipl-b"]},
    {"id": "svc-b", "src_router": "router_rewa", "dst_router": "router_jhansi",
     "demand_gbps": 100.0, "working_path": ["ipl-c"], "protection_path": []},
)
POINTS = {"storm-svc-1": (25.0, 81.0), "svc-b": (25.6, 81.0)}


@pytest.fixture
def scenario(write_scenario, example_scenario_yaml):
    return load_scenario(write_scenario(example_scenario_yaml))


def test_lead_time_is_zero_for_ip_reroute_and_the_scenario_value_otherwise():
    assert lead_time_hours_for("ip_reroute", 1) == 0
    assert lead_time_hours_for("hybrid", 1) == 1
    assert lead_time_hours_for("optical_reroute", 2) == 2
    with pytest.raises(ValueError):
        lead_time_hours_for("teleportation", 1)


def test_latest_issuance_at_t0_is_the_t0_issuance(scenario):
    assert latest_issuance(scenario, "t0").issued_at == "t0"


def test_latest_issuance_at_t2_is_still_the_t1_issuance(scenario):
    # t2 publishes no advisory of its own; the agent carries t1's forward.
    assert latest_issuance(scenario, "t1").issued_at == "t1"
    assert latest_issuance(scenario, "t2").issued_at == "t1"


def test_observation_at_t0_leaks_no_later_issuance(scenario):
    obs = build_observation(scenario, "t0", service_points=POINTS,
                            services=SERVICES, spares_on_hand=1)
    payload = json.dumps(obs.to_dict())
    # The t1 issuance revises the cone centre to lat 25.2. If that number
    # appears anywhere in the t0 observation, waiting has become free.
    assert "25.2" not in payload
    assert obs.issuance.issued_at == "t0"


def test_observation_never_leaks_the_realized_cuts(scenario):
    obs = build_observation(scenario, "t0", service_points=POINTS,
                            services=SERVICES, spares_on_hand=1)
    assert "fiber_004" not in json.dumps(obs.to_dict())


def test_exposure_is_reported_per_service_per_horizon_with_a_cut_probability(
    scenario,
):
    obs = build_observation(scenario, "t0", service_points=POINTS,
                            services=SERVICES, spares_on_hand=1)
    at_t3 = obs.exposure["storm-svc-1"]["t3"]
    assert at_t3["hours_ahead"] == 3
    assert 0.0 < at_t3["p_cut"] <= 1.0
    # svc-b sits ~67 km north of the t3 cone axis, storm-svc-1 sits on it,
    # so storm-svc-1 must be the more exposed of the two.
    assert obs.exposure["svc-b"]["t3"]["p_cut"] < at_t3["p_cut"]


def test_risk_group_ids_are_carried_through_for_the_constraint_decision(
    scenario,
):
    obs = build_observation(scenario, "t0", service_points=POINTS,
                            services=SERVICES, spares_on_hand=1,
                            risk_group_ids={"t3": "rg_EXAMPLE_A_t0_t3"})
    assert obs.risk_group_ids["t3"] == "rg_EXAMPLE_A_t0_t3"
    assert obs.to_dict()["risk_group_ids"]["t3"] == "rg_EXAMPLE_A_t0_t3"


def test_hours_remaining_counts_down(scenario):
    assert build_observation(scenario, "t0", service_points=POINTS,
                             services=SERVICES,
                             spares_on_hand=1).hours_remaining == 3
    assert build_observation(scenario, "t3", service_points=POINTS,
                             services=SERVICES,
                             spares_on_hand=1).hours_remaining == 0


ACTIONS = ({"hour": "t1", "lever": "optical_reroute", "pairs": 1,
            "avoid": {"risk_groups": ["rg_EXAMPLE_A_t1_t3"]},
            "effective_at_index": 2},)


def test_the_observation_reports_the_actions_already_committed(scenario):
    obs = build_observation(scenario, "t2", service_points=POINTS,
                            services=SERVICES, spares_on_hand=0,
                            actions_taken=ACTIONS, spares_spent=1)
    assert obs.actions_taken == ACTIONS
    assert obs.spares_spent == 1
    payload = obs.to_dict()
    assert payload["actions_taken"] == [dict(ACTIONS[0])]
    assert payload["spares_spent"] == 1


def test_an_episode_with_no_commits_yet_reports_an_empty_action_list(scenario):
    obs = build_observation(scenario, "t0", service_points=POINTS,
                            services=SERVICES, spares_on_hand=1)
    assert obs.actions_taken == ()
    assert obs.to_dict()["actions_taken"] == []
    assert obs.to_dict()["spares_spent"] == 0
