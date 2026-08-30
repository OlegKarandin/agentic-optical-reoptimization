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
# ((lat, lon), (lat, lon)) per span, already filtered to the storm-cuttable
# ones by the caller -- observation.py stays free of event vocabulary and of
# the server.
SPANS = {
    "storm-svc-1": (((25.0, 81.0), (25.2, 81.0)),),
    "svc-b": (((25.6, 81.0), (25.8, 81.0)),),
}


@pytest.fixture
def scenario(write_scenario, example_scenario_yaml):
    return load_scenario(write_scenario(example_scenario_yaml))


# Task 10 fixtures: a roster wide enough to exercise co-terminating grouping.
# svc-b and svc-c share both endpoints (satna <-> raipur) -- a real
# co-terminating pair a single bidirectional lightpath restores together --
# while svc-d terminates nowhere near the depot (kolkata <-> mumbai) and so
# cannot compete for a satna spare regardless of its own exposure. All three
# spans sit near the t1 issuance's t3 cone centre (25.2, 81.0) so every
# service gets a nonzero exposure row there.
WIDE_SPANS = {
    "storm-svc-1": (((25.0, 81.0), (25.2, 81.0)),),
    "svc-b": (((25.6, 81.0), (25.8, 81.0)),),
    "svc-c": (((25.55, 81.0), (25.75, 81.0)),),
    "svc-d": (((25.65, 81.05), (25.85, 81.05)),),
}
WIDE_SERVICES = (
    {"id": "storm-svc-1", "src_router": "router_satna",
     "dst_router": "router_allahabad", "demand_gbps": 300.0,
     "working_path": ["ipl-a"], "protection_path": ["ipl-b"]},
    {"id": "svc-b", "src_router": "router_satna", "dst_router": "router_raipur",
     "demand_gbps": 100.0, "working_path": ["ipl-c"], "protection_path": []},
    {"id": "svc-c", "src_router": "router_satna", "dst_router": "router_raipur",
     "demand_gbps": 80.0, "working_path": ["ipl-d"], "protection_path": []},
    {"id": "svc-d", "src_router": "router_kolkata",
     "dst_router": "router_mumbai", "demand_gbps": 120.0,
     "working_path": ["ipl-e"], "protection_path": []},
)


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
    obs = build_observation(scenario, "t0", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1)
    payload = json.dumps(obs.to_dict())
    # The t1 issuance revises the cone centre to lat 25.2. If that number
    # appears anywhere in the t0 observation, waiting has become free.
    assert "25.2" not in payload
    assert obs.issuance.issued_at == "t0"


def test_observation_never_leaks_the_realized_cuts(scenario):
    obs = build_observation(scenario, "t0", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1)
    assert "fiber_004" not in json.dumps(obs.to_dict())


def test_exposure_is_reported_per_service_per_horizon_with_a_cut_probability(
    scenario,
):
    obs = build_observation(scenario, "t0", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1)
    at_t3 = obs.exposure["storm-svc-1"]["t3"]
    assert at_t3["hours_ahead"] == 3
    assert 0.0 < at_t3["p_cut"] <= 1.0
    # svc-b sits ~67 km north of the t3 cone axis, storm-svc-1 sits on it,
    # so storm-svc-1 must be the more exposed of the two.
    assert obs.exposure["svc-b"]["t3"]["p_cut"] < at_t3["p_cut"]


def test_each_exposure_row_carries_its_own_expected_capacity_at_risk(scenario):
    # D1 (remediation spec lines 447-456). p_cut x demand_gbps is the one
    # quantity every gold.rationale is arithmetic over (cone.py:73); asking
    # the model to multiply it across 8 services x 2 horizons in its head is
    # asking a deterministic step of a judgement engine.
    obs = build_observation(scenario, "t1", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1)
    entry = obs.exposure["storm-svc-1"]["t3"]
    assert set(entry) == {"hours_ahead", "offset_km", "width_km", "p_cut",
                          "demand_gbps", "expected_capacity_at_risk_gbps"}
    # Derived from the ROUNDED p_cut printed in the same dict, so a reader
    # multiplying the two numbers shown gets the number shown.
    assert entry["expected_capacity_at_risk_gbps"] == pytest.approx(
        round(entry["p_cut"] * entry["demand_gbps"], 3))


def test_the_derived_field_appears_for_every_service_and_every_horizon(scenario):
    obs = build_observation(scenario, "t1", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1)
    for svc, per_horizon in obs.exposure.items():
        for horizon, entry in per_horizon.items():
            assert "expected_capacity_at_risk_gbps" in entry, (
                f"{svc}/{horizon} is missing the derived field")


def test_horizon_totals_split_the_sut_from_everyone_else(scenario):
    # D2 (remediation spec lines 470-490), updated by Task 10: with no
    # endpoint_sites/depot_site supplied (as here), NO non-SUT service can be
    # shown depot-eligible, so everyone else's expected capacity at risk
    # lands in `non_sut_ineligible_ecar_gbps` and no restorable group exists.
    obs = build_observation(scenario, "t1", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1)
    totals = obs.horizon_totals["t3"]
    assert set(totals) == {"sut_ecar_gbps", "largest_restorable_group_ecar_gbps",
                           "non_sut_ineligible_ecar_gbps"}
    assert totals["sut_ecar_gbps"] == pytest.approx(
        obs.exposure["storm-svc-1"]["t3"]["expected_capacity_at_risk_gbps"])
    others = sum(per_horizon["t3"]["expected_capacity_at_risk_gbps"]
                 for svc, per_horizon in obs.exposure.items()
                 if svc != "storm-svc-1")
    assert totals["non_sut_ineligible_ecar_gbps"] == pytest.approx(
        others, abs=1e-3)
    assert totals["largest_restorable_group_ecar_gbps"] == pytest.approx(0.0)
    assert obs.restorable_groups.get("t3", ()) == ()
    assert obs.to_dict()["horizon_totals"] == obs.horizon_totals
    assert obs.to_dict()["restorable_groups"] == obs.restorable_groups


def test_horizon_totals_cover_exactly_the_issuances_horizons(scenario):
    obs = build_observation(scenario, "t1", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1)
    assert sorted(obs.horizon_totals) == sorted(obs.issuance.horizons)


def test_risk_group_ids_are_carried_through_for_the_constraint_decision(
    scenario,
):
    obs = build_observation(scenario, "t0", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1,
                            risk_group_ids={"t3": "rg_EXAMPLE_A_t0_t3"})
    assert obs.risk_group_ids["t3"] == "rg_EXAMPLE_A_t0_t3"
    assert obs.to_dict()["risk_group_ids"]["t3"] == "rg_EXAMPLE_A_t0_t3"


def test_hours_remaining_counts_down(scenario):
    assert build_observation(scenario, "t0", service_spans=SPANS,
                             services=SERVICES,
                             spares_on_hand=1).hours_remaining == 3
    assert build_observation(scenario, "t3", service_spans=SPANS,
                             services=SERVICES,
                             spares_on_hand=1).hours_remaining == 0


ACTIONS = ({"hour": "t1", "lever": "optical_reroute", "pairs": 1,
            "avoid": {"risk_groups": ["rg_EXAMPLE_A_t1_t3"]},
            "effective_at_index": 2, "effective_at_hour": "t2"},)


def test_the_observation_reports_the_actions_already_committed(scenario):
    obs = build_observation(scenario, "t2", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=0,
                            actions_taken=ACTIONS, spares_spent=1)
    assert obs.actions_taken == ACTIONS
    assert obs.spares_spent == 1
    payload = obs.to_dict()
    assert payload["actions_taken"] == [dict(ACTIONS[0])]
    assert payload["spares_spent"] == 1


def test_an_episode_with_no_commits_yet_reports_an_empty_action_list(scenario):
    obs = build_observation(scenario, "t0", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1)
    assert obs.actions_taken == ()
    assert obs.to_dict()["actions_taken"] == []
    assert obs.to_dict()["spares_spent"] == 0


def test_the_wire_names_the_actionable_service_not_the_one_under_test(
        scenario):
    # "service under test" is eval-harness vocabulary leaking into the
    # operational world; being named it reads as "this is the important one",
    # which is exactly the framing the control run's root cause #5 accuses
    # (eval-fairness design, §5.3).
    payload = build_observation(scenario, "t0", service_spans=SPANS,
                                services=SERVICES, spares_on_hand=1).to_dict()
    assert payload["actionable_service"] == "storm-svc-1"
    assert "service_under_test" not in payload


def test_the_roster_marks_which_row_the_tools_can_act_on(scenario):
    rows = build_observation(scenario, "t0", service_spans=SPANS,
                             services=SERVICES,
                             spares_on_hand=1).to_dict()["services"]
    actionable = [r for r in rows if r.get("actionable")]
    assert [r["id"] for r in actionable] == ["storm-svc-1"]


def test_the_python_attribute_keeps_the_harness_name(scenario):
    obs = build_observation(scenario, "t0", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1)
    assert obs.service_under_test == "storm-svc-1"


def test_a_service_with_no_cuttable_span_gets_no_exposure_row(scenario):
    # p_cut is exactly 0 and offset_km would have to be None or inf. A service
    # with no representative point has always been omitted the same way; this
    # is that rule, applied to the filtered span list.
    obs = build_observation(scenario, "t0",
                            service_spans={**SPANS, "svc-b": ()},
                            services=SERVICES, spares_on_hand=1)
    assert "svc-b" not in obs.exposure
    assert "storm-svc-1" in obs.exposure


def test_offset_km_is_the_distance_to_the_nearest_cuttable_span(scenario):
    # NOT to a representative midpoint. The t3 cone of the t1 issuance is
    # centred at (25.2, 81.0), which is one END of storm-svc-1's only span --
    # so the offset is zero and the containment test the forecast-blind
    # baseline runs reads "inside".
    obs = build_observation(scenario, "t1", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1)
    assert obs.exposure["storm-svc-1"]["t3"]["offset_km"] == pytest.approx(0.0)
    # Measured 0.9618 at width_km=90/damage_radius_km=74 -- comfortably
    # "near-certain" without pinning a brittle exact float.
    assert obs.exposure["storm-svc-1"]["t3"]["p_cut"] > 0.95


def test_p_cut_is_the_region_probability_not_a_point_one(scenario):
    from storm_reoptimizer.eval.cone import p_cut_region
    obs = build_observation(scenario, "t1", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1)
    expected = p_cut_region(SPANS["svc-b"], 25.2, 81.0, 90.0, 74.0)
    assert obs.exposure["svc-b"]["t3"]["p_cut"] == pytest.approx(
        round(expected, 4), abs=1e-9)


ENDPOINT_SITES = {"storm-svc-1": ("satna", "allahabad"),
                  "svc-b": ("satna", "raipur"),
                  "svc-c": ("satna", "raipur"),
                  "svc-d": ("kolkata", "mumbai")}


def test_one_spare_buys_one_lightpath_so_groups_are_maxed_not_summed(scenario):
    # svc-b and svc-c co-terminate (satna <-> raipur, both directions), so one
    # bidirectional lightpath genuinely restores both and their ECARs add.
    # svc-d terminates nowhere near satna: restoring it needs transponders at
    # kolkata and mumbai, and a satna line card cannot serve it, so it does
    # not compete for this depot at all.
    obs = build_observation(scenario, "t1", service_spans=WIDE_SPANS,
                            services=WIDE_SERVICES, spares_on_hand=1,
                            endpoint_sites=ENDPOINT_SITES, depot_site="satna")
    totals = obs.horizon_totals["t3"]
    assert set(totals) == {"sut_ecar_gbps",
                           "largest_restorable_group_ecar_gbps",
                           "non_sut_ineligible_ecar_gbps"}
    b = obs.exposure["svc-b"]["t3"]["expected_capacity_at_risk_gbps"]
    c = obs.exposure["svc-c"]["t3"]["expected_capacity_at_risk_gbps"]
    d = obs.exposure["svc-d"]["t3"]["expected_capacity_at_risk_gbps"]
    assert totals["largest_restorable_group_ecar_gbps"] == pytest.approx(
        b + c, abs=1e-3)
    assert totals["non_sut_ineligible_ecar_gbps"] == pytest.approx(d, abs=1e-3)


def test_the_groups_are_enumerated_not_just_totalled(scenario):
    obs = build_observation(scenario, "t1", service_spans=WIDE_SPANS,
                            services=WIDE_SERVICES, spares_on_hand=1,
                            endpoint_sites=ENDPOINT_SITES, depot_site="satna")
    groups = {g["endpoints"]: g for g in obs.restorable_groups["t3"]}
    assert groups[("raipur", "satna")]["members"] == ("svc-b", "svc-c")
    assert ("kolkata", "mumbai") not in groups


def test_the_actionable_service_is_not_its_own_competing_claim(scenario):
    obs = build_observation(scenario, "t1", service_spans=WIDE_SPANS,
                            services=WIDE_SERVICES, spares_on_hand=1,
                            endpoint_sites=ENDPOINT_SITES, depot_site="satna")
    members = {m for g in obs.restorable_groups["t3"] for m in g["members"]}
    assert "storm-svc-1" not in members
