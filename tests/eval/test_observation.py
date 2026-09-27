"""Observation building, and above all issuance isolation (eval design spec,
"Testing": "the observation built for hour t contains issued_t and no later
issuance. This is the one leak that would silently invalidate every timing
result, so it is asserted rather than assumed")."""
import json

import pytest

from storm_reoptimizer.eval.cone import p_cut_service
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


@pytest.fixture
def services():
    return SERVICES


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
    assert "fiber_rewa_satna_0" not in json.dumps(obs.to_dict())


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
        round(expected, 3), abs=1e-9)


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


def test_a_group_over_capacity_is_capped_to_the_admitted_members(scenario):
    from storm_reoptimizer.eval.observation import _restorable_groups
    exposure = {
        "svc-b": {"t3": {"expected_capacity_at_risk_gbps": 500.0,
                         "demand_gbps": 500.0}},
        "svc-c": {"t3": {"expected_capacity_at_risk_gbps": 400.0,
                         "demand_gbps": 400.0}},
        "svc-d": {"t3": {"expected_capacity_at_risk_gbps": 100.0,
                         "demand_gbps": 100.0}},
    }
    endpoint_sites = {"svc-b": ("satna", "raipur"),
                      "svc-c": ("satna", "raipur"),
                      "svc-d": ("satna", "raipur")}
    groups = _restorable_groups(
        exposure, endpoint_sites, depot_site="satna",
        service_under_test="storm-svc-1", lightpath_capacity_gbps=800.0)
    [group] = groups["t3"]
    # Highest-ECAR-first: svc-b (500) admitted, svc-c (400) would push the
    # running sum to 900 > 800 and is left out, svc-d (100) fits in the
    # remaining 300 and is admitted.
    assert group["members"] == ("svc-b", "svc-d")
    assert group["ecar_gbps"] == pytest.approx(600.0)


def test_the_t1_shape_100g_each_direction_is_not_capped(scenario):
    from storm_reoptimizer.eval.observation import _restorable_groups
    exposure = {
        "c-fwd": {"t3": {"expected_capacity_at_risk_gbps": 90.0,
                        "demand_gbps": 100.0}},
        "c-rev": {"t3": {"expected_capacity_at_risk_gbps": 85.0,
                        "demand_gbps": 100.0}},
    }
    endpoint_sites = {"c-fwd": ("jalgaon", "dhulia"),
                      "c-rev": ("dhulia", "jalgaon")}
    groups = _restorable_groups(
        exposure, endpoint_sites, depot_site="jalgaon",
        service_under_test="storm-svc-1")     # default capacity (800.0)
    [group] = groups["t3"]
    assert group["members"] == ("c-fwd", "c-rev")
    assert group["ecar_gbps"] == pytest.approx(175.0)


def test_the_observation_carries_the_episodes_damage_radius(
    write_scenario, example_scenario_yaml,
):
    """The baseline's containment test and the agent's prompt both need it,
    and it is scenario data the decider is entitled to see -- an operator
    reading a published hazard area knows how far the damage reaches."""
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    obs = build_observation(
        scenario, "t1", service_spans={}, services=(), spares_on_hand=1)
    assert obs.damage_radius_km == 74.0
    assert obs.to_dict()["damage_radius_km"] == 74.0


def test_protected_service_p_cut_is_joint_rounded_to_3_and_has_no_per_leg_entry(
        scenario, services):
    working = {"storm-svc-1": (((24.6, 80.8), (24.5, 81.3)),)}
    protection = {"storm-svc-1": (((24.6, 80.8), (24.9, 80.6)),)}
    obs = build_observation(scenario, "t1", service_spans=working, services=services,
                            spares_on_hand=1, protection_spans=protection)
    row = obs.exposure["storm-svc-1"]["t3"]
    cone = latest_issuance(scenario, "t1").horizons["t3"]
    assert row["p_cut"] == round(p_cut_service(working["storm-svc-1"], protection["storm-svc-1"],
                                               cone.center["lat"], cone.center["lon"],
                                               cone.width_km, scenario.damage_radius_km), 3)
    assert row["expected_capacity_at_risk_gbps"] == round(row["p_cut"] * 300.0, 3)
    # Spec 5.3 (decided 2026-09-06): no per-leg display at all; the joint
    # probability is the only exposure number a protected service shows.
    assert "legs" not in row


def test_issuance_schedule_and_deadline(scenario, services):
    obs = build_observation(scenario, "t0", service_spans={}, services=services, spares_on_hand=1)
    assert obs.issuance_schedule == ("t0", "t1")
    # lead_time_hours in the fixture is 1 and the only horizon is t3 -> deadline t2
    assert obs.deadline_hour == {"ip_reroute": "t3", "hybrid": "t2", "optical_reroute": "t2"}
    assert obs.to_dict()["deadline_hour"]["optical_reroute"] == "t2"


def test_deadline_is_none_once_passed(scenario, services):
    obs = build_observation(scenario, "t3", service_spans={}, services=services, spares_on_hand=1)
    assert obs.deadline_hour["optical_reroute"] is None


def test_next_issuance_names_the_hour_a_revision_is_still_coming(
        example_scenario_yaml, write_scenario):
    """EXAMPLE_A publishes at t0 and t1 over hours [t0..t3]. At t0 a revision
    is still scheduled; at t1 the issuance in force IS the last one."""
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    at_t0 = build_observation(scenario, "t0", service_spans={}, services=(),
                              spares_on_hand=1)
    assert at_t0.next_issuance == {"hour": "t1"}
    assert at_t0.to_dict()["next_issuance"] == {"hour": "t1"}
    for hour in ("t1", "t2", "t3"):
        later = build_observation(scenario, hour, service_spans={},
                                  services=(), spares_on_hand=1)
        assert later.next_issuance is None, hour


def test_a_timing_observation_carries_no_decision_shell(
        example_scenario_yaml, write_scenario):
    """`decided_this_hour: null` at the timing step would read as 'you have
    already decided nothing this hour'. Absent means not yet asked."""
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    payload = build_observation(scenario, "t0", service_spans={}, services=(),
                                spares_on_hand=1).to_dict()
    assert "decided_this_hour" not in payload
    assert "attempts_this_hour" not in payload
    assert payload["standing_claim_priority"] == []


def test_the_carried_decision_and_attempts_survive_to_dict(
        example_scenario_yaml, write_scenario):
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    decided = {"timing": {"action": "act", "reasoning": "spend",
                          "contested_claim": None, "claim_priority": ["s"]},
               "probe_answers": [{"service_id": "s",
                                  "risk_group_id": "rg_ref",
                                  "status": "ok",
                                  "full_restore_candidates": 3,
                                  "min_spares_needed_by_site": {"satna": 1},
                                  "levers": ["optical_reroute"]}]}
    attempts = ({"avoid": {"risk_groups": ["rg_ref"]},
                 "menu_status": "no_solution", "menu_size": 0,
                 "choice": "infeasible", "outcome": "declared_infeasible"},)
    payload = build_observation(
        scenario, "t0", service_spans={}, services=(), spares_on_hand=1,
        decided_this_hour=decided, attempts_this_hour=attempts,
        standing_claim_priority=("c", "s")).to_dict()
    assert payload["decided_this_hour"] == decided
    assert payload["attempts_this_hour"] == list(attempts)
    assert payload["standing_claim_priority"] == ["c", "s"]


def test_the_band_is_shown_only_while_a_revision_is_still_scheduled(
        example_scenario_yaml, write_scenario):
    """`p_cut` says how likely the cut is if this issuance is right; the band
    says how much that can move when it is revised. At the last issuance
    there is nothing left to revise, so the field is withheld rather than
    shown as a zero-width band that would read as certainty."""
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    spans = {"storm-svc-1": (((25.0, 81.0), (25.1, 81.2)),)}
    services = ({"id": "storm-svc-1", "demand_gbps": 300.0},)
    at_t0 = build_observation(scenario, "t0", service_spans=spans,
                              services=services, spares_on_hand=1)
    row = at_t0.exposure["storm-svc-1"]["t1"]
    assert set(row["p_cut_if_track_revised"]) == {
        "revision_radius_km", "min", "max", "mean"}
    # 30 km per hour of lead x 1 hour ahead.
    assert row["p_cut_if_track_revised"]["revision_radius_km"] == 30.0
    at_t1 = build_observation(scenario, "t1", service_spans=spans,
                              services=services, spares_on_hand=1)
    assert "p_cut_if_track_revised" not in at_t1.exposure["storm-svc-1"]["t3"]


def test_the_band_scales_with_how_far_ahead_the_horizon_is(
        example_scenario_yaml, write_scenario):
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    spans = {"storm-svc-1": (((25.0, 81.0), (25.1, 81.2)),)}
    services = ({"id": "storm-svc-1", "demand_gbps": 300.0},)
    obs = build_observation(scenario, "t0", service_spans=spans,
                            services=services, spares_on_hand=1)
    near = obs.exposure["storm-svc-1"]["t1"]["p_cut_if_track_revised"]
    far = obs.exposure["storm-svc-1"]["t3"]["p_cut_if_track_revised"]
    assert near["revision_radius_km"] == 30.0
    assert far["revision_radius_km"] == 90.0


def test_probe_answers_carry_across_the_episode_on_the_observation(scenario):
    """ClaudeDecider opens a FRESH conversation for every decision
    (agent.py:770), so without this field a probe answer bought at t0 does not
    exist at t0's own constraints call, let alone at t1's timing call. T3b run
    B seed 0 probed buldhana at t0, used the answer, and reverted at t1
    (2026-09-12 failure analysis, finding 7). Same argument as `actions_taken`,
    which the codebase already accepted."""
    carried = ({"hour": "t0", "decision": "timing", "service_id": "svc-b",
                "risk_group_id": "rg_t3", "status": "no_solution",
                "full_restore_candidates": 0,
                "min_spares_needed_by_site": None, "levers": [],
                "scope": "answered while avoiding every asset in rg_t3; "
                         "narrower avoid sets were not evaluated"},)
    obs = build_observation(scenario, "t1", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1,
                            probe_answers_this_episode=carried)
    assert obs.probe_answers_this_episode == carried
    assert obs.to_dict()["probe_answers_this_episode"] == list(carried)


def test_the_carried_probe_field_is_always_emitted_even_when_empty(scenario):
    """Always present, unlike `decided_this_hour`/`attempts_this_hour`: an
    absent key then means "this trace predates the field" and an empty list
    means "nothing has been asked yet", which are different facts (the viewer's
    own standing lesson, harness explainer §13)."""
    obs = build_observation(scenario, "t0", service_spans=SPANS,
                            services=SERVICES, spares_on_hand=1)
    assert obs.to_dict()["probe_answers_this_episode"] == []
