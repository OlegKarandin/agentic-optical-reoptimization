"""JevDecider (spec 2026-09-28). Unit tests only: a fake TypeSafe client, no
network, no cost, no `typesafe-sdk` install."""
import json
import re

from storm_reoptimizer.eval.agent import SYSTEM_PROMPT, ClaudeDecider
from storm_reoptimizer.eval.jev import (
    CLAIM_LEVELS,
    CLAIM_SCORE_INSTRUCTION,
    CONSTRAINT_INSTRUCTION,
    DEFAULT_JEV_MODEL,
    HOLD_CRITERION,
    INFEASIBLE_CRITERION,
    OBJECTIVE_INSTRUCTION,
    PROBE_GATE_INSTRUCTION,
    TIMING_ACTION_CRITERIA,
    TIMING_ACTION_INSTRUCTION,
    TOTALS_SENTENCE,
    JevDecider,
)
from storm_reoptimizer.eval.observation import Observation
from storm_reoptimizer.eval.scenario_file import ConeAtHorizon, Issuance

HORIZON = "t3"
DEPOT_SITE = "jalgaon"
SUT = "t1-svc-jalgaon-indore"
RG = "rg_T1a_t0_t3"
CLAIMANTS = [("claim-a", 0.30, 100.0), ("claim-b", 0.20, 100.0)]


def _obs(*, others=CLAIMANTS, risk_group_ids=None, risk_group_assets=(),
         iteration=0, last_rejection=None):
    """T1's shape at t0: the SUT plus depot-eligible claimants, each in its
    own co-terminating group, with horizon_totals populated so the `totals`
    arm has something to add."""
    exposure = {SUT: {HORIZON: {"hours_ahead": 3, "offset_km": 0.0,
                                "width_km": 200.0, "p_cut": 0.38,
                                "demand_gbps": 300.0}}}
    services = [{"id": SUT, "demand_gbps": 300.0,
                 "src_router": "router_jalgaon", "dst_router": "router_indore",
                 "working_path": ["w"], "protection_path": ["p"]}]
    groups = []
    for svc, p_cut, demand in others:
        exposure[svc] = {HORIZON: {"hours_ahead": 3, "offset_km": 50.0,
                                   "width_km": 200.0, "p_cut": p_cut,
                                   "demand_gbps": demand}}
        services.append({"id": svc, "demand_gbps": demand,
                         "src_router": "router_jalgaon",
                         "dst_router": f"router_{svc}",
                         "working_path": [], "protection_path": []})
        groups.append({"endpoints": (DEPOT_SITE, svc), "members": (svc,),
                       "ecar_gbps": round(p_cut * demand, 3)})
    return Observation(
        scenario_id="T1a", service_under_test=SUT, hour="t0", hour_index=0,
        hours_remaining=4,
        issuance=Issuance(issued_at="t0", horizons={
            HORIZON: ConeAtHorizon(cone={"type": "Polygon", "coordinates": []},
                                   width_km=200.0,
                                   center={"lat": 21.0, "lon": 75.5})}),
        exposure=exposure, services=tuple(services), spares_on_hand=1,
        lead_time_hours=1,
        risk_group_ids=(risk_group_ids if risk_group_ids is not None
                        else {HORIZON: RG}),
        risk_group_assets=tuple(risk_group_assets),
        iteration=iteration, last_rejection=last_rejection,
        restorable_groups={HORIZON: tuple(groups)} if groups else {},
        horizon_totals={HORIZON: {"sut_ecar_gbps": 114.0,
                                  "largest_restorable_group_ecar_gbps": 30.0,
                                  "non_sut_ineligible_ecar_gbps": 0.0}})


def test_arm_names_are_distinct_and_filename_safe():
    raw, totals = JevDecider(), JevDecider(include_totals=True)
    assert raw.name == f"jev:{DEFAULT_JEV_MODEL}"
    assert totals.name == f"jev:{DEFAULT_JEV_MODEL}+totals"
    assert DEFAULT_JEV_MODEL != "jev-latest"
    for name in (raw.name, totals.name):
        assert re.fullmatch(r"[A-Za-z0-9._+-]+", name.replace(":", "_"))


def test_construction_needs_neither_the_extra_nor_a_key():
    decider = JevDecider()          # must not import typesafe_sdk
    assert decider._client is None


def test_raw_projection_is_claudes_projection():
    obs = _obs()
    jev, claude = JevDecider(), ClaudeDecider(client=object())
    for kwargs in ({}, {"include_risk_group_assets": True}):
        assert jev._project(obs, **kwargs) == claude._project(obs, **kwargs)
        assert jev.last_projection == claude.last_projection
    assert "horizon_totals" not in jev.last_projection


def test_totals_projection_adds_exactly_horizon_totals():
    obs = _obs()
    raw = JevDecider()._project(obs)
    totals = JevDecider(include_totals=True)._project(obs)
    assert set(totals) - set(raw) == {"horizon_totals"}
    assert {k: v for k, v in totals.items() if k != "horizon_totals"} == raw
    assert totals["horizon_totals"] == obs.horizon_totals


def test_with_totals_appends_only_the_totals_sentence():
    assert JevDecider()._with_totals("Q?") == "Q?"
    assert JevDecider(include_totals=True)._with_totals("Q?") == \
        f"Q? {TOTALS_SENTENCE}"
    assert "horizon_totals" in TOTALS_SENTENCE


def _all_question_text():
    parts = [TIMING_ACTION_INSTRUCTION, CLAIM_SCORE_INSTRUCTION,
             CONSTRAINT_INSTRUCTION, OBJECTIVE_INSTRUCTION,
             PROBE_GATE_INSTRUCTION, *CLAIM_LEVELS,
             json.dumps(TIMING_ACTION_CRITERIA), json.dumps(HOLD_CRITERION),
             json.dumps(INFEASIBLE_CRITERION)]
    return "\n".join(parts)


def test_rule2_raw_text_never_names_horizon_totals():
    assert "horizon_totals" not in _all_question_text()


def test_rule2_observation_paths_name_only_fields_claude_is_told_about():
    claude_fields = set(re.findall(r"`([a-z_]+)`", SYSTEM_PROMPT))
    paths = re.findall(r"`observation\.([a-z_]+)", _all_question_text())
    assert paths, "the constants should point at evidence"
    assert set(paths) <= claude_fields, set(paths) - claude_fields


def test_rule4_confusable_options_carry_structured_criteria():
    for crit in (TIMING_ACTION_CRITERIA["act"], TIMING_ACTION_CRITERIA["wait"],
                 HOLD_CRITERION, INFEASIBLE_CRITERION):
        assert set(crit) == {"what", "not_for", "examples"}
        assert crit["examples"]
    assert set(TIMING_ACTION_CRITERIA) == {"act", "wait"}


def test_rule5_score_levels_carry_no_numbers():
    assert 2 <= len(CLAIM_LEVELS) <= 10
    assert not any(ch.isdigit() for level in CLAIM_LEVELS for ch in level)
