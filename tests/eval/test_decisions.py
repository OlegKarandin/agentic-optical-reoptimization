"""The three decision interfaces (eval design spec, Decisions 1-3)."""
import pytest

from storm_reoptimizer.eval.decisions import (
    CONSTRAINT_JSON_SCHEMA, COST_TERMS, ConstraintDecision, DecisionError,
    OBJECTIVE_JSON_SCHEMA, ObjectiveDecision, TIMING_JSON_SCHEMA,
    TimingDecision, candidate_index, rank_by_priority,
)


def test_timing_decision_round_trips():
    d = TimingDecision.from_dict({"action": "wait", "reasoning": "cone is wide"})
    assert d.action == "wait"
    assert d.to_dict() == {"action": "wait", "reasoning": "cone is wide",
                           "contested_claim": None, "claim_priority": []}


def test_timing_rejects_an_unknown_action():
    with pytest.raises(DecisionError, match="action"):
        TimingDecision.from_dict({"action": "panic", "reasoning": "x"})


def test_reasoning_is_mandatory_on_all_three_decisions():
    with pytest.raises(DecisionError, match="reasoning"):
        TimingDecision.from_dict({"action": "act"})
    with pytest.raises(DecisionError, match="reasoning"):
        ObjectiveDecision.from_dict({"choice": "candidate_0", "priority": None})


def test_constraint_decision_defaults_the_protection_posture():
    """The default is the posture this project actually uses, NOT the
    server signature's own. It used to be protected=True/srlg/srlg, and the
    whole-branch review (2026-08-23, finding I5) confirmed that ZERO of the
    7+ real call sites on this branch wanted that -- every one of them
    passed protected=False/basis="physical"/level="link" explicitly, so the
    old default was a pure trap for anything constructing a
    ConstraintDecision without naming every field."""
    d = ConstraintDecision.from_dict(
        {"avoid": {"risk_groups": ["rg_t3"]}, "reasoning": "clear of t+3"})
    assert d.avoid == {"risk_groups": ["rg_t3"]}
    assert (d.protected, d.best_effort, d.basis, d.level) == (
        False, False, "physical", "link")
    # The bare constructor and from_dict must agree -- they are two of the
    # three paths into this object (ScriptedDecider's own fallback is the
    # third) and a divergence between them would be invisible.
    bare = ConstraintDecision(avoid={}, reasoning="x")
    assert (bare.protected, bare.best_effort, bare.basis, bare.level) == (
        False, False, "physical", "link")


def test_risk_group_is_a_legal_constraint_level():
    """T2a/T3a/T3b's gold decisions only validate under basis="risk_group"/
    level="risk_group". _LEVELS omitted it until 2026-08-23 (whole-branch
    review finding C3), which made those three gold answers unrepresentable
    by anything that goes through from_dict or the JSON schema -- i.e. by
    step 6's schema-constrained LLM decider. The tests that construct them
    directly never noticed, because the constructor does no validation."""
    payload = {"avoid": {"risk_groups": ["rg_T2a_t1_t6"]},
               "reasoning": "avoid the full t+6 cone",
               "protected": False, "best_effort": False,
               "basis": "risk_group", "level": "risk_group",
               "contested_claim": None}
    d = ConstraintDecision.from_dict(payload)
    assert (d.basis, d.level) == ("risk_group", "risk_group")
    assert d.to_dict() == payload
    assert ConstraintDecision.from_dict(d.to_dict()) == d
    assert "risk_group" in CONSTRAINT_JSON_SCHEMA["properties"]["level"]["enum"]
    assert "risk_group" in CONSTRAINT_JSON_SCHEMA["properties"]["basis"]["enum"]


def test_constraint_decision_rejects_an_unknown_level():
    with pytest.raises(DecisionError, match="level"):
        ConstraintDecision.from_dict(
            {"avoid": {}, "reasoning": "x", "level": "continent"})


def test_constraint_decision_rejects_an_unknown_avoid_key():
    with pytest.raises(DecisionError, match="avoid"):
        ConstraintDecision.from_dict(
            {"avoid": {"riskgroups": ["rg_t3"]}, "reasoning": "typo"})


def test_objective_priority_must_name_only_real_cost_terms():
    ok = ObjectiveDecision.from_dict(
        {"choice": "candidate_1", "priority": ["transponders", "dropped_traffic"],
         "reasoning": "keep the spare"})
    assert ok.priority == ("transponders", "dropped_traffic")
    with pytest.raises(DecisionError, match="priority"):
        ObjectiveDecision.from_dict(
            {"choice": "candidate_1", "priority": ["vibes"], "reasoning": "x"})


def test_declining_to_state_a_priority_is_legitimate():
    d = ObjectiveDecision.from_dict(
        {"choice": "candidate_2", "priority": None,
         "reasoning": "not a strict ordering"})
    assert d.priority is None


def test_candidate_index_parses_the_choice_label():
    assert candidate_index("candidate_0") == 0
    assert candidate_index("candidate_12") == 12
    assert candidate_index("infeasible") is None
    with pytest.raises(DecisionError):
        candidate_index("candidate_x")


def test_rank_by_priority_is_lexicographic_and_treats_margin_as_a_benefit():
    cands = [
        {"cost_vector": {"transponders": 4.0, "total_margin": 1.0}},
        {"cost_vector": {"transponders": 2.0, "total_margin": 0.0}},
        {"cost_vector": {"transponders": 2.0, "total_margin": 9.0}},
    ]
    # Fewer transponders first; among ties, MORE margin first.
    assert rank_by_priority(cands, ("transponders", "total_margin")) == [2, 1, 0]


def test_cost_terms_are_the_seven_the_server_reports():
    assert COST_TERMS == (
        "spectrum_used", "transponders", "max_util", "dropped_traffic",
        "added_latency", "total_margin", "services_at_risk")


def test_a_decision_carries_the_rival_claim_it_weighed_or_null():
    d = TimingDecision.from_dict({
        "action": "wait", "reasoning": "the cone sharpens next hour",
        "contested_claim": {"service_id": "d0462",
                            "expected_capacity_at_risk_gbps": 88.3}})
    assert d.contested_claim == {"service_id": "d0462",
                                 "expected_capacity_at_risk_gbps": 88.3}


def test_a_decision_with_no_rival_claim_says_so_explicitly():
    d = TimingDecision.from_dict({
        "action": "act", "reasoning": "nothing else is exposed",
        "contested_claim": None})
    assert d.contested_claim is None
    assert d.to_dict()["contested_claim"] is None
    assert d.to_dict()["claim_priority"] == []


def test_claim_priority_round_trips_and_validates():
    d = TimingDecision.from_dict({"action": "wait", "reasoning": "hold",
                                  "contested_claim": None,
                                  "claim_priority": ["c", "s"]})
    assert d.claim_priority == ("c", "s")
    assert d.to_dict()["claim_priority"] == ["c", "s"]
    with pytest.raises(DecisionError):
        TimingDecision.from_dict(
            {"action": "wait", "reasoning": "x", "claim_priority": "c"})


def test_an_absent_contested_claim_is_the_same_as_null():
    # baseline.py and every gold decision construct these positionally with no
    # claim; they must keep working untouched.
    assert TimingDecision.from_dict(
        {"action": "act", "reasoning": "x"}).contested_claim is None
    assert TimingDecision("act", "x").contested_claim is None
    assert ObjectiveDecision("candidate_0", None, "x").contested_claim is None
    assert ConstraintDecision(avoid={}, reasoning="x").contested_claim is None


@pytest.mark.parametrize("bad, match", [
    ({"service_id": "d0462"}, "expected_capacity_at_risk_gbps"),
    ({"expected_capacity_at_risk_gbps": 1.0}, "service_id"),
    ({"service_id": "", "expected_capacity_at_risk_gbps": 1.0},
     "service_id"),
    ({"service_id": "d0462", "expected_capacity_at_risk_gbps": "big"},
     "expected_capacity_at_risk_gbps"),
    ("d0462", "contested_claim"),
])
def test_a_malformed_contested_claim_is_rejected(bad, match):
    with pytest.raises(DecisionError, match=match):
        TimingDecision.from_dict(
            {"action": "act", "reasoning": "x", "contested_claim": bad})


def test_all_three_decisions_require_the_field():
    for schema in (TIMING_JSON_SCHEMA, CONSTRAINT_JSON_SCHEMA,
                   OBJECTIVE_JSON_SCHEMA):
        assert "contested_claim" in schema["required"]
        assert "contested_claim" in schema["properties"]
