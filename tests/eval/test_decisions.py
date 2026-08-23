"""The three decision interfaces (eval design spec, Decisions 1-3)."""
import pytest

from storm_reoptimizer.eval.decisions import (
    CONSTRAINT_JSON_SCHEMA, COST_TERMS, ConstraintDecision, DecisionError,
    ObjectiveDecision, TimingDecision, candidate_index, rank_by_priority,
)


def test_timing_decision_round_trips():
    d = TimingDecision.from_dict({"action": "wait", "reasoning": "cone is wide"})
    assert d.action == "wait"
    assert d.to_dict() == {"action": "wait", "reasoning": "cone is wide"}


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
               "basis": "risk_group", "level": "risk_group"}
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
