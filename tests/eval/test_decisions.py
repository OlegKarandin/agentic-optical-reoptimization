"""The three decision interfaces (eval design spec, Decisions 1-3)."""
import pytest

from storm_reoptimizer.eval.decisions import (
    COST_TERMS, ConstraintDecision, DecisionError, ObjectiveDecision,
    TimingDecision, candidate_index, rank_by_priority,
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
    d = ConstraintDecision.from_dict(
        {"avoid": {"risk_groups": ["rg_t3"]}, "reasoning": "clear of t+3"})
    assert d.avoid == {"risk_groups": ["rg_t3"]}
    assert (d.protected, d.best_effort, d.basis, d.level) == (
        True, False, "srlg", "srlg")


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
