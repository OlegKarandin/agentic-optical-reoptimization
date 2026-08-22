"""The one-variable check (eval design spec, "The one-variable check"), and
the fixtures that prove it fires. "An assertion never seen to fire is not
known to work, and this one's whole job is to fire during authoring." """
from pathlib import Path

import pytest

from storm_reoptimizer.eval.assertions import (
    PairInvalid, assert_no_single_variable_rule_solves,
)
from storm_reoptimizer.eval.rules import (
    OBSERVABLE_VARS, best_rule, candidate_rules, score_rule,
)
from storm_reoptimizer.eval.scenario_file import load_all_scenarios

DISCARDED = Path(__file__).parent / "fixtures" / "discarded"


def test_the_enumerated_observables_are_the_five_the_spec_names():
    assert OBSERVABLE_VARS == (
        "cone_width_km", "cone_motion_kmh", "n_future_claimants",
        "exposure_horizon_hours", "spares_on_hand")


def test_the_enumeration_includes_parameter_free_greedy_policies():
    episodes = list(load_all_scenarios(DISCARDED).values())
    names = {r.name for r in candidate_rules(episodes)}
    assert "greedy:widest_feasible_avoid" in names
    assert "greedy:always_lo" in names


def test_the_check_rejects_the_discarded_drafts(tmp_path):
    """T1's gold was a pure function of "is my service in the cone?", T2's of
    cone_motion, T3's of n_future_claimants. Each is one observable, so a
    one-line rule reproduces it -- the check must say so."""
    episodes = list(load_all_scenarios(DISCARDED).values())
    rule, score = best_rule(episodes)
    assert score == pytest.approx(1.0), (
        f"the discarded drafts are supposed to be solvable by one rule; "
        f"best was {rule.name} at {score}")
    with pytest.raises(PairInvalid, match="single-variable"):
        assert_no_single_variable_rule_solves(episodes)


def test_the_greedy_policy_alone_beats_the_discarded_T2(tmp_path):
    episodes = [e for e in load_all_scenarios(DISCARDED).values()
                if e.pair == "T2"]
    greedy = next(r for r in candidate_rules(episodes)
                  if r.name == "greedy:widest_feasible_avoid")
    assert score_rule(greedy, episodes) == pytest.approx(1.0)


def test_the_shipped_suite_is_not_solved_by_any_single_rule():
    """The real check, over the real episodes. Runs green only once Tasks
    14-17 have landed all seven."""
    episodes = list(load_all_scenarios().values())
    if len(episodes) < 7:
        pytest.skip("episodes not authored yet (Tasks 14-17)")
    assert_no_single_variable_rule_solves(episodes)
