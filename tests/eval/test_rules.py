"""The one-variable check (eval design spec, "The one-variable check"), and
the fixtures that prove it fires. "An assertion never seen to fire is not
known to work, and this one's whole job is to fire during authoring." """
import asyncio
from pathlib import Path

import pytest

from storm_reoptimizer.eval.assertions import (
    PairInvalid, assert_no_single_variable_rule_solves,
)
from storm_reoptimizer.eval.derived import derived_scalars_for
from storm_reoptimizer.eval.rules import (
    OBSERVABLE_VARS, _distinct_within_tolerance, best_rule, candidate_rules,
    score_rule, split_points,
)
from storm_reoptimizer.eval.scenario_file import load_all_scenarios
from storm_reoptimizer.mcp_client import connect_server

DISCARDED = Path(__file__).parent / "fixtures" / "discarded"
TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)


def test_the_enumerated_observables_are_the_five_the_spec_names():
    assert OBSERVABLE_VARS == (
        "cone_width_km", "cone_motion_kmh", "n_future_claimants",
        "exposure_horizon_hours", "spares_on_hand")


def test_distinct_within_tolerance_collapses_float_noise_not_real_gaps():
    """Regression for the failure mode found rebuilding T1's geometry
    (whole-branch fix, Step 2): two derived values that agree to ~1e-13 --
    well inside the tolerance passed here (1e-6), real float slack from a numerical
    solve, not a real difference -- must collapse to ONE split-point
    candidate, or a threshold rule can sit on the noise and falsely "solve"
    a pair that was built to be unsolvable. A gap that is actually
    meaningful (1e-3, three orders above the tolerance) must NOT collapse --
    the check still has to catch a real confound."""
    assert _distinct_within_tolerance(
        [0.1349420979, 0.1349420979 + 1e-13], tol=1e-6) == [0.1349420979]
    assert _distinct_within_tolerance(
        [0.5, 0.5 + 1e-3], tol=1e-6) == pytest.approx([0.5, 0.501])


def test_split_points_are_midpoints_between_adjacent_distinct_values():
    assert split_points([1.0, 3.0, 7.0]) == [2.0, 5.0]


def test_split_points_collapse_values_inside_the_tolerance():
    # The float-noise failure mode _distinct_within_tolerance exists for: two
    # values equal BY CONSTRUCTION but not bit-for-bit must not yield a split.
    assert split_points([1.0, 1.0 + 1e-12, 3.0]) == [2.0]


def test_the_enumeration_includes_parameter_free_greedy_policies():
    episodes = list(load_all_scenarios(DISCARDED).values())
    names = {r.name for r in candidate_rules(episodes)}
    assert "greedy:widest_feasible_avoid" in names
    assert "greedy:always_lo" in names


def test_the_default_tolerance_is_one_thousandth():
    # Spec 5.3: p_cut is a 1/65535-quantum step function; float-identical
    # equality forced nested solves for no evaluative gain.
    from storm_reoptimizer.eval.derived import DERIVED_TOLERANCE
    assert DERIVED_TOLERANCE == 1e-3
    assert split_points([0.2480, 0.2485, 0.9]) == [pytest.approx((0.2480 + 0.9) / 2)]


def test_the_check_rejects_the_discarded_drafts(tmp_path):
    """T1's gold was a pure function of "is my service in the cone?", T2's of
    cone_motion, T3's of n_future_claimants. Each is one observable, so a
    one-line rule reproduces it -- the check must say so."""
    from collections import defaultdict
    episodes = list(load_all_scenarios(DISCARDED).values())

    # Verify that each pair is solvable by a single-variable rule
    groups: dict[str, list] = defaultdict(list)
    for e in episodes:
        key = e.pair or e.id
        groups[key].append(e)

    for pair_key, pair_episodes in groups.items():
        if len(pair_episodes) > 1:
            rule, score = best_rule(pair_episodes)
            assert score == pytest.approx(1.0), (
                f"pair {pair_key} should be solvable by one rule; "
                f"best was {rule.name} at {score}")

    # The check should reject because each pair is solvable by a single variable
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


def test_the_shipped_suite_is_not_solved_by_any_rule_over_DERIVED_geometry(
    loaded_state_path, local_server_command, local_server_env,
):
    """The same check, over the numbers the episodes' `forecast` blocks
    actually imply rather than the ones their authors typed into `metadata`
    (whole-branch review 2026-08-23, finding C2).

    The test above it is the one that was green through four reviews of T1
    while `p_cut(storm-svc-1)` at T1's exposure horizon sat at 0.1349 in one
    half and 0.8834 in the other -- a bare threshold on a single number,
    no comparison to the competing claimant and no reasoning of any kind,
    answering the pair 2/2. It could not see that, because `OBSERVABLE_VARS`
    only ever read author-declared metadata and nobody had declared it. An
    episode author cannot be trusted to declare the variable that breaks
    their own episode.

    MCP-backed because the derived value needs the service under test's REAL
    working-path coordinates (`runner.service_points`), which only the server
    knows. One connection for the whole suite: every shipped episode names the
    same `state_file`, so the point is the same number seven times, and
    `derived_scalars_for` asserts that rather than assuming it.

    ALL SEVEN episodes are checked here, T1 included. T1 was temporarily
    excluded (2026-09-05 plan, Task 11) while its pair was mid-redesign:
    Tasks 3/4 of that plan moved `sut_p_cut_at_exposure_horizon` to a joint
    (working-AND-protection) probability for protected services, which broke
    the equality T1a/T1b had been tuned for against the OLD working-leg-only
    metric and left a bare threshold on that one scalar solving the pair 2/2.
    Task 15 then rebuilt T1 from scratch on a jalgaon-homed SUT whose working
    AND protection legs both leave the depot on aerial spans inside the cone,
    and the two halves tie again under the joint metric for real -- both read
    `sut_p_cut_at_exposure_horizon == 0.11718928816662852` (see
    `tests/eval/fixtures/frozen_derived_scalars.json`), so no threshold on it
    can separate them. The exclusion was therefore removed
    (whole-branch review 2026-09-05, Important finding 2), and this test
    passing with T1 in the list is itself the confirmation that the redesign
    restored the property."""
    episodes = list(load_all_scenarios().values())
    if len(episodes) < 7:
        pytest.skip("episodes not authored yet (Tasks 14-17)")

    async def _derived():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            return await derived_scalars_for(client, episodes,
                                             topology_path=TOPOLOGY_PATH)

    derived = asyncio.run(_derived())
    assert_no_single_variable_rule_solves(episodes, derived)
