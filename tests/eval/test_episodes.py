"""The shipped episodes must pass every pre-flight assertion before any
rollout is scored (eval design spec, "Twin-pair discipline"). A pair failing
these is cut, not shipped."""
import asyncio
import json
from pathlib import Path

import pytest

from storm_reoptimizer.eval import oracle
from storm_reoptimizer.eval.assertions import (
    assert_both_legs_exposed, assert_flip_dominates,
    assert_pair_derived_geometry_is_equal, assert_probe_flips,
    assert_each_baseline_variant_ties, assert_gold_choices_differ,
    assert_gold_matches_outcomes, assert_gold_spare_action_is_grounded,
    assert_issuance_prefix_shared,
    assert_menus_identical, assert_no_global_policy_solves_the_suite,
    assert_non_flip_decisions_non_binding, assert_revision_band_equal_at_t0,
    assert_risk_group_assets_cover_realized, assert_spend_is_real,
    assert_risk_group_covers_measurable_exposure, assert_shared_scalars_equal,
    assert_wait_gold_has_no_free_escape,
)
from storm_reoptimizer.eval.gold import enumerate_outcomes
from storm_reoptimizer.eval.decisions import (
    ConstraintDecision, TimingDecision,
)
from storm_reoptimizer.eval.derived import (
    derived_scalars_for_suite, flip_scalars_for_suite,
)
from storm_reoptimizer.eval.risk_assets import fiber_span_index, risk_group_rows
from storm_reoptimizer.eval.runner import (
    EVENT_TYPE, horizon_risk_group_asset_ids, service_geometry,
)
from storm_reoptimizer.eval.scenario_file import (
    SCENARIOS_DIR as SCENARIOS, load_all_scenarios, load_scenario,
)
from storm_reoptimizer.events.filters import get_filter
from storm_reoptimizer.geo_mapper import load_edges
from storm_reoptimizer.mcp_client import call_tool_json

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)

# Extended by Task 17 to add "D1" (a diagnostic singleton, not a pair).
#
# NARROWED (2026-09-05 plan, Task 11): T2/T3's live invariants below were
# built and frozen against the OLD working-leg-only exposure numbers. Tasks
# 3/4 of this same plan changed the model to a joint (working-AND-protection)
# probability for protected services, so several of T2/T3's checks now fail
# on a model change, not a real regression -- ruling: "don't worry about old
# tests or T2/T3, we'll update those later". PAIRS keeps running the pairs
# whose invariants are current; STALE_PAIRS still runs (so the baselines
# don't silently stop executing) but every test that reads it is skipped via
# the `stale_pair` marker (see conftest.py's `pytest_collection_modifyitems`)
# with a reason pointing back here, pending T2/T3's own redesign turn.
#
# T2 REBUILT (2026-09-06 plan, Task 10) on the T1 spend-or-hold machinery
# (spec docs/superpowers/specs/2026-09-06-t2-t3-probe-redesign-design.md
# §4.1): its live invariants are current again, so it moves out of
# STALE_PAIRS and into PAIRS.
#
# T3 REBUILT (2026-09-06 plan, Task 11) on the same machinery, from §4.2's
# documented FALLBACK (the two-corridor restorability variant -- Task 8
# checked the primary design's zero-spare groom live and it is not offered).
# STALE_PAIRS is now EMPTY: every shipped pair's live invariants are current.
PAIRS = ("T1", "T2", "T3")
STALE_PAIRS = ()


def _pair_params(*, live=PAIRS, stale=STALE_PAIRS):
    """Every pair id in `live` unmarked, followed by every pair id in `stale`
    marked `stale_pair` -- the shape `pytest.mark.parametrize` wants for a
    pair-parametrised test that must still enumerate the stale pairs (so
    they show up as skipped, not silently absent from the report)."""
    return [pytest.param(p) for p in live] + [
        pytest.param(p, marks=pytest.mark.stale_pair) for p in stale]


@pytest.mark.parametrize("pair", _pair_params())
def test_pair_passes_the_static_assertions(pair):
    episodes = load_all_scenarios()
    a = episodes[f"{pair}a"]
    b = episodes[f"{pair}b"]
    assert_gold_choices_differ(a, b)
    assert_issuance_prefix_shared(a, b)
    assert_shared_scalars_equal(a, b)


async def _menus(a, b, connect_for):
    async with connect_for(a.state_file)() as client_a, \
            connect_for(b.state_file)() as client_b:
        await assert_menus_identical(client_a, client_b, a, b)


@pytest.mark.parametrize("pair", _pair_params())
def test_pair_menus_are_identical_under_reference_avoid(pair, connect_for):
    episodes = load_all_scenarios()
    asyncio.run(_menus(episodes[f"{pair}a"], episodes[f"{pair}b"], connect_for))


async def _derived(a, b, connect_for):
    async with connect_for(a.state_file)() as client_a, \
            connect_for(b.state_file)() as client_b:
        await assert_pair_derived_geometry_is_equal(
            client_a, client_b, a, b, topology_path=TOPOLOGY_PATH)


@pytest.mark.parametrize("pair", _pair_params())
def test_pair_derived_geometry_is_equal_across_the_halves(pair, connect_for):
    """test_pair_passes_the_static_assertions checks the scalars the episode's
    AUTHOR declared. This checks the ones its `forecast` block actually
    implies -- the service under test's real p_cut at the exposure horizon,
    the decision-hour issuance's own within-issuance cone motion, and every
    horizon's real width (whole-branch review 2026-08-23, finding C2).

    MCP-backed, and one fresh connection per half, because the p_cut needs the
    service under test's real working-path coordinates -- the same contract
    test_pair_menus_are_identical_under_reference_avoid already has."""
    episodes = load_all_scenarios()
    asyncio.run(_derived(episodes[f"{pair}a"], episodes[f"{pair}b"], connect_for))


@pytest.mark.parametrize("pair", _pair_params())
def test_the_revision_band_is_equal_across_the_halves_at_t0(pair, connect_for):
    """Spec 8.2. The one new derived quantity, checked for leakage the same
    way the geometry is."""
    episodes = load_all_scenarios()
    a, b = episodes[f"{pair}a"], episodes[f"{pair}b"]

    async def _run():
        async with connect_for(a.state_file)() as client_a, \
                connect_for(b.state_file)() as client_b:
            await assert_revision_band_equal_at_t0(
                client_a, client_b, a, b, topology_path=TOPOLOGY_PATH)

    asyncio.run(_run())


@pytest.mark.parametrize("pair", _pair_params())
def test_each_baseline_variant_scores_exactly_one_half(pair, connect_for):
    episodes = load_all_scenarios()
    a, b = episodes[f"{pair}a"], episodes[f"{pair}b"]

    async def _run():
        # connect_for(state_file) is already a zero-arg async-context-manager
        # factory -- matches assert_each_baseline_variant_ties's
        # client_factory_a/client_factory_b signature directly (fixed Task 11
        # this session: reusing one connection across both baseline-variant
        # replays corrupted the second replay with state the first had
        # mutated).
        await assert_each_baseline_variant_ties(
            connect_for(a.state_file), connect_for(b.state_file), a, b,
            topology_path=TOPOLOGY_PATH)

    asyncio.run(_run())


@pytest.mark.parametrize("pair", _pair_params())
def test_pair_probe_flips(pair, connect_for):
    """Spec 5.3: the pair's flip, checked by the code path that answers the
    agent. T1 declares no probe_flip, so its claimants must probe
    identically in both halves; T2 flips on restorability."""
    episodes = load_all_scenarios()
    a, b = episodes[f"{pair}a"], episodes[f"{pair}b"]

    async def _run():
        async with connect_for(a.state_file)() as client_a, \
                connect_for(b.state_file)() as client_b:
            await assert_probe_flips(client_a, client_b, a, b,
                                     topology_path=TOPOLOGY_PATH)

    asyncio.run(_run())


# T1/T2's flip variable (2026-09-05 redesign, extended to T2 by the 2026-09-06
# probe redesign) is the SPEND/HOLD decision, which the harness expresses as
# the timing decision plus a real, spare-consuming objective pick --
# `metadata.label_rule: spare_action_by_deadline`, graded by
# `scoring._spare_action_by_deadline_label`. The one decision this pair is NOT
# testing is therefore `constraints`: which avoid set the escape is searched
# under is free, as long as a genuine escape is still found and spent.
#
# The gold replay is built from `oracle`'s OWN pieces rather than a
# hand-written candidate index, because those pieces ARE the gold: Task 10's
# enumerator computed each half's `gold.label` by running exactly
# `oracle.spend_decider`/`oracle.hold_decider` (see `gold.enumerate_outcomes`),
# so a hand-copied approximation here could disagree with the very rollout the
# label came from. Both halves declare `gold.decision_at_t0: wait`, so the
# gold TIMING DEFAULT is "wait" in both and the spend half's "act" is scripted
# at the decision hour only -- passing it as a bare default would replay a
# rollout that also acts at t0, which is NOT the gold rollout (that mistake
# was live in T2/T3's fixtures below until 2026-08-23, where it really did
# move the graded label).
#
# RENAMED from `_t1_gold_replay` (2026-09-06 plan, Task 10): the body reads
# everything off the scenario object, not off anything T1-specific, so T2's
# halves reuse it unchanged once they share the same `label_rule`.
def _spend_or_hold_gold_replay(scenario):
    """`(gold_decisions, gold_by_hour, objective_fn)` reproducing whichever
    of `oracle.spend_decider`/`oracle.hold_decider` this half's own
    `gold.label` names."""
    d = scenario.decision_hour
    sut = scenario.service_under_test
    claimants = tuple(scenario.metadata.get("claimant_services", ()))
    if scenario.gold.label == "spend":
        horizon, rg_id = oracle.spend_risk_group(scenario)
        priority = (sut, *claimants)
        return (
            {"timing": TimingDecision(
                "wait", "gold: wait outside the decision hour",
                claim_priority=priority),
             "constraints": ConstraintDecision(
                 avoid={"risk_groups": [rg_id]},
                 reasoning=f"gold: oracle.spend_decider's own avoid of "
                          f"{rg_id!r} ({horizon!r})")},
            {"timing": {d: TimingDecision(
                "act", "gold: spend the depot's spare at the decision hour",
                claim_priority=priority)}},
            oracle.escape_objective)
    priority = (*claimants, sut)
    return (
        {"timing": TimingDecision(
            "wait", "gold: hold the depot's spare for the claimants",
            claim_priority=priority),
         # Never consulted -- `run_episode` only reaches the constraints
         # decision after a timing decision of "act", and hold never acts.
         "constraints": ConstraintDecision(
             avoid={}, reasoning="gold: hold never acts")},
        None, None)


@pytest.mark.parametrize("half", ("T1a", "T1b", "T2a", "T2b", "T3a", "T3b"))
def test_half_non_flip_decisions_are_non_binding(half, connect_for):
    scenario = load_all_scenarios()[half]
    gold_decisions, gold_by_hour, objective_fn = _spend_or_hold_gold_replay(scenario)

    async def _run():
        await assert_non_flip_decisions_non_binding(
            connect_for(scenario.state_file), scenario,
            topology_path=TOPOLOGY_PATH,
            gold_decisions=gold_decisions, gold_by_hour=gold_by_hour,
            non_flip=("constraints",), objective_fn=objective_fn)

    asyncio.run(_run())


@pytest.mark.parametrize("half", ("T1a", "T1b", "T2a", "T2b", "T3a", "T3b"))
def test_half_both_legs_exposed(half, connect_for):
    """Invariant 9a (redesign spec 4.7): the SUT's WORKING and PROTECTION
    legs each carry a non-trivial cut probability at the decision-hour
    issuance's own latest horizon, in BOTH halves. This is what makes each
    pair's SUT CLAUDE.md's canonical case rather than a technicality -- a
    protection leg that sits outside every cone would let switchover absorb
    everything and the spend/hold comparison would test nothing."""
    scenario = load_all_scenarios()[half]

    async def _run():
        async with connect_for(scenario.state_file)() as client:
            await assert_both_legs_exposed(client, scenario,
                                           topology_path=TOPOLOGY_PATH)

    asyncio.run(_run())


@pytest.mark.parametrize("half", ("T1a", "T1b", "T2a", "T2b", "T3a", "T3b"))
def test_half_spend_is_real(half, connect_for):
    """Invariant 9b (redesign spec 4.7): the escape `oracle.spend_decider`
    would actually take exists, moves the service, does not ride its own
    protection corridor, costs exactly one transponder pair at the depot,
    and lands the service clear of the cone. Run on BOTH halves, not just
    the spend one: the two halves share a state file and a menu, so "spend"
    has to be a real, available option in the half where gold says to HOLD
    it too -- otherwise the hold half's gold would be correct by absence of
    an alternative rather than by comparison."""
    scenario = load_all_scenarios()[half]

    async def _run():
        async with connect_for(scenario.state_file)() as client:
            await assert_spend_is_real(client, scenario,
                                       topology_path=TOPOLOGY_PATH)

    asyncio.run(_run())


@pytest.mark.parametrize("half", ("T1a", "T1b", "T2a", "T2b", "T3a", "T3b"))
def test_half_gold_matches_oracle_outcomes(half, connect_for):
    """The frozen `gold.label`/`gold.outcome_gbps_h` are re-enumerated LIVE
    (`gold.enumerate_outcomes`, the same function `tools/compute_gold.py`
    wrote them with) and must still argmin to the frozen label by at least
    `gold.min_margin_gbps_h`. This is the check that keeps the authored gold
    a MEASUREMENT rather than a prediction: any harness or state change that
    moves the two rollouts' real Gbps-hours shows up here."""
    scenario = load_all_scenarios()[half]

    async def _run():
        return await enumerate_outcomes(connect_for(scenario.state_file),
                                        scenario, topology_path=TOPOLOGY_PATH)

    outcomes = asyncio.run(_run())
    assert_gold_matches_outcomes(
        scenario, {choice: data["total"] for choice, data in outcomes.items()})
    assert {c: d["total"] for c, d in outcomes.items()} == \
        scenario.gold.outcome_gbps_h


# stale_pair (2026-09-05 plan, Task 11): D1-specific, and D1's live
# invariants are known-stale on the joint-exposure model -- see
# PAIRS/STALE_PAIRS above.
@pytest.mark.stale_pair
def test_d1_menu_contains_no_ip_reroute_candidate(connect_for):
    """A zero-lead-time option makes waiting free and silently inverts D1's
    gold answer. A topology or seed change would do it, so this is asserted
    rather than assumed (eval design spec, "D1's menu precondition")."""
    d1 = load_all_scenarios()["D1"]

    async def _menu():
        async with connect_for(d1.state_file)() as client:
            from storm_reoptimizer.eval.assertions import menu_at_decision_hour
            return await menu_at_decision_hour(client, d1)

    menu = asyncio.run(_menu())
    levers = {c["lever"] for c in menu["candidates"]}
    assert "ip_reroute" not in levers, (
        f"D1's menu offers {sorted(levers)}; an ip_reroute has zero lead "
        f"time, which makes waiting free and flips D1's gold label to 'wait'")


def test_the_claimant_aggregates_are_derivable_for_every_twin_half(connect_for):
    """W1.1's acceptance: the numbers F1's arithmetic is built on, measured
    against a live server rather than asserted from the spec."""
    episodes = [load_scenario(SCENARIOS / f"{name}.yaml")
                for name in ("T1a", "T1b", "T2a", "T2b", "T3a", "T3b")]

    async def _run():
        return await flip_scalars_for_suite(connect_for, episodes,
                                            topology_path=TOPOLOGY_PATH)

    flips = asyncio.run(_run())

    assert set(flips) == {"T1a", "T1b", "T2a", "T2b", "T3a", "T3b"}
    for scenario_id, flip in flips.items():
        assert flip.claimant_ecar_peak_over_horizons > 0.0, scenario_id
        assert flip.sut_ecar_by_horizon, scenario_id
    # Record the measured values so the next task's sweep can be read against
    # real numbers instead of the spec's table.
    print("\n".join(
        f"{sid:<5} at={f.claimant_ecar_at_exposure_horizon:9.1f}  "
        f"before={f.claimant_ecar_before_exposure_horizon:9.1f}  "
        f"peak={f.claimant_ecar_peak_over_horizons:9.1f}  "
        f"sut={f.sut_ecar_by_horizon}"
        for sid, f in sorted(flips.items())))


def test_no_global_policy_solves_the_shipped_suite(connect_for):
    """GATE A, as a real assertion: no ONE threshold, on ONE of
    `derived.FLIP_VARS`, under ONE fixed orientation, answers all six twin
    halves.

    RE-DERIVED 2026-09-07 (T2/T3 probe redesign plan, Task 13) against the
    THREE jalgaon-homed pairs (T1, T2, T3 -- the satna-homed placeholder
    claimants this docstring used to cite are retired; see CLAUDE.md's "The
    T2/T3 pairs live on their own state files" section) and `global_policy_
    report` (`assertions.py`), the reporting function this gate itself now
    calls. Every number below is `tools/sweep_flip_vars.py`'s own printed
    output against the live server -- run it to reproduce digit for digit --
    and matches `docs/superpowers/plans/notes/2026-09-06-t2-t3-authoring.md`.
    "Best achievable" is what an exhaustive sweep of every threshold under
    both orientations actually reaches; `global_policy_report` also names
    WHY each variable falls short (`blocked_by`), which this table repeats
    verbatim rather than re-deriving:

    | Variable | Blocked by | Best achievable (of 6 halves) |
    |---|---|---|
    | `claimant_ecar_at_exposure_horizon` | REVERSAL | 5/6 |
    | `claimant_ecar_before_exposure_horizon` | TIE | 0/6 |
    | `claimant_ecar_peak_over_horizons` | REVERSAL | 5/6 |
    | `claimant_ecar_min_over_horizons` | REVERSAL | 5/6 |
    | `largest_restorable_group_ecar_gbps` | TIE | 4/6 |

    Every pair now publishes exactly ONE horizon (`t3`) per issuance, so
    `..._at_exposure_horizon`, `..._peak_over_horizons` and `..._min_over_
    horizons` are the SAME column, digit for digit: T1b=831.070 (spend),
    T3b=1090.169 (spend), T1a=2456.858 (conserve), T2a=3991.015 (conserve),
    T3a=4094.957 (conserve), T2b=6737.691 (spend). `_reversed_pair_exists`
    is what names this REVERSAL, not a tie or a plain interleave: T1's own
    orientation is "more claimant exposure -> conserve" (spend 831.070 <
    conserve 2456.858) and T3's agrees (spend 1090.169 < conserve 4094.957),
    but T2 is the OPPOSITE way round (conserve 3991.015 < spend 6737.691) --
    exactly spec `2026-09-06-t2-t3-probe-redesign-design.md` §3's design ("T2
    reverses the orientation any function of claimant exposure would need").
    `..._before_exposure_horizon` is a TIE at 0.0 across all six halves (no
    pair publishes a horizon before its own single exposure horizon), so
    `rules.split_points` finds no split point to sweep at all and "best
    achievable" is reported as 0/6, not the 3/6 a naive default-to-one-label
    policy would actually score outside this sweep's own threshold search.

    `largest_restorable_group_ecar_gbps` is the variable T3's construction is
    built to tie ON: T3a and T3b both read 114.763 G (T2a matches it too --
    not coincidence, T3's half A reuses T2's own half-A cone anchor exactly,
    bearing 84.0/radius 68.0 -- but T2b does not, 197.076 G) -- sorted:
    T1b=24.643 (spend), T2a=114.763 (conserve),
    T3a=114.763 (conserve), T3b=114.763 (spend), T1a=148.931 (conserve),
    T2b=197.076 (spend). T3a and T3b tying on their OWN pair is what
    `global_policy_report` reports as `blocked_by == "tie"`, and per the
    design spec's corrected §4.2, T1 and T2 alone already interleave on this
    same variable (T1b < T2a < T1a < T2b, conserve/spend/conserve/spend) --
    T3's tie is a real, additional blocker, not the sole one an earlier
    draft of that spec claimed."""
    episodes = [load_scenario(SCENARIOS / f"{n}.yaml")
                for n in ("T1a", "T1b", "T2a", "T2b", "T3a", "T3b")]

    async def _run():
        return await flip_scalars_for_suite(connect_for, episodes,
                                            topology_path=TOPOLOGY_PATH)

    flips = asyncio.run(_run())
    flip_values = {sid: f.values() for sid, f in flips.items()}
    assert_no_global_policy_solves_the_suite(episodes, flip_values)


@pytest.mark.parametrize("pair", _pair_params())
def test_the_flip_dominates_every_equal_signal(pair, connect_for):
    """W1.5. No equal-in-both-halves signal about the service under test may
    outweigh the flip -- the check that would have caught T1 before a suite
    run was spent on it."""
    a = load_scenario(SCENARIOS / f"{pair}a.yaml")
    b = load_scenario(SCENARIOS / f"{pair}b.yaml")

    async def _run():
        return await flip_scalars_for_suite(connect_for, [a, b],
                                            topology_path=TOPOLOGY_PATH)

    flips = asyncio.run(_run())
    assert_flip_dominates(a, b, flips[a.id], flips[b.id])


def _unwrap_lone_exception(exc: BaseException) -> BaseException:
    """`connect_server`'s `stdio_client` (anyio, over an asyncio TaskGroup)
    wraps ANY exception raised inside its `async with` body in one or more
    nested `ExceptionGroup`s on cleanup -- confirmed live (Finding #3,
    2026-08-26 re-review): a bare `PairInvalid` raised by an assertion inside
    that body reaches `asyncio.run(_run())`'s caller as `ExceptionGroup(
    ExceptionGroup(PairInvalid))`, not a plain `PairInvalid`. It was added so
    the caller's then-`pytest.mark.xfail(raises=PairInvalid)` could match at
    all; that marker is gone (2026-09-05, Task 15 review -- see the comment on
    `test_a_conserve_gold_has_no_free_escape`), and the unwrap is kept because
    it is still what makes a real assertion failure legible in the report
    instead of a two-deep ExceptionGroup traceback. Peel off only
    SINGLE-exception nesting: a concurrent failure -- more than one exception
    in a group -- is a genuinely different situation and is left as the
    group."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


# THE `xfail` MARKER IS GONE (2026-09-05, Task 15 review) -- and the reason it
# is gone matters more than the fact, because this test now looks trivial and
# a future reader must not mistake that for an oversight.
#
# The W1.6 finding this used to track was specific to T1's OLD `label_rule`,
# `timing_at_decision_hour`: under that rule `assertions._label_if_committed`
# returns "act" for ANY commit, so a free (0-pair) ip_reroute onto
# storm-svc-1's own static protection lightpath read "act" against a gold of
# "wait" and really was an exploitable escape.
#
# T1's redesigned label rule is `spare_action_by_deadline`, under which
# `_label_if_committed` returns `"spend" if depot_spares_needed else "hold"`.
# `_check_no_free_escape`'s outer gate only ever reaches candidates whose
# TOTAL `spares_needed` is 0, so `depot_spares_needed` is necessarily 0 too
# and the label is necessarily "hold" -- which IS T1a's gold. The condition
# this check hunts for is therefore structurally IMPOSSIBLE for a
# `spare_action_by_deadline` conserve half, not merely absent from today's
# menu: a candidate that costs nothing cannot read "spend", and "spend" is the
# only label that would differ from gold here.
#
# So the test cannot fail on the finding any more, and keeping `xfail(strict=
# False, raises=PairInvalid)` would be actively harmful: an XPASS every run
# (noise that reads as "the finding is fixed" when it is really "unreachable"),
# and -- worse -- any genuinely UNRELATED `PairInvalid`, e.g.
# `_current_working_lightpaths` raising because the server reports no such
# service, would be silently absorbed as "expected". Running it unmarked keeps
# the live plumbing (menu fetch, current-lightpath resolution) exercised and
# makes any future breakage a hard failure. `_unwrap_lone_exception` below
# stays for the same reason it was added (Finding #3, 2026-08-26): anyio wraps
# whatever is raised inside `stdio_client` in nested ExceptionGroups, and the
# unwrap is what makes a real `PairInvalid` legible in the report.
#
# If a later pair re-introduces a conserve half on a label rule where a
# zero-pair commit CAN change the graded label, this check becomes meaningful
# again with no change to it.
#
# T2a/T3a (2026-09-06) grade the same `spare_action_by_deadline` rule, so the
# same structural argument applies.
@pytest.mark.parametrize("scenario_id", ("T1a", "T2a", "T3a"))
def test_a_conserve_gold_has_no_free_escape(scenario_id, connect_for):
    """W1.6 -- see the module-level comment above for why this one is
    unmarked and currently unfalsifiable."""
    scenario = load_scenario(SCENARIOS / f"{scenario_id}.yaml")

    async def _run():
        async with connect_for(scenario.state_file)() as client:
            await assert_wait_gold_has_no_free_escape(client, scenario)

    try:
        asyncio.run(_run())
    except BaseExceptionGroup as eg:
        raise _unwrap_lone_exception(eg) from None


# Finding #4 (2026-08-26 re-review): `gold_spare_action` was never checked
# against what the harness's own gold-decision replay would actually commit.
# `committed_lever` is the LEVER each half's own gold.rationale text names as
# the one that actually gets committed; `None` for a half whose gold never
# commits anything at all (a pure-wait timing half). See `assert_gold_spare_
# action_is_grounded`'s docstring for the full investigation: T2a/T3a/T2b/T3b
# tie this down structurally via test_episodes.py's own non-flip gold fixtures
# and their scenario YAMLs' rationale prose.
#
# T1 did NOT, at the time of that finding: its non-flip fixture then used a
# fixed `candidate_0` (an ip_reroute costing 0 pairs) for BOTH halves -- a
# deliberate, non-representative placeholder, not a claim about what T1b's
# real committed action costs. That gap is closed since the 2026-09-05
# redesign (Task 15): T1's gold replay is now `_spend_or_hold_gold_replay`
# above (renamed from `_t1_gold_replay` when T2 joined it, Task 10 of the
# 2026-09-06 plan), which builds the spend half's objective from
# `oracle.escape_objective` -- the same RULE `oracle.spend_decider` (and
# therefore the enumerator that computed `gold.label`) is wired to, so it
# necessarily picks a candidate that changes the working path, clears the
# SUT's own protection, and charges a real transponder pair.
# `test_half_spend_is_real` checks that exact pick live.
_GOLD_COMMITTED_LEVER = {
    "T1a": None,                 # gold label "hold": the gold rollout never
                                  # acts on the SUT at all, so it commits no
                                  # lever
    "T1b": "optical_reroute",    # gold label "spend": the only candidate that
                                  # both escapes the cone and charges a real
                                  # transponder pair at the depot is an
                                  # optical_reroute (assert_spend_is_real
                                  # checks exactly that candidate, live)
    "T2a": None,                  # gold label "hold" (2026-09-06 redesign):
                                  # T2's gold is enumerated the same way T1's
                                  # is -- the gold rollout never acts on the
                                  # SUT, so it commits no lever
    "T2b": "optical_reroute",    # gold label "spend": the spend half's only
                                  # real escape is a new lightpath over a
                                  # buried spur, same shape as T1b
    "T3a": None,                  # gold label "hold" (2026-09-06 redesign,
                                  # §4.2's fallback): the gold rollout never
                                  # acts on the SUT, so it commits no lever
    "T3b": "optical_reroute",    # gold label "spend": same shape as T1b/T2b --
                                  # a new lightpath over a buried jalgaon spur
}


@pytest.mark.parametrize(
    "scenario_id", ("T1a", "T1b", "T2a", "T2b", "T3a", "T3b"))
def test_gold_spare_action_is_grounded_in_a_real_candidate(
    scenario_id, connect_for,
):
    """The test that would have caught Finding #4: for every half, the
    declared `metadata.gold_spare_action` must be achievable by a REAL
    candidate on the live menu carrying the lever gold's own rationale names
    as the one that gets committed -- not merely asserted in metadata and
    never checked against anything real."""
    scenario = load_all_scenarios()[scenario_id]

    async def _run():
        async with connect_for(scenario.state_file)() as client:
            await assert_gold_spare_action_is_grounded(
                client, scenario,
                committed_lever=_GOLD_COMMITTED_LEVER[scenario_id])

    asyncio.run(_run())


FROZEN_SCALARS_PATH = (
    Path(__file__).parent / "fixtures" / "frozen_derived_scalars.json")


# RE-FROZEN 2026-09-05 (Task 15), and un-skipped with it. Tasks 3/4 of this
# plan moved `p_cut` to a joint (working-AND-protection) probability for
# protected services, which moved `sut_p_cut_at_exposure_horizon` for T2 and
# T3 as well as for T1 -- so the pre-Task-3 snapshot this test read was stale
# beyond T1's own rebuild, and Task 11 skipped the test rather than re-freeze
# it mid-plan. Task 15 re-froze it against the rebuilt state
# (`tools/freeze_derived_scalars.py --state eval/states/loaded-s17.json`),
# which also picks up T1's brand-new SUT/claimant pins.
#
# Exactly what moved, diffed entry by entry against the pre-Task-15 snapshot:
# `derived` for T1a/T1b/T2a/T2b/T3a/T3b (the joint-p_cut change, plus T1's new
# geometry) and `flip` for T1a/T1b only (T2/T3's claimant aggregates are
# unchanged). D1 moved in NEITHER section -- both its scalars are saturated
# (`sut_p_cut_at_exposure_horizon: 1.0`, claimant aggregates 1100.0/200.0), so
# the model change could not shift them.
#
# RE-FROZEN AGAIN 2026-09-06 (T2/T3 probe redesign plan, Task 10), for the
# same reason and by the same tool (`tools/freeze_derived_scalars.py`, which
# Task 9 generalised to read each scenario's OWN `state_file` now that the
# suite spans several): T2a/T2b were re-authored from scratch on the jalgaon
# machinery against `eval/states/t2-jalgaon-s17.json`, so their scalars are
# about a different SUT, a different state file and a different forecast
# block than the ones this snapshot held.
#
# Exactly what moved, diffed entry by entry against the pre-Task-10 snapshot:
# `derived` and `flip` for T2a and T2b ONLY. D1, T1a, T1b, T3a and T3b are
# bit-identical across the re-freeze -- which is the point: the probability
# model did not move, one pair's geometry did, and the diff proves the
# difference. (T3a/T3b are still the 2026-08-31 episodes at this point; Task
# 11 re-authors them and will have to re-freeze once more, with the same
# entry-by-entry diff.)
#
# RE-FROZEN A THIRD TIME 2026-09-06 (same plan, Task 11), for the same reason
# and by the same tool: T3a/T3b were re-authored on the jalgaon machinery
# against `eval/states/t3-jalgaon-s17.json` (spec §4.2's fallback, two
# claimant corridors -- khandwa and buldhana), so their scalars are about a
# different SUT, a different state file and a different forecast block.
#
# Exactly what moved, diffed entry by entry against the pre-Task-11 snapshot:
# `derived` and `flip` for T3a and T3b ONLY. D1, T1a, T1b, T2a and T2b are
# bit-identical across this re-freeze, in BOTH sections. The moves are
# `sut_p_cut_at_exposure_horizon` 0.18147554741741054 -> 0.1446707866025788
# and `within_issuance_cone_motion_kmh` ~290.0 -> 0.0 in both halves (the new
# T3, like T1 and T2, publishes a SINGLE horizon per issuance, so there is no
# within-issuance motion and no horizon before the exposure one), plus the
# claimant aggregates 444.08/370.81 -> 4094.96 (T3a) and 444.08/672.69 ->
# 1090.17 (T3b), and `largest_restorable_group_ecar_gbps` 51.408 -> 114.763
# in BOTH halves -- that last one is T3's deliberate tie (see
# tools/derive_t3.py), which is what blocks a global threshold on it.
#
# The guarantee below is unchanged and bites again from this snapshot forward.
def test_the_probability_model_scalars_are_unmoved(connect_for):
    """cone.py's outputs are frozen against a snapshot taken before the
    hazard-footprint seam fix (plan 2026-09-01, Task 1).

    This is the load-bearing guarantee that the fix touches ROUTING
    behaviour -- which candidates survive `avoid`, whether the baseline acts
    -- and not the probability model. `derived.FLIP_VARS` feeds
    `assert_no_global_policy_solves_the_suite`, whose interleave margins are
    under 1 G; if any of these moved, Claim 2 would have to be re-argued
    rather than merely re-run.

    Bit-identical, not approximate. The scalars are deterministic (SOBOL_M is
    a contract, the Sobol sequence is unscrambled and built once at import),
    so any tolerance here would only hide a real move.
    """
    frozen = json.loads(FROZEN_SCALARS_PATH.read_text(encoding="utf-8"))
    episodes = load_all_scenarios()
    scenarios = list(episodes.values())

    async def _run():
        derived = await derived_scalars_for_suite(
            connect_for, scenarios, topology_path=TOPOLOGY_PATH)
        flip = await flip_scalars_for_suite(
            connect_for, scenarios, topology_path=TOPOLOGY_PATH)
        return derived, {sid: f.values() for sid, f in flip.items()}

    derived, flip = asyncio.run(_run())
    assert derived == frozen["derived"]
    assert flip == frozen["flip"]


@pytest.mark.parametrize("scenario_id", sorted(load_all_scenarios()))
def test_no_episode_defines_an_empty_risk_group_where_something_is_at_risk(
    scenario_id, connect_for,
):
    """The hazard-footprint invariant (spec 2026-08-31 §3.3). This FAILED on
    T1a before the seam fix: both its issuances' cones contain no aerial span
    at all -- the t0 issuance's t3 cone holds two BURIED edges and the t1
    revision's holds no edge whatsoever, nearest aerial span 66.9 km away --
    while the SUT's own p_cut at that horizon is 0.1349."""
    scenario = load_all_scenarios()[scenario_id]

    async def _run():
        async with connect_for(scenario.state_file)() as client:
            from storm_reoptimizer.mcp_client import call_tool_json
            oms = (await call_tool_json(
                client, "get_topology", {"layer": "optical"}))["oms"]
            await assert_risk_group_covers_measurable_exposure(
                client, scenario, topology_path=TOPOLOGY_PATH, oms=oms)

    asyncio.run(_run())


@pytest.mark.parametrize("scenario_id", sorted(load_all_scenarios()))
def test_the_risk_group_assets_cover_every_realized_cut(scenario_id, connect_for):
    """Spec 8.2."""
    scenario = load_all_scenarios()[scenario_id]

    async def _run():
        async with connect_for(scenario.state_file)() as client:
            await assert_risk_group_assets_cover_realized(
                client, scenario, topology_path=TOPOLOGY_PATH)

    asyncio.run(_run())


def test_d1s_reference_avoid_is_nameable_from_the_constraints_observation(
        connect_for):
    """Spec 10.6's live test. D1's gold avoid is eight named fibres and the
    named group disconnects satna; the whole point of showing the group's
    contents is that this avoid set becomes expressible."""
    d1 = load_all_scenarios()["D1"]

    async def _rows():
        async with connect_for(d1.state_file)() as client:
            geometry = await service_geometry(client, TOPOLOGY_PATH)
            topo = await call_tool_json(client, "get_topology",
                                        {"layer": "optical"})
            cone = d1.forecast["t0"].horizons["t1"]
            return {r["asset_id"] for r in risk_group_rows(
                horizon_risk_group_asset_ids(
                    cone, d1.damage_radius_km,
                    edges=load_edges(TOPOLOGY_PATH), oms=topo["oms"],
                    filter_fn=get_filter(EVENT_TYPE)),
                fiber_span_index(topo["oms"], geometry.coords),
                center_lat=cone.center["lat"], center_lon=cone.center["lon"],
                width_km=cone.width_km, damage_radius_km=d1.damage_radius_km,
                working_edges=set(), protection_edges=set())}

    named = asyncio.run(_rows())
    assert set(d1.reference_avoid["assets"]) <= named
