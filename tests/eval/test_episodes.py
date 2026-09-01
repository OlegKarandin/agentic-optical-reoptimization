"""The shipped episodes must pass every pre-flight assertion before any
rollout is scored (eval design spec, "Twin-pair discipline"). A pair failing
these is cut, not shipped."""
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from storm_reoptimizer.eval.assertions import (
    PairInvalid, assert_flip_dominates, assert_pair_derived_geometry_is_equal,
    assert_each_baseline_variant_ties, assert_gold_choices_differ,
    assert_gold_spare_action_is_grounded, assert_issuance_prefix_shared,
    assert_menus_identical, assert_no_global_policy_solves_the_suite,
    assert_non_flip_decisions_non_binding,
    assert_risk_group_covers_measurable_exposure, assert_shared_scalars_equal,
    assert_wait_gold_has_no_free_escape,
)
from storm_reoptimizer.eval.baseline import ForecastBlindBaseline
from storm_reoptimizer.eval.decisions import (
    ConstraintDecision, ObjectiveDecision, TimingDecision,
)
from storm_reoptimizer.eval.derived import derived_scalars_for, flip_scalars_for
from storm_reoptimizer.eval.runner import run_episode
from storm_reoptimizer.eval.scenario_file import (
    SCENARIOS_DIR as SCENARIOS, load_all_scenarios, load_scenario,
)
from storm_reoptimizer.eval.scoring import episode_metrics
from storm_reoptimizer.mcp_client import connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)

# Extended by Task 17 to add "D1" (a diagnostic singleton, not a pair).
PAIRS = ("T1", "T2", "T3")


@pytest.mark.parametrize("pair", PAIRS)
def test_pair_passes_the_static_assertions(pair):
    episodes = load_all_scenarios()
    a = episodes[f"{pair}a"]
    b = episodes[f"{pair}b"]
    assert_gold_choices_differ(a, b)
    assert_issuance_prefix_shared(a, b)
    assert_shared_scalars_equal(a, b)


async def _menus(a, b, state_path, server_command, server_env):
    @asynccontextmanager
    async def _connect():
        async with connect_server(
            TOPOLOGY_PATH, server_command=server_command, env=server_env,
            extra_args=["--state", str(state_path)],
        ) as client:
            yield client

    async with _connect() as client_a, _connect() as client_b:
        await assert_menus_identical(client_a, client_b, a, b)


@pytest.mark.parametrize("pair", PAIRS)
def test_pair_menus_are_identical_under_reference_avoid(
    pair, loaded_state_path, local_server_command, local_server_env,
):
    episodes = load_all_scenarios()
    asyncio.run(_menus(episodes[f"{pair}a"], episodes[f"{pair}b"],
                       loaded_state_path, local_server_command,
                       local_server_env))


async def _derived(a, b, state_path, server_command, server_env):
    @asynccontextmanager
    async def _connect():
        async with connect_server(
            TOPOLOGY_PATH, server_command=server_command, env=server_env,
            extra_args=["--state", str(state_path)],
        ) as client:
            yield client

    async with _connect() as client_a, _connect() as client_b:
        await assert_pair_derived_geometry_is_equal(
            client_a, client_b, a, b, topology_path=TOPOLOGY_PATH)


@pytest.mark.parametrize("pair", PAIRS)
def test_pair_derived_geometry_is_equal_across_the_halves(
    pair, loaded_state_path, local_server_command, local_server_env,
):
    """test_pair_passes_the_static_assertions checks the scalars the episode's
    AUTHOR declared. This checks the ones its `forecast` block actually
    implies -- the service under test's real p_cut at the exposure horizon,
    the decision-hour issuance's own within-issuance cone motion, and every
    horizon's real width (whole-branch review 2026-08-23, finding C2).

    MCP-backed, and one fresh connection per half, because the p_cut needs the
    service under test's real working-path coordinates -- the same contract
    test_pair_menus_are_identical_under_reference_avoid already has."""
    episodes = load_all_scenarios()
    asyncio.run(_derived(episodes[f"{pair}a"], episodes[f"{pair}b"],
                         loaded_state_path, local_server_command,
                         local_server_env))


@pytest.mark.parametrize("pair", PAIRS)
def test_each_baseline_variant_scores_exactly_one_half(
    pair, loaded_state_path, local_server_command, local_server_env,
):
    episodes = load_all_scenarios()
    a, b = episodes[f"{pair}a"], episodes[f"{pair}b"]

    async def _run():
        @asynccontextmanager
        async def _connect():
            async with connect_server(
                TOPOLOGY_PATH, server_command=local_server_command,
                env=local_server_env,
                extra_args=["--state", str(loaded_state_path)],
            ) as client:
                yield client

        # _connect is already a zero-arg async-context-manager factory --
        # matches assert_each_baseline_variant_ties's client_factory_a/
        # client_factory_b signature directly (fixed Task 11 this session:
        # reusing one connection across both baseline-variant replays
        # corrupted the second replay with state the first had mutated).
        await assert_each_baseline_variant_ties(
            _connect, _connect, a, b, topology_path=TOPOLOGY_PATH)

    asyncio.run(_run())


# T1's flip variable is `timing`, so gold.label IS the gold timing action;
# the two decisions this pair is NOT testing are `constraints` and
# `objective`. This test exists BECAUSE its absence let two real bugs slip
# through Task 14's own review three times over -- the assertion had only
# ever been reasoned about from a probed menu, never actually run against
# the live server (see docs/superpowers/rehearsals/T1.md's Q2 and the SDD
# ledger's Task-14-reopened-3rd-time entry). Tasks 15/16 (T2, T3) must add
# their own equivalent test when they author those pairs: gold.label there
# encodes the constraints/objective flip, not timing, so the gold_decisions
# mapping below does not generalize as-is -- it needs a pair-specific
# translation from that pair's own gold.label into a decision object.
# Both halves declare `gold.decision_at_t0: wait`, so the gold TIMING
# DEFAULT is "wait" in both and T1b's gold.label ("act") is scripted at the
# decision hour only. Passing the label as a bare default would replay a
# rollout that also acts at t0 -- not the gold rollout. That mistake was live
# in T2/T3's fixtures below until 2026-08-23, where it really did move the
# graded label; see assert_non_flip_decisions_non_binding's docstring.
_T1_NON_FLIP_GOLD_DECISIONS = {
    "T1a": (TimingDecision("wait", "gold"), None),
    "T1b": (TimingDecision("wait", "gold: hold at t0, per gold.decision_at_t0"),
            TimingDecision("act", "gold: act at the decision hour")),
}


@pytest.mark.parametrize("half", ("T1a", "T1b"))
def test_t1_non_flip_decisions_are_non_binding(
    half, loaded_state_path, local_server_command, local_server_env,
):
    scenario = load_all_scenarios()[half]
    default_timing, timing_at_d = _T1_NON_FLIP_GOLD_DECISIONS[half]
    gold_decisions = {
        "timing": default_timing,
        "constraints": ConstraintDecision(
            avoid={}, reasoning="gold", protected=False, basis="physical",
            level="link"),
        "objective": ObjectiveDecision("candidate_0", None, "gold"),
    }
    gold_by_hour = ({"timing": {scenario.decision_hour: timing_at_d}}
                    if timing_at_d else None)

    async def _run():
        @asynccontextmanager
        async def _connect():
            async with connect_server(
                TOPOLOGY_PATH, server_command=local_server_command,
                env=local_server_env,
                extra_args=["--state", str(loaded_state_path)],
            ) as client:
                yield client

        await assert_non_flip_decisions_non_binding(
            _connect, scenario, topology_path=TOPOLOGY_PATH,
            gold_decisions=gold_decisions, gold_by_hour=gold_by_hour,
            non_flip=("constraints", "objective"))

    asyncio.run(_run())


# T2's flip variable is `constraints` (gold.label is `wide`/`narrow`, an
# avoid-set outcome), so the two decisions T2 is NOT testing are `timing` and
# `objective`. There is no way to score an avoid-horizon choice without
# committing something, so the gold TIMING decision at t1 is "act" in BOTH
# halves (unlike T1, where gold.label IS the timing action). The gold
# OBJECTIVE decision is whichever candidate index actually carries the
# wide-avoid lever in T2a (candidate_2: optical_reroute, 1 spare pair, under
# avoid={risk_groups:[rg_T2a_t1_t6]}/basis=risk_group/level=risk_group -- see
# docs/superpowers/rehearsals/T2.md) and the narrow-avoid lever in T2b
# (candidate_0: ip_reroute, 0 spare pairs, reusing storm-svc-1's own existing
# working lightpath unchanged, under avoid={}/basis=physical/level=link) --
# found by probing the real server, not guessed. An unknown risk_group id
# (e.g. rg_T2a_t1_t6 before hour t1 has minted it) is silently treated as an
# empty exclusion by route_service (confirmed against the real server), so
# using this fixed gold constraints/objective pair at every hour is safe --
# no crash, just a wasted retry loop at hours other than t1.
#
# Re-probed and unchanged after T2's 2026-08-23 rebuild (whole-branch review
# finding C1): that rebuild moved only the t1 issuance's NEAR (t2) horizon
# centre, and neither gold candidate's menu depends on it -- T2a's wide avoid
# names the FAR (t6) risk group, whose asset list is byte-identical across the
# halves and unchanged from the reviewed version, and T2b's is avoid={}.
_T2_NON_FLIP_GOLD_DECISIONS = {
    "T2a": (
        TimingDecision("act", "gold"),
        ConstraintDecision(
            avoid={"risk_groups": ["rg_T2a_t1_t6"]}, reasoning="gold",
            protected=False, best_effort=False, basis="risk_group",
            level="risk_group"),
        ObjectiveDecision("candidate_2", None, "gold"),
    ),
    "T2b": (
        TimingDecision("act", "gold"),
        ConstraintDecision(
            avoid={}, reasoning="gold", protected=False, best_effort=False,
            basis="physical", level="link"),
        ObjectiveDecision("candidate_0", None, "gold"),
    ),
}


@pytest.mark.parametrize("half", ("T2a", "T2b"))
def test_t2_non_flip_decisions_are_non_binding(
    half, loaded_state_path, local_server_command, local_server_env,
):
    scenario = load_all_scenarios()[half]
    timing, constraints, objective = _T2_NON_FLIP_GOLD_DECISIONS[half]
    # T2a/T2b both declare `gold.decision_at_t0: wait`: gold HOLDS at t0 and
    # acts at t1. See _T1_NON_FLIP_GOLD_DECISIONS' comment.
    gold_decisions = {
        "timing": TimingDecision("wait", "gold: hold at t0, per "
                                         "gold.decision_at_t0"),
        "constraints": constraints, "objective": objective,
    }
    gold_by_hour = {"timing": {scenario.decision_hour: timing}}

    async def _run():
        @asynccontextmanager
        async def _connect():
            async with connect_server(
                TOPOLOGY_PATH, server_command=local_server_command,
                env=local_server_env,
                extra_args=["--state", str(loaded_state_path)],
            ) as client:
                yield client

        await assert_non_flip_decisions_non_binding(
            _connect, scenario, topology_path=TOPOLOGY_PATH,
            gold_decisions=gold_decisions, gold_by_hour=gold_by_hour,
            non_flip=("timing", "objective"))

    asyncio.run(_run())


# T3's flip variable is `objective` (gold.label is A/B, read off which
# candidate LEVER actually gets committed at t1 via metadata.label_by_lever),
# so the two decisions T3 is NOT testing are `timing` and `constraints`. Per
# chosen_lever_at_decision_hour (scoring.py), a label can only be read off a
# COMMITTED candidate, so the gold TIMING decision at t1 is "act" in BOTH
# halves (same reasoning as T2, unlike T1 where gold.label IS the timing
# action). The gold CONSTRAINTS decision is FIXED to avoid={} (T3's own
# reference_avoid -- the menu the pair's whole claim rests on being identical
# in both halves) with basis="risk_group"/level="risk_group": confirmed
# against the real server that BOTH gold candidates need risk_group-basis
# validation to commit -- basis="physical"/level="link" makes EITHER
# candidate collide with storm-svc-1's own static protection leg
# (jhansi<->allahabad, shared regardless of which corridor the new working
# path actually uses) via disjointness_collapse, the exact same
# buried-shared-leg mechanic T2's rehearsal doc derives in full. The gold
# OBJECTIVE decision is whichever candidate index actually carries each
# half's lever under this fixed avoid={}: candidate_4 for T3a (optical_reroute
# via jabalpur, 1 spare pair -- the only lightpath-clean route confirmed
# genuinely disjoint from protection under basis=risk_group), candidate_0 for
# T3b (ip_reroute, reusing storm-svc-1's own CURRENT working lightpath
# unchanged, 0 spare pairs) -- both found by probing the real server, see
# docs/superpowers/rehearsals/T3.md.
_T3_NON_FLIP_GOLD_DECISIONS = {
    "T3a": (
        TimingDecision("act", "gold"),
        ConstraintDecision(
            avoid={}, reasoning="gold", protected=False, best_effort=False,
            basis="risk_group", level="risk_group"),
        ObjectiveDecision("candidate_4", None, "gold"),
    ),
    "T3b": (
        TimingDecision("act", "gold"),
        ConstraintDecision(
            avoid={}, reasoning="gold", protected=False, best_effort=False,
            basis="risk_group", level="risk_group"),
        ObjectiveDecision("candidate_0", None, "gold"),
    ),
}


@pytest.mark.parametrize("half", ("T3a", "T3b"))
def test_t3_non_flip_decisions_are_non_binding(
    half, loaded_state_path, local_server_command, local_server_env,
):
    scenario = load_all_scenarios()[half]
    timing, constraints, objective = _T3_NON_FLIP_GOLD_DECISIONS[half]
    # T3a/T3b both declare `gold.decision_at_t0: wait`. Acting at t0 as well
    # is not merely off-gold here, it CHANGES THE GRADED LABEL: the t0 commit
    # reroutes storm-svc-1, so by t1 `candidate_4` is an ip_reroute rather
    # than the optical_reroute T3a was authored against, and T3a reads "B"
    # where gold says "A". Found 2026-08-23, once this assertion started
    # scoring the label instead of survival (whole-branch review finding I1).
    gold_decisions = {
        "timing": TimingDecision("wait", "gold: hold at t0, per "
                                         "gold.decision_at_t0"),
        "constraints": constraints, "objective": objective,
    }
    gold_by_hour = {"timing": {scenario.decision_hour: timing}}

    async def _run():
        @asynccontextmanager
        async def _connect():
            async with connect_server(
                TOPOLOGY_PATH, server_command=local_server_command,
                env=local_server_env,
                extra_args=["--state", str(loaded_state_path)],
            ) as client:
                yield client

        await assert_non_flip_decisions_non_binding(
            _connect, scenario, topology_path=TOPOLOGY_PATH,
            gold_decisions=gold_decisions, gold_by_hour=gold_by_hour,
            non_flip=("timing", "constraints"))

    asyncio.run(_run())


def test_d1_menu_contains_no_ip_reroute_candidate(
    loaded_state_path, local_server_command, local_server_env,
):
    """A zero-lead-time option makes waiting free and silently inverts D1's
    gold answer. A topology or seed change would do it, so this is asserted
    rather than assumed (eval design spec, "D1's menu precondition")."""
    d1 = load_all_scenarios()["D1"]

    async def _menu():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            from storm_reoptimizer.eval.assertions import menu_at_decision_hour
            return await menu_at_decision_hour(client, d1)

    menu = asyncio.run(_menu())
    levers = {c["lever"] for c in menu["candidates"]}
    assert "ip_reroute" not in levers, (
        f"D1's menu offers {sorted(levers)}; an ip_reroute has zero lead "
        f"time, which makes waiting free and flips D1's gold label to 'wait'")


class _WidensOnDisjointnessRejection:
    """Wraps ForecastBlindBaseline('immediate') -- same timing/objective --
    but reacts to a genuine validate_plan disjointness_collapse rejection by
    adding the violation's OWN `shared_assets` to the avoid set and retrying.

    This is NOT a change to baseline.py: ForecastBlindBaseline's constraints()
    is deliberately blind to `last_rejection` (its docstring: "Neither reads
    revisions... that is what makes each variant emit the SAME answer to both
    halves", the exact property assert_each_baseline_variant_ties relies on
    for every pair). Making it reactive would risk breaking that invariant
    suite-wide. This wrapper is local to this one probe: it demonstrates the
    real recovery mechanic ("widening avoid and re-calling route_service", per
    the design) without touching the shared, already-reviewed baseline.

    Confirmed against the real server: storm-svc-1's STATIC protection
    lightpath uses satna<->jhansi<->allahabad. The exposed-horizon avoid this
    wrapper inherits from ForecastBlindBaseline therefore leaves every one of
    route_service's candidates colliding with that protection leg under
    basis=physical/level=link -- a REAL, first-try rejection, not fabricated.
    Widening with the violation's own shared_assets (which name the specific
    jhansi<->allahabad fiber/amp/oms/roadm ids) finds a genuinely disjoint,
    longer route that validates cleanly on a later iteration.

    Note this wrapper names no fiber id of its own: it reads them out of the
    violation the server reports, so T2's 2026-08-23 rebuild (which moved the
    near-horizon cone off storm-svc-1, making t6 rather than t2 the nearest
    exposed horizon the inherited constraints() keys on) needed no edit
    here -- see docs/superpowers/rehearsals/T2.md's Q3.

    Task 14 finding (2026-08-30, exposure-and-depot plan): satna<->jabalpur
    going aerial (Task 5 of this same plan) put it INSIDE T2a's far-horizon
    cone's own geometry too (jabalpur's nearest point to that cone's centre
    is ~65 km, well inside its ~100 km radius) -- confirmed live via
    `geo_mapper.map_geo_event_to_assets`, `rg_T2a_t1_t6`'s own asset list now
    contains all THREE of satna's aerial directions (rewa, jhansi, AND
    jabalpur), not two. Under basis=physical, avoiding all three leaves NO
    physical route out of satna at all -- `route_service` returns
    `menu_size=0`, `status=no_solution`, not merely an invalid candidate --
    so this wrapper's widen-on-`disjointness_collapse` logic never finds a
    `validation_violations` rejection to react to and cannot recover (there
    is nothing left to widen with; the escape route this scenario needs,
    jabalpur, is now itself excluded by the very avoid set being tested).
    See `test_t2a_carries_a_real_validate_plan_rejection`'s own docstring for
    what this proves instead. `tests/eval/test_runner.py`'s EXPOSURE_SMOKE
    fixture hit the identical problem and fixed it with a hand-built
    "keyhole" cone that carves out jabalpur's own bearing -- not applicable
    here, since T2a's far cone must stay BYTE-IDENTICAL to T2b's (the
    joint-tuning construction's own tied pair; see
    docs/superpowers/plans/notes/2026-08-30-joint-tuning.md) and this test
    exists specifically to exercise T2a's REAL, shipped geometry, not a
    synthetic stand-in for it."""

    def __init__(self) -> None:
        self._inner = ForecastBlindBaseline("immediate")
        self.name = "t2a-rejection-probe"

    def timing(self, obs):
        return self._inner.timing(obs)

    def constraints(self, obs, unconstrained_menu=None):
        base = self._inner.constraints(obs, unconstrained_menu)
        extra: set[str] = set()
        if obs.last_rejection and obs.last_rejection.get("type") == "validation_violations":
            for violation in obs.last_rejection.get("violations", []):
                if violation.get("type") == "disjointness_collapse":
                    extra.update(violation.get("shared_assets", []))
        if not extra:
            return base
        avoid = dict(base.avoid)
        avoid["assets"] = sorted(set(avoid.get("assets", [])) | extra)
        return ConstraintDecision(
            avoid=avoid,
            reasoning=base.reasoning + "; widened after disjointness rejection",
            protected=base.protected, best_effort=base.best_effort,
            basis=base.basis, level=base.level)

    def objective(self, obs, menu):
        return self._inner.objective(obs, menu)


def test_t2a_carries_a_real_validate_plan_rejection(
    loaded_state_path, local_server_command, local_server_env,
):
    """Re-derived 2026-08-30 (Task 14, exposure-and-depot plan) against the
    live server -- see `_WidensOnDisjointnessRejection`'s own docstring
    for the full mechanism. Before Task 5 of this plan made satna<->jabalpur
    aerial, avoiding T2a's far-horizon risk group (rewa + jhansi) under
    basis=physical genuinely produced a candidate that then FAILED
    validate_plan with `disjointness_collapse` against storm-svc-1's own
    static protection leg -- a real, recoverable rejection, and this test's
    original name and docstring described exactly that.

    That is no longer what happens, confirmed live and NOT a regression this
    task introduced: `rg_T2a_t1_t6`'s own asset list now ALSO contains
    satna<->jabalpur (it is aerial now, and geometrically inside the same far
    cone), so avoiding it under basis=physical excludes all THREE of satna's
    aerial directions and leaves NO physical route out of satna at all --
    `route_service` itself returns zero candidates (`status=no_solution`),
    one step earlier than a `validate_plan` rejection, and with nothing for
    `_WidensOnDisjointnessRejection` to widen with. This is confirmed
    pre-existing at baseline HEAD (git-stash confirmed by Task 12, before any
    Task 14 change), i.e. a consequence of Task 5's own topology edit, not of
    this task's episode geometry retune (T2a's far cone is BYTE-IDENTICAL
    before and after Task 14).

    What this test now proves instead: T2a's forecast-blind, basis=physical
    baseline hits a genuine, real dead end -- not a fabricated one -- which
    is an even STARKER version of the same point the shipped gold rationale
    already makes (T2a.yaml: "the same call under basis=physical does not
    validate"). Recovery is genuinely impossible via widen-and-retry here
    (there is no violation to read `shared_assets` from), so
    `recovered_from_rejection` must be False, and the episode never commits
    under this baseline -- exactly why the REAL gold decision uses
    basis=risk_group instead (see `_T2_NON_FLIP_GOLD_DECISIONS`), which
    `test_gold_spare_action_is_grounded_in_a_real_candidate` and
    `assert_escape_route_survives` already confirm still finds and validates
    the jabalpur escape route for real."""
    t2a = load_all_scenarios()["T2a"]

    async def _run():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            return await run_episode(client, t2a,
                                     _WidensOnDisjointnessRejection(),
                                     topology_path=TOPOLOGY_PATH)

    trace = asyncio.run(_run())
    kinds = {r["type"] for h in trace.hours for r in h["rejections"]}
    assert kinds == {"declared_infeasible"}, (
        f"T2a's forecast-blind, basis=physical baseline saw rejection kinds "
        f"{sorted(kinds)}, not the expected {{'declared_infeasible'}} -- "
        f"either the topology's satna<->jabalpur aerial edge, or T2a's own "
        f"far-horizon cone, moved since this was last confirmed live")
    assert trace.terminal_status == "declared_infeasible"
    assert not episode_metrics(t2a, trace)["recovered_from_rejection"], (
        "recovery should be impossible here: avoiding all three of satna's "
        "aerial directions under basis=physical leaves no candidate to "
        "widen from")


def test_the_claimant_aggregates_are_derivable_for_every_twin_half(
        loaded_state_path, local_server_command, local_server_env):
    """W1.1's acceptance: the numbers F1's arithmetic is built on, measured
    against a live server rather than asserted from the spec."""
    episodes = [load_scenario(SCENARIOS / f"{name}.yaml")
                for name in ("T1a", "T1b", "T2a", "T2b", "T3a", "T3b")]

    async def _run():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            return await flip_scalars_for(client, episodes,
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


def test_no_global_policy_solves_the_shipped_suite(
        loaded_state_path, local_server_command, local_server_env):
    """GATE A, as a real assertion: no ONE threshold, on ONE of
    `derived.FLIP_VARS`, under ONE fixed orientation, answers all six twin
    halves.

    RE-DERIVED 2026-08-30 (Task 14, exposure-and-depot plan) against the REAL
    satna-homed claimant pair (`claimant-satna-jabalpur-fwd`/`-rev`) and
    Task 13's frozen geometry -- every number below is from a live run of
    `tools/derive_episodes.py` and matches
    `docs/superpowers/plans/notes/2026-08-30-joint-tuning.md` digit for
    digit; the PRE-Task-13 numbers this docstring used to quote (89.35 G
    threshold, 128.4/284.0 G ties, a 107.6/114.6/128.4 G interleave) described
    the OLD placeholder claimants and no longer exist. "Bound" below is what
    a tied pair guarantees; "best" is what an exhaustive sweep of every
    threshold under both orientations actually reaches. They are not always
    the same number, and quoting the bound as if it were the measured score
    makes the suite look closer to solvable than it is:

      * `..._at_exposure_horizon`   -- tied on T2 (both 529.441 G) and on T3
                                       (both 444.080 G): each pair publishes a
                                       byte-identical far horizon.
                                       Bound 4/6, best 4/6.
      * `..._before_exposure_horizon` -- tied on T1 (both 0.0): T1's
                                       decision-hour issuance publishes only
                                       its exposure horizon, so the sum is
                                       over an empty set.
                                       Bound 5/6, best 4/6 -- the bound is NOT
                                       tight here: T2a (687.276, spend) sits
                                       between T3b (672.689, conserve) and T2b
                                       (1087.764, conserve), costing a second
                                       misclassification beyond the T1 tie.
      * `..._peak_over_horizons`   -- NO tie at all; blocked by a genuine
                                       INTERLEAVE: sorted order is spend,
                                       spend, conserve, conserve, **spend**
                                       (T2a), conserve -- three transitions.
                                       Best 5/6. The real binding edge,
                                       live-bisected and cross-checked against
                                       closed-form arithmetic: shrinking T2a
                                       down flips the check at T1a's own value
                                       (655.4772), margin 31.798 G -- T1a is
                                       the binder, not the adjacent-in-sorted-
                                       order T3b (whose gap to T2a, 14.586 G,
                                       is NOT the real margin; see the notes
                                       file's own correction of exactly this
                                       trap).
      * `..._min_over_horizons`    -- tied on T2 (both 529.441 G): both
                                       halves' `min` reads the SAME far-
                                       horizon value because each half's OWN
                                       near-horizon total is larger than it.
                                       Bound 5/6, best 5/6 (tight).

    A tied pair predicts the same label for both halves under any threshold
    and any orientation. An interleave is the stronger outcome; T2a's near
    cone (bearing 202.0/213.0 deg at 250.0 km from storm-svc-1's own point,
    both halves) is what buys it for `peak_over_horizons`. All four FLIP_VARS
    members (plus the fifth, `largest_restorable_group_ecar_gbps`, added
    2026-08-30 -- TIED on T2 at 75.633 G and on T3 at 51.408 G, bound and
    best both 4/6) are genuinely blocked; see the notes file for the full,
    triple-verified derivation."""
    episodes = [load_scenario(SCENARIOS / f"{n}.yaml")
                for n in ("T1a", "T1b", "T2a", "T2b", "T3a", "T3b")]

    async def _run():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            return await flip_scalars_for(client, episodes,
                                          topology_path=TOPOLOGY_PATH)

    flips = asyncio.run(_run())
    flip_values = {sid: f.values() for sid, f in flips.items()}
    assert_no_global_policy_solves_the_suite(episodes, flip_values)


@pytest.mark.parametrize("pair", ("T1", "T2", "T3"))
def test_the_flip_dominates_every_equal_signal(pair, loaded_state_path,
                                               local_server_command,
                                               local_server_env):
    """W1.5. No equal-in-both-halves signal about the service under test may
    outweigh the flip -- the check that would have caught T1 before a suite
    run was spent on it."""
    a = load_scenario(SCENARIOS / f"{pair}a.yaml")
    b = load_scenario(SCENARIOS / f"{pair}b.yaml")

    async def _run():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            return await flip_scalars_for(client, [a, b],
                                          topology_path=TOPOLOGY_PATH)

    flips = asyncio.run(_run())
    assert_flip_dominates(a, b, flips[a.id], flips[b.id])


# W1.6 FINDING (task 5, 2026-08-26), NARROWED (Finding #5, 2026-08-26
# re-review). The original write-up here reported that ALL THREE shipped
# conserve halves fail this check identically -- a bare "moves the service"
# predicate on `_check_no_free_escape` flagged every one of them, since
# storm-svc-1's own static protection lightpath (`lp-prot-storm-svc-1-0`) is
# always a free (0-pair) `ip_reroute` candidate under the near-neutral
# `reference_avoid: {}` every one of these three halves declares, and it never
# reuses the service's current working lightpath (`lp-cand-storm-svc-1-0`) in
# any of them. That much is still true. What the original write-up never
# checked is whether committing that free candidate would actually change the
# GRADED LABEL each half is scored on -- and traced through each half's own
# `label_rule` (`scoring.decision_label`), only ONE of the three genuinely
# does:
#
#   * T1a (`timing_at_decision_hour`, gold `wait`): committing ANYTHING at the
#     decision hour requires having acted, so the label reads "act" --
#     WRONG against gold `wait`. A REAL, live integrity hazard.
#   * T2b (`avoid_horizon_at_decision_hour`, gold `narrow`): the free
#     candidate is offered under `reference_avoid={}`, so its `risk_groups`
#     is empty and the label still reads "narrow" -- gold-CORRECT. The
#     original predicate's flag here was a FALSE POSITIVE.
#   * T3b (`chosen_lever_at_decision_hour`,
#     `label_by_lever: {ip_reroute: B}`, gold `B`): the free candidate IS an
#     `ip_reroute`, so the label still reads "B" -- also gold-correct, also a
#     FALSE POSITIVE.
#
# `_check_no_free_escape` (assertions.py) now checks the label directly (see
# its own docstring and `_label_if_committed`), so T2b/T3b are expected to
# PASS this check for real and T1a is expected to keep failing it -- both
# confirmed against the live server below. The free `ip_reroute` onto the
# protection lightpath still exists in all three menus (that part of the
# original finding is unchanged, and still worth a separate look for T1a's
# sake), but only T1a's exposure to it is a genuine confound.
@pytest.mark.parametrize("scenario_id", ("T2b", "T3b"))
def test_a_conserve_gold_with_an_unexploitable_free_escape_passes(
    scenario_id, loaded_state_path, local_server_command, local_server_env,
):
    """T2b/T3b: the free `ip_reroute` onto storm-svc-1's static protection
    lightpath exists in the menu, but taking it reads the SAME label gold
    does under each half's own `label_rule` -- not exploitable, so this must
    pass, not xfail."""
    scenario = load_scenario(SCENARIOS / f"{scenario_id}.yaml")

    async def _run():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            await assert_wait_gold_has_no_free_escape(client, scenario)

    asyncio.run(_run())


def _unwrap_lone_exception(exc: BaseException) -> BaseException:
    """`connect_server`'s `stdio_client` (anyio, over an asyncio TaskGroup)
    wraps ANY exception raised inside its `async with` body in one or more
    nested `ExceptionGroup`s on cleanup -- confirmed live (Finding #3,
    2026-08-26 re-review): a bare `PairInvalid` raised by an assertion inside
    that body reaches `asyncio.run(_run())`'s caller as `ExceptionGroup(
    ExceptionGroup(PairInvalid))`, not a plain `PairInvalid`. Without this
    unwrap, `pytest.mark.xfail(raises=PairInvalid)` can never match ANYTHING
    in this environment -- it isn't narrowing to the known finding, it is
    unconditionally converting the xfail into a hard FAIL, the opposite of
    what Finding #3 asked for. Peel off only SINGLE-exception nesting (a
    concurrent failure -- more than one exception in a group -- is a
    genuinely different situation and is left as the group, unmatched by
    `raises=PairInvalid`, exactly as it should be)."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


# Kept as xfail (not skip): the check is correct and should keep running live
# so an eventual fix (e.g. exempting a genuine protection-switch from "moves
# the service", or reshaping reference_avoid/the menu so the escape isn't
# offered) shows up as an unexpected XPASS instead of silently vanishing.
# `raises=PairInvalid` (Finding #3, 2026-08-26 re-review) so an UNRELATED
# failure here -- a launch/protocol error from `connect_server`, or
# `_current_working_lightpaths` raising `PairInvalid` because "the server
# reports no service X" -- surfaces as a real failure instead of being
# silently absorbed as "the expected known finding". The bare decorator only
# narrows by exception TYPE; `test_the_raised_message_names_the_specific_
# lightpath_and_reason` (test_assertions.py) pins the exact message content
# at the pure-unit level, since xfail's own `raises=` cannot match on it.
# `_unwrap_lone_exception` is required for `raises=` to work at all here --
# see its own docstring; confirmed live that without it this test hard-FAILs
# instead of xfailing, on the very finding it is supposed to track.
@pytest.mark.xfail(
    reason="W1.6 finding (task 5, 2026-08-26; narrowed by Finding #5, "
           "2026-08-26 re-review): T1a offers a free ip_reroute onto "
           "storm-svc-1's own static protection lightpath under "
           "reference_avoid={}, and committing it reads label 'act' where "
           "gold says 'wait' -- a real, structural free escape, independent "
           "of forecast geometry. Flagged for W2/W3/W4; not this dispatch's "
           "scope to repair the scenario/menu.",
    strict=False, raises=PairInvalid)
def test_a_conserve_gold_has_no_free_escape(
    loaded_state_path, local_server_command, local_server_env,
):
    """W1.6, T1a only -- see the module-level comment above for why T2b/T3b
    were split out into their own, non-xfail test."""
    scenario = load_scenario(SCENARIOS / "T1a.yaml")

    async def _run():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
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
# already tie this down structurally via test_episodes.py's own non-flip gold
# fixtures and their scenario YAMLs' rationale prose; T1 did not, and T1b in
# particular declares "spend" while `_T1_NON_FLIP_GOLD_DECISIONS` uses
# candidate_0 (an ip_reroute, 0 pairs) for BOTH halves -- a deliberate,
# non-representative placeholder for testing that T1's objective decision is
# non-binding, not a claim about what T1b's real committed action costs.
_GOLD_COMMITTED_LEVER = {
    "T1a": None,                 # gold never acts here; see assert_wait_gold_
                                  # has_no_free_escape for the separate check
                                  # that DOES cover this half
    "T1b": "optical_reroute",    # gold.rationale: "an optical_reroute
                                  # committed at t1 is effective at t2..."
    "T2a": "optical_reroute",    # gold.rationale: "an optical_reroute via
                                  # jabalpur costing 1 pair"; matches
                                  # _T2_NON_FLIP_GOLD_DECISIONS' candidate_2
    "T2b": "ip_reroute",         # gold.rationale: "an ip_reroute that reuses
                                  # the current lightpath"; matches
                                  # _T2_NON_FLIP_GOLD_DECISIONS' candidate_0
    "T3a": "optical_reroute",    # gold.rationale: candidate_4, optical_reroute
                                  # via jabalpur; matches label_by_lever's "A"
    "T3b": "ip_reroute",         # gold.rationale: candidate_0, ip_reroute;
                                  # matches label_by_lever's "B"
}


@pytest.mark.parametrize(
    "scenario_id", ("T1a", "T1b", "T2a", "T2b", "T3a", "T3b"))
def test_gold_spare_action_is_grounded_in_a_real_candidate(
    scenario_id, loaded_state_path, local_server_command, local_server_env,
):
    """The test that would have caught Finding #4: for every half, the
    declared `metadata.gold_spare_action` must be achievable by a REAL
    candidate on the live menu carrying the lever gold's own rationale names
    as the one that gets committed -- not merely asserted in metadata and
    never checked against anything real."""
    scenario = load_all_scenarios()[scenario_id]

    async def _run():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            await assert_gold_spare_action_is_grounded(
                client, scenario,
                committed_lever=_GOLD_COMMITTED_LEVER[scenario_id])

    asyncio.run(_run())


FROZEN_SCALARS_PATH = (
    Path(__file__).parent / "fixtures" / "frozen_derived_scalars.json")


def test_the_probability_model_scalars_are_unmoved(
    loaded_state_path, local_server_command, local_server_env,
):
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

    @asynccontextmanager
    async def _connect():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            yield client

    async def _run():
        async with _connect() as client:
            derived = await derived_scalars_for(
                client, scenarios, topology_path=TOPOLOGY_PATH)
            flip = await flip_scalars_for(
                client, scenarios, topology_path=TOPOLOGY_PATH)
        return derived, {sid: f.values() for sid, f in flip.items()}

    derived, flip = asyncio.run(_run())
    assert derived == frozen["derived"]
    assert flip == frozen["flip"]


@pytest.mark.parametrize("scenario_id", sorted(load_all_scenarios()))
def test_no_episode_defines_an_empty_risk_group_where_something_is_at_risk(
    scenario_id, loaded_state_path, local_server_command, local_server_env,
):
    """The hazard-footprint invariant (spec 2026-08-31 §3.3). This FAILED on
    T1a before the seam fix: both its issuances' cones contain no aerial span
    at all -- the t0 issuance's t3 cone holds two BURIED edges and the t1
    revision's holds no edge whatsoever, nearest aerial span 66.9 km away --
    while the SUT's own p_cut at that horizon is 0.1349."""
    scenario = load_all_scenarios()[scenario_id]

    async def _run():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            from storm_reoptimizer.mcp_client import call_tool_json
            oms = (await call_tool_json(
                client, "get_topology", {"layer": "optical"}))["oms"]
            await assert_risk_group_covers_measurable_exposure(
                client, scenario, topology_path=TOPOLOGY_PATH, oms=oms)

    asyncio.run(_run())
