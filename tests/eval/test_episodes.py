"""The shipped episodes must pass every pre-flight assertion before any
rollout is scored (eval design spec, "Twin-pair discipline"). A pair failing
these is cut, not shipped."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from storm_reoptimizer.eval.assertions import (
    assert_each_baseline_variant_ties, assert_gold_choices_differ,
    assert_issuance_prefix_shared, assert_menus_identical,
    assert_non_flip_decisions_non_binding, assert_shared_scalars_equal,
)
from storm_reoptimizer.eval.baseline import ForecastBlindBaseline
from storm_reoptimizer.eval.decisions import (
    ConstraintDecision, ObjectiveDecision, TimingDecision,
)
from storm_reoptimizer.eval.runner import run_episode
from storm_reoptimizer.eval.scenario_file import load_all_scenarios
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
_T1_NON_FLIP_GOLD_DECISIONS = {
    "T1a": TimingDecision("wait", "gold"),
    "T1b": TimingDecision("act", "gold"),
}


@pytest.mark.parametrize("half", ("T1a", "T1b"))
def test_t1_non_flip_decisions_are_non_binding(
    half, loaded_state_path, local_server_command, local_server_env,
):
    scenario = load_all_scenarios()[half]
    gold_decisions = {
        "timing": _T1_NON_FLIP_GOLD_DECISIONS[half],
        "constraints": ConstraintDecision(
            avoid={}, reasoning="gold", protected=False, basis="physical",
            level="link"),
        "objective": ObjectiveDecision("candidate_0", None, "gold"),
    }

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
            gold_decisions=gold_decisions,
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
    gold_decisions = {
        "timing": timing, "constraints": constraints, "objective": objective,
    }

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
            gold_decisions=gold_decisions,
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
    gold_decisions = {
        "timing": timing, "constraints": constraints, "objective": objective,
    }

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
            gold_decisions=gold_decisions,
            non_flip=("timing", "constraints"))

    asyncio.run(_run())


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

    Confirmed against the real server (see task-15-report.md): storm-svc-1's
    STATIC protection lightpath uses satna<->jhansi<->allahabad. Avoiding only
    the near/currently-exposed corridor (satna<->rewa) leaves every one of
    route_service's 9 candidates colliding with that protection leg under
    basis=physical/level=link -- a REAL, first-try rejection, not fabricated.
    Widening with the violation's own shared_assets (which name the specific
    jhansi<->allahabad fiber/amp/oms/roadm ids) finds a genuinely disjoint,
    longer route via jabalpur that validates cleanly on the very next
    iteration."""

    def __init__(self) -> None:
        self._inner = ForecastBlindBaseline("immediate")
        self.name = "t2a-rejection-probe"

    def timing(self, obs):
        return self._inner.timing(obs)

    def constraints(self, obs):
        base = self._inner.constraints(obs)
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
    """storm-svc-1's first-choice candidate under the near-horizon-only avoid
    genuinely fails validate_plan (disjointness_collapse against its own
    static protection leg, which shares the satna<->jhansi<->allahabad
    corridor with every cheap alternative to the exposed satna<->rewa
    corridor) -- a real rejection, not fabricated. Recovery means widening
    avoid with the violation's own shared_assets and re-calling route_service,
    which is what makes recovered_from_rejection a metric that can actually
    fire. See docs/superpowers/rehearsals/T2.md and task-15-report.md for the
    full derivation."""
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
    assert "validation_violations" in kinds, (
        f"T2a produced no validate_plan rejection (saw {sorted(kinds)}); the "
        f"near-horizon avoid must genuinely collide with storm-svc-1's own "
        f"protection leg, or recovered_from_rejection can never fire")
    assert episode_metrics(t2a, trace)["recovered_from_rejection"]
