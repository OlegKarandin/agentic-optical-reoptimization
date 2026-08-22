# src/storm_reoptimizer/eval/assertions.py
"""Twin-pair validity, checked BEFORE any episode runs (eval design spec,
"Twin-pair discipline"). A pair failing any of these is CUT, NOT SHIPPED.

This is the single most valuable rule in the harness: it fires before tokens
are spent, and it prevents publishing a table on which the decision tree
ties.

Two things that are easy to get backwards:

  * The issuance-prefix boundary is STRICTLY BEFORE d, not through it. The
    twins must be the same situation on arrival at the decision hour -- same
    network state, same commitments, same ledger, same forecast history --
    and the issuance read AT d is the first thing that differs. A pair
    sharing the issuance at d gives the agent identical information in both
    halves, and identical information cannot have two opposite correct
    answers: that pair is ill-posed, not hard.
  * REFERENCE_AVOID is a DECLARED per-pair constant (scenario.reference_avoid),
    not "the gold avoid set". In T2 the avoid set IS the flip variable, so the
    latter phrase names two different things. The reference exists solely to
    make the two menus comparable.

The non-flip-decisions check exists because of a concrete failure: the
forecast-blind baseline avoids only the currently-exposed risk group, so if a
timing pair's reroute needed a FORWARD-looking avoid set, that baseline would
be re-cut later and fail that half for a CONSTRAINT reason while failing the
other for a TIMING reason -- 0/2, not 1/2. The exactly-50% arithmetic holds
only when the baseline's constant answer IS one of the two gold answers."""
from __future__ import annotations

from pathlib import Path

from mcp.client import Client

from ..mcp_client import call_tool_json
from .baseline import BASELINE_VARIANTS, ForecastBlindBaseline, ScriptedDecider
from .decisions import ConstraintDecision, ObjectiveDecision, TimingDecision
from .runner import run_episode
from .scenario_file import ScenarioFile
from .scoring import decision_label, episode_metrics

# The scalars a one-line rule could key on. Held EQUAL across the halves, so
# no surface correlation is left to key on.
SHARED_SCALARS = ("cone_width_km", "cone_motion_kmh", "n_future_claimants",
                  "exposure_horizon_hours", "spares_on_hand")

# Replayed for the two decisions a pair is NOT testing; the outcome must not
# move. Keyed by decision name.
PLAUSIBLE_ALTERNATIVES = {
    "timing": [TimingDecision("act", "alternative: act at once"),
               TimingDecision("wait", "alternative: hold")],
    "constraints": [
        ConstraintDecision(avoid={}, reasoning="alternative: unconstrained",
                           protected=False, basis="physical", level="link"),
        ConstraintDecision(avoid={"risk_groups": []},
                           reasoning="alternative: empty risk-group avoid",
                           protected=False, basis="physical", level="link")],
    "objective": [
        ObjectiveDecision("candidate_0", None, "alternative: first candidate"),
        ObjectiveDecision("candidate_1", None, "alternative: second candidate")],
}


class PairInvalid(AssertionError):
    """A twin pair the harness refuses to run."""


def assert_gold_choices_differ(a: ScenarioFile, b: ScenarioFile) -> None:
    if a.gold.label == b.gold.label:
        raise PairInvalid(
            f"{a.id}/{b.id}: same gold label {a.gold.label!r} in both halves; "
            f"a twin pair must have opposite correct answers")


def assert_issuance_prefix_shared(a: ScenarioFile, b: ScenarioFile) -> None:
    """Identical issuance history strictly before the decision hour, and a
    DIFFERENT issuance at it."""
    if a.decision_hour != b.decision_hour:
        raise PairInvalid(
            f"{a.id}/{b.id}: different decision hours "
            f"({a.decision_hour!r} vs {b.decision_hour!r})")
    d = a.hours.index(a.decision_hour)
    for hour in a.hours[:d]:
        if _issuance_payload(a, hour) != _issuance_payload(b, hour):
            raise PairInvalid(
                f"{a.id}/{b.id}: issuances differ at {hour!r}, which is "
                f"strictly before the decision hour {a.decision_hour!r}; the "
                f"halves must be the same situation on arrival")
    if _issuance_payload(a, a.decision_hour) == _issuance_payload(b, b.decision_hour):
        raise PairInvalid(
            f"{a.id}/{b.id}: identical issuance AT the decision hour "
            f"{a.decision_hour!r}; identical information cannot carry two "
            f"opposite correct answers -- the pair is ill-posed, not hard")


def _issuance_payload(scenario: ScenarioFile, hour: str):
    issuance = scenario.forecast.get(hour)
    if issuance is None:
        return None
    return sorted((h, c.width_km, c.center["lat"], c.center["lon"])
                  for h, c in issuance.horizons.items())


def assert_shared_scalars_equal(a: ScenarioFile, b: ScenarioFile) -> None:
    """Hold every scalar the one-variable check enumerates EQUAL across the
    twins, and let the flip live in the relation between them."""
    for name in SHARED_SCALARS:
        if a.metadata.get(name) != b.metadata.get(name):
            raise PairInvalid(
                f"{a.id}/{b.id}: {name} differs ({a.metadata.get(name)!r} vs "
                f"{b.metadata.get(name)!r}); a threshold on it would separate "
                f"the halves, so the flip is a threshold, not a comparison")


async def menu_at_decision_hour(client: Client, scenario: ScenarioFile) -> dict:
    """Replay this half's realized cuts strictly before the decision hour,
    then read the menu under the pair's declared REFERENCE_AVOID."""
    d = scenario.hours.index(scenario.decision_hour)
    for hour in scenario.hours[:d]:
        cuts = scenario.realized.get(hour, ())
        if cuts:
            await call_tool_json(client, "inject_failure",
                                 {"asset_ids": list(cuts)})
    return await call_tool_json(client, "route_service", {
        "service_id": scenario.service_under_test, "protected": False,
        "basis": "physical", "level": "link", "best_effort": False,
        "avoid": scenario.reference_avoid})


async def assert_menus_identical(client_a: Client, client_b: Client,
                                 a: ScenarioFile, b: ScenarioFile) -> None:
    """Same candidates and same cost vectors under REFERENCE_AVOID. This is
    what makes "no function of the candidate menu alone can score above 50%"
    provable rather than argued -- the distinguishing information provably
    lives outside the list. Each client must be freshly connected against the
    half's own state file."""
    menu_a = await menu_at_decision_hour(client_a, a)
    menu_b = await menu_at_decision_hour(client_b, b)
    if menu_a["candidates"] != menu_b["candidates"]:
        raise PairInvalid(
            f"{a.id}/{b.id}: menus differ under REFERENCE_AVOID "
            f"{a.reference_avoid!r}; the halves are not observationally "
            f"identical at the decision hour")


async def assert_each_baseline_variant_ties(
    client_factory_a, client_factory_b, a: ScenarioFile, b: ScenarioFile, *,
    topology_path: str | Path,
) -> None:
    """Per variant, separately: each individual fixed policy must answer both
    halves identically. Not that the two variants agree with each other.
    `client_factory_a`/`client_factory_b` are async context manager factories
    yielding a FRESH server connection per replay -- each run_episode call
    mutates state, so reusing one connection across variants would replay
    the second variant against state already mutated by the first."""
    for variant in BASELINE_VARIANTS:
        labels = []
        for client_factory, scenario in ((client_factory_a, a),
                                         (client_factory_b, b)):
            async with client_factory() as client:
                trace = await run_episode(client, scenario,
                                          ForecastBlindBaseline(variant),
                                          topology_path=topology_path)
            labels.append(decision_label(scenario, trace))
        if labels[0] != labels[1]:
            raise PairInvalid(
                f"{a.id}/{b.id}: baseline variant {variant!r} answered the "
                f"halves differently ({labels[0]!r} vs {labels[1]!r}); its "
                f"input is supposed to be identical across them")


async def assert_non_flip_decisions_non_binding(
    client_factory, scenario: ScenarioFile, *, topology_path: str | Path,
    gold_decisions: dict, non_flip: tuple[str, ...],
) -> None:
    """Replay this half with the gold decision for the flip variable and each
    plausible alternative for the two decisions the pair is NOT testing; the
    outcome must not move. `client_factory` is an async context manager
    factory yielding a FRESH server connection per replay -- each replay
    mutates state."""
    for name in non_flip:
        for alternative in PLAUSIBLE_ALTERNATIVES[name]:
            decider = ScriptedDecider(
                f"non-binding:{name}",
                default_timing=gold_decisions.get("timing"),
                default_constraints=gold_decisions.get("constraints"),
                default_objective=gold_decisions.get("objective"),
                **{f"{name}_by_hour": {scenario.decision_hour: alternative}})
            async with client_factory() as client:
                trace = await run_episode(client, scenario, decider,
                                          topology_path=topology_path)
            survived = episode_metrics(scenario, trace)["services_survived"]
            if sorted(survived) != sorted(scenario.gold.survived):
                raise PairInvalid(
                    f"{scenario.id}: the {name} decision BINDS -- replacing it "
                    f"with {alternative!r} changed the outcome to {survived!r} "
                    f"(gold {list(scenario.gold.survived)!r}). The pair is "
                    f"confounded: a lost episode would not say which decision "
                    f"lost it.")
