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

from collections import defaultdict
from pathlib import Path

from mcp.client import Client

from ..mcp_client import call_tool_json
from .baseline import BASELINE_VARIANTS, ForecastBlindBaseline, ScriptedDecider
from .decisions import ConstraintDecision, ObjectiveDecision, TimingDecision
from .derived import DERIVED_TOLERANCE, FLIP_VARS, DerivedGeometry, derived_geometry
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
    gold_by_hour: dict[str, dict] | None = None,
) -> None:
    """Replay this half with the gold decision for the flip variable and each
    plausible alternative for the two decisions the pair is NOT testing; the
    GRADED ANSWER must not move. `client_factory` is an async context manager
    factory yielding a FRESH server connection per replay -- each replay
    mutates state.

    **What "the outcome" means here, and why it changed (2026-08-23,
    whole-branch review finding I1).** This used to score each replay on
    `set(gold.survived) <= set(services_survived)`, and the whole-branch
    review found that check could not fail on ANY shipped episode: T2/T3
    declare `realized: {}`, which makes it trivially true, and D1/T1b's
    single-corridor realized cut is auto-reconverged by the untouched
    PROTECTION corridor inside `simulate_ip_routing`, so the service under
    test survives no matter what the decider does (documented in D1.yaml's
    own `realized` comment). An assertion that cannot fire is not a check.

    It now scores on `decision_label` -- the one categorical
    `cross_twin_metrics.pair_solved` is actually built from (`label_correct`
    on both halves). Precisely: a replay may return NO label, but it must
    never return a DIFFERENT, WRONG one.

      * No label (`None`) is tolerated. It means the alternative declined to
        act, or its candidate never validated, so nothing committed and the
        episode is simply ungraded. That is a null result, not a flipped one:
        it cannot make the wrong half look right. It is also structural for
        T2/T3, whose gold timing at the decision hour is "act" in BOTH halves
        precisely because an avoid-horizon or lever choice cannot be read off
        an episode that never commits -- so `TimingDecision("wait", ...)` in
        PLAUSIBLE_ALTERNATIVES necessarily yields `None` there, by design,
        not by defect.
      * A wrong non-`None` label FAILS. That is the confound the assertion
        exists to catch: an episode lost for a reason other than the decision
        the pair is testing, which would make a lost episode uninterpretable.

    The old survival subset check is KEPT, deliberately, as a secondary
    guard. It is insufficient alone (that is finding I1) but it is not wrong,
    it costs nothing on top of a replay already performed, and it would still
    catch a gross regression on an episode whose `realized` block does bite.
    Both are reported with the same `PairInvalid`.

    **`gold_decisions` are DEFAULTS at every hour; `gold_by_hour` is the
    per-hour override, and an episode whose gold answer differs BETWEEN hours
    needs it.** Every twin half declares `gold.decision_at_t0` separately from
    `gold.label`, and for five of the six they differ: T1b/T2a/T2b/T3a/T3b all
    say "wait at t0, act at the decision hour". Passing a bare
    `TimingDecision("act", ...)` as the gold timing therefore replays a
    rollout that acts at t0 as well -- which is NOT the gold rollout, and on
    T3a genuinely changed the graded label (the t0 commit reroutes
    storm-svc-1, so by t1 `candidate_4` is an `ip_reroute` rather than the
    `optical_reroute` the pair was authored against, reading label "B" where
    gold says "A"). That was a live defect in this assertion's own test
    fixtures, invisible for as long as the check scored survival instead of
    the label. Script the decision hour through `gold_by_hour` and leave the
    default at the episode's `gold.decision_at_t0`.

    **Where this is provably vacuous, stated so nobody mistakes green for
    evidence.** For T1 the flip variable IS `timing`, and T1's `label_rule`
    is `timing_at_decision_hour` -- the graded label is read straight off the
    timing decision the replay holds at gold. No constraints or objective
    alternative can move it. For T1a it is doubly vacuous: gold timing is
    "wait" at every hour, so the rollout never acts and the
    constraints/objective alternatives are never even consulted (the
    `TimingDecision("act", ...)` alternative does exercise them, but timing
    is T1's flip variable and so is never in `non_flip`). That is a
    structural property of a timing pair, not a hole in this check -- the
    check does real work on T2 (where a constraints/objective alternative
    could commit under a different avoid horizon) and on T3 (where a
    constraints alternative could commit a different LEVER, which is exactly
    what `chosen_lever_at_decision_hour` grades)."""
    for name in non_flip:
        for alternative in PLAUSIBLE_ALTERNATIVES[name]:
            # Gold's own per-hour script, with the alternative substituted for
            # gold at the decision hour for the one decision under test.
            by_hour = {key: dict(value)
                       for key, value in (gold_by_hour or {}).items()}
            by_hour.setdefault(name, {})[scenario.decision_hour] = alternative
            decider = ScriptedDecider(
                f"non-binding:{name}",
                default_timing=gold_decisions.get("timing"),
                default_constraints=gold_decisions.get("constraints"),
                default_objective=gold_decisions.get("objective"),
                timing_by_hour=by_hour.get("timing"),
                constraints_by_hour=by_hour.get("constraints"),
                objective_by_hour=by_hour.get("objective"))
            async with client_factory() as client:
                trace = await run_episode(client, scenario, decider,
                                          topology_path=topology_path)
            metrics = episode_metrics(scenario, trace)

            # PRIMARY: the graded categorical. None means "ungraded", which
            # is tolerated; a different non-None label is the real confound.
            label = metrics["decision_label"]
            if label is not None and label != scenario.gold.label:
                raise PairInvalid(
                    f"{scenario.id}: the {name} decision BINDS -- replacing it "
                    f"with {alternative!r} changed the GRADED LABEL to "
                    f"{label!r} (gold {scenario.gold.label!r}). The pair is "
                    f"confounded: a lost episode would not say which decision "
                    f"lost it.")

            # SECONDARY, and known-weak: see the docstring. NOT exact-set
            # equality -- see scoring.py's survived_matches_gold comment:
            # services_survived unions in the whole loaded network's roster
            # (~500+ services on the real eval state), while gold.survived
            # only ever names the 1-3 services this episode is actually
            # about. The real check is whether every named service survived,
            # not whether the two sets are identical.
            survived = metrics["services_survived"]
            if not set(scenario.gold.survived) <= set(survived):
                raise PairInvalid(
                    f"{scenario.id}: the {name} decision BINDS -- replacing it "
                    f"with {alternative!r} changed survival to {survived!r} "
                    f"(gold {list(scenario.gold.survived)!r}). The pair is "
                    f"confounded: a lost episode would not say which decision "
                    f"lost it.")


async def assert_pair_derived_geometry_is_equal(
    client_a: Client, client_b: Client, a: ScenarioFile, b: ScenarioFile, *,
    topology_path: str | Path,
) -> None:
    """`assert_shared_scalars_equal`, but for the quantities the episode's
    author did NOT declare -- the ones its `forecast` block actually implies.

    Added 2026-08-23 for the whole-branch review's finding C2. Every confound
    check the harness had read only author-typed `metadata`, so nothing ever
    noticed that T1's halves differ by `p_cut(storm-svc-1) = 0.1349` vs
    `0.8834` at the exposure horizon -- a single number that answers the pair
    2/2 with no reasoning at all -- or that their decision-hour issuance's own
    cones move 126.1 km/h apart in one half and 0.0 in the other while both
    halves declare `cone_motion_kmh: 63.0`. An episode author cannot be
    trusted to declare the variable that breaks their own episode; the check
    has to derive it. See derived.py.

    Three derived quantities, each held EQUAL across the halves to
    DERIVED_TOLERANCE, for the same reason `assert_shared_scalars_equal`
    holds the declared ones equal -- a bare scalar difference is a threshold,
    and the flip is supposed to live in a RELATION between two services:

      1. `sut_p_cut_at_exposure_horizon` -- the service under test's real cut
         probability at the horizon `exposure_horizon_hours` names, computed
         from its real working-path coordinates on a live server against the
         real cone. The single most exploitable number in an episode.
      2. `within_issuance_cone_motion_kmh` -- how far the decision-hour
         issuance's own cone centre travels between its own horizons. Not the
         same quantity as the declared `cone_motion_kmh`, which describes the
         t0 -> d REVISION.
      3. the decision-hour issuance's per-horizon `width_km`. The declared
         `cone_width_km` is one number naming one horizon; this checks every
         horizon the issuance actually publishes, which is what catches a
         pair whose halves differ ONLY by a width epsilon.

    Each client must be freshly connected against the half's own state file,
    the same contract `assert_menus_identical` has. ALL mismatches are
    collected and reported together: a half being rebuilt needs the whole
    list, not the first item."""
    derived_a = await derived_geometry(client_a, a, topology_path=topology_path)
    derived_b = await derived_geometry(client_b, b, topology_path=topology_path)
    problems = _derived_mismatches(a, derived_a, b, derived_b)
    if problems:
        raise PairInvalid(
            f"{a.id}/{b.id}: the halves' DERIVED forecast geometry differs, so "
            f"a one-line threshold on a number nobody declared separates them:"
            + "".join(f"\n  - {p}" for p in problems)
            + f"\n(SUT representative points: {a.id} {derived_a.sut_point}, "
              f"{b.id} {derived_b.sut_point}; exposure horizons "
              f"{derived_a.exposure_horizon!r}/{derived_b.exposure_horizon!r}.) "
              f"Rebuild the pair so every derived scalar is EQUAL across its "
              f"halves and the flip lives in the relation between two "
              f"services.")


def _derived_mismatches(a: ScenarioFile, derived_a: DerivedGeometry,
                        b: ScenarioFile, derived_b: DerivedGeometry
                        ) -> list[str]:
    """The human-readable list of derived quantities that are not equal."""
    problems: list[str] = []
    for name, value_a, value_b in (
        ("sut_p_cut_at_exposure_horizon",
         derived_a.sut_p_cut_at_exposure_horizon,
         derived_b.sut_p_cut_at_exposure_horizon),
        ("within_issuance_cone_motion_kmh",
         derived_a.within_issuance_cone_motion_kmh,
         derived_b.within_issuance_cone_motion_kmh),
    ):
        if abs(value_a - value_b) > DERIVED_TOLERANCE:
            problems.append(
                f"{name}: {value_a!r} ({a.id}) vs {value_b!r} ({b.id}), "
                f"|delta| = {abs(value_a - value_b):.6g} > "
                f"{DERIVED_TOLERANCE:g}")
    widths_a, widths_b = derived_a.horizon_widths_km, derived_b.horizon_widths_km
    if set(widths_a) != set(widths_b):
        problems.append(
            f"decision-hour issuance publishes different horizons: "
            f"{sorted(widths_a)} ({a.id}) vs {sorted(widths_b)} ({b.id})")
    for horizon in sorted(set(widths_a) & set(widths_b)):
        if abs(widths_a[horizon] - widths_b[horizon]) > DERIVED_TOLERANCE:
            problems.append(
                f"width_km at horizon {horizon!r}: {widths_a[horizon]!r} "
                f"({a.id}) vs {widths_b[horizon]!r} ({b.id})")
    # The SUT's own exposure at EVERY horizon, not just the declared exposure
    # one (scoped re-review, Important #3): a pair whose flip lives in a
    # horizon other than exposure_horizon_hours could otherwise move the
    # SUT's exposure there unchecked.
    exposure_a = derived_a.sut_exposure_by_horizon
    exposure_b = derived_b.sut_exposure_by_horizon
    for horizon in sorted(set(exposure_a) & set(exposure_b)):
        offset_a, p_cut_a = exposure_a[horizon]
        offset_b, p_cut_b = exposure_b[horizon]
        if abs(p_cut_a - p_cut_b) > DERIVED_TOLERANCE:
            problems.append(
                f"sut_p_cut at horizon {horizon!r}: {p_cut_a!r} ({a.id}) vs "
                f"{p_cut_b!r} ({b.id}), offsets {offset_a:.3f}km/{offset_b:.3f}km, "
                f"|delta| = {abs(p_cut_a - p_cut_b):.6g} > {DERIVED_TOLERANCE:g}")
    return problems


# The common axis W1.2's ONE global orientation is scored against. Declared
# per half in `metadata.gold_spare_action`, because the three pairs draw gold
# labels from three vocabularies (act/wait, wide/narrow, A/B) and a single
# orientation needs a single vocabulary.
SPARE_ACTIONS = ("spend", "conserve")


def assert_no_global_policy_solves_the_suite(
    episodes: list[ScenarioFile],
    flip_values: dict[str, dict[str, float]],
) -> None:
    """No ONE fixed threshold, on ONE claimant-side scalar, under ONE fixed
    orientation, answers every twin half.

    This is the second half of the two-check doctrine -- see rules.py's module
    docstring. `assert_no_single_variable_rule_solves` asks whether any scalar
    the halves are supposed to SHARE differs; it is per-pair, and the flip
    variable must be EXCLUDED from it (with two values per pair, per-pair
    orientation makes any differing variable score 1.0, permanently and
    unfixably). This one asks the opposite question over the whole suite, and
    the flip variable must be ENUMERATED in it.

    **Why the orientation rule differs from `rules.score_rule`.** That
    function grants per-pair orientation deliberately: for a confound check,
    generosity makes a surviving pair strong evidence. It is wrong here. A
    fixed policy an operator could actually deploy has ONE orientation, not
    one per pair, so granting per-pair orientation would let this check pass a
    rule that is a genuine cheat.

    `flip_values` is `{scenario_id: {var: value}}`, the shape
    `derived.FlipScalars.values()` produces. Only episodes belonging to a pair
    are scored -- diagnostic episodes have no twin and no flip."""
    from .rules import split_points       # lazy: see the import note below

    halves = [e for e in episodes if e.pair]
    if not halves:
        return

    actions: dict[str, str] = {}
    for episode in halves:
        action = episode.metadata.get("gold_spare_action")
        if action not in SPARE_ACTIONS:
            raise PairInvalid(
                f"{episode.id}: metadata.gold_spare_action must be one of "
                f"{list(SPARE_ACTIONS)}, got {action!r}. W1.2 needs one "
                f"common axis across pairs whose gold labels use three "
                f"different vocabularies.")
        actions[episode.id] = action

    by_pair: dict[str, set[str]] = defaultdict(set)
    for episode in halves:
        by_pair[episode.pair].add(actions[episode.id])
    for pair, declared in sorted(by_pair.items()):
        if len(declared) < 2:
            raise PairInvalid(
                f"pair {pair}: both halves declare gold_spare_action="
                f"{declared.pop()!r}. A twin pair's halves must land on "
                f"opposite sides of the one spare pair, or there is no flip.")

    for var in FLIP_VARS:
        values = {e.id: flip_values.get(e.id, {}).get(var) for e in halves}
        supplied = [v for v in values.values() if v is not None]
        if not supplied:
            # Nobody supplied this var at all -- not swept, not a partial
            # sweep. `derived.FlipScalars.values()` (the real caller) always
            # populates all of FLIP_VARS together, so this branch only ever
            # fires for a synthetic/unit-test `flip_values` that deliberately
            # exercises one variable at a time.
            continue
        if len(supplied) < len(values):
            missing = sorted(k for k, v in values.items() if v is None)
            raise PairInvalid(
                f"{var}: no value supplied for {missing}; the whole-suite "
                f"check cannot be run on a partial sweep")
        for threshold in split_points(list(values.values())):
            for lo_action in SPARE_ACTIONS:
                hi_action = ("conserve" if lo_action == "spend" else "spend")
                predicted = {
                    eid: (lo_action if value < threshold else hi_action)
                    for eid, value in values.items()}
                if predicted == actions:
                    raise PairInvalid(
                        f"a fixed global policy solves the suite {len(halves)}"
                        f"/{len(halves)}: one global threshold at "
                        f"{threshold:.4g} on {var}, with lo -> {lo_action} and "
                        f"hi -> {hi_action}. Values: "
                        f"{ {k: round(v, 1) for k, v in sorted(values.items())} }. "
                        f"Such a policy never reads the service under test and "
                        f"performs no comparison, so an agent scoring "
                        f"{len(halves)}/{len(halves)} would prove nothing. "
                        f"Retune the geometry until the values INTERLEAVE "
                        f"across the labels.")


def assert_flip_dominates(a: ScenarioFile, b: ScenarioFile,
                          flip_a: "FlipScalars", flip_b: "FlipScalars") -> None:
    """The largest EQUAL-IN-BOTH-HALVES signal about the service under test
    must not outweigh the flip itself.

    Every other confound check in this module asks whether something DIFFERS
    between the halves. None of them asks whether the thing that differs is
    the largest thing on the board -- and an equal-in-both-halves signal that
    dominates the flip makes every reasoner answer identically while passing
    every equality assertion. That is exactly how T1 shipped with a `t2`
    nowcast worth 265.0 G sitting on top of a 86.1 G flip (remediation spec,
    finding F2).

    "Flip magnitude" is the largest inter-half difference over any claimant
    aggregate: the pairs disagree about WHICH horizon carries the competing
    claim (T1's is at the exposure horizon, T2's and T3's are before it), so
    taking the max is what makes one check cover all three."""
    magnitude = max(
        abs(flip_a.values()[var] - flip_b.values()[var]) for var in FLIP_VARS)

    shared = set(flip_a.sut_ecar_by_horizon) & set(flip_b.sut_ecar_by_horizon)
    equal = {h: flip_a.sut_ecar_by_horizon[h] for h in shared
             if abs(flip_a.sut_ecar_by_horizon[h]
                    - flip_b.sut_ecar_by_horizon[h]) <= DERIVED_TOLERANCE}
    if not equal:
        return

    horizon, largest = max(equal.items(), key=lambda item: item[1])
    if largest > magnitude:
        raise PairInvalid(
            f"{a.id}/{b.id}: the service under test's own exposure at horizon "
            f"{horizon!r} is {largest:.1f} G and is EQUAL in both halves, "
            f"while the flip magnitude is only {magnitude:.1f} G -- the "
            f"distractor dominates the flip {largest / max(magnitude, 1e-9):.1f}"
            f"x. Every expected-value reasoner answers the same thing in both "
            f"halves, so the graded label is unreachable, and every equality "
            f"assertion passes anyway. Remove or relocate that horizon, or "
            f"enlarge the flip.")


def _check_no_free_escape(scenario_id: str, menu: dict,
                          current: set[str]) -> None:
    """The pure half of W1.6: given a menu and the service's CURRENT working
    lightpath ids, no zero-pair candidate may move the service.

    **Why the cheap variant.** The strict version needs each candidate's own
    exposure, which means recomputing a representative point from its OMS
    sequence -- reimplementing `runner.service_points` against hypothetical
    paths. The cheap version captures the distinction that matters: a 0-pair
    candidate that does not move you buys nothing, so gold is safe."""
    from .ledger import pairs_needed

    for index, candidate in enumerate(menu.get("candidates") or []):
        if pairs_needed(candidate) != 0:
            continue
        reused = set(candidate.get("reused_lightpaths") or ())
        if not current <= reused:
            raise PairInvalid(
                f"{scenario_id}: gold declines to spend the pair, but "
                f"candidate_{index} ({candidate.get('lever')}) costs ZERO "
                f"pairs and does not reuse the service's current working "
                f"lightpath(s) {sorted(current)} (it reuses "
                f"{sorted(reused)}). A free lever that improves the service's "
                f"position makes 'take the free thing and keep the pair' the "
                f"correct answer -- which the label rule scores as WRONG. "
                f"'Act' and 'spend' are different events; gold argues about "
                f"spending and the label reads acting.")


async def _current_working_lightpaths(client: Client,
                                      service_id: str) -> set[str]:
    """The lightpath ids the service's WORKING path rides today. Derived the
    same way runner.service_points walks it: IP link -> lightpath_id."""
    ip = await call_tool_json(client, "get_topology", {"layer": "ip"})
    services = await call_tool_json(client, "get_services")
    lp_by_link = {link["id"]: link.get("lightpath_id")
                  for link in ip["ip_links"]}
    service = next((s for s in services["services"] if s["id"] == service_id),
                   None)
    if service is None:
        raise PairInvalid(
            f"the server reports no service {service_id!r}; its current "
            f"working path cannot be read")
    return {lp_by_link[link] for link in service["working_path"]
            if lp_by_link.get(link)}


async def assert_wait_gold_has_no_free_escape(client: Client,
                                              scenario: ScenarioFile) -> None:
    """For a half whose gold declines to spend the spare pair, every zero-pair
    candidate at the decision hour must REUSE the service's current working
    lightpath -- i.e. no free candidate actually moves the service.

    Gold's reasoning is always "spending the pair isn't worth it", but ACT and
    SPEND are different events. A free lever that improves the SUT's position
    makes "take the free thing and keep the pair" correct, and the label rule
    scores that as wrong -- which is exactly how the agent beat T1a's label
    while satisfying every other gold criterion (remediation spec, finding
    F2). The client must be freshly connected against this half's own state
    file, the same contract `assert_menus_identical` has."""
    if scenario.metadata.get("gold_spare_action") != "conserve":
        return
    current = await _current_working_lightpaths(
        client, scenario.service_under_test)
    menu = await menu_at_decision_hour(client, scenario)
    _check_no_free_escape(scenario.id, menu, current)


def assert_no_single_variable_rule_solves(
    episodes: list[ScenarioFile],
    derived: dict[str, dict[str, float]] | None = None,
) -> None:
    """No threshold on any observable, and no parameter-free greedy policy,
    solves the suite. Static over gold labels and metadata -- cheap, and it
    fails the build during authoring rather than measuring after the fact.

    `derived` (added 2026-08-23, whole-branch review finding C2) is an
    optional `{scenario_id: {var: value}}` map of scalars computed from the
    episodes' REAL forecast geometry rather than read from their author-typed
    `metadata` -- see derived.py, and `derived.DerivedGeometry.scalars()`,
    which produces exactly this shape. Passing it widens the enumeration by
    `rules.DERIVED_VARS`. It is optional only because those values need a
    live server to read the service under test's real coordinates; a caller
    that CAN supply them always should, because the confound this check
    missed for three rounds of review of T1 was exactly a derived one.

    **Per-pair interpretation (not whole-suite):** This check verifies that no
    individual PAIR is 100% solvable by a single-variable rule. This is the
    correct read because:
    1. The design rationale (eval design spec, line 115) is inherently per-pair:
       "T1 flips on width... T2 on motion... T3 on claimant count" — each pair
       fails for its own distinct single-variable reason.
    2. The brief's own fixture table, when run against whole-suite semantics
       (best_rule over all 6 mixed episodes), maxes out at 0.667 — mathematically
       impossible to reach the original test's 1.0 assertion. Empirical
       measurement: cone_width threshold solves T1 (2/2) but can't separate T2
       or T3 (both hold cone_width=90); cone_motion solves T2; n_future_claimants
       solves T3. No single rule reaches 6/6 = 1.0.
    3. Per-pair logic is defensible and stronger: it catches the design flaw
       (each pair reduces to one variable) that whole-suite logic couldn't prove."""
    from collections import defaultdict
    from .rules import best_rule

    # Group by pair and check if any pair is completely solvable by a single rule
    groups: dict[str, list[ScenarioFile]] = defaultdict(list)
    for episode in episodes:
        pair_key = episode.pair or episode.id
        groups[pair_key].append(episode)

    # Check if any pair is 100% solvable by a single-variable rule
    for pair_key, pair_episodes in groups.items():
        if len(pair_episodes) > 1:  # Only check actual pairs (2+ episodes)
            rule, score = best_rule(pair_episodes, derived)
            if score >= 1.0:
                raise PairInvalid(
                    f"pair {pair_key}: a single-variable rule solves it completely: {rule.name} "
                    f"scores {score:.2f}. The flip is a threshold (or a greedy "
                    f"one-liner), not a comparison -- rebuild the pair so every "
                    f"enumerated scalar is EQUAL across its halves and the flip lives "
                    f"in the relation between two of them.")
