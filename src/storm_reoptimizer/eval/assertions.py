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

from ..events.filters import get_filter
from ..geo_mapper import load_edges
from ..mcp_client import call_tool_json
from .agent import P_CUT_ENUMERATION_THRESHOLD, project_observation
from .baseline import BASELINE_VARIANTS, ForecastBlindBaseline, ScriptedDecider
from .cone import p_cut_region
from .decisions import (
    ConstraintDecision, ObjectiveDecision, TimingDecision, candidate_index,
)
from .derived import (
    DERIVED_TOLERANCE, FLIP_VARS, DerivedGeometry, FlipScalars,
    derived_geometry,
)
from .observation import build_observation, latest_issuance
from .runner import (
    EVENT_TYPE, horizon_risk_group_asset_ids, menu_for_prompt,
    menu_with_path_facts, run_episode, service_geometry,
)
from .scenario_file import ScenarioFile
from .scoring import decision_label, episode_metrics

# multilayer_optical_network.model.modes.default_modes()'s largest
# transceiver mode's bitrate -- the most a single lightpath can carry in the
# best available mode. A plain constant, not an import of the sibling
# library from src/ (CLAUDE.md's hard seam): confirmed against the real
# server by tools/probe_claimants.py ("max lightpath capacity: 800.0 Gbps"),
# also recorded in docs/superpowers/plans/notes/2026-08-30-claimant-
# family.md.
MAX_LIGHTPATH_CAPACITY_GBPS: float = 800.0

# The scalars a one-line rule could key on. Held EQUAL across the halves, so
# no surface correlation is left to key on.
SHARED_SCALARS = ("cone_width_km", "cone_motion_kmh", "n_future_claimants",
                  "exposure_horizon_hours", "spares_on_hand",
                  "track_revision_km_per_hour_ahead")


def _shared_scalar(scenario: ScenarioFile, name: str):
    """A SHARED_SCALARS entry, resolved against the ScenarioFile's own fields
    first and its `metadata` second. Most of these are author-declared
    metadata; `track_revision_km_per_hour_ahead` is a real top-level field
    the harness reads, so it is checked where it actually lives rather than
    duplicated into metadata where the two copies could drift."""
    return getattr(scenario, name, None) if hasattr(scenario, name) \
        else scenario.metadata.get(name)


# Replayed for the two decisions a pair is NOT testing; the outcome must not
# move. Keyed by decision name.
PLAUSIBLE_ALTERNATIVES = {
    "timing": [TimingDecision("act", "alternative: act at once"),
               TimingDecision("wait", "alternative: hold")],
    "constraints": [
        ConstraintDecision(avoid={}, reasoning="alternative: unconstrained"),
        ConstraintDecision(avoid={"risk_groups": []},
                           reasoning="alternative: empty risk-group avoid")],
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
        value_a, value_b = _shared_scalar(a, name), _shared_scalar(b, name)
        if value_a != value_b:
            raise PairInvalid(
                f"{a.id}/{b.id}: {name} differs ({value_a!r} vs "
                f"{value_b!r}); a threshold on it would separate "
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
    objective_fn=None,
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
    check does real work on T2 and T3 too, wherever a constraints or
    objective alternative could commit a different candidate against the
    label rule each pair's own scenarios declare.

    **`objective_fn`** (optional; None keeps every existing caller's
    behaviour exactly) hands the replay decider a RULE for the objective
    instead of a fixed `candidate_N` -- `baseline.ScriptedDecider` has
    supported this since Task 9 and this only exposes it. A
    `spare_action_by_deadline` pair needs it: that label rule has no `None`
    (ungraded) reading -- `scoring._spare_action_by_deadline_label` returns
    "hold" whenever no qualifying decider debit exists -- so the "no label
    is tolerated" escape hatch above does not apply to it, and a spend half
    replayed under a DIFFERENT avoid set must still find and commit a real
    escape rather than whatever `candidate_0` happens to be under that set.
    `oracle.escape_objective` is exactly that rule."""
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
                objective_by_hour=by_hour.get("objective"),
                objective_fn=objective_fn)
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
            + f"\n(SUT storm-cuttable spans: {a.id} {derived_a.sut_spans}, "
              f"{b.id} {derived_b.sut_spans}; exposure horizons "
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


async def assert_revision_band_equal_at_t0(
    client_a: Client, client_b: Client, a: ScenarioFile, b: ScenarioFile, *,
    topology_path: str | Path,
) -> None:
    """For every service shown at t0, `p_cut_if_track_revised` is identical
    across the two halves.

    The no-leakage guarantee for the one new derived quantity: the halves
    must be the same situation on arrival at the decision hour, and a band
    that differed at t0 would hand the agent the flip an hour before it is
    supposed to be readable. Compared at `DERIVED_TOLERANCE`, the same bar
    `assert_pair_derived_geometry_is_equal` uses -- and the band is already
    rounded to 3 decimals, so a difference this tolerates is one the agent
    cannot see either."""
    bands = []
    for client, scenario in ((client_a, a), (client_b, b)):
        geometry = await service_geometry(client, topology_path)
        services = tuple((await call_tool_json(client, "get_services"))["services"])
        obs = build_observation(
            scenario, scenario.hours[0],
            service_spans=geometry.cuttable_spans, services=services,
            spares_on_hand=scenario.spares_on_hand,
            endpoint_sites=geometry.endpoint_sites,
            depot_site=scenario.depot_site,
            protection_spans=geometry.protection_cuttable_spans)
        bands.append({
            (svc, horizon): entry["p_cut_if_track_revised"]
            for svc, per_horizon in project_observation(obs)["exposure"].items()
            for horizon, entry in per_horizon.items()
            if "p_cut_if_track_revised" in entry})
    if set(bands[0]) != set(bands[1]):
        raise PairInvalid(
            f"{a.id}/{b.id}: different services carry a revision band at t0 "
            f"({sorted(set(bands[0]) ^ set(bands[1]))}); the halves must show "
            f"the same rows before the decision hour")
    for key, band_a in bands[0].items():
        band_b = bands[1][key]
        for field in ("revision_radius_km", "min", "max", "mean"):
            if abs(band_a[field] - band_b[field]) > DERIVED_TOLERANCE:
                raise PairInvalid(
                    f"{a.id}/{b.id}: p_cut_if_track_revised[{field!r}] "
                    f"differs at t0 for {key} ({band_a[field]} vs "
                    f"{band_b[field]}); the one new derived field leaks the "
                    f"flip before the decision hour")


# The common axis W1.2's ONE global orientation is scored against. Declared
# per half in `metadata.gold_spare_action`, because the three pairs draw gold
# labels from three vocabularies (act/wait, wide/narrow, A/B) and a single
# orientation needs a single vocabulary.
SPARE_ACTIONS = ("spend", "conserve")


def _reversed_pair_exists(
    pairs: list[tuple[ScenarioFile, ScenarioFile]],
    values: dict[str, float],
    actions: dict[str, str],
) -> bool:
    """True when two pairs order their spend/conserve halves OPPOSITELY on
    `values` -- one pair's spend half reads lower than its conserve half,
    another pair's spend half reads higher. A pair whose two halves tie
    (within `_distinct_within_tolerance`'s bar) carries no orientation of its
    own and is skipped rather than forced either way; `global_policy_report`
    only calls this once it has already confirmed no pair ties, so in
    practice every pair contributes an orientation here."""
    from .rules import _distinct_within_tolerance

    orientations: set[bool] = set()
    for a, b in pairs:
        spend_id, conserve_id = (
            (a.id, b.id) if actions[a.id] == "spend" else (b.id, a.id))
        spend_value, conserve_value = values[spend_id], values[conserve_id]
        if len(_distinct_within_tolerance([spend_value, conserve_value])) == 1:
            continue
        orientations.add(spend_value < conserve_value)
    return len(orientations) > 1


def global_policy_report(
    episodes: list[ScenarioFile],
    flip_values: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """Per `FLIP_VARS` member: the values per half, the best count any ONE
    threshold under ONE orientation reaches, and why it falls short.

    `blocked_by`: `"tie"` when some pair's two halves carry the same value
    (within `_distinct_within_tolerance`'s tolerance), `"reversal"` when no
    tie exists but the best orientation still misses because a pair's halves
    are ordered the OTHER way round from another pair's, `"interleave"` when
    it misses for any other reason, `"solved"` when `best == n`.

    Extracted from `assert_no_global_policy_solves_the_suite` (which now
    delegates to this and raises on any `"solved"` entry) so the per-variable
    bound/best table `tools/sweep_flip_vars.py` prints comes from the gate's
    own arithmetic, not a hand-transcribed copy of it.

    `flip_values` is `{scenario_id: {var: value}}`, the shape
    `derived.FlipScalars.values()` produces. Only episodes belonging to a
    pair are scored -- diagnostic episodes have no twin and no flip. A
    variable nobody supplied ANY value for is OMITTED from the report
    entirely (see the loop below); this mirrors the equally-silent skip the
    pre-extraction code had, and only ever fires for a synthetic/unit-test
    `flip_values` that deliberately exercises one variable at a time --
    `derived.FlipScalars.values()`, the real caller, always populates every
    `FLIP_VARS` member together."""
    # Lazy, not for circular-import reasons (there are none: rules.py never
    # imports this module, directly or transitively -- only suite.py imports
    # assertions.py). It mirrors the equally-lazy `from .rules import
    # best_rule` in assert_no_single_variable_rule_solves below, kept local
    # so this module's own top-level import surface does not gain rules.py
    # (and, through it, nothing new -- rules.py only adds derived.py, which
    # this module already imports) for callers that never run either check.
    from .rules import _distinct_within_tolerance, split_points

    halves = [e for e in episodes if e.pair]
    if not halves:
        return {}

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

    by_pair: dict[str, list[ScenarioFile]] = defaultdict(list)
    for episode in halves:
        by_pair[episode.pair].append(episode)
    for pair, group in sorted(by_pair.items()):
        declared = {actions[e.id] for e in group}
        if len(declared) < 2:
            raise PairInvalid(
                f"pair {pair}: both halves declare gold_spare_action="
                f"{declared.pop()!r}. A twin pair's halves must land on "
                f"opposite sides of the one spare pair, or there is no flip.")
    pairs = [tuple(group) for group in by_pair.values()]

    report: dict[str, dict] = {}
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
        best, best_threshold, best_orientation = 0, None, None
        for threshold in split_points(list(values.values())):
            for lo_action in SPARE_ACTIONS:
                hi_action = ("conserve" if lo_action == "spend" else "spend")
                predicted = {
                    eid: (lo_action if value < threshold else hi_action)
                    for eid, value in values.items()}
                score = sum(
                    predicted[eid] == actions[eid] for eid in values)
                if score > best:
                    best, best_threshold, best_orientation = (
                        score, threshold, lo_action)
        tied = any(
            len(_distinct_within_tolerance([values[a.id], values[b.id]])) == 1
            for a, b in pairs)
        if best == len(values):
            blocked_by = "solved"
        elif tied:
            blocked_by = "tie"
        elif _reversed_pair_exists(pairs, values, actions):
            blocked_by = "reversal"
        else:
            blocked_by = "interleave"
        report[var] = {
            "values": values, "n": len(values), "best": best,
            "threshold": best_threshold, "lo_action": best_orientation,
            "blocked_by": blocked_by}
    return report


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
    are scored -- diagnostic episodes have no twin and no flip.

    Delegates the actual sweep to `global_policy_report` (which raises the
    same `PairInvalid`s this function used to raise directly for an invalid
    `gold_spare_action` or a partial sweep) and raises iff any variable's
    report entry is `blocked_by == "solved"`."""
    report = global_policy_report(episodes, flip_values)
    for var, entry in report.items():
        if entry["blocked_by"] != "solved":
            continue
        hi_action = "conserve" if entry["lo_action"] == "spend" else "spend"
        raise PairInvalid(
            f"a fixed global policy solves the suite {entry['best']}"
            f"/{entry['n']}: one global threshold at "
            f"{entry['threshold']:.4g} on {var}, with lo -> "
            f"{entry['lo_action']} and hi -> {hi_action}. Values: "
            f"{ {k: round(v, 1) for k, v in sorted(entry['values'].items())} }. "
            f"Such a policy never reads the service under test and "
            f"performs no comparison, so an agent scoring "
            f"{entry['best']}/{entry['n']} would prove nothing. "
            f"Retune the geometry until the values INTERLEAVE "
            f"across the labels.")


async def assert_gold_spare_action_is_grounded(
    client: Client, scenario: ScenarioFile, *, committed_lever: str | None,
) -> None:
    """Cross-checks `metadata.gold_spare_action` against a REAL candidate on
    the live menu, closing a gap Finding #4 (2026-08-26 whole-branch
    re-review) found: nothing anywhere checked the declared "spend"/
    "conserve" declaration against what the harness's own gold-decision
    replay would actually commit.

    **Why this was missing, and what actually grounds it now.**
    `assert_no_global_policy_solves_the_suite` scores every episode against
    `metadata.gold_spare_action`, but that field is author-typed and nothing
    ever verified it against reality. This assertion is the check, called by
    `test_episodes.py::test_gold_spare_action_is_grounded_in_a_real_candidate`
    for all six halves, driven by that module's own `_GOLD_COMMITTED_LEVER`
    dict (NOT the `_T1_NON_FLIP_GOLD_DECISIONS`/`_T2_NON_FLIP_GOLD_DECISIONS`/
    `_T3_NON_FLIP_GOLD_DECISIONS` candidate-index fixtures an earlier revision
    of this docstring described -- Task 12 of the T2/T3 probe redesign plan
    deleted all three along with the retired pre-redesign T2/T3 pair they
    supported). As of the redesign, all three pairs share ONE shape:
    `T1a`/`T2a`/`T3a` (gold `hold`) commit `None` -- the gold rollout never
    acts on the SUT at all in the hold half of any pair -- and `T1b`/`T2b`/
    `T3b` (gold `spend`) each commit `optical_reroute` at ONE transponder
    pair, over a new lightpath on a buried escape spur (T1b's `jhansi`-side
    escape, T2b's and T3b's `jalgaon <-> aurangabad`). Each scenario's own
    YAML `gold.rationale` spells out that lever and cost in prose --
    confirmed by reading every one of the six halves' rationale text. T1's
    entry needed no placeholder to make this true, unlike an earlier revision
    of the pre-redesign fixtures this docstring used to cite: T1's
    `gold.rationale` for the spend half already says "an optical_reroute
    committed at t1 is effective at t2, strictly before the t3 cut", the same
    lever `_GOLD_COMMITTED_LEVER["T1b"]` declares. Investigated rather than
    assumed: searched the whole module and runner/scoring for any OTHER place
    that reads `gold_spare_action` or checks `pairs_needed` against it --
    there is none; `assert_no_global_policy_solves_the_suite` is the only
    consumer, and it trusts the declaration verbatim.

    `committed_lever` is the LEVER gold's own rationale names as the one that
    actually gets committed in this half, supplied by the caller per half (not
    read from scenario metadata -- there is no such field, by design, to keep
    this fix's footprint to assertions.py/tests rather than the scenario
    files). `None` means gold never commits anything in this half at all (a
    pure-wait timing half, e.g. T1a): trivially "conserve", and separately
    guarded already by `assert_wait_gold_has_no_free_escape`.

    Deliberately checks EXISTENCE of a same-lever candidate at the declared
    cost under `reference_avoid`, not the exact candidate index test_
    episodes.py's non-flip fixtures use -- T2a's REAL gold avoid (the wide
    risk-group set) differs from `reference_avoid` (neutral), so the SAME
    candidate index would not necessarily appear under `reference_avoid`.
    That is not a gap here: `pairs_needed` in this harness is a property of
    the LEVER (`ip_reroute` costs 0, `optical_reroute` costs one new
    lightpath, regardless of which avoid produced the candidate -- confirmed
    against `ledger.pairs_needed`'s own contract, "one pair per new
    lightpath"), not of which specific avoid produced it, so the neutral
    `reference_avoid` menu is enough to confirm the declared action is
    achievable at all, without needing the exact gold avoid reproduced here."""
    action = scenario.metadata.get("gold_spare_action")
    if committed_lever is None:
        if action != "conserve":
            raise PairInvalid(
                f"{scenario.id}: gold never commits anything in this half "
                f"(committed_lever=None) but metadata.gold_spare_action="
                f"{action!r}; a half that never acts cannot 'spend'")
        return

    from .ledger import spares_needed

    menu = await menu_at_decision_hour(client, scenario)
    oms_nodes = await _oms_nodes(client)
    candidates = [c for c in menu.get("candidates") or []
                 if c.get("lever") == committed_lever]
    if not candidates:
        raise PairInvalid(
            f"{scenario.id}: no {committed_lever!r} candidate exists in the "
            f"real menu under reference_avoid {scenario.reference_avoid!r}; "
            f"gold_spare_action {action!r} cannot be grounded against reality")

    costs = {sum(spares_needed(c, oms_nodes).values()) for c in candidates}
    if action == "spend" and not any(cost > 0 for cost in costs):
        raise PairInvalid(
            f"{scenario.id}: every real {committed_lever!r} candidate under "
            f"reference_avoid costs 0 pairs ({sorted(costs)}), but metadata."
            f"gold_spare_action='spend' -- the declaration is not grounded "
            f"in a real spend")
    if action == "conserve" and not any(cost == 0 for cost in costs):
        raise PairInvalid(
            f"{scenario.id}: every real {committed_lever!r} candidate under "
            f"reference_avoid costs more than 0 pairs ({sorted(costs)}), but "
            f"metadata.gold_spare_action='conserve' -- the declaration is "
            f"not grounded in a real zero-cost lever")


def assert_flip_dominates(a: ScenarioFile, b: ScenarioFile,
                          flip_a: FlipScalars, flip_b: FlipScalars) -> None:
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


def _label_if_committed(*, label_rule: str, candidate: dict,
                        avoid_used: dict,
                        changes_working_path: bool = True,
                        depot_spares_needed: int = 0) -> str | None:
    """The label `scoring.decision_label` would read off a HYPOTHETICAL commit
    of `candidate` under `avoid_used`, without a trace or a rollout -- the same
    `label_rule` branches that function implements, applied to one candidate
    instead of a replayed hour's `iterations` record.

    `depot_spares_needed` is the `spare_action_by_deadline` branch's own
    input: this function never sees a ledger or an oms_nodes map, so the
    caller (`_check_no_free_escape`, whose only candidates reaching here
    already cost zero pairs total -- see its own docstring) passes the
    depot's own share of that total. Defaulted to 0 so every other
    `label_rule`'s existing callers are unaffected.

    `changes_working_path` is the analogous input for `timing_at_decision_
    hour` (Task 5): this function never sees the service's OWN current
    working lightpaths either (only `_check_no_free_escape` does, as
    `current`), and recomputing runner.py's OMS-level `path_delta` here
    would mean reimplementing `_candidate_oms`/`_path_delta` against a raw
    menu candidate that carries no `path_delta` of its own. Defaulted to
    True so a caller that (like every one shipped today) already filtered
    down to candidates that move the service does not have to pass it.

    Finding #5 (2026-08-26 re-review of W1.6) is why this function exists at
    all; the two retired rules' branches lived here until 2026-09-06.

    `timing_at_decision_hour` no longer reads unconditionally as "act"
    (Task 5): `scoring.timing_at_decision_hour` now reads `timing_effective`,
    which downgrades an INERT commit -- one that neither moves the working
    path nor needs a depot spare, mirroring `runner.py`'s own `step["inert"]`
    check (runner.py:1078-1080) -- to "wait", on the reasoning that an act
    that commits nothing (or commits inertly) is a wait. A candidate reaching
    this function under that rule therefore reads "wait" when it is inert by
    that same test, else "act"."""
    if label_rule == "timing_at_decision_hour":
        inert = not changes_working_path and depot_spares_needed == 0
        return "wait" if inert else "act"
    if label_rule == "spare_action_by_deadline":
        return "spend" if depot_spares_needed else "hold"
    raise ValueError(f"unknown label_rule {label_rule!r}")


def _check_no_free_escape(scenario_id: str, menu: dict, current: set[str], *,
                          label_rule: str, gold_label: str,
                          reference_avoid: dict,
                          oms_nodes: dict | None = None) -> None:
    """The pure half of W1.6: given a menu and the service's CURRENT working
    lightpath ids, no zero-pair candidate that would actually flip the GRADED
    LABEL away from gold's may exist.

    **Why the label check, not just "moves the service" (Finding #5, 2026-
    08-26 re-review).** A zero-pair candidate that does not reuse the current
    working lightpath is necessary but not sufficient evidence of a real
    escape: whether it is exploitable depends on whether committing it would
    actually change what `scoring.decision_label` reads for this half, which
    depends on `label_rule`. Traced against the shipped suite's own
    `gold_spare_action=conserve` halves: T1a (`timing_at_decision_hour`,
    gold `wait`) -- ANY commit reads "act" (see `_label_if_committed`) --
    REAL escape, since gold is "wait".

    See `test_a_conserve_gold_has_no_free_escape` in test_episodes.py.

    **Why the cheap variant, otherwise.** The strict version would need each
    candidate's own exposure, which means recomputing a representative point
    from its OMS sequence -- reimplementing `runner.service_points` against
    hypothetical paths. The cheap version captures the distinction that
    matters: a 0-pair candidate that does not move you buys nothing, so gold
    is safe regardless of label_rule."""
    from .ledger import spares_needed

    oms_nodes = oms_nodes or {}
    for index, candidate in enumerate(menu.get("candidates") or []):
        if sum(spares_needed(candidate, oms_nodes).values()) != 0:
            continue
        reused = set(candidate.get("reused_lightpaths") or ())
        if current <= reused:
            continue
        label = _label_if_committed(
            label_rule=label_rule, candidate=candidate,
            avoid_used=reference_avoid,
            # Always True here: the `current <= reused` gate just above
            # already `continue`d away every candidate that reuses everything
            # currently working, so every candidate reaching this call drops
            # at least one of the service's own current lightpaths -- this
            # file's own cheap proxy for path_delta.changes_working_path.
            changes_working_path=True,
            # Always 0 here: the gate above already requires this
            # candidate's total spares_needed (summed across every site) to
            # be zero, so its depot-specific share can never be nonzero
            # either. Passed explicitly rather than left to the default so a
            # future caller of _label_if_committed cannot assume the default
            # is always the right answer for it too.
            depot_spares_needed=0)
        if label == gold_label:
            continue    # free AND moves the service, but grades the same
        raise PairInvalid(
            f"{scenario_id}: gold declines to spend the pair, but "
            f"candidate_{index} ({candidate.get('lever')}) costs ZERO "
            f"pairs, does not reuse the service's current working "
            f"lightpath(s) {sorted(current)} (it reuses {sorted(reused)}), "
            f"and committing it under label_rule {label_rule!r} would read "
            f"the GRADED LABEL as {label!r} where gold says {gold_label!r}. "
            f"A free lever that improves the service's position makes 'take "
            f"the free thing and keep the pair' the correct answer -- which "
            f"the label rule scores as WRONG. 'Act' and 'spend' are "
            f"different events; gold argues about spending and the label "
            f"reads acting.")


async def _oms_nodes(client: Client) -> dict[str, list[str]]:
    """oms_id -> [src_node_id, dst_node_id], the same static optical adjacency
    `runner.service_geometry` builds -- what `ledger.spares_needed` needs to
    resolve a candidate's new lightpaths to endpoint SITES. A live read
    rather than a cached one: assertions.py runs once per twin pair against a
    freshly connected client, so there is no per-hour loop to amortize this
    against, unlike runner.run_episode's own oms_nodes."""
    optical = await call_tool_json(client, "get_topology", {"layer": "optical"})
    return {o["id"]: [o["src_node_id"], o["dst_node_id"]]
            for o in optical["oms"]}


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
    """For a half whose gold declines to spend the spare pair, no zero-pair
    candidate at the decision hour may both move the service off its current
    working lightpath AND flip the graded label away from gold's -- i.e. no
    free candidate is a real, exploitable escape. See `_check_no_free_escape`
    for the per-`label_rule` reasoning (Finding #5, 2026-08-26 re-review:
    "moves the service" alone over-flagged two of the suite's three conserve
    halves).

    Gold's reasoning is always "spending the pair isn't worth it", but ACT and
    SPEND are different events. A free lever that improves the SUT's position
    makes "take the free thing and keep the pair" correct, and the label rule
    scores that as wrong -- which is exactly how the agent beat T1a's label
    while satisfying every other gold criterion (remediation spec, finding
    F2). The client must be freshly connected against this half's own state
    file, the same contract `assert_menus_identical` has.

    **Ordering (Finding #11, 2026-08-26 re-review).** The menu replay must
    happen BEFORE `current` is read: `menu_at_decision_hour` replays this
    half's realized cuts strictly before the decision hour, and if a future
    scenario ever realizes a cut there, the service's current working
    lightpath could change as a result. Reading `current` first would then
    compare a PRE-replay lightpath id against a POST-replay menu -- silently
    comparing two different network states. Inert for every half shipped
    today (none realize a cut before their own decision hour -- see each
    pair's own `realized` block), but asserted here rather than merely
    ordered-and-hoped-for, so a future violation fails loudly instead of
    silently misreading."""
    if scenario.metadata.get("gold_spare_action") != "conserve":
        return
    d = scenario.hours.index(scenario.decision_hour)
    assert not any(scenario.realized.get(hour) for hour in scenario.hours[:d]), (
        f"{scenario.id}: realizes a cut strictly before its own decision "
        f"hour {scenario.decision_hour!r}; `current` must be read AFTER "
        f"menu_at_decision_hour's replay, not before -- update the ordering "
        f"here instead of relying on this assumption")
    menu = await menu_at_decision_hour(client, scenario)
    current = await _current_working_lightpaths(
        client, scenario.service_under_test)
    oms_nodes = await _oms_nodes(client)
    _check_no_free_escape(
        scenario.id, menu, current,
        label_rule=scenario.metadata.get("label_rule"),
        gold_label=scenario.gold.label,
        reference_avoid=scenario.reference_avoid,
        oms_nodes=oms_nodes)


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


# --------------------------------------------------------------------------
# Task 12 (exposure-and-depot plan, §6 invariants 1-8): the dimensional-
# coherence class. Every defect this plan fixed was a physically-typed
# quantity (a site, a lightpath, a mount type, a span) collapsed into an
# untyped scalar, and every one survived review because the suite checked
# equality and derivation but never asked whether the units made sense.
# These eight invariants check exactly that class, so a future episode
# author cannot reintroduce it. All raise `PairInvalid`, same as every check
# above; wired into `suite.main()`'s pre-flight, after the two existing
# checks.
# --------------------------------------------------------------------------


def claimant_service_ids(scenario: ScenarioFile) -> tuple[str, ...]:
    """The services `metadata.claimant_services` names, cross-checked against
    `gold.rationale` so a declared list cannot silently drift from the prose
    it is supposed to summarize -- every listed id must appear literally in
    the rationale text. Turns "did the author's arithmetic name a real,
    eligible, exposed service" into a checkable question without parsing
    English (invariants 2-4's shared parsing note).

    `metadata.claimant_services` is a strict, required scenario-file key
    (scenario_file.py, same treatment as `depot_site`/`spare_inventory`) --
    always present, possibly empty. An empty list is honest for an episode
    whose rationale names no claimant by id (e.g. D1), not an error."""
    ids = tuple(scenario.metadata.get("claimant_services") or ())
    missing = [sid for sid in ids if sid not in scenario.gold.rationale]
    if missing:
        raise PairInvalid(
            f"{scenario.id}: metadata.claimant_services names {missing!r}, "
            f"which do not appear literally in gold.rationale -- the "
            f"declared claimant list and the rationale prose must agree")
    return ids


# Invariant 1.
def assert_realized_cuts_pass_the_event_filter(
    scenario: ScenarioFile, *, topology_path: str | Path,
    oms_by_id: dict[str, dict],
) -> None:
    """Every fiber named in a `realized` block belongs to an edge the
    episode's own event filter admits (aerial, for a storm) -- a storm does
    not physically cut buried conduit, so a `realized` block that says
    otherwise is asserting a physically impossible cut, no matter how
    internally consistent the rest of the episode is.

    `oms_by_id` is `{oms_id: {"src_node_id", "dst_node_id", "elements"}}`,
    the same shape `get_topology(layer="optical")`'s `oms` list gives (see
    `runner._define_horizon_risk_groups`) -- passed in rather than fetched
    here so the caller (suite.main()'s pre-flight) can read it ONCE and
    reuse it across every episode sharing the same state file, instead of
    this function opening its own connection.

    T1a fails this TODAY: its `realized` block injects
    fiber_allahabad_fatehpur_0/_1 and fiber_fatehpur_allahabad_0, and that
    link is BURIED (Task 14, not this task, fixes the episode data)."""
    filter_fn = get_filter(EVENT_TYPE)
    edge_mount: dict[tuple[str, str], object] = {}
    for edge in load_edges(topology_path):
        edge_mount[(edge.src, edge.dst)] = edge
        edge_mount[(edge.dst, edge.src)] = edge

    fiber_to_oms: dict[str, dict] = {}
    for oms in oms_by_id.values():
        for element_id in oms.get("elements") or ():
            if element_id.startswith("fiber_"):
                fiber_to_oms[element_id] = oms

    for hour, asset_ids in scenario.realized.items():
        for asset_id in asset_ids:
            oms = fiber_to_oms.get(asset_id)
            if oms is None:
                raise PairInvalid(
                    f"{scenario.id}: realized cut {asset_id!r} at hour "
                    f"{hour!r} names no fiber in any OMS's `elements` -- "
                    f"cannot check it against the event filter")
            edge = edge_mount.get((oms["src_node_id"], oms["dst_node_id"]))
            if edge is None:
                raise PairInvalid(
                    f"{scenario.id}: realized cut {asset_id!r} at hour "
                    f"{hour!r} resolves to OMS {oms.get('id')!r} "
                    f"({oms['src_node_id']}<->{oms['dst_node_id']}), which "
                    f"names no edge in the local topology")
            if not filter_fn(edge):
                raise PairInvalid(
                    f"{scenario.id}: realized cut {asset_id!r} at hour "
                    f"{hour!r} is on edge {edge.src}<->{edge.dst}, "
                    f"mount_type={edge.mount_type!r} -- the event's own "
                    f"vulnerability filter does not admit it, so this event "
                    f"cannot physically have cut it. The `realized` block "
                    f"asserts a physically impossible cut.")


# Invariant 2.
def _claimant_exposure_violations(
    claimant_ids: tuple[str, ...], cuttable_spans: dict[str, tuple],
) -> list[str]:
    """Claimant ids with no storm-cuttable span on their working path -- a
    service the event's own filter can never touch is not exposed by THIS
    event at all, so naming it in the arithmetic argues about a threat that
    cannot physically materialize (would have caught D4)."""
    return [sid for sid in claimant_ids if not cuttable_spans.get(sid)]


async def assert_claimants_have_filterable_exposure(
    client: Client, scenario: ScenarioFile, *, topology_path: str | Path,
) -> None:
    """Every service `metadata.claimant_services` names has at least one
    span the event's own filter admits, read the same way `runner.
    service_geometry` derives it for the real rollout -- so the check and
    the episode's own exposure computation can never disagree.

    No-op for an empty claimant list (see `claimant_service_ids`)."""
    ids = claimant_service_ids(scenario)
    if not ids:
        return
    geometry = await service_geometry(client, topology_path)
    violations = _claimant_exposure_violations(ids, geometry.cuttable_spans)
    if violations:
        raise PairInvalid(
            f"{scenario.id}: metadata.claimant_services names "
            f"{violations!r}, which have NO storm-cuttable span on their "
            f"working path -- a service the event's own filter can never "
            f"touch cannot honestly compete for the depot's spare")


# Invariant 9 (hazard-footprint plan, 2026-09-01).
def _risk_group_coverage_violations(
    exposure: dict[str, dict[str, dict[str, float]]],
    group_assets_by_horizon: dict[str, list[str]], *, threshold: float,
) -> list[tuple[str, str, float]]:
    """`(horizon, service_id, p_cut)` for every horizon whose risk group
    names NOTHING while some service's cut probability there reaches
    `threshold`.

    Deliberately ONE-DIRECTIONAL. A horizon at which nothing is measurably
    at risk is entitled to an empty group -- that is an honest statement
    about a quiet forecast hour. The failure this catches is the reverse:
    the agent handed a risk-group id for a horizon at which something IS
    measurably at risk, and the group names nothing, so `avoid` on it prunes
    nothing and the whole synthesized-risk-group story is inert.

    Every existing suite assertion tests a QUANTITY (p_cut values, flip
    margins, gold rationales) on one side of the seam or the other. Nothing
    tested the RELATIONSHIP between the two hazard geometries, which is
    exactly why a suite that asserts hard on both sides never saw the
    defect."""
    out: list[tuple[str, str, float]] = []
    for horizon, assets in group_assets_by_horizon.items():
        if assets:
            continue
        at_risk = [(sid, float(per[horizon]["p_cut"]))
                   for sid, per in exposure.items()
                   if horizon in per
                   and float(per[horizon]["p_cut"]) >= threshold]
        out.extend((horizon, sid, p_cut) for sid, p_cut
                   in sorted(at_risk, key=lambda item: -item[1]))
    return out


async def assert_risk_group_covers_measurable_exposure(
    client: Client, scenario: ScenarioFile, *, topology_path: str | Path,
    oms: list[dict],
) -> None:
    """For every issuance and every horizon of `scenario`: if any service's
    `p_cut` there reaches `P_CUT_ENUMERATION_THRESHOLD`, the risk group the
    runner would define for that horizon must name at least one asset.

    Exposure comes from `build_observation` -- the SAME function the real
    rollout calls -- and the group contents from
    `runner.horizon_risk_group_asset_ids` -- the SAME function
    `_define_horizon_risk_groups` calls. Neither side is reimplemented here,
    so this check and the payload the agent is actually handed cannot
    disagree.

    Every issuance is covered by walking `scenario.forecast`'s own issue
    hours: an issuance IS the latest issuance at its own issue hour, which is
    what `build_observation` selects.

    `oms` is `get_topology(layer="optical")["oms"]`, read once by the caller
    and reused across the episodes sharing a state file."""
    geometry = await service_geometry(client, topology_path)
    services = tuple(
        (await call_tool_json(client, "get_services"))["services"])
    edges = load_edges(topology_path)
    filter_fn = get_filter(EVENT_TYPE)

    for issue_hour in scenario.forecast:
        obs = build_observation(
            scenario, issue_hour, service_spans=geometry.cuttable_spans,
            services=services, spares_on_hand=scenario.spares_on_hand)
        groups = {
            horizon: horizon_risk_group_asset_ids(
                cone, scenario.damage_radius_km, edges=edges, oms=oms,
                filter_fn=filter_fn)
            for horizon, cone in obs.issuance.horizons.items()}
        violations = _risk_group_coverage_violations(
            obs.exposure, groups, threshold=P_CUT_ENUMERATION_THRESHOLD)
        if violations:
            worst = violations[0]
            raise PairInvalid(
                f"{scenario.id}: the {issue_hour} issuance defines an EMPTY "
                f"risk group at horizon {worst[0]!r}, but {len(violations)} "
                f"service-horizon pair(s) there reach p_cut "
                f"{P_CUT_ENUMERATION_THRESHOLD} -- worst is {worst[1]} at "
                f"p_cut {worst[2]}. The agent is being handed a risk-group id "
                f"that names nothing, so avoiding it prunes nothing.")


# Invariant 3.
def _claimant_depot_violations(
    claimant_ids: tuple[str, ...],
    endpoint_sites: dict[str, tuple[str, str]], depot_site: str,
) -> list[str]:
    """Claimant ids that do not terminate at `depot_site` -- restoring them
    would draw on THEIR OWN sites' depots, not this one, so they are no part
    of the contest `observation._restorable_groups` builds (its own
    eligibility rule, mirrored here)."""
    return [sid for sid in claimant_ids
            if depot_site not in (endpoint_sites.get(sid) or ())]


async def assert_claimants_depot_eligible(
    client: Client, scenario: ScenarioFile, *, topology_path: str | Path,
) -> None:
    """Every service `metadata.claimant_services` names terminates at
    `scenario.depot_site` -- a satna line card cannot restore
    kolkata<->mumbai, so a claimant that does not terminate at the depot
    cannot honestly compete for it.

    No-op for an empty claimant list (see `claimant_service_ids`)."""
    ids = claimant_service_ids(scenario)
    if not ids:
        return
    geometry = await service_geometry(client, topology_path)
    violations = _claimant_depot_violations(
        ids, geometry.endpoint_sites, scenario.depot_site)
    if violations:
        raise PairInvalid(
            f"{scenario.id}: metadata.claimant_services names "
            f"{violations!r}, which do not terminate at depot_site "
            f"{scenario.depot_site!r} -- restoring them would draw on "
            f"THEIR OWN sites' depots, not this one, so they cannot "
            f"compete for this episode's spare")


# Invariant 4.
def assert_claim_is_one_lightpath(
    scenario: ScenarioFile, groups: dict[str, tuple[dict, ...]],
) -> None:
    """The gold rationale's claimed aggregate ECAR
    (`metadata.claimed_competing_ecar_gbps`, when declared) must not exceed
    the LARGEST of the CLAIM'S OWN co-terminating groups' ECAR at the
    claim's OWN declared horizon (`metadata.claimed_competing_ecar_at`) --
    one spare buys ONE lightpath, so the honest competing figure across
    co-terminating groups is a MAXIMUM, never a sum
    (`observation._restorable_groups`'s own docstring). This is the check
    that would have caught D2: 108.8 G billed across three co-terminating
    groups where the honest figure was the largest single group, 55.3 G.

    **"The claim's OWN groups", not "the largest group anywhere in the
    network at that horizon" -- fixed 2026-08-30, round 2 review.** The
    first cut of this check took `max` over EVERY group `groups[horizon]`
    lists, regardless of who is in it. That is a different, wrong question:
    it let T3a's claim pass by comparing it against an unrelated group its
    own named claimants (`metadata.claimant_services`) are not even members
    of -- a pass with nothing to do with T3a's own claim, made possible only
    because today's live state happens to have exactly one depot-eligible
    group at that horizon; a second, unrelated group would make it worse,
    not better. Confirmed against two independent sources: the design
    spec's own worked D2 example computes "Honest (max group)" over the
    CLAIM's OWN groups, not the network's; and the brief's own Step 1 test
    text -- "its honest figure is the largest of THOSE [the claim's own
    three named groups]" -- names the claim's groups specifically. So the
    comparison set is now `groups[horizon]` FILTERED to only the groups
    whose `members` intersect `claimant_service_ids(scenario)` -- the same
    named list the claim itself is about.

    **The claim's horizon is NOT `derived.exposure_horizon_hour`,
    fixed 2026-08-30, round 1 review.** That function returns the SERVICE
    UNDER TEST'S OWN exposure horizon, which only happens to be the claim's
    horizon for T1 -- `assert_flip_dominates`'s own docstring states the
    general rule: "T1's is at the exposure horizon, T2's and T3's are
    before it" (T2a/T3a bill their competing claim at the NEAR horizon,
    e.g. `t2`, while the SUT's own exposure is graded at the FAR one, e.g.
    `t6`). Using the SUT's horizon for T2/T3 would silently compare the
    claim against the wrong horizon's groups -- once real numbers exist,
    that could pass a real over-claim or reject an honest one. So the claim
    declares its OWN horizon explicitly (`metadata.claimed_competing_ecar_
    at`) rather than borrowing the SUT's.

    `groups` is `observation._restorable_groups`'s own output shape,
    `dict[horizon, tuple[group_dict, ...]]`, so the check and the
    observation the agent reads can never disagree about what the largest
    group is.

    No-op when `claimant_service_ids(scenario)` is empty (an episode with
    no named claimants, e.g. D1) -- gated on the PARSED claimant list, not
    on the raw `claimed_competing_ecar_gbps` metadata key, so a stray
    leftover value on an otherwise-empty-claimant episode (schema does not
    forbid it; `scenario_file.py` only makes the two claim keys required
    TOGETHER when the claimant list is non-empty) cannot fall through to a
    bare `KeyError` on the missing horizon key below."""
    claimant_ids = set(claimant_service_ids(scenario))
    if not claimant_ids:
        return
    claimed = scenario.metadata.get("claimed_competing_ecar_gbps")
    horizon = scenario.metadata.get("claimed_competing_ecar_at")
    if claimed is None or horizon is None:
        raise PairInvalid(
            f"{scenario.id}: metadata.claimant_services is non-empty "
            f"({sorted(claimant_ids)!r}) but claimed_competing_ecar_gbps/"
            f"claimed_competing_ecar_at is missing -- scenario_file.py's "
            f"load_scenario is supposed to require both together whenever "
            f"claimant_services is non-empty; a hand-built ScenarioFile "
            f"(e.g. in a test) likely bypassed it")
    own_groups = [g for g in groups.get(horizon, ())
                 if claimant_ids & set(g.get("members") or ())]
    largest = max((g["ecar_gbps"] for g in own_groups), default=0.0)
    if claimed > largest + DERIVED_TOLERANCE:
        raise PairInvalid(
            f"{scenario.id}: metadata.claimed_competing_ecar_gbps="
            f"{claimed:.1f} exceeds the largest of the CLAIM'S OWN "
            f"co-terminating groups (the ones whose members intersect "
            f"metadata.claimant_services {sorted(claimant_ids)!r}) at its "
            f"declared horizon (metadata.claimed_competing_ecar_at="
            f"{horizon!r}), {largest:.1f} -- one spare buys ONE lightpath, "
            f"so the honest competing claim is the largest single group "
            f"among the claim's OWN named services, not a sum across "
            f"groups and not an unrelated group elsewhere in the network")


# Invariant 5.
def assert_group_fits_one_lightpath(
    scenario: ScenarioFile, groups: dict[str, tuple[dict, ...]],
) -> None:
    """Every co-terminating group's own ECAR, at every horizon, must not
    exceed `MAX_LIGHTPATH_CAPACITY_GBPS` -- a group's total claim cannot
    physically need more than one lightpath's capacity in the best available
    mode, since one spare buys exactly one lightpath."""
    for horizon, entries in groups.items():
        for g in entries:
            if g["ecar_gbps"] > MAX_LIGHTPATH_CAPACITY_GBPS:
                raise PairInvalid(
                    f"{scenario.id}: co-terminating group "
                    f"{g['endpoints']!r} (members {g['members']!r}) has "
                    f"ECAR {g['ecar_gbps']:.1f} G at horizon {horizon!r}, "
                    f"exceeding one lightpath's "
                    f"{MAX_LIGHTPATH_CAPACITY_GBPS:.0f} G capacity -- a "
                    f"group's total claim cannot physically need more than "
                    f"one lightpath")


# Invariant 6.
def _binding_site_violations(
    candidates, oms_nodes: dict, *, depot_site: str,
    spare_inventory: dict[str, int], default_spares_per_site: int,
) -> list[tuple[int, str, int, int]]:
    """`(candidate_index, site, needed, available)` for every
    candidate/site pair where a site OTHER than `depot_site` cannot cover
    its own charge -- i.e. a site other than the declared depot is ALSO
    scarce, so the depot would not be the only site a real candidate can
    bind on."""
    from .ledger import spares_needed

    violations: list[tuple[int, str, int, int]] = []
    for index, candidate in enumerate(candidates):
        needed = spares_needed(candidate, oms_nodes)
        for site, count in needed.items():
            if site == depot_site:
                continue
            available = spare_inventory.get(site, default_spares_per_site)
            if count > available:
                violations.append((index, site, count, available))
    return violations


async def assert_depot_is_the_binding_site(
    client: Client, scenario: ScenarioFile, *, topology_path: str | Path,
) -> None:
    """No candidate in the real menu under `reference_avoid` is limited by a
    site OTHER than `scenario.depot_site`, given `scenario.spare_inventory`
    plus the ledger's own default stock elsewhere
    (`ledger.SpareLedger.default_spares_per_site`) -- the episode is built to
    make the depot THE scarce resource; a candidate that also runs into
    scarcity somewhere else would make the harness silently reject that
    candidate for a reason the episode never intended to test."""
    from .ledger import SpareLedger

    menu = await menu_at_decision_hour(client, scenario)
    oms_nodes = await _oms_nodes(client)
    violations = _binding_site_violations(
        menu.get("candidates") or (), oms_nodes,
        depot_site=scenario.depot_site,
        spare_inventory=scenario.spare_inventory,
        default_spares_per_site=SpareLedger.default_spares_per_site)
    if violations:
        index, site, count, available = violations[0]
        raise PairInvalid(
            f"{scenario.id}: candidate_{index} needs {count} spare "
            f"transponder(s) at {site!r} ({available} on hand there, per "
            f"scenario.spare_inventory/the ledger's default), but {site!r} "
            f"is not depot_site ({scenario.depot_site!r}) -- only the "
            f"depot site may be the scarce, binding constraint")


# Invariant 7.
# The escape direction the exposure-and-depot design added a third aerial
# edge at (satna<->jabalpur, CLAUDE.md's 2026-08-30 topology note) -- the
# corridor T2's and T3's gold `optical_reroute` candidates escape along. The
# default for every scenario that does not declare its own
# `metadata.escape_route_node` (T1 spend-or-hold redesign, Task 9): T1 shares
# T2/T3's topology and this same corridor, but a future scenario built
# against a different topology, or one whose escape genuinely runs through a
# different node, need not share T2/T3's hardcoded assumption.
_ESCAPE_ROUTE_NODE = "jabalpur"


def _escape_route_node(scenario: ScenarioFile) -> str:
    """The corridor node `assert_escape_route_survives` requires an
    `optical_reroute` candidate to touch, for THIS scenario -- declared via
    `metadata.escape_route_node`, defaulting to `_ESCAPE_ROUTE_NODE`
    (T2's/T3's own `jabalpur`) for every scenario that does not override
    it."""
    return scenario.metadata.get("escape_route_node", _ESCAPE_ROUTE_NODE)


def _jabalpur_optical_reroute_candidate(
    candidates, oms_nodes: dict[str, list[str]], *,
    node: str = _ESCAPE_ROUTE_NODE,
) -> tuple[int | None, dict | None]:
    """The first `optical_reroute` candidate whose new lightpath's OMS route
    touches `node` (default `_ESCAPE_ROUTE_NODE`, T2's/T3's own corridor), or
    `(None, None)` if none exists."""
    for index, candidate in enumerate(candidates):
        if candidate.get("lever") != "optical_reroute":
            continue
        touched: set[str] = set()
        for lightpath in candidate.get("new_lightpaths") or ():
            for oms_id in lightpath.get("oms_sequence") or ():
                touched.update(oms_nodes.get(oms_id, ()))
        if node in touched:
            return index, candidate
    return None, None


async def assert_escape_route_survives(
    client: Client, scenario: ScenarioFile,
) -> None:
    """The `_escape_route_node(scenario)` `optical_reroute` candidate T2's
    and T3's gold decisions rely on still exists in the real menu under
    `basis=risk_group`/`level=risk_group` (the basis those gold decisions
    actually validate under -- grounded live by
    `test_gold_spare_action_is_grounded_in_a_real_candidate`'s
    `_GOLD_COMMITTED_LEVER` dict in test_episodes.py, which names
    `optical_reroute` as T2b's/T3b's committed lever), and still validates.
    `metadata.escape_route_node` is `aurangabad` for both T2 and T3 (moved
    off `surat`, which carries no live `optical_reroute` candidate for
    either SUT -- see CLAUDE.md's "T2/T3 pairs live on their own state
    files" section and `tools/build_eval_state.py`'s `T2_PINS`/`T3_PINS`
    comments). Guards the jalgaon pin set's spectrum consumption on that one
    buried spur: `T2_PINS`/`T3_PINS` and their claimant corridors all draw
    on the same `jalgaon` depot, and a future pin change that adds more
    demand over `jalgaon <-> aurangabad` could silently consume the last of
    the spare capacity this escape candidate needs."""
    node = _escape_route_node(scenario)
    menu = await call_tool_json(client, "route_service", {
        "service_id": scenario.service_under_test, "protected": False,
        "basis": "risk_group", "level": "risk_group", "best_effort": False,
        "avoid": scenario.reference_avoid})
    oms_nodes = await _oms_nodes(client)
    index, candidate = _jabalpur_optical_reroute_candidate(
        menu.get("candidates") or (), oms_nodes, node=node)
    if candidate is None:
        raise PairInvalid(
            f"{scenario.id}: no optical_reroute candidate via {node} exists "
            f"in the real menu under basis=risk_group/level=risk_group, "
            f"avoid={scenario.reference_avoid!r} -- the escape route T2's "
            f"and T3's gold optical_reroute candidates rely on may have "
            f"been lost to the Phase 2 spectrum pins")

    from .plans import build_topology_index, plan_from_candidate

    topo_index = await build_topology_index(client)
    plan = plan_from_candidate(
        topo_index, candidate, scenario.service_under_test,
        prefix=f"assert-escape-{scenario.id}")
    report = await call_tool_json(client, "validate_plan", {
        "plan": plan, "basis": "risk_group", "level": "risk_group"})
    if not report["ok"]:
        raise PairInvalid(
            f"{scenario.id}: the {node} optical_reroute candidate_{index} "
            f"exists but does not validate under basis=risk_group/"
            f"level=risk_group: {report['violations']!r}")


# Invariant 8.
def assert_sampling_error_within_margin(
    flip_values: dict[str, dict[str, float]], *, measured_error: float,
) -> None:
    """The suite's smallest flip margin -- the smallest gap between any two
    DISTINCT values (`rules._distinct_within_tolerance`, so numerically-
    solved near-duplicates don't count as a margin) any `derived.FLIP_VARS`
    scalar takes across the shipped episodes -- must be at least one order
    of magnitude (10x) larger than `measured_error`, the pipeline's own
    measured sampling error at its configured `m` (e.g. Task 2's 2.56e-4 at
    SOBOL_M=16). If the sampler's own noise floor sits within an order of
    magnitude of the smallest real gap the suite relies on to discriminate,
    the sampler could flip a gold label on noise alone.

    `flip_values` is `{scenario_id: {var: value}}`, exactly the shape
    `derived.FlipScalars.values()` produces and
    `assert_no_global_policy_solves_the_suite` already consumes.

    Wired into `suite.main()`'s pre-flight by Task 14 (exposure-and-depot
    plan, 2026-08-30), against the real, frozen per-episode FLIP_VARS values
    -- confirmed live: the smallest sorted-adjacent gap across all five
    FLIP_VARS members (shared between `claimant_ecar_before_exposure_horizon`
    and `claimant_ecar_peak_over_horizons`, both T3b vs T2a) is 14.586 G,
    clearing `measured_error * 10` (2.56e-3 G at the shipped
    `measured_error=2.56e-4`) by ~56,977x."""
    from .rules import _distinct_within_tolerance

    margins: list[float] = []
    for var in FLIP_VARS:
        values = [v[var] for v in flip_values.values()
                 if v.get(var) is not None]
        distinct = _distinct_within_tolerance(values)
        margins.extend(hi - lo for lo, hi in zip(distinct, distinct[1:]))
    if not margins:
        return

    smallest = min(margins)
    if smallest < measured_error * 10:
        raise PairInvalid(
            f"the smallest flip margin across the suite, {smallest:.6g}, "
            f"clears the measured sampling error {measured_error:.6g} by "
            f"only {smallest / max(measured_error, 1e-300):.2f}x -- an "
            f"order of magnitude (10x) margin is required, or the "
            f"sampler's own noise floor could flip a gold label")


# Invariant 9 (T1 spend-or-hold redesign, Task 9). `assert_both_legs_exposed`
# and `assert_spend_is_real` are the sanity checks a later task (Task 15)
# runs against the actual authored T1 scenario to prove it is a REAL
# decision -- both legs genuinely at risk, and the labeled-correct "spend"
# escape route genuinely reduces exposure -- rather than a scenario where the
# labeled-correct choice is trivial (protection already absorbs everything)
# or fake (the "escape" leaves the service about as exposed as before).
async def assert_both_legs_exposed(
    client: Client, scenario: ScenarioFile, *, topology_path: str | Path,
    floor: float = 0.05,
) -> None:
    """Both the service under test's WORKING leg (the corridor the storm
    threatens) and its PROTECTION leg (the corridor the automatic 1:1
    switchover falls back to) must independently read a non-trivial cut
    probability -- at least `floor` -- at the decision-hour issuance's own
    latest horizon (the same horizon `oracle.escape_objective` scores
    residual exposure against). A service whose protection sat entirely
    clear of the storm would make "the storm threatens this service" true
    only on a technicality: protection would absorb every realistic cut
    regardless of what the decider chooses, and the spend/hold comparison
    would not be testing anything.

    `p_cut_region`, not the joint `p_cut_service` observation.py's own
    exposure row uses for a protected service's `p_cut` -- that quantity is
    the probability BOTH legs are cut together, which stays small whenever
    EITHER leg alone is barely exposed. This check needs the two legs' own,
    separate probabilities, which is exactly what `p_cut_region` computes
    over one span union.

    Applies to protected SUTs (`metadata.exposure_legs: both`, the default)
    in full; for `exposure_legs: working` only the working leg is checked
    (spec 5.3)."""
    from .oracle import latest_horizon

    edges = load_edges(topology_path)
    geometry = await service_geometry(client, topology_path, edges=edges)
    issuance = latest_issuance(scenario, scenario.decision_hour)
    if not issuance.horizons:
        raise PairInvalid(
            f"{scenario.id}: the {scenario.decision_hour!r} issuance "
            f"publishes no horizon to check leg exposure against")
    horizon = latest_horizon(issuance)
    cone = issuance.horizons[horizon]
    sut = scenario.service_under_test
    working = geometry.cuttable_spans.get(sut, ())
    protection = geometry.protection_cuttable_spans.get(sut, ())
    p_working = p_cut_region(working, cone.center["lat"], cone.center["lon"],
                             cone.width_km, scenario.damage_radius_km)
    p_protection = p_cut_region(
        protection, cone.center["lat"], cone.center["lon"], cone.width_km,
        scenario.damage_radius_km)
    legs = scenario.metadata.get("exposure_legs", "both")
    if legs == "working":
        # Unprotected SUT (T3, spec 4.2): the working leg alone must carry
        # the exposure; there is no protection leg to check.
        if p_working < floor:
            raise PairInvalid(
                f"{scenario.id}: {sut}'s working-leg exposure at {horizon!r} "
                f"of the {scenario.decision_hour!r} issuance is "
                f"{p_working:.4f}, below the floor {floor}")
        return
    if legs != "both":
        raise PairInvalid(
            f"{scenario.id}: metadata.exposure_legs must be 'both' or "
            f"'working', got {legs!r}")
    if p_working < floor or p_protection < floor:
        raise PairInvalid(
            f"{scenario.id}: {sut}'s own leg exposure at {horizon!r} of the "
            f"{scenario.decision_hour!r} issuance is working={p_working:.4f}, "
            f"protection={p_protection:.4f}, and at least one is below the "
            f"floor {floor} -- not both legs are genuinely at risk, so "
            f"'spend the spare on the SUT' is not a real bet")


async def assert_spend_is_real(
    client: Client, scenario: ScenarioFile, *, topology_path: str | Path,
    residual_ceiling: float = 0.05,
) -> None:
    """The SPEND half of the T1 comparison must be a genuine escape, not a
    paper one. Replays this half's realized cuts strictly before the
    decision hour (the same discipline `menu_at_decision_hour` uses),
    defines the decision-hour issuance's own latest-horizon risk group
    EXACTLY as `oracle.spend_decider`'s own constraints decision would (via
    the SAME `oracle.spend_risk_group` helper, so the two can never drift
    apart), reads the real menu under it, and runs `oracle.escape_objective`
    over it -- the SAME function `oracle.spend_decider` is wired to.

    The pick must exist (not `"infeasible"` -- and never `"hold"` either:
    that literal is a real decider's own third exit at the objective step
    (decisions.py, HOLD_CHOICE), not something `oracle.escape_objective`'s
    fixed escape rule ever answers), charge exactly one spare
    transponder pair at `scenario.depot_site` (a real, single-lightpath
    escape -- not a free reuse, and not a multi-lightpath spend the
    episode's own `spares_on_hand` could not actually afford), and land the
    service at a residual p_cut at or below `residual_ceiling` -- an
    "escape" that is still nearly as exposed as before would not be a real
    spend."""
    from .oracle import escape_objective, spend_risk_group

    edges = load_edges(topology_path)
    geometry = await service_geometry(client, topology_path, edges=edges)
    d = scenario.decision_hour
    d_index = scenario.hours.index(d)
    for hour in scenario.hours[:d_index]:
        cuts = scenario.realized.get(hour, ())
        if cuts:
            await call_tool_json(client, "inject_failure",
                                 {"asset_ids": list(cuts)})

    try:
        horizon, rg_id = spend_risk_group(scenario)
    except ValueError as e:
        raise PairInvalid(str(e)) from e
    issuance = scenario.forecast[d]

    topo = await call_tool_json(client, "get_topology", {"layer": "optical"})
    filter_fn = get_filter(EVENT_TYPE)
    fiber_ids = horizon_risk_group_asset_ids(
        issuance.horizons[horizon], scenario.damage_radius_km, edges=edges,
        oms=topo["oms"], filter_fn=filter_fn)
    await call_tool_json(client, "define_risk_group", {
        "rg_id": rg_id, "asset_ids": fiber_ids,
        "metadata": {"event_type": EVENT_TYPE, "scenario": scenario.id,
                     "issued_at": d, "horizon": horizon}})

    services = tuple(
        (await call_tool_json(client, "get_services"))["services"])
    demand = next(s["demand_gbps"] for s in services
                 if s["id"] == scenario.service_under_test)
    obs = build_observation(
        scenario, d, service_spans=geometry.cuttable_spans, services=services,
        spares_on_hand=scenario.spares_on_hand, risk_group_ids={horizon: rg_id},
        endpoint_sites=geometry.endpoint_sites, depot_site=scenario.depot_site,
        protection_spans=geometry.protection_cuttable_spans)

    menu = await call_tool_json(client, "route_service", {
        "service_id": scenario.service_under_test, "protected": False,
        "basis": "risk_group", "level": "risk_group", "best_effort": False,
        "avoid": {"risk_groups": [rg_id]}})
    menu = menu_with_path_facts(
        menu, geometry, scenario.service_under_test, issuance=issuance,
        damage_radius_km=scenario.damage_radius_km, demand_gbps=demand)
    menu = menu_for_prompt(menu, geometry.oms_nodes)

    choice = escape_objective(obs, menu)
    index = candidate_index(choice.choice)
    if index is None:
        raise PairInvalid(
            f"{scenario.id}: oracle.escape_objective found no real escape "
            f"in the decision-hour menu under the spend decider's own avoid "
            f"(risk_groups=[{rg_id!r}]) -- {choice.reasoning}")
    candidate = menu["candidates"][index]
    needed = candidate.get("spares_needed") or {}
    if needed.get(scenario.depot_site) != 1:
        raise PairInvalid(
            f"{scenario.id}: the escape_objective pick (candidate_{index}) "
            f"charges {needed!r} at the depot, not exactly one spare "
            f"transponder pair at {scenario.depot_site!r}")
    residual = candidate["residual_exposure"][horizon]["p_cut"]
    if residual > residual_ceiling:
        raise PairInvalid(
            f"{scenario.id}: the escape_objective pick (candidate_{index}) "
            f"still reads a residual p_cut of {residual} at {horizon!r}, "
            f"above the ceiling {residual_ceiling} -- not a genuine escape")


def assert_gold_matches_outcomes(
    scenario: ScenarioFile, outcomes: dict[str, float],
) -> None:
    """The oracle enumerator's cross-check (T1 spend-or-hold redesign,
    Task 9/10): `scenario.gold.label` must be the ARGMIN of `outcomes` --
    the actual, simulated Gbps-hours lost under each of the two scripted
    deciders (`oracle.spend_decider`/`hold_decider`, run for real by a later
    task) -- and the best outcome must be separated from every other by at
    least `scenario.gold.min_margin_gbps_h`, so the label is not merely
    correct but correct by a margin an author could not have hit by
    accident.

    A hand-authored `gold.label` is a PREDICTION about which of the two
    scripted deciders' rollouts loses less; this is what actually checks it
    against a simulated ANSWER, not a second guess. `outcomes` is
    `{"spend": ..., "hold": ...}` -- the same shape `Gold.outcome_gbps_h`
    stores -- but takes any number of keys, so a future policy set is not
    hardcoded to exactly two.

    `scenario.gold.min_margin_gbps_h` of `None` (every episode not using the
    `spare_action_by_deadline` label rule) skips the margin check entirely --
    there is no simulated outcome pair for those episodes to compare a
    margin against in the first place."""
    if not outcomes:
        raise PairInvalid(
            f"{scenario.id}: assert_gold_matches_outcomes given no outcomes "
            f"to compare gold.label against")
    best = min(outcomes, key=outcomes.get)
    if scenario.gold.label != best:
        raise PairInvalid(
            f"{scenario.id}: gold.label={scenario.gold.label!r} but the "
            f"simulated outcomes {outcomes!r} argmin to {best!r} -- the "
            f"authored label does not match the actual best decision")
    if scenario.gold.min_margin_gbps_h is None:
        return
    ordered = sorted(outcomes.values())
    margin = ordered[1] - ordered[0] if len(ordered) > 1 else float("inf")
    if margin < scenario.gold.min_margin_gbps_h:
        raise PairInvalid(
            f"{scenario.id}: outcomes {outcomes!r} separate {best!r} from "
            f"the rest by only {margin:.3f} Gbps-h, below "
            f"gold.min_margin_gbps_h={scenario.gold.min_margin_gbps_h!r}")


# T2/T3 probe redesign (2026-09-06, spec 5.3): the pair's flip lives in the
# probe's answer, so the gate that proves the flip exists runs the SAME
# code path that answers the agent (`probe.answer_probe`) and compares.
PROBE_FLIP_KINDS = ("restorable", "spares_needed")


def probe_reading(kind: str, answer: dict) -> str:
    """The categorical fact one probe answer states, for one `probe_flip`
    kind: `restorable`/`not_restorable`, or (`spares_needed`) `spare`/`free`
    when a full restore exists and `not_restorable` when none does."""
    if kind not in PROBE_FLIP_KINDS:
        raise ValueError(f"unknown probe_flip kind {kind!r}")
    if answer["full_restore_candidates"] == 0:
        return "not_restorable"
    if kind == "restorable":
        return "restorable"
    needed = answer["min_spares_needed_by_site"] or {}
    return "free" if sum(needed.values()) == 0 else "spare"


def probe_flip_mismatches(a: ScenarioFile, answers_a: dict[str, dict],
                          b: ScenarioFile, answers_b: dict[str, dict]
                          ) -> list[str]:
    """The pure half of `assert_probe_flips`: given each half's probe answer
    for every declared claimant, everything that is not the declared flip.

    Each half declares `metadata.probe_flip = {kind, claimant, expected}`.
    Checked: (1) both halves declare the same kind; (2) each half's reading
    on ITS OWN named claimant equals its declared expectation; (3) the two
    expectations differ (that is the flip); (4) every claimant named by
    NEITHER half answers identically in both halves. T2 names the same
    claimant in both halves (restorable vs not); T3 names a different
    claimant per half (the exposed one: spare vs free), so (4) is vacuous
    there and (2)+(3) carry the check."""
    problems: list[str] = []
    if a.metadata.get("probe_flip") is None and b.metadata.get("probe_flip") is None:
        # A pair whose flip is NOT in the probe (T1: the fact is in the
        # observation). The probe must then say the same thing about every
        # claimant in both halves, or the probe carries an undeclared flip.
        for claimant in sorted(set(answers_a) | set(answers_b)):
            if answers_a.get(claimant) != answers_b.get(claimant):
                problems.append(
                    f"{claimant!r}: neither half declares a probe_flip, yet "
                    f"the probe answers differently: {answers_a.get(claimant)!r} "
                    f"({a.id}) vs {answers_b.get(claimant)!r} ({b.id})")
        return problems
    flips = {}
    for half in (a, b):
        flip = half.metadata.get("probe_flip")
        if not isinstance(flip, dict) or not {"kind", "claimant", "expected"} <= set(flip):
            problems.append(f"{half.id}: metadata.probe_flip must be "
                            f"{{kind, claimant, expected}}, got {flip!r}")
            continue
        flips[half.id] = flip
    if len(flips) < 2:
        return problems
    fa, fb = flips[a.id], flips[b.id]
    if fa["kind"] != fb["kind"]:
        problems.append(f"probe_flip kinds differ: {fa['kind']!r} ({a.id}) vs "
                        f"{fb['kind']!r} ({b.id})")
        return problems
    kind = fa["kind"]
    for half, flip, answers in ((a, fa, answers_a), (b, fb, answers_b)):
        answer = answers.get(flip["claimant"])
        if answer is None:
            problems.append(f"{half.id}: probe_flip names {flip['claimant']!r}, "
                            f"which is not a declared claimant with an answer")
            continue
        reading = probe_reading(kind, answer)
        if reading != flip["expected"]:
            problems.append(
                f"{half.id}: the probe reads {reading!r} for "
                f"{flip['claimant']!r} under its own decision-hour group, but "
                f"metadata.probe_flip expects {flip['expected']!r} "
                f"(answer: {answer!r})")
    if fa["expected"] == fb["expected"]:
        problems.append(
            f"{a.id}/{b.id}: both halves expect {fa['expected']!r}; the "
            f"expectations must differ or there is no probe flip")
    named = {fa["claimant"], fb["claimant"]}
    for claimant in sorted(set(answers_a) | set(answers_b)):
        if claimant in named:
            continue
        if answers_a.get(claimant) != answers_b.get(claimant):
            problems.append(
                f"{claimant!r} is named by neither half's probe_flip but "
                f"answers differently: {answers_a.get(claimant)!r} ({a.id}) "
                f"vs {answers_b.get(claimant)!r} ({b.id})")
    return problems


async def claimant_probe_answers(client: Client, scenario: ScenarioFile, *,
                                 topology_path: str | Path) -> dict[str, dict]:
    """Every declared claimant's probe answer under the decision-hour
    issuance's own latest-horizon risk group, defined EXACTLY as
    `oracle.spend_risk_group` + `runner.horizon_risk_group_asset_ids` define
    it (the same discipline `assert_spend_is_real` follows). Pre-decision
    realized cuts are replayed first, as `menu_at_decision_hour` does."""
    import functools
    from .oracle import spend_risk_group
    from .probe import answer_probe

    edges = load_edges(topology_path)
    d = scenario.decision_hour
    for hour in scenario.hours[:scenario.hours.index(d)]:
        cuts = scenario.realized.get(hour, ())
        if cuts:
            await call_tool_json(client, "inject_failure",
                                 {"asset_ids": list(cuts)})
    try:
        horizon, rg_id = spend_risk_group(scenario)
    except ValueError as e:
        raise PairInvalid(str(e)) from e
    issuance = scenario.forecast[d]
    geometry = await service_geometry(client, topology_path, edges=edges)
    topo = await call_tool_json(client, "get_topology", {"layer": "optical"})
    await call_tool_json(client, "define_risk_group", {
        "rg_id": rg_id,
        "asset_ids": horizon_risk_group_asset_ids(
            issuance.horizons[horizon], scenario.damage_radius_km,
            edges=edges, oms=topo["oms"], filter_fn=get_filter(EVENT_TYPE)),
        "metadata": {"event_type": EVENT_TYPE, "scenario": scenario.id,
                     "issued_at": d, "horizon": horizon}})
    services = (await call_tool_json(client, "get_services"))["services"]
    demands = {s["id"]: float(s["demand_gbps"]) for s in services}
    call = functools.partial(call_tool_json, client)
    answers = {}
    for claimant in scenario.metadata.get("claimant_services", ()):
        answer = await answer_probe(
            call, service_id=claimant, risk_group_id=rg_id, geometry=geometry,
            issuance=issuance, damage_radius_km=scenario.damage_radius_km,
            demands=demands)
        answers[claimant] = answer.to_dict()
    return answers


async def assert_probe_flips(client_a: Client, client_b: Client,
                             a: ScenarioFile, b: ScenarioFile, *,
                             topology_path: str | Path) -> None:
    """Spec 5.3: the scripted probe for each declared claimant under the
    decision-hour issuance's latest-horizon group differs between the halves
    in the declared way (`metadata.probe_flip`) and is identical for every
    other claimant. Each client must be freshly connected against the
    half's own state file."""
    answers_a = await claimant_probe_answers(client_a, a, topology_path=topology_path)
    answers_b = await claimant_probe_answers(client_b, b, topology_path=topology_path)
    problems = probe_flip_mismatches(a, answers_a, b, answers_b)
    if problems:
        raise PairInvalid(f"{a.id}/{b.id}: the probe does not flip as declared:\n  "
                          + "\n  ".join(problems))
