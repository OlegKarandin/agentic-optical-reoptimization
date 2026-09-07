# src/storm_reoptimizer/eval/rules.py
"""Candidate one-variable and greedy policies, for the check that no such
policy solves the suite (eval design spec, "The one-variable check").

This replaces a second baseline ARM with a static check over the scenarios'
gold labels and metadata -- no server, no rollout, no agent, no fitting
against outcomes. It is the strictly better artifact: an arm measures one
number after the fact; this FAILS THE BUILD while the episodes are still
being authored, which is when the information is actionable.

Greedy policies are in the enumeration because the discarded T2 draft was
beaten by "avoid the widest horizon that still returns a solution" -- a
policy with no threshold at all, which slips past a threshold-only sweep.

Per-pair orientation is deliberately GENEROUS: the three pairs draw gold
labels from three different vocabularies (act/wait, wide/narrow, A/B), so a
rule emits a SIDE and scoring grants it the better side->label mapping within
each pair. A rule that still fails is genuinely unable to separate the
halves, not merely mislabelled.

**The two-check doctrine -- read this before adding the claimant scalar to
DERIVED_VARS.** `derived.FLIP_VARS` (the claimant-side expected-capacity-at-
risk aggregates: at the exposure horizon, before it, peaked over all
horizons, and MINIMISED over all horizons) is deliberately EXCLUDED from
`DERIVED_VARS`, and therefore from `ENUMERATED_VARS`, even though it is
exactly the kind of derived scalar this module otherwise wants to catch. The
exclusion is not an oversight; adding it here would be actively wrong, for
three reasons:

1. This check is PER-PAIR. `best_rule` is scored one pair at a time,
   `threshold_rules` draws its split points from that pair's two values, and
   `score_rule` (deliberately, see above) grants the rule its better
   side->label orientation WITHIN THAT PAIR. With exactly two values per
   variable per pair, ANY variable that differs at all between the halves
   scores a clean 1.0 -- there is no threshold placement or orientation that
   can fail to separate two points. The claimant aggregate *is* the flip (it
   is the whole reason the two halves have opposite gold labels), so it is
   guaranteed to differ, so it would score 1.0 on every pair, forever. No
   amount of retuning the forecast geometry can ever fix that: the check
   would not be measuring whether a bare scalar solves the pair, it would be
   re-deriving the tautology that the flip variable correlates with the
   flip.
2. The question this module answers -- "does a single-variable rule solve
   THIS PAIR" -- is not the question that matters for the claimant scalar.
   The question that matters for it is whether ONE FIXED THRESHOLD, applied
   UNIFORMLY, answers the WHOLE SUITE -- i.e. whether an operator could
   deploy a bare number and skip the agent's per-pair reasoning entirely.
   That is a different check with a different scoring rule, and it lives in
   `assertions.assert_no_global_policy_solves_the_suite`. It enumerates
   exactly the variables this module excludes (`derived.FLIP_VARS`), and it
   grants the rule exactly ONE global orientation for the entire suite, not
   one per pair -- because a policy an operator could actually run has one
   orientation, period. See that function's docstring for the orientation
   argument in full, and derived.py's module docstring for the empirical
   result of running it. The short version, and it is not the reassuring one
   this docstring used to give: from the day it was built until the day it
   was reviewed, the whole-suite check passed only because `FLIP_VARS` listed
   three variants and there are four. The
   fourth, `claimant_ecar_min_over_horizons`, solved the shipped suite 6/6 at
   a single global threshold of 89.35 G -- reproducing the eval design spec's
   own predicted "89.4 G, 6/6" finding, which had been written off as an
   artifact of mixed per-pair reading. It is not an artifact: `min`
   mechanises exactly that mixed reading, in one uniform rule. The check now
   passes for real, over all four variants, because T2's near-horizon
   geometry was retuned (2026-08-26) until the min values INTERLEAVE across
   the labels. Three of the four are blocked by a structural tie
   (`..._at_exposure_horizon` ties on BOTH T2 and T3; `..._before_exposure_
   horizon` ties on T1; `..._peak_over_horizons` ties on T3); the fourth has
   no tie at all and is blocked by that engineered interleave. Swept live,
   the best any of them actually achieves is 4/6, 5/6, 4/6 and 5/6
   respectively -- note that a tie gives an upper BOUND, not the score, and
   for `..._peak_over_horizons` the bound (5/6) is not tight. The moral for
   anyone adding a variable here or there: a passing enumeration check is
   evidence about the enumeration first, and about the episodes only second
   -- a fifth summary of the same map (the aggregate at the EARLIEST
   published horizon) also solved the pre-retune suite 6/6, and was found
   only because someone went looking after the fact.
3. The ratio `claimant_ecar / sut_ecar` is enumerated by NEITHER check, on
   purpose. Computing that ratio and reasoning about it is the agent's
   INTENDED behavior, not a shortcut to be fenced off -- every gold
   rationale in this suite *is* that comparison. The line both checks police
   is bare, uncompared scalars; a rule that performs the comparison itself
   is not a confound, it is the task."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from .derived import DERIVED_TOLERANCE, DERIVED_VARS
from .scenario_file import ScenarioFile

# What an episode's AUTHOR declared.
OBSERVABLE_VARS = ("cone_width_km", "cone_motion_kmh", "n_future_claimants",
                   "exposure_horizon_hours", "spares_on_hand")

# ...plus what the forecast geometry actually IMPLIES, whether or not anyone
# declared it. `DERIVED_VARS` (derived.py) is the second list, and the reason
# it exists: for three rounds of review, T1's halves differed by
# p_cut(storm-svc-1) = 0.1349 vs 0.8834 at the exposure horizon -- a bare
# threshold that answers the pair 2/2 with no reasoning at all -- and this
# enumeration could not see it, because nobody had typed that number into
# `metadata`. An episode author cannot be trusted to declare the variable
# that breaks their own episode; the check has to derive it. Derived values
# are supplied per episode by the caller (they need a live server to read the
# service under test's real coordinates), so every function here takes an
# optional `derived` map and falls back to metadata alone when none is given.
ENUMERATED_VARS = OBSERVABLE_VARS + DERIVED_VARS

# scenario id -> {var: value}, as produced by
# derived.DerivedGeometry.scalars().
DerivedMap = dict[str, dict[str, float]]


def observables(episode: ScenarioFile,
                derived: DerivedMap | None = None) -> dict:
    """One episode's declared metadata, overlaid with any derived-geometry
    scalars supplied for it. Derived wins on a name collision: it is the
    measured value, and the declaration is the claim about it."""
    values = dict(episode.metadata)
    values.update((derived or {}).get(episode.id, {}))
    return values


@dataclass(frozen=True)
class Rule:
    """A candidate policy. `side` returns "lo" or "hi"; the orientation onto
    a pair's two gold labels is chosen per pair by score_rule."""
    name: str
    side: Callable[[ScenarioFile], str]


def _distinct_within_tolerance(values, tol: float = DERIVED_TOLERANCE) -> list[float]:
    """Collapse values that agree to within `tol` into one representative,
    instead of treating every bit-pattern as its own distinct value.

    Without this, two derived quantities that are equal BY CONSTRUCTION (e.g.
    two cone centres solved numerically to agree to 1e-13, not bit-for-bit)
    still differ as raw floats, and a threshold rule can split exactly
    between them -- a real failure mode hit while rebuilding T1's geometry
    (whole-branch fix, Step 2): a numerically-solved pair that passed
    `assert_pair_derived_geometry_is_equal`'s `DERIVED_TOLERANCE` equality
    check was still reported "solved 1.00" by a threshold sitting on the last
    mantissa bits. `DERIVED_TOLERANCE` is already the tolerance the pair
    assertion uses to call two derived values "equal"; a rule search that
    uses a tighter bar than the equality check it's supposed to validate
    against is checking floating-point noise, not the episode."""
    ordered = sorted(values)
    distinct: list[float] = []
    for v in ordered:
        if not distinct or v - distinct[-1] > tol:
            distinct.append(v)
    return distinct


def split_points(values: list[float],
                 tol: float = DERIVED_TOLERANCE) -> list[float]:
    """Every threshold worth testing over a set of values: the midpoints
    between adjacent DISTINCT values, where "distinct" means further apart
    than `tol`. Extracted from `threshold_rules` so the whole-suite check
    (assertions.assert_no_global_policy_solves_the_suite) sweeps exactly the
    same split points the per-pair check does, and cannot drift from them."""
    distinct = _distinct_within_tolerance(values, tol)
    return [(lo + hi) / 2.0 for lo, hi in zip(distinct, distinct[1:])]


def threshold_rules(episodes: list[ScenarioFile], var: str,
                    derived: DerivedMap | None = None) -> list[Rule]:
    """Every split point over the suite for one observable. Split points are
    the midpoints between adjacent distinct values -- exhaustive, because the
    episode set is small enough to score every one of them. Values within
    `DERIVED_TOLERANCE` of each other are treated as one value, not two (see
    `_distinct_within_tolerance`)."""
    seen = [observables(e, derived) for e in episodes]
    thresholds = split_points([v[var] for v in seen if var in v])
    rules = []
    for threshold in thresholds:
        rules.append(Rule(
            name=f"threshold:{var}<{threshold:g}",
            side=lambda e, v=var, t=threshold, d=derived: (
                "lo" if observables(e, d).get(v, 0) < t else "hi")))
    return rules


def greedy_rules() -> list[Rule]:
    """Parameter-free policies. `widest_feasible_avoid` is the one that beat
    the discarded T2: take the widest avoid horizon that still returns a
    solution, backing off only where it does not."""
    return [
        Rule("greedy:always_lo", lambda e: "lo"),
        Rule("greedy:always_hi", lambda e: "hi"),
        Rule("greedy:widest_feasible_avoid",
             lambda e: "hi" if e.metadata.get("widest_avoid_feasible") else "lo"),
        Rule("greedy:hoard_spare",
             lambda e: "lo" if e.metadata.get("spares_on_hand", 0) <= 1 else "hi"),
    ]


def candidate_rules(episodes: list[ScenarioFile],
                    derived: DerivedMap | None = None) -> list[Rule]:
    rules: list[Rule] = []
    for var in ENUMERATED_VARS:
        rules.extend(threshold_rules(episodes, var, derived))
    rules.extend(greedy_rules())
    return rules


def _by_pair(episodes: list[ScenarioFile]) -> dict[str, list[ScenarioFile]]:
    groups: dict[str, list[ScenarioFile]] = defaultdict(list)
    for episode in episodes:
        groups[episode.pair or episode.id].append(episode)
    return dict(groups)


def score_rule(rule: Rule, episodes: list[ScenarioFile]) -> float:
    """Fraction of episodes this rule gets right, granting it the better
    side->label orientation within each pair."""
    correct = 0
    for group in _by_pair(episodes).values():
        labels = sorted({e.gold.label for e in group})
        orientations = ([{"lo": labels[0], "hi": labels[0]}] if len(labels) == 1
                        else [{"lo": labels[0], "hi": labels[-1]},
                              {"lo": labels[-1], "hi": labels[0]}])
        correct += max(
            sum(1 for e in group if mapping[rule.side(e)] == e.gold.label)
            for mapping in orientations)
    return correct / len(episodes)


def best_rule(episodes: list[ScenarioFile],
              derived: DerivedMap | None = None) -> tuple[Rule, float]:
    scored = [(rule, score_rule(rule, episodes))
              for rule in candidate_rules(episodes, derived)]
    return max(scored, key=lambda item: item[1])
