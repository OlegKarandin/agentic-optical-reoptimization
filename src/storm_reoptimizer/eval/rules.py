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
halves, not merely mislabelled."""
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
    `assert_pair_derived_geometry_is_equal`'s 1e-6 equality check
    was still reported "solved 1.00" by a threshold sitting on the last
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


def threshold_rules(episodes: list[ScenarioFile], var: str,
                    derived: DerivedMap | None = None) -> list[Rule]:
    """Every split point over the suite for one observable. Split points are
    the midpoints between adjacent distinct values -- exhaustive, because the
    episode set is small enough to score every one of them. Values within
    `DERIVED_TOLERANCE` of each other are treated as one value, not two (see
    `_distinct_within_tolerance`)."""
    seen = [observables(e, derived) for e in episodes]
    values = _distinct_within_tolerance([v[var] for v in seen if var in v])
    thresholds = [(lo + hi) / 2.0 for lo, hi in zip(values, values[1:])]
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
