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

from .scenario_file import ScenarioFile

OBSERVABLE_VARS = ("cone_width_km", "cone_motion_kmh", "n_future_claimants",
                   "exposure_horizon_hours", "spares_on_hand")


@dataclass(frozen=True)
class Rule:
    """A candidate policy. `side` returns "lo" or "hi"; the orientation onto
    a pair's two gold labels is chosen per pair by score_rule."""
    name: str
    side: Callable[[ScenarioFile], str]


def threshold_rules(episodes: list[ScenarioFile], var: str) -> list[Rule]:
    """Every split point over the suite for one observable. Split points are
    the midpoints between adjacent distinct values -- exhaustive, because the
    episode set is small enough to score every one of them."""
    values = sorted({e.metadata[var] for e in episodes
                     if var in e.metadata})
    thresholds = [(lo + hi) / 2.0 for lo, hi in zip(values, values[1:])]
    rules = []
    for threshold in thresholds:
        rules.append(Rule(
            name=f"threshold:{var}<{threshold:g}",
            side=lambda e, v=var, t=threshold: (
                "lo" if e.metadata.get(v, 0) < t else "hi")))
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


def candidate_rules(episodes: list[ScenarioFile]) -> list[Rule]:
    rules: list[Rule] = []
    for var in OBSERVABLE_VARS:
        rules.extend(threshold_rules(episodes, var))
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


def best_rule(episodes: list[ScenarioFile]) -> tuple[Rule, float]:
    scored = [(rule, score_rule(rule, episodes))
              for rule in candidate_rules(episodes)]
    return max(scored, key=lambda item: item[1])
