# src/storm_reoptimizer/eval/gold.py
"""The gold enumerator (T1 spend-or-hold redesign spec, 2026-09-05, section
4.6; plan Task 10): the mechanism that turns a `spare_action_by_deadline`
episode's `gold.label` from an author's PREDICTION into an actual MEASURED
outcome.

`enumerate_outcomes` runs the SAME scenario twice through the real harness
(`runner.run_episode`, exactly as `tests/eval/test_episodes.py` does for
every other episode) -- once under `oracle.spend_decider`, once under
`oracle.hold_decider` -- and reports each choice's real, simulated
Gbps-hours lost (`scoring.gbps_hours_lost`). `gold_from_outcomes` turns
those two numbers into an actual `Gold`: the argmin choice is the label,
the two totals become `outcome_gbps_h`, and the rationale is the
enumerator's own table, not hand-written prose (spec 4.6: "gold.rationale
is the enumerator's table, not prose")."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .oracle import hold_decider, spend_decider
from .runner import run_episode
from .scenario_file import Gold, ScenarioFile
from .scoring import decision_label, gbps_hours_lost

# The two scripted policies T1's `spare_action_by_deadline` label rule
# distinguishes. A plain tuple, not a dict, so iteration order (and so
# `outcomes`' own key order, and so which choice `min()` picks on an exact
# tie) is fixed and readable top to bottom in the rationale table.
CHOICES: tuple[str, ...] = ("spend", "hold")

_DECIDERS: dict[str, Callable[[ScenarioFile], Any]] = {
    "spend": spend_decider, "hold": hold_decider,
}


async def enumerate_outcomes(
    connect: Callable[[], Any], scenario: ScenarioFile, *,
    topology_path: str | Path,
) -> dict[str, dict]:
    """Run `scenario` once per choice in `CHOICES`, each against a FRESH
    server connection, and report the real outcome.

    `connect` is a zero-arg async-context-manager factory yielding a
    connected client -- the same `connect()` idiom `test_episodes.py` uses
    (see e.g. its `_menus`/`_derived` helpers' local `_connect`) -- called
    ONCE PER CHOICE, never reused across them: `run_episode` mutates server
    state (commits reroutes, injects failures, debits the ledger), and the
    two rollouts must each start from the scenario's own untouched state,
    not from whatever the other rollout left behind.

    Returns `{"spend": {"gbps_hours_lost": {service_id: float, ...},
    "total": float, "label": str | None}, "hold": {...}}` -- `"label"` is
    `scoring.decision_label(scenario, trace)`, this ONE rollout's own graded
    categorical (almost always exactly the choice name itself, since a
    `spend`-decider rollout that never actually lands a decider-origin debit
    by the deadline reads "hold" like any other non-spend rollout would;
    that is a real, not a defect -- an infeasible escape spends nothing, so
    scoring correctly reads it as "hold" regardless of which decider was
    running)."""
    outcomes: dict[str, dict] = {}
    for choice in CHOICES:
        decider = _DECIDERS[choice](scenario)
        async with connect() as client:
            trace = await run_episode(
                client, scenario, decider, topology_path=topology_path)
        losses = gbps_hours_lost(scenario, trace)
        outcomes[choice] = {
            "gbps_hours_lost": losses,
            "total": sum(losses.values()),
            "label": decision_label(scenario, trace),
        }
    return outcomes


def _scope(scenario: ScenarioFile) -> tuple[str, ...]:
    """The services the rationale table reports on: the service under test
    first, then every declared claimant, in the scenario's own declared
    order -- the same set `oracle.spend_decider`/`hold_decider`'s own
    `claim_priority` is built from (`oracle._claimants`), so the rationale
    reads about exactly the services either decider could actually act on
    or compete for."""
    claimants = tuple(scenario.metadata.get("claimant_services", ()))
    return (scenario.service_under_test, *claimants)


def _rationale_table(scenario: ScenarioFile, outcomes: dict[str, dict],
                     scope: tuple[str, ...], label: str,
                     totals: dict[str, float]) -> str:
    """A fixed-width table: one row per service in `scope`, one column per
    choice, its Gbps-hours lost under that choice -- plus a TOTAL row and a
    one-line verdict. Every `scope` entry (the SUT and every declared
    claimant) appears as a literal row label, which is what satisfies
    `assertions.claimant_service_ids`'s requirement that every
    `metadata.claimant_services` id appear literally in `gold.rationale`."""
    choices = list(outcomes)
    name_w = max([len("service"), len("TOTAL")] + [len(s) for s in scope])
    col_w = max([len(c) for c in choices] + [12])

    def row(name: str, values: dict[str, float]) -> str:
        cells = " ".join(f"{values[c]:>{col_w}.3f}" for c in choices)
        return f"{name:<{name_w}} {cells}"

    lines = [f"{'service':<{name_w}} "
            + " ".join(f"{c:>{col_w}}" for c in choices)]
    for sid in scope:
        lines.append(row(sid, {c: outcomes[c]["gbps_hours_lost"].get(sid, 0.0)
                               for c in choices}))
    lines.append(row("TOTAL", totals))

    ordered = sorted(totals.values())
    margin = ordered[1] - ordered[0] if len(ordered) > 1 else float("inf")
    other = next(c for c in choices if c != label) if len(choices) > 1 else None
    lines.append(
        f"gold: {label} (argmin total Gbps-hours lost"
        + (f", margin {margin:.3f} over {other!r}" if other else "") + ")")
    return "\n".join(lines) + "\n"


def gold_from_outcomes(
    scenario: ScenarioFile, outcomes: dict[str, dict], *,
    min_margin_fraction: float = 0.25,
) -> Gold:
    """Turn `enumerate_outcomes`'s output into an actual `Gold`.

    `label` is the argmin choice by `outcomes[choice]["total"]`.
    `outcome_gbps_h` is the flat `{choice: total}` map `Gold.outcome_gbps_h`
    stores (and `assertions.assert_gold_matches_outcomes` re-checks against
    a fresh enumeration). `min_margin_gbps_h` is `min_margin_fraction` of
    the smaller total, floored at 1.0 Gbps-hour so a near-zero optimal loss
    cannot demand a near-zero margin (spec 4.6's own open-question proposal:
    "the smaller of the two halves' margins must be at least 25% of that
    half's optimal loss"). `survived` is every service in `_scope` whose
    loss under the GOLD choice is exactly zero -- not necessarily every
    service in `scope`, since the gold choice itself may still cost the SUT
    or a claimant something. `rationale` is `_rationale_table`'s own text,
    not hand-written prose, per spec 4.6."""
    totals = {choice: data["total"] for choice, data in outcomes.items()}
    label = min(totals, key=totals.get)
    min_total = min(totals.values())
    min_margin_gbps_h = max(min_margin_fraction * min_total, 1.0)

    scope = _scope(scenario)
    gold_losses = outcomes[label]["gbps_hours_lost"]
    survived = tuple(sid for sid in scope if gold_losses.get(sid, 0.0) == 0.0)

    return Gold(
        survived=survived,
        max_spares_wasted=0,
        decision_at_t0="wait",
        label=label,
        rationale=_rationale_table(scenario, outcomes, scope, label, totals),
        outcome_gbps_h=dict(totals),
        min_margin_gbps_h=min_margin_gbps_h,
    )
