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
from .spare_value import best_hold_ranking, decision_facts

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
    hold_ranking: tuple[str, ...] | None = None,
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
    running).

    The hold rollout runs under `hold_ranking` as its `claim_priority`
    (2026-09-27 spec 4.6: hold judged at its best). `None` computes it:
    `spare_value.best_hold_ranking` of `spare_value.decision_facts`, read on
    ONE EXTRA fresh `connect()` of its own -- `decision_facts` replays
    pre-decision cuts and defines a risk group on the server it reads, so it
    never shares a connection with either rollout. `outcomes["hold"]` then
    carries the ranking it ran under as `"claim_priority"` (a tuple), which
    `gold_from_outcomes` states in the rationale. The spend rollout is
    unchanged (`oracle.spend_decider`, SUT first)."""
    if hold_ranking is None:
        async with connect() as client:
            hold_ranking = best_hold_ranking(await decision_facts(
                client, scenario, topology_path=topology_path))
    hold_ranking = tuple(hold_ranking)

    outcomes: dict[str, dict] = {}
    for choice in CHOICES:
        if choice == "hold":
            decider = hold_decider(scenario, claim_priority=hold_ranking)
        else:
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
    outcomes["hold"]["claim_priority"] = hold_ranking
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


def _differing(outcomes: dict[str, dict], scope: tuple[str, ...]
               ) -> tuple[str, ...]:
    """Every service outside `scope` whose Gbps-hours lost is not the same
    under every choice, sorted by id. Since the replay restores every SHOWN
    service (2026-09-27 spec 4.5), a non-claimant can carry part of the
    spend/hold difference; a service whose loss is identical under both
    choices carries none of it and is left out."""
    ids = set().union(*(data["gbps_hours_lost"] for data in outcomes.values()))
    return tuple(sorted(
        sid for sid in ids - set(scope)
        if len({data["gbps_hours_lost"].get(sid, 0.0)
                for data in outcomes.values()}) > 1))


def _rationale_table(scenario: ScenarioFile, outcomes: dict[str, dict],
                     scope: tuple[str, ...], label: str,
                     totals: dict[str, float]) -> str:
    """A fixed-width table: one row per service in `scope`, then one row per
    service outside it whose loss DIFFERS between the choices (`_differing`,
    sorted), one column per choice, its Gbps-hours lost under that choice
    -- plus a TOTAL row, a one-line verdict, and, when the hold rollout
    recorded one (`enumerate_outcomes` always does), a `hold ranking: a > b
    > ...` line naming the `claim_priority` it ran under. Every `scope`
    entry (the SUT and every declared claimant) appears as a literal row
    label, which is what satisfies `assertions.claimant_service_ids`'s
    requirement that every `metadata.claimant_services` id appear literally
    in `gold.rationale`."""
    choices = list(outcomes)
    rows = (*scope, *_differing(outcomes, scope))
    name_w = max([len("service"), len("TOTAL")] + [len(s) for s in rows])
    col_w = max([len(c) for c in choices] + [12])

    def row(name: str, values: dict[str, float]) -> str:
        cells = " ".join(f"{values[c]:>{col_w}.3f}" for c in choices)
        return f"{name:<{name_w}} {cells}"

    lines = [f"{'service':<{name_w}} "
            + " ".join(f"{c:>{col_w}}" for c in choices)]
    for sid in rows:
        lines.append(row(sid, {c: outcomes[c]["gbps_hours_lost"].get(sid, 0.0)
                               for c in choices}))
    lines.append(row("TOTAL", totals))

    ordered = sorted(totals.values())
    margin = ordered[1] - ordered[0] if len(ordered) > 1 else float("inf")
    other = next(c for c in choices if c != label) if len(choices) > 1 else None
    lines.append(
        f"gold: {label} (argmin total Gbps-hours lost"
        + (f", margin {margin:.3f} over {other!r}" if other else "") + ")")
    ranking = outcomes.get("hold", {}).get("claim_priority")
    if ranking:
        lines.append("hold ranking: " + " > ".join(ranking))
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
    the smaller SCOPE-ONLY total -- the sum of the losses borne by `_scope`
    (the service under test and its declared claimants) alone, NOT the
    network-wide total -- floored at 1.0 Gbps-hour so a near-zero optimal
    loss cannot demand a near-zero margin (spec 4.6's own open-question
    proposal: "the smaller of the two halves' margins must be at least 25%
    of that half's optimal loss").

    The scope-only denominator is the point (whole-branch review 2026-09-05,
    Important finding 1; see `docs/superpowers/plans/notes/2026-09-05-t1-
    authoring.md`, "The margin floor's denominator", for the numbers and the
    workaround this replaced). A `total` is the
    WHOLE harness's loss, and is dominated by background services that ride
    the cut fibres under BOTH choices -- ~12000 Gbps-h identical in both
    columns, carrying no information about the decision. Scaling the floor
    off that demands a margin thousands of Gbps-h wide from a policy
    difference whose entire scale is hundreds, which no geometry can meet;
    scaling it off the scope total asks exactly what the spec asks, of
    exactly the services the decision moves.

    The denominator stays `_scope` even though the rationale table now also
    lists every other service whose loss differs between the choices
    (2026-09-27 spec 4.5/4.6): the table reports where the difference
    landed; the floor is still "25% of the SUT and claimants' own optimal
    loss". Widening it to the table's rows would put T2b's floor above its
    own projected 600 Gbps-h margin (2026-09-27 plan, Task 5).

    `label` is deliberately still the argmin of the NETWORK-WIDE `totals`
    (unchanged): the gold answer is which choice costs the network less, and
    only the FLOOR's denominator was ever wrong.

    `survived` is every service in `_scope` whose loss under the GOLD choice
    is exactly zero -- not necessarily every service in `scope`, since the
    gold choice itself may still cost the SUT or a claimant something.
    `rationale` is `_rationale_table`'s own text, not hand-written prose,
    per spec 4.6."""
    totals = {choice: data["total"] for choice, data in outcomes.items()}
    label = min(totals, key=totals.get)

    scope = _scope(scenario)
    scope_totals = {
        choice: sum(data["gbps_hours_lost"].get(sid, 0.0) for sid in scope)
        for choice, data in outcomes.items()
    }
    min_margin_gbps_h = max(
        min_margin_fraction * min(scope_totals.values()), 1.0)

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
