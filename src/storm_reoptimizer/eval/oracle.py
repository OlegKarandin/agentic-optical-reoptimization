# src/storm_reoptimizer/eval/oracle.py
"""The two SCRIPTED (non-LLM) oracle deciders (T1 spend-or-hold redesign
spec, 2026-09-05; plan Task 9).

A hand-authored `gold.label` for a `spare_action_by_deadline` episode is a
PREDICTION about which of two policies -- spend the depot's last spare
transponder pair on the service under test right now, or hold it for a
future claimant -- loses fewer Gbps-hours over the episode. This module is
the machinery a later task (Task 10) uses to actually CHECK that prediction:
run the same episode twice, once under `spend_decider` and once under
`hold_decider`, score both with `scoring.gbps_hours_lost`, and compare.

Both deciders are `ScriptedDecider`s (baseline.py) -- no LLM, no randomness,
same tools and same harness as the agent and the baselines. They differ from
every other `ScriptedDecider` use in the suite (the pre-flight
non-flip-decision replay) only in carrying a genuine OBJECTIVE RULE
(`escape_objective`) rather than a fixed per-hour answer, since "spend" is
not one candidate but "whichever real candidate escapes" -- the menu differs
hour to hour and issuance to issuance."""
from __future__ import annotations

from .baseline import ScriptedDecider
from .decisions import ConstraintDecision, ObjectiveDecision, TimingDecision
from .observation import Observation
from .scenario_file import ScenarioFile


def escape_objective(obs: Observation, menu: dict) -> ObjectiveDecision:
    """The SPEND policy's picking rule: the candidate that is a REAL escape
    from the storm, not a free or fake one.

    A candidate qualifies only if it clears all four of:
      - `path_delta.changes_working_path` -- it actually moves the service;
        an inert reuse of the exact fibre the episode is about protecting
        cannot be "spending the spare on an escape" (the T1 inert-reroute
        finding, 2026-08-31, is exactly this defect).
      - NOT `collides_with_protection.collides` -- it does not ride the
        service's own protection corridor; collapsing working and
        protection onto one corridor trades one invisible failure mode for
        another.
      - `spares_needed` is non-empty -- it actually lights something and so
        actually spends a transponder pair; an ip_reroute onto an existing
        lightpath costs the depot nothing and is not a "spend" decision at
        all, whatever else it does to the path.
      - `shortfall_gbps == 0` -- it fully restores the service; a partial
        restore is not the "the service is safe now" outcome the spend
        policy is supposed to buy.

    Among the survivors, the one with the LOWEST `residual_exposure[H]
    ["p_cut"]` at `H`, the issuance's own LATEST horizon (`obs.issuance.
    horizons`, in the order the issuance publishes them -- the same order
    `runner._residual_exposure` builds each candidate's own
    `residual_exposure` dict in, so the last key here is always the last key
    there). Scoring at the latest horizon, not the nearest, mirrors
    `ConstraintDecision`'s own reasoning (decisions.py): an escape that is
    only clear of the NEAR cone is re-exposed as the storm advances, and
    that is precisely the failure mode a real spend must avoid.

    `"infeasible"` if no candidate clears all four gates, or if the issuance
    publishes no horizon at all to score residual exposure against."""
    candidates = menu.get("candidates") or []
    horizons = obs.issuance.horizons
    if not horizons:
        return ObjectiveDecision(
            "infeasible", None,
            "oracle.escape_objective: this issuance publishes no horizon to "
            "score residual exposure against")
    latest = next(reversed(horizons))
    survivors = [
        i for i, c in enumerate(candidates)
        if (c.get("path_delta") or {}).get("changes_working_path")
        and not (c.get("collides_with_protection") or {}).get("collides")
        and c.get("spares_needed")
        and c.get("shortfall_gbps") == 0
    ]
    if not survivors:
        return ObjectiveDecision(
            "infeasible", None,
            f"oracle.escape_objective: none of the {len(candidates)} "
            f"candidate(s) both actually escapes (moves off the working "
            f"path, clear of the service's own protection, fully restores "
            f"it) and spends a real, transponder-consuming lightpath")
    best = min(survivors,
              key=lambda i: candidates[i]["residual_exposure"][latest]["p_cut"])
    p_cut = candidates[best]["residual_exposure"][latest]["p_cut"]
    return ObjectiveDecision(
        f"candidate_{best}", None,
        f"oracle.escape_objective: candidate_{best} escapes cleanly "
        f"(spends {candidates[best]['spares_needed']!r}) with the lowest "
        f"residual p_cut ({p_cut}) at {latest!r}, this issuance's latest "
        f"horizon")


def _claimants(scenario: ScenarioFile) -> tuple[str, ...]:
    return tuple(scenario.metadata.get("claimant_services", []))


def spend_decider(scenario: ScenarioFile) -> ScriptedDecider:
    """Try a genuine escape at the decision hour, exactly once, and wait
    every other hour.

    Constraints avoid the decision-hour issuance's own LATEST horizon's risk
    group -- `f"rg_{{scenario.id}}_{{decision_hour}}_{{H}}"`, the SAME id
    `runner._define_horizon_risk_groups` mints for that (issuance, horizon)
    when `issuance.issued_at == decision_hour` (true by the twin-pair
    discipline: "the issuance read AT d is the first thing that differs",
    assertions.py) -- under `basis="risk_group"`/`level="risk_group"`, so
    this decider's own escape genuinely clears the full forecast cone rather
    than only current exposure.

    `claim_priority` states this decider's own bias for the harness's
    post-cut restoration replay (replay.py, Task 7): the service under test
    FIRST, then the claimants in the scenario's own declared order -- spend
    reasons the SUT's own risk outweighs the claimants', so if a LATER cut
    this same hour also drops a claimant, the SUT still gets first call on
    whatever spare remains. Stated on every `TimingDecision` this decider
    can return (the decision-hour `act` and the `wait` default alike):
    `run_episode` reads THIS HOUR's own timing decision for restore
    ordering, not only the decision-hour one, so a cut at any other hour
    must see the same bias."""
    d = scenario.decision_hour
    issuance = scenario.forecast.get(d)
    if issuance is None or not issuance.horizons:
        raise ValueError(
            f"{scenario.id}: decision hour {d!r} has no forecast issuance of "
            f"its own (or that issuance has no horizon) -- oracle."
            f"spend_decider's risk-group id assumes one, matching "
            f"runner._define_horizon_risk_groups's own `issuance.issued_at`")
    horizon = next(reversed(issuance.horizons))
    rg_id = f"rg_{scenario.id}_{d}_{horizon}"
    claim_priority = (scenario.service_under_test, *_claimants(scenario))
    return ScriptedDecider(
        f"oracle:spend:{scenario.id}",
        timing_by_hour={d: TimingDecision(
            "act",
            f"oracle:spend: try a genuine escape route at the decision "
            f"hour {d!r}",
            claim_priority=claim_priority)},
        constraints_by_hour={d: ConstraintDecision(
            avoid={"risk_groups": [rg_id]},
            reasoning=f"oracle:spend: avoid {rg_id!r}, the decision-hour "
                     f"issuance's own latest horizon ({horizon!r}), so the "
                     f"escape is not re-exposed as the storm advances",
            protected=False, best_effort=False, basis="risk_group",
            level="risk_group")},
        default_timing=TimingDecision(
            "wait", "oracle:spend: no action outside the decision hour",
            claim_priority=claim_priority),
        objective_fn=escape_objective)


def hold_decider(scenario: ScenarioFile) -> ScriptedDecider:
    """Never act on the service under test -- the HOLD policy. `timing`
    always waits, so `constraints`/`objective` are never consulted by
    `run_episode` (only reached when `timing.action == "act"`); no
    `objective_fn` is wired for the same reason.

    `claim_priority` states the opposite bias from `spend_decider`: the
    claimants first, in the scenario's own declared order, then the service
    under test -- hold reasons the claimants' risk outweighs the SUT's, so
    the harness's post-cut restoration replay spends whatever spare remains
    on them before it ever reaches the SUT."""
    claim_priority = (*_claimants(scenario), scenario.service_under_test)
    return ScriptedDecider(
        f"oracle:hold:{scenario.id}",
        default_timing=TimingDecision(
            "wait",
            "oracle:hold: hold the depot's spare for the claimants; never "
            "spend it on the service under test",
            claim_priority=claim_priority))
