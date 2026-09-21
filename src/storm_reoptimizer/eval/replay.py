# src/storm_reoptimizer/eval/replay.py
"""Deterministic post-cut restoration replay (T1 spend-or-hold redesign spec,
2026-09-05, section 5.1; plan Task 7).

Before this module existed, nothing in the harness ever consumed a spare the
decider chose to HOLD: a realized cut dropped a service and the episode
simply recorded that fact. "Hold the spare for a claimant" therefore had no
simulated consequence, and the timing decision's `claim_priority` field
(decisions.py) was purely decorative. This module is the consequence: after
every hour's realized cuts are injected, the harness -- not the decider --
spends whatever spares remain to restore actually-dropped services, in
EXACTLY the order the decider itself stated via `claim_priority` (falling
back to largest-demand-first for anyone it left unranked).

No decider call anywhere in this loop. It is pure harness orchestration over
the same primitives the decision-3 loop in run_episode already uses
(`menu_with_path_facts`, `try_commit`, `SpareLedger`) -- deliberately reused
rather than duplicated, so a candidate that would count as inert or
unaffordable in the hourly loop counts the same way here.

`Action`, `menu_with_path_facts` and `try_commit` are imported from `.runner`
LAZILY, inside `restore_after_cuts`, rather than at module level: runner.py
imports `restore_after_cuts` from this module to wire it into the per-hour
cut block, and a module-level import back here would be a cycle. By the time
`restore_after_cuts` actually RUNS, `runner` has finished importing, so the
lazy import always succeeds."""
from __future__ import annotations

from typing import Any

from .ledger import spares_needed
from .observation import lead_time_hours_for


def _ordered_scope(scope: set[str], priority: tuple[str, ...],
                   demand_by_service: dict[str, float]) -> list[str]:
    """`scope`, in restoration order: every id `priority` names, in the order
    it names them, first; then every other scope member, largest
    `demand_gbps` first (ties broken by service id, for determinism).

    A `priority` id outside `scope` is not this episode's to restore (it
    names some OTHER service the decider was shown) and is silently
    ignored -- `decisions._claim_priority`'s own validation is only that the
    id was SHOWN to the decider, not that it is in this episode's claimant
    roster.

    `priority` itself is deduplicated first-occurrence-wins:
    `decisions._claim_priority` only validates SHAPE (a list of strings),
    never uniqueness, so a decider -- including the real LLM-driven
    `ClaudeDecider` -- can legally repeat an id. Without the dedupe here, a
    repeated id would appear twice in the returned order and the caller's
    loop would attempt to restore the same, already-restored service a
    second time."""
    deduped_priority = dict.fromkeys(priority)
    listed = [svc for svc in deduped_priority if svc in scope]
    unlisted = sorted(scope - set(listed),
                      key=lambda svc: (-demand_by_service.get(svc, 0.0), svc))
    return listed + unlisted


def _empty_record(hour: str, service: str, outcome: str) -> dict[str, Any]:
    return {"hour": hour, "service_id": service, "outcome": outcome,
            "lever": None, "spares": {}, "effective_at_hour": None,
            "rejection": None}


async def restore_after_cuts(
    counting, *, scenario, hour: str, hour_index: int, affected: list[str],
    priority: tuple[str, ...], ledger, geometry, issuance, rg_for_cut_hour,
    index_factory,
) -> tuple[list["Action"], list[dict]]:
    """Restore whichever of `{scenario.service_under_test} ∪
    metadata.claimant_services` this hour's realized cut actually dropped
    (`affected`, the runner's `dropped_after_cut` for this hour), spending
    `ledger`'s remaining spares in `priority` order.

    `priority` is the STANDING claim-priority ranking (spec 7.4) minus the
    actionable service -- run_episode's own last-stated `claim_priority`
    across hours, not necessarily this cut hour's own decision, since a cut
    hour may have no decision of its own under the decidable-hours rule.
    `_ordered_scope` below still ignores any id outside `scope` and still
    dedupes.

    Only a service actually IN `affected` is attempted -- one whose scope
    membership is real but which the cut left untouched (protection absorbed
    it, or it simply wasn't on the cut asset) gets no record at all, the same
    as a service never mentioned this episode.

    For each attempted service, in order: `route_service` under the SAME
    posture the hourly decision loop uses (`protected=False`,
    `basis="physical"`, `level="link"`, `best_effort=False`), avoiding this
    hour's own risk group if one was defined; annotate with
    `menu_with_path_facts` so the selection below can tell a real reroute
    from an inert one; pick the CHEAPEST-in-spares candidate that would fully
    restore the service (`shortfall_gbps == 0`), actually move it
    (`path_delta.changes_working_path` -- an inert candidate restores
    nothing, so it is never a valid pick here), and that the ledger can
    afford (menu order among equals). `try_commit` (runner.py, Task 2) does the validate/commit pair
    under `baseline="standing"`; on success the ledger is debited with
    `origin="harness"` (SpareLedger.debit, Task 2) so later scoring can tell
    a decider's own spend from the harness's, and `index_factory` is called
    again so the NEXT service in this same replay sees the topology this
    commit just changed (a fresh lightpath/IP-link id, spectrum consumed).

    Outcomes, one per attempted service:
      - "restored" -- committed; `lever`/`spares`/`effective_at_hour` filled.
      - "unaffordable" -- a structurally workable candidate (full restore,
        real path change) existed but the ledger could not afford ANY of
        them.
      - "no_candidate" -- no candidate in the menu both fully restores the
        service and actually changes its path (an all-inert or
        all-partial-restore menu, including an empty one).
      - "rejected" -- the ledger could afford the chosen candidate but
        `try_commit` refused it (validation violations or a non-"committed"
        commit status); `rejection` carries the same typed dict
        `run_episode`'s own hourly loop records.
      - "skipped" -- the service is in `affected` and in scope, but this
        hour's topology index has no record of it at all (defensive: an
        unresolvable service cannot be routed or costed, so it is recorded
        and left alone rather than raising out of the replay and losing
        every later service's restoration in the same hour).

    Returns `(actions, records)`: the harness `Action`s this replay itself
    committed (origin="harness", for `run_episode` to fold into the
    episode's own action list and `actions_taken` projection) and one record
    dict per attempted service, in attempt order."""
    from .runner import Action, menu_with_path_facts, try_commit

    scope = {scenario.service_under_test,
             *scenario.metadata.get("claimant_services", [])}
    dropped = set(affected)
    index = await index_factory()

    actions: list["Action"] = []
    records: list[dict] = []

    # Built ONCE, from this initial index, even though `index` is reassigned
    # after every commit below. Safe: a demand_gbps figure is a property of
    # the SERVICE, not of the topology a commit changes, and nothing in this
    # replay (or the plan ops try_commit issues) creates or removes a
    # service -- only reroutes one -- so a service present in the initial
    # index stays present, with the same demand, through every later commit
    # in this same hour.
    demand_by_service = {svc: index.service_by_id[svc]["demand_gbps"]
                         for svc in scope if svc in index.service_by_id}

    for service in _ordered_scope(scope, priority, demand_by_service):
        if service not in dropped:
            continue
        if service not in index.service_by_id:
            records.append(_empty_record(hour, service, "skipped"))
            continue

        avoid = {"risk_groups": [rg_for_cut_hour]} if rg_for_cut_hour else {}
        menu = await counting.call("route_service", {
            "service_id": service, "protected": False, "basis": "physical",
            "level": "link", "best_effort": False, "avoid": avoid})
        menu = menu_with_path_facts(
            menu, geometry, service, issuance=issuance,
            damage_radius_km=scenario.damage_radius_km,
            demand_gbps=demand_by_service[service])

        # The one candidate-selection predicate the spec states: fully
        # restores the service AND actually moves it. Affordability is
        # checked separately below so "structurally fine but the depot can't
        # cover it" (unaffordable) is distinguishable from "nothing in the
        # menu would even help" (no_candidate).
        workable = [c for c in (menu.get("candidates") or [])
                   if c.get("shortfall_gbps") == 0
                   and c.get("path_delta", {}).get("changes_working_path")]
        # Spec 5.4 (T2/T3 probe redesign): CHEAPEST in spares first, not
        # first in menu order, so a zero-spare ip_reroute over an existing
        # lightpath with headroom is taken over a spare-charging lightpath.
        # `sorted` is stable, so equal-cost candidates keep menu order. This
        # is also what makes probe.answer_probe's min_spares_needed_by_site
        # equal to what this replay would actually spend -- `lit_runs=
        # ledger.lit_runs` (transponder-pairing spec, 2026-09-21) is what
        # keeps that equality true for a mate pair too: c-rev's own cost
        # here must already reflect whatever c-fwd's earlier restoration
        # in this SAME hour just lit.
        workable.sort(key=lambda c: sum(
            spares_needed(c, ledger.oms_nodes, lit_runs=ledger.lit_runs).values()))
        if not workable:
            records.append(_empty_record(hour, service, "no_candidate"))
            continue

        candidate = next((c for c in workable if ledger.can_afford(c)), None)
        if candidate is None:
            records.append(_empty_record(hour, service, "unaffordable"))
            continue

        # `split_transient_outage=True` is what makes this replay able to
        # restore anything at all against the real server: every service
        # reaching this line is ALREADY DOWN, and a two-op restoration plan
        # for a down service is refused wholesale for the outage it is
        # repairing. See `runner.try_commit`'s docstring for the full
        # mechanism and the live evidence, and
        # `docs/superpowers/plans/notes/2026-09-05-t1-authoring.md`'s "Known
        # follow-up: `split_transient_outage` has no rollback on
        # partial-commit failure" for the tracked record of its one open
        # gap. The hourly decision loop keeps
        # the default (False): a decider acting PRE-EMPTIVELY is not
        # repairing an outage, so a transient drop there would be damage
        # its own plan introduced.
        commit, rejection = await try_commit(
            counting, index, candidate, service,
            prefix=f"restore-{hour}-{service}", basis="physical",
            level="link", split_transient_outage=True)
        if rejection is not None:
            records.append({**_empty_record(hour, service, "rejected"),
                            "lever": candidate["lever"],
                            "rejection": rejection})
            continue

        spares = ledger.debit(candidate, hour=hour, service_id=service,
                              origin="harness")
        lead = lead_time_hours_for(candidate["lever"], scenario.lead_time_hours)
        effective_at_index = hour_index + lead
        actions.append(Action(
            hour=hour, hour_index=hour_index, lever=candidate["lever"],
            effective_at_index=effective_at_index, spares=spares,
            service_id=service, origin="harness"))
        effective_at_hour = (scenario.hours[effective_at_index]
                             if effective_at_index < len(scenario.hours)
                             else None)
        records.append({**_empty_record(hour, service, "restored"),
                        "lever": candidate["lever"], "spares": spares,
                        "effective_at_hour": effective_at_hour})
        # The next service in this SAME replay must see what this commit
        # just did -- a new lightpath/IP-link id, spectrum now occupied.
        index = await index_factory()

    return actions, records
