# src/storm_reoptimizer/eval/scoring.py
"""Episode and cross-twin scoring (eval design spec, "Scoring").

`pair_solved` is the README's headline number, and it needs ONE categorical
per episode to compare against gold.label. The three pairs are decided on
three different things -- T1 on WHEN the agent acted, T2 on HOW WIDE its
avoid set was, T3 on WHICH candidate it took -- so each scenario declares
`metadata.label_rule` and this module implements the three named readers
rather than inferring which applies.

Survival is `the model did not drop it` AND `the action was not too late`.
Lead time is bookkeeping in the runner (a commit lands immediately in the
model), so the timing penalty has to be applied here or not at all.

Flip-variable citation is NECESSARY, NOT SUFFICIENT. It is cheap entity
matching, and it catches "right answer, unrelated reason", which a bare
outcome metric cannot see. It is not reasoning verification: an agent
prompted to name every variable passes it with unrelated reasoning. Say so
wherever the number is reported."""
from __future__ import annotations

from collections import Counter

from .runner import EpisodeTrace
from .scenario_file import ScenarioFile

LABEL_RULES = ("timing_at_decision_hour", "avoid_horizon_at_decision_hour",
               "chosen_lever_at_decision_hour")


def _hour_record(trace: EpisodeTrace, hour: str) -> dict | None:
    return next((h for h in trace.hours if h["hour"] == hour), None)


def decision_label(scenario: ScenarioFile, trace: EpisodeTrace) -> str | None:
    """The one categorical this episode is scored on, read off the trace by
    the rule the scenario declares."""
    rule = scenario.metadata.get("label_rule")
    if rule not in LABEL_RULES:
        raise ValueError(
            f"{scenario.id}: metadata.label_rule must be one of "
            f"{list(LABEL_RULES)}, got {rule!r}")
    record = _hour_record(trace, scenario.decision_hour)
    if record is None:
        return None

    if rule == "timing_at_decision_hour":
        return record.get("timing", {}).get("action")

    committed = next((s for s in record.get("iterations", [])
                      if s.get("outcome") == "committed"), None)
    if committed is None:
        return None

    if rule == "avoid_horizon_at_decision_hour":
        wide = scenario.metadata["wide_avoid_risk_group"]
        chosen = committed["constraints"]["avoid"].get("risk_groups", [])
        return "wide" if wide in chosen else "narrow"

    lever = committed.get("lever")
    return scenario.metadata["label_by_lever"].get(lever)


def episode_metrics(scenario: ScenarioFile, trace: EpisodeTrace) -> dict:
    """Everything the spec's per-episode table names."""
    hours = scenario.hours
    sut = scenario.service_under_test

    ever_affected: set[str] = set()
    for affected in trace.affected_by_hour.values():
        ever_affected.update(affected)

    cut_index = next(
        (i for i, h in enumerate(hours)
         if sut in trace.affected_by_hour.get(h, ())), None)
    effective = min((a.effective_at_index for a in trace.actions), default=None)
    acted_too_late = bool(
        cut_index is not None and effective is not None
        and effective > cut_index)

    dropped = {d["service_id"]
               for d in trace.final_routing.get("dropped", {}).get("services", [])}
    survived = _known_service_ids(trace) - dropped
    if acted_too_late:
        survived.discard(sut)

    reexposed = any(
        sut in trace.affected_by_hour.get(h, ())
        for a in trace.actions
        for h in hours[a.hour_index + 1:])

    spares_wasted = sum(d["pairs"] for d in trace.ledger_debits
                        if d["service_id"] not in ever_affected)

    recovered = any(h.get("rejections") and h.get("committed")
                    for h in trace.hours)

    t0 = _hour_record(trace, hours[0]) or {}
    return {
        "scenario_id": scenario.id,
        "decider": trace.decider_name,
        "services_survived": sorted(survived),
        # NOT exact-set equality: _known_service_ids unions in every service
        # named in ANY hour's real get_services() roster (the whole loaded
        # network -- ~500+ background services on the real eval state, by
        # design, since episode_metrics also needs the full roster for
        # spares_wasted's ever_affected check below). gold.survived is
        # always a short, specific list naming the 1-3 services THIS
        # episode is actually about (e.g. [storm-svc-1, svc-b]) -- it was
        # never meant to enumerate the whole network. Exact equality can
        # never be true once the loaded network carries more than a
        # handful of services (confirmed: it never was, on the real
        # eval/states/loaded-s17.json state -- found during Task 14's
        # authoring, 2026-08-22). The correct check is whether every
        # service gold.survived NAMES actually survived.
        "survived_matches_gold": set(scenario.gold.survived) <= survived,
        "decision_label": decision_label(scenario, trace),
        "label_correct": decision_label(scenario, trace) == scenario.gold.label,
        "spares_wasted": spares_wasted,
        "within_max_spares_wasted": spares_wasted <= scenario.gold.max_spares_wasted,
        "acted_too_late": acted_too_late,
        "reexposed": reexposed,
        "first_shot_correct":
            t0.get("timing", {}).get("action") == scenario.gold.decision_at_t0,
        "recovered_from_rejection": recovered,
        "spares_remaining": trace.spares_remaining,
        "lever_mix": dict(Counter(a.lever for a in trace.actions)),
        "terminal_status": trace.terminal_status,
        "iterations": sum(len(h.get("iterations", [])) for h in trace.hours),
        "tool_calls": trace.tool_calls,
        "wall_clock_s": trace.wall_clock_s,
        "cites_flip_variable": cites_flip_variable(trace, scenario.flip_variable),
    }


def _known_service_ids(trace: EpisodeTrace) -> set[str]:
    """Every service this episode could have kept alive. simulate_ip_routing
    reports utilizations per IP LINK, not per service, so the roster comes
    from the per-hour service lists the runner records, unioned with the
    dropped list and everything a failure affected."""
    seen: set[str] = set()
    for hour in trace.hours:
        seen.update(hour.get("services", []))
    seen.update(d["service_id"]
                for d in trace.final_routing.get("dropped", {}).get("services", []))
    for affected in trace.affected_by_hour.values():
        seen.update(affected)
    return seen


def cites_flip_variable(trace: EpisodeTrace,
                        flip_variable: tuple[str, ...]) -> bool:
    """Does the reasoning reference the fact that actually distinguishes the
    twin? Cheap entity matching; see the module docstring on why this is a
    filter and not a verification."""
    text = " ".join(
        str(hour.get("timing", {}).get("reasoning", ""))
        + " " + " ".join(
            str(step.get("constraints", {}).get("reasoning", ""))
            + " " + str(step.get("objective", {}).get("reasoning", ""))
            for step in hour.get("iterations", []))
        for hour in trace.hours).lower()
    return all(token.lower() in text for token in flip_variable)


def cross_twin_metrics(a: ScenarioFile, trace_a: EpisodeTrace,
                       b: ScenarioFile, trace_b: EpisodeTrace) -> dict:
    """`pair_solved` is the README's headline number. `coherent` reads the
    stated priorities against the direction the pair's design says they
    should move -- declared as metadata.coherence, since only the pair's
    author knows which way is defensible."""
    ma, mb = episode_metrics(a, trace_a), episode_metrics(b, trace_b)
    halves = int(ma["label_correct"]) + int(mb["label_correct"])
    return {
        "pair": a.pair,
        "halves_correct": halves,
        "pair_solved": halves == 2,
        "coherent": _coherent(a, trace_a, b, trace_b),
        "halves": {a.id: ma, b.id: mb},
    }


def _coherent(a: ScenarioFile, trace_a: EpisodeTrace,
              b: ScenarioFile, trace_b: EpisodeTrace) -> bool | None:
    """None when either half stated no priority -- declining to impose an
    ordering is a legitimate answer, not an incoherent one."""
    spec = a.metadata.get("coherence")
    if not spec:
        return None
    term, higher_in = spec["term"], spec["higher_priority_in"]
    ranks = {}
    for scenario, trace in ((a, trace_a), (b, trace_b)):
        record = _hour_record(trace, scenario.decision_hour) or {}
        step = next((s for s in record.get("iterations", [])
                     if s.get("outcome") == "committed"), None)
        priority = (step or {}).get("objective", {}).get("priority")
        if not priority or term not in priority:
            return None
        ranks[scenario.id] = priority.index(term)
    other = b.id if higher_in == a.id else a.id
    return ranks[higher_in] < ranks[other]
