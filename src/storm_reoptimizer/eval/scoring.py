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
wherever the number is reported.

T1 spend-or-hold redesign (Task 8) adds a fourth rule, `spare_action_by_
deadline`, and the OUTCOME metric that makes T1 gradeable on something other
than a raw label match: `gbps_hours_lost` turns a rollout into one number per
service actually cut -- demand times how long it stayed down before anything
(a harness restoration, or nothing) brought it back -- and `regret_gbps_h`
compares the episode's own total against the best of the two GOLD outcomes
(`Gold.outcome_gbps_h`). Both are defined ONLY over consecutive `t0..tN` hour
labels: T2/T3's own scenarios use a non-positional label set
(`[t0, t1, t2, t6]`), and the loss arithmetic below indexes hours
positionally -- reading it against those would silently misread which hour a
given index names. `episode_metrics` therefore only populates the loss keys
for episodes whose `metadata.label_rule` is `spare_action_by_deadline`."""
from __future__ import annotations

from collections import Counter

from .runner import EpisodeTrace
from .scenario_file import ScenarioFile

LABEL_RULES = ("timing_at_decision_hour", "spare_action_by_deadline")


def _hour_record(trace: EpisodeTrace, hour: str) -> dict | None:
    return next((h for h in trace.hours if h["hour"] == hour), None)


def _assert_consecutive_hours(scenario: ScenarioFile) -> None:
    """The loss arithmetic (`_cut_index_by_service`, `gbps_hours_lost`)
    indexes `scenario.hours` positionally -- `restore_index` and `cut_index`
    are both raw ints into that tuple. That is only a safe thing to do when
    the labels themselves are `t0..tN` in order; T2/T3's own scenarios use
    `[t0, t1, t2, t6]`, where index 3 names the hour LABELLED t6, and reading
    this arithmetic against that would silently score the wrong hour."""
    expected = tuple(f"t{i}" for i in range(len(scenario.hours)))
    if scenario.hours != expected:
        raise ValueError(
            f"{scenario.id}: spare_action_by_deadline/gbps_hours_lost "
            f"require consecutive hour labels {expected}, got "
            f"{scenario.hours}")


def _cut_index_by_service(scenario: ScenarioFile,
                          trace: EpisodeTrace) -> dict[str, int]:
    """First hour index at which each service was ACTUALLY cut (Loss
    definition, T1 spend-or-hold redesign spec 5.1): the first hour whose
    `dropped_after_cut` names it, plus one special case for the service under
    test alone -- a SUT that is never in `dropped_after_cut` (nothing the
    server calls a drop) but whose own decider spent a spare too late against
    a REAL realized cut still counts as cut, at the hour the cut was
    realized, not at some later hour a tardy action pretends to fix."""
    _assert_consecutive_hours(scenario)
    cut_index: dict[str, int] = {}
    for i, hour in enumerate(scenario.hours):
        record = _hour_record(trace, hour) or {}
        for svc in record.get("dropped_after_cut") or ():
            cut_index.setdefault(svc, i)

    sut = scenario.service_under_test
    if sut not in cut_index:
        c_realized = next(
            (i for i, hour in enumerate(scenario.hours)
             if sut in trace.affected_by_hour.get(hour, ())), None)
        if c_realized is not None and any(
                a.service_id == sut and a.origin == "decider" and a.spares
                and a.effective_at_index > c_realized
                for a in trace.actions):
            cut_index[sut] = c_realized
    return cut_index


def _demands_by_service(trace: EpisodeTrace) -> dict[str, float]:
    """The per-service Gbps figure `runner.run_episode` now writes every
    hour (`record["demands"]`, from the same `get_services` roster fetch the
    hour's `record["services"]` ids already come from). Demand does not
    change mid-episode, so the first hour that carries the field is enough."""
    for hour in trace.hours:
        demands = hour.get("demands")
        if demands:
            return demands
    return {}


def gbps_hours_lost(scenario: ScenarioFile, trace: EpisodeTrace) -> dict[str, float]:
    """Gbps-hours lost per service actually cut this episode: demand times
    how many hours it stayed down, from its own `cut_index` up to whichever
    action next touches it (a harness restoration or a decider reroute) or
    episode end, whichever comes first. See `_cut_index_by_service` and the
    module docstring's Loss definition."""
    hours = scenario.hours
    cut_index = _cut_index_by_service(scenario, trace)
    demands = _demands_by_service(trace)
    losses: dict[str, float] = {}
    for svc, cut in cut_index.items():
        restore = min(
            (a.effective_at_index for a in trace.actions
             if a.service_id == svc and a.effective_at_index > cut),
            default=len(hours))
        span = min(restore, len(hours)) - cut
        losses[svc] = demands.get(svc, 0.0) * span
    return losses


def _spare_action_by_deadline_label(scenario: ScenarioFile,
                                    trace: EpisodeTrace) -> str:
    """"spend" iff a DECIDER-origin ledger debit on the SUT is backed by an
    action whose `effective_at_index` lands at or before the deadline: the
    SUT's own `cut_index` when it was actually cut (including the
    acted-too-late special case), or the exposure horizon from the decision
    hour when it never was. A harness-origin debit -- the harness's own
    deterministic post-cut restoration -- never counts; only the DECIDER's
    own choice is the graded decision."""
    sut = scenario.service_under_test
    cut_index = _cut_index_by_service(scenario, trace)
    deadline = cut_index.get(sut)
    if deadline is None:
        deadline = (scenario.hours.index(scenario.decision_hour)
                   + int(scenario.metadata["exposure_horizon_hours"]))
    for debit in trace.ledger_debits:
        if debit["service_id"] != sut or debit.get("origin") != "decider":
            continue
        action = next(
            (a for a in trace.actions
             if a.hour == debit["hour"] and a.service_id == debit["service_id"]
             and a.origin == debit.get("origin", "decider")
             and a.spares == debit["spares"]), None)
        if action is not None and action.effective_at_index <= deadline:
            return "spend"
    return "hold"


def decision_label(scenario: ScenarioFile, trace: EpisodeTrace) -> str | None:
    """The one categorical this episode is scored on, read off the trace by
    the rule the scenario declares. Under `timing_at_decision_hour`, an act
    that commits nothing (or commits inertly) is a wait -- see
    `timing_effective`."""
    rule = scenario.metadata.get("label_rule")
    if rule not in LABEL_RULES:
        raise ValueError(
            f"{scenario.id}: metadata.label_rule must be one of "
            f"{list(LABEL_RULES)}, got {rule!r}")

    if rule == "spare_action_by_deadline":
        return _spare_action_by_deadline_label(scenario, trace)

    record = _hour_record(trace, scenario.decision_hour)
    if record is None:
        return None

    if rule == "timing_at_decision_hour":
        # Same fallback expression `first_shot_correct` uses (:251-253).
        return record.get("timing_effective", record.get("timing", {}).get("action"))

    raise ValueError(f"unhandled label_rule {rule!r}")


def reexposed(scenario: ScenarioFile, trace: EpisodeTrace) -> bool:
    """Whether the service under test's OWN path, at any hour strictly
    after one of this episode's actions took EFFECT, sits inside a
    still-published cone -- read from the SUT's own per-hour recomputed
    exposure (`observation.build_observation`'s `p_cut`, stored on the
    trace's `hour["observation"]["exposure"]`), not from `affected_by_hour`
    (realized-cut membership against the PRE-action path).

    `affected_by_hour` fires on every successful pre-emptive reroute -- it
    is realized-cut membership against the path the service rode BEFORE
    the action, which is precisely the event the reroute was issued
    against. Measured on the 2026-09-15 paid run: `reexposed: true` on all
    six T1b rows (three agent seeds, both baselines) while the committed
    candidate's own `residual_exposure` already read
    `{"p_cut": 0.0, "ecar_gbps": 0.0}` and the SUT appeared nowhere in
    `gbps_hours_lost` -- the old metric was a synonym for `acted`, not a
    report of what CLAUDE.md promises ("reroute clear of the full forecast
    cone, so the reroute isn't re-exposed at t+3h").

    Keyed off `effective_at_index`, not `hour_index`: between commit and
    effectiveness the service is still on its OLD path by construction, so
    the honest question is whether the path the reroute LANDED sits in a
    later cone, not whether the OLD path (which the episode is trying to
    leave) does.

    Reach: `observation.exposure` is computed against whichever forecast
    issuance is in force at that hour. An episode that publishes no
    issuance after the action hour can never report a re-exposure this
    metric would catch -- T1 publishes only t0 and t1, so it can only ever
    confirm the reroute was clear of the t1 issuance's own t3 cone. Reading
    False there is a fact about the EPISODE's own issuance schedule, not
    evidence the moving-cone hazard was tested."""
    sut = scenario.service_under_test
    hours = scenario.hours
    return any(
        float(entry["p_cut"]) > 0.0
        for a in trace.actions
        for h in hours[a.effective_at_index + 1:]
        for entry in (_hour_record(trace, h) or {})
            .get("observation", {}).get("exposure", {}).get(sut, {}).values())


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

    # Total transponders across every charged SITE, not just the depot: a
    # wasted spend is wasted wherever it lands, and a lightpath's true cost is
    # one transponder at EACH of its two endpoints (ledger.spares_needed).
    spares_wasted = sum(sum(d["spares"].values()) for d in trace.ledger_debits
                        if d["service_id"] not in ever_affected)

    recovered = any(h.get("rejections") and h.get("committed")
                    for h in trace.hours)

    t0 = _hour_record(trace, hours[0]) or {}

    # T1 spend-or-hold redesign (Task 8): the OUTCOME metrics, populated only
    # for episodes graded on `spare_action_by_deadline` -- the loss
    # arithmetic requires consecutive `t0..tN` hours (`_assert_consecutive_
    # hours`), which T2/T3's own scenarios (`[t0, t1, t2, t6]`) do not have.
    if scenario.metadata.get("label_rule") == "spare_action_by_deadline":
        losses = gbps_hours_lost(scenario, trace)
        losses_total = sum(losses.values())
        sut_cut_index = _cut_index_by_service(scenario, trace).get(sut)
        regret = (losses_total - min(scenario.gold.outcome_gbps_h.values())
                 if scenario.gold.outcome_gbps_h else None)
        sut_working_cut_hour = (hours[sut_cut_index]
                               if sut_cut_index is not None else None)
    else:
        losses, losses_total, regret, sut_working_cut_hour = {}, 0.0, None, None

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
        "reexposed": reexposed(scenario, trace),
        "first_shot_correct":
            t0.get("timing_effective", t0.get("timing", {}).get("action"))
            == scenario.gold.decision_at_t0,
        "recovered_from_rejection": recovered,
        "spares_remaining": trace.spares_remaining,
        "lever_mix": dict(Counter(a.lever for a in trace.actions)),
        "terminal_status": trace.terminal_status,
        "iterations": sum(len(h.get("iterations", [])) for h in trace.hours),
        "inert_commits": sum(1 for h in trace.hours
                             for s in h.get("iterations", [])
                             if s.get("inert")),
        "tool_calls": trace.tool_calls,
        "wall_clock_s": trace.wall_clock_s,
        "cites_flip_variable": cites_flip_variable(trace, scenario.flip_variable),
        "gbps_hours_lost": losses,
        "gbps_hours_lost_total": losses_total,
        "regret_gbps_h": regret,
        "sut_working_cut_hour": sut_working_cut_hour,
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
