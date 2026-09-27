# src/storm_reoptimizer/eval/observation.py
"""What the agent sees at each hour of a rollout (eval design spec, "The
episode model"), and the two things it must NOT see.

The agent never sees `realized` -- the cuts the harness will inject -- and
never sees a future issuance. Waiting is what buys the next issuance; if the
observation at hour t could already read issued_(t+1), waiting would cost
nothing, WAIT would collapse into "act as late as lead time allows", and
every timing gold label in the suite would be scored against information the
agent should not have had. Neither exclusion is a comment asking the reader
to be careful: `build_observation` reads only from `scenario.forecast` at or
before `hour`, and `Observation` has no field that could carry `realized`.

Lead time is a harness rule, not a server capability (the server model has no
concept of time at all -- grepping multilayer_optical_network for `duration`,
`lead_time`, `valid_at` returns zero hits). It is a function of the chosen
candidate's LEVER, not a scenario constant: an IP reroute is a config change
and lands immediately; lighting a new optical path is provisioning and may be
a truck roll."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .cone import (
    SAMPLE_COUNT, Segment, expected_capacity_at_risk_gbps,
    nearest_span_offset_km, service_cut_mask,
)
from .revision import revision_band
from .scenario_file import Issuance, ScenarioFile

# The rule, stated as data so the trace can carry it. `zero` -> 0 hours;
# `scenario` -> the episode's own lead_time_hours.
LEAD_TIME_BY_LEVER = {
    "ip_reroute": "zero",
    "hybrid": "scenario",
    "optical_reroute": "scenario",
}


def lead_time_hours_for(lever: str, scenario_lead_time_hours: int) -> int:
    """Hours between issuing an action on this lever and it being effective."""
    try:
        rule = LEAD_TIME_BY_LEVER[lever]
    except KeyError:
        raise ValueError(
            f"unknown lever {lever!r} (known: {sorted(LEAD_TIME_BY_LEVER)})"
        ) from None
    return 0 if rule == "zero" else scenario_lead_time_hours


# multilayer_optical_network.model.modes.default_modes()'s largest
# transceiver mode's bitrate -- the most a single lightpath can carry in the
# best available mode. A plain constant, not an import of the sibling
# library from src/ (CLAUDE.md's hard seam): confirmed against the real
# server by tools/probe_claimants.py ("max lightpath capacity: 800.0 Gbps"),
# also recorded in docs/superpowers/plans/notes/2026-08-30-claimant-
# family.md. Owned here (moved from assertions.py, 2026-09-21 transponder-
# pairing spec) because _restorable_groups is the RUNTIME consumer;
# assertions.py's own static invariant now imports it from here instead of
# duplicating it.
MAX_LIGHTPATH_CAPACITY_GBPS: float = 800.0

# The rare-row merge threshold for a `cut_outcomes` table, and the enumeration
# cut-off `agent.project_observation` already used for a single service's
# peak p_cut (scenarios/T3a.yaml's own authoring note: its three named
# claimants "are the only non-SUT services above p_cut 0.005 at this horizon
# in EITHER half"). Moved here from agent.py (2026-09-27 T2 correlated-claims
# spec, same precedent as MAX_LIGHTPATH_CAPACITY_GBPS above) because
# `cut_outcome_rows` is now a second RUNTIME consumer of the same threshold;
# agent.py re-imports it rather than duplicating the value.
P_CUT_ENUMERATION_THRESHOLD = 0.005


def latest_issuance(scenario: ScenarioFile, hour: str) -> Issuance:
    """The most recent forecast issuance at or before `hour`. An hour with no
    advisory of its own carries the previous one forward -- which is exactly
    why waiting past an issuance hour buys nothing and waiting THROUGH one
    buys everything."""
    index = scenario.hours.index(hour)
    for candidate in reversed(scenario.hours[: index + 1]):
        if candidate in scenario.forecast:
            return scenario.forecast[candidate]
    raise ValueError(
        f"{scenario.id}: no forecast issuance at or before {hour!r}; every "
        f"episode must publish one at its first hour")


def _restorable_groups(exposure: dict, endpoint_sites: dict, *,
                       depot_site: str | None, service_under_test: str,
                       lightpath_capacity_gbps: float = MAX_LIGHTPATH_CAPACITY_GBPS,
                       ) -> dict[str, tuple[dict, ...]]:
    """Per horizon, the depot-eligible services grouped by what one new
    lightpath could restore.

    ONE spare transponder buys ONE lightpath. A set of services is jointly
    restored by it only if they CO-TERMINATE -- share both endpoints, so a
    single bidirectional lightpath serves all of them -- AND their combined
    same-direction demand fits inside `lightpath_capacity_gbps`. Before the
    2026-09-21 transponder-pairing fix, "a single bidirectional lightpath
    serves all of them" was not yet true even for a genuine mate pair (each
    direction billed its own card); it is true now, which is why the
    unbounded sum this function used to compute is wrong even for a
    perfectly co-terminating group once its own demand exceeds one
    lightpath's bitrate. Otherwise they COMPETE for the spare and only one
    is restored, which makes summing across groups an overstatement: the
    honest figure across groups is the MAXIMUM.

    Depot-eligible means terminating at `depot_site`. A service that does not
    may be badly exposed, but restoring it draws on ITS OWN sites' depots --
    a satna line card cannot restore kolkata <-> mumbai -- so it is no part
    of this contest. The service under test is excluded: it is the claim
    being weighed, not a claim against itself.

    `lightpath_capacity_gbps` is an OPTIMISTIC upper bound -- the real
    restoring mode depends on the chosen route's own GSNR, unknowable
    before routing -- deliberately generous, since overstating the
    competing claim is the conservative direction for a figure the agent
    weighs AGAINST acting on the SUT. Per direction (the group's sorted
    endpoint pair against each member's own `endpoint_sites` order),
    members are considered highest-ECAR-first; a member is admitted if
    admitting it keeps the running `demand_gbps` sum within capacity, else
    it is left out of both `members` and `ecar_gbps` for that horizon and
    the NEXT (smaller) member is still tried against the remaining
    headroom -- filling the lightpath as full as the priority order allows
    is the more optimistic (larger) reading of the competing claim, the
    conservative direction per the paragraph above, rather than stopping
    admission at the first member that doesn't fit."""
    raw: dict[str, dict[tuple[str, str], list[dict]]] = {}
    for svc_id, per_horizon in exposure.items():
        if svc_id == service_under_test:
            continue
        sites = endpoint_sites.get(svc_id)
        if sites is None or depot_site not in sites:
            continue
        key = tuple(sorted(sites))
        direction = "fwd" if sites == key else "bwd"
        for horizon, entry in per_horizon.items():
            raw.setdefault(horizon, {}).setdefault(key, []).append({
                "service_id": svc_id, "direction": direction,
                "ecar_gbps": entry["expected_capacity_at_risk_gbps"],
                "demand_gbps": entry["demand_gbps"]})

    groups: dict[str, tuple[dict, ...]] = {}
    for horizon, by_key in raw.items():
        entries = []
        for key, members in by_key.items():
            admitted: list[dict] = []
            for direction in ("fwd", "bwd"):
                running = 0.0
                ranked = sorted(
                    (m for m in members if m["direction"] == direction),
                    key=lambda m: (-m["ecar_gbps"], m["service_id"]))
                for m in ranked:
                    # `continue`, not `break`: a smaller, lower-priority
                    # member still fits in whatever headroom remains after
                    # a bigger one was skipped -- see the docstring
                    # paragraph above for why the fuller (more optimistic)
                    # reading is the one wanted here.
                    if running + m["demand_gbps"] > lightpath_capacity_gbps:
                        continue
                    running += m["demand_gbps"]
                    admitted.append(m)
            entries.append({
                "endpoints": key,
                "members": tuple(sorted(m["service_id"] for m in admitted)),
                "ecar_gbps": round(
                    sum(m["ecar_gbps"] for m in admitted), 3)})
        groups[horizon] = tuple(sorted(
            entries, key=lambda g: (-g["ecar_gbps"], g["endpoints"])))
    return groups


def _horizon_totals(exposure: dict, endpoint_sites: dict, *,
                    depot_site: str | None, service_under_test: str,
                    groups: dict[str, tuple[dict, ...]]
                    ) -> dict[str, dict[str, float]]:
    """Per horizon, the three operands every gold rationale in the suite now
    compares: the service under test's own expected capacity at risk, the
    LARGEST restorable group's expected capacity at risk (the honest
    competing claim -- one spare buys one lightpath, so the figure across
    co-terminating groups is a MAXIMUM, not a network-wide sum), and the
    summed expected capacity at risk of the depot-INELIGIBLE services -- ones
    that cannot compete for this depot's spare regardless of their own
    exposure, tracked separately so they are neither silently dropped nor
    wrongly counted as a competing claim.

    `groups` is `_restorable_groups`'s own output over the SAME exposure,
    endpoint_sites and depot_site -- passed in rather than recomputed so the
    two can never disagree about which services are depot-eligible.

    Both totals are summed over EVERY service with a representative point,
    not merely the ones a projection later chooses to list -- the aggregate
    is a fact about the network, and a total that silently covered only the
    visible rows would be worse than no total at all.

    Deliberately NOT shared with derived._ecar_at_cone, which computes the
    same shape from the UNROUNDED cut probability. That one feeds FLIP_VARS
    and assert_no_global_policy_solves_the_suite, whose interleave margins
    are under 1 G; this one is agent-facing and must agree with the rounded
    per-row numbers printed beside it. Two readers, two roundings, one
    quantity -- keep them apart."""
    totals: dict[str, dict[str, float]] = {}
    for svc_id, per_horizon in exposure.items():
        is_sut = svc_id == service_under_test
        sites = None if is_sut else endpoint_sites.get(svc_id)
        eligible = (not is_sut) and sites is not None and depot_site in sites
        for horizon, entry in per_horizon.items():
            slot = totals.setdefault(
                horizon, {"sut_ecar_gbps": 0.0,
                          "non_sut_ineligible_ecar_gbps": 0.0})
            ecar = entry["expected_capacity_at_risk_gbps"]
            if is_sut:
                slot["sut_ecar_gbps"] += ecar
            elif not eligible:
                slot["non_sut_ineligible_ecar_gbps"] += ecar
    for horizon, slot in totals.items():
        slot["largest_restorable_group_ecar_gbps"] = max(
            (g["ecar_gbps"] for g in groups.get(horizon, ())), default=0.0)
    return {horizon: {k: round(v, 3) for k, v in slot.items()}
            for horizon, slot in totals.items()}


@dataclass(frozen=True)
class Observation:
    """Everything the decider is allowed to read at one hour of one episode."""
    scenario_id: str
    service_under_test: str
    hour: str
    hour_index: int
    hours_remaining: int
    issuance: Issuance
    # service_id -> horizon hour -> {"hours_ahead", "offset_km", "p_cut",
    #                               "width_km", "demand_gbps",
    #                               "expected_capacity_at_risk_gbps",
    #                               "p_cut_if_track_revised"?}
    # `offset_km` is the distance from the cone centre to the NEAREST point
    # of the service's storm-cuttable span union (0.0 if the centre lies on
    # one), and `p_cut` is the probability that union is cut -- not a
    # representative-midpoint reading of either. A service with no
    # storm-cuttable span on its working path has no key here at all: see
    # `build_observation`. `p_cut_if_track_revised` (revision.revision_band)
    # is present only while another issuance is still scheduled -- see
    # `build_observation`'s own comment at the point it is computed.
    exposure: dict[str, dict[str, dict[str, float]]]
    services: tuple[dict, ...]
    spares_on_hand: int              # transponder PAIRS
    lead_time_hours: int             # the optical_reroute value
    # The episode's own damage radius, so a reader of this payload can
    # reconstruct the DAMAGE FOOTPRINT (width_km/2 + damage_radius_km) and
    # not merely the track-containment circle. Without it,
    # baseline._nearest_exposed_horizon could only test containment against
    # `width_km`, which is where the storm CENTRE goes rather than what it
    # breaks -- the 2026-08-31 seam defect, in the baseline's copy of it.
    # Defaulted to 0.0 so the hand-built Observations in
    # tests/eval/test_agent.py and tests/eval/test_baseline.py keep
    # constructing, and so that a caller who supplies none gets the strictly
    # narrower (old) containment test rather than a silently wrong one.
    damage_radius_km: float = 0.0
    # horizon hour -> the risk-group id the runner defined for that cone.
    # Decision 2's output is a risk-group id list, so the ids have to be in
    # the observation for the decider to be able to name one.
    risk_group_ids: dict[str, str] = field(default_factory=dict)
    iteration: int = 0
    last_rejection: dict | None = None
    # What the decider has ALREADY DONE this episode. Without it the runner
    # calls timing() every hour with no record that a plan was committed an
    # hour earlier, and the decider argues against its own escape route
    # (remediation spec, F4). Plain dicts, not runner.Action objects: runner
    # imports this module and not the reverse, and the whole Observation has
    # to be JSON-serializable for the trace and the prompt.
    actions_taken: tuple[dict, ...] = ()
    spares_spent: int = 0            # cumulative transponder PAIRS debited
    # horizon hour -> {"sut_ecar_gbps", "largest_restorable_group_ecar_gbps",
    # "non_sut_ineligible_ecar_gbps"}: the three operands of the comparison
    # every gold rationale makes. One spare buys one lightpath, so the
    # competing claim across co-terminating groups is a MAXIMUM, not a
    # network-wide sum (Task 10) -- `largest_restorable_group_ecar_gbps` is
    # that maximum, and `non_sut_ineligible_ecar_gbps` separately tracks
    # services that cannot compete for THIS depot at all. Present on every
    # Observation and in the trace; whether the AGENT is shown it is
    # project_observation's decision, gated per arm (remediation spec, W3.2).
    # Defaulted so the hand-built Observations in tests/eval/test_agent.py and
    # tests/eval/test_baseline.py keep constructing.
    horizon_totals: dict[str, dict[str, float]] = field(default_factory=dict)
    # horizon hour -> tuple of {"endpoints", "members", "ecar_gbps"}: the
    # co-terminating groups `largest_restorable_group_ecar_gbps` maxes over,
    # enumerated rather than only totalled so a reader (or the viewer) can see
    # WHICH services would share the restoring lightpath. Defaulted for the
    # same hand-built-Observation reason as `horizon_totals` above.
    restorable_groups: dict[str, tuple[dict, ...]] = field(default_factory=dict)
    # The forecast's OWN issue hours, in hour-label order -- when a new
    # advisory publishes, as against `issuance.issued_at`, which is only the
    # one currently in force. Realistic: a real storm centre runs on a fixed
    # advisory cadence, and knowing WHEN the next one lands is what makes
    # "wait for more information" a reasoned bet rather than an open-ended
    # stall. Defaulted for the same hand-built-Observation reason as
    # `horizon_totals` above.
    issuance_schedule: tuple[str, ...] = ()
    # lever -> the LAST hour that lever can still be issued and still land
    # (lead time included) at or before the latest horizon the CURRENT
    # issuance publishes -- `None` once that hour has already passed, i.e.
    # the lever can no longer act in time for anything this issuance has
    # forecast. This is what turns "wait" into a bounded bet rather than a
    # free option: a decider reading `hours_remaining` alone cannot tell
    # whether a slower lever (`hybrid`/`optical_reroute`, whose lead time
    # eats into the runway) has already run out of road. Defaulted for the
    # same hand-built-Observation reason as `horizon_totals` above.
    deadline_hour: dict[str, str | None] = field(default_factory=dict)
    # {"hour": "t1"} while an issuance LATER than the one currently in force
    # is still scheduled, else None. `issuance_schedule` already lists every
    # issue hour, but reading "is one still coming" off it requires knowing
    # which one is in force AND the hour ordering -- and the 2026-09-09 run
    # shows that inference failing in the direction that matters: three
    # separate timing calls opened with "no further issuance is coming,
    # waiting buys nothing" at an hour where one WAS scheduled. This states
    # it. Defaulted to None for the hand-built Observations in
    # tests/eval/test_agent.py and test_baseline.py.
    next_issuance: dict | None = None
    # The last non-empty `claim_priority` stated anywhere in this episode, or
    # (). An empty `claim_priority` on a later timing decision KEEPS this one
    # rather than clearing it, and `replay.restore_after_cuts` reads THIS at
    # a cut hour -- not that hour's own decision, which under the
    # decidable-hours rule may not exist at all.
    standing_claim_priority: tuple[str, ...] = ()
    # On the constraints and objective observations only: the timing decision
    # already made THIS HOUR, with the probe answers obtained alongside it.
    # `ClaudeDecider` is stateless by construction (one fresh single-turn
    # request per decision), so this is the only channel by which the step
    # that EXECUTES a decision can see the decision.  None at the timing
    # step, and omitted from `to_dict()` there -- an explicit null would read
    # as "you have already decided nothing this hour".
    decided_this_hour: dict | None = None
    # On the constraints and objective observations only: one entry per
    # COMPLETED iteration this hour -- the avoid set tried, the menu it
    # produced, and what was answered. Supersedes `last_rejection` as the
    # carrier of iteration history (`last_rejection` stays, for the ledger
    # and validation detail a rejection carries). D1 spent ten iterations on
    # ten identical avoid sets because nothing told it what it had already
    # tried.
    attempts_this_hour: tuple[dict, ...] = ()
    # ONE entry per horizon of the current issuance, populated only on the
    # ITERATION-loop `build_observation` call (spec 5.1): {"horizon": ...,
    # "assets": [...]} -- each asset a {"asset_id", "p_cut", "on"} row
    # (risk_assets.risk_group_rows). `risk_group_ids` told the agent a group
    # EXISTS; this is what it is actually MADE OF, so `avoid.assets` can
    # narrow to named spans instead of the whole group. Left at the default
    # `()` on the hour-level Observation (the timing decision has no use for
    # it), and even where populated it is large -- dozens of fibres per
    # horizon -- so `agent.project_observation` shows it only at the
    # constraints step, gated per DECISION, not by this field alone.
    risk_group_assets: tuple[dict, ...] = ()
    # Every probe answer ACCEPTED anywhere in this episode so far, oldest
    # first, each stamped with the `hour` and the `decision` it was bought at.
    # The same argument as `actions_taken` above, which the codebase already
    # accepted: ClaudeDecider opens a fresh conversation for every decision
    # (agent.py:770), so a probe answer bought at t0's timing call does not
    # exist at t0's own constraints call, let alone at t1's. T3b run B seed 0
    # probed buldhana at t0, used the answer correctly, and at t1 reverted to
    # the raw ECAR ordering the answer contradicted -- not a self-
    # contradiction, the information was genuinely gone (2026-09-12 failure
    # analysis, finding 7).
    #
    # Distinct from `decided_this_hour.probe_answers`, which is THIS hour's,
    # rebuilt per iteration: one says "what I have asked in this decision",
    # this one "what I have ever asked".
    #
    # Stored here WITHOUT a `current` flag -- `to_dict` adds it per entry at
    # serialization time, since "current" is a fact about THIS hour's own
    # `risk_group_ids`, not about the stored answer, and would go stale the
    # moment it was cached on the tuple instead (spec 4.4).
    probe_answers_this_episode: tuple[dict, ...] = ()
    # service_id -> horizon hour -> the per-sample boolean cut mask
    # (cone.service_cut_mask) `p_cut` at that (service, horizon) already
    # averages. Stored ONLY where `p_cut > 0` (design note, 2026-09-27 T2
    # correlated-claims spec §4.2): a missing entry means "never cut at that
    # horizon", not "cut mask not computed". This is what lets
    # `outcome_rows_from_masks` reduce several services' masks to a JOINT
    # table of who goes down TOGETHER, over the SAME sample space each
    # service's own `p_cut` already draws from -- not a second model.
    # `compare=False, repr=False` and absent from `to_dict`: a raw
    # `SAMPLE_COUNT`-length array per (service, horizon) is not
    # JSON-serializable and not itself the wire payload -- `cut_outcomes`
    # (agent.project_observation) is the reduction that goes on the wire.
    cut_masks: dict[str, dict[str, np.ndarray]] = field(
        default_factory=dict, compare=False, repr=False)
    # horizon hour -> {"unrestored": hours from this horizon to the last
    # episode hour inclusive, "restored_after_cut": the optical lead time,
    # capped at "unrestored"}. Converts the ECAR the agent compares (Gbps)
    # into what gold scores (Gbps-hours) -- the unit gap CLAUDE.md's
    # 2026-09-21 section records for T1, now stated on the wire instead of
    # left for the agent to reconstruct from `lead_time_hours` and
    # `hours_remaining` by hand.
    hours_down_if_cut: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, for the trace and for step 6's prompt."""
        payload = {
            "scenario_id": self.scenario_id,
            # "service under test" is eval-harness vocabulary; on the wire it
            # reads as "this is the important one", which is the framing the
            # control run's root cause #5 accuses. The Python attribute keeps
            # the harness name -- it is the correct one for a harness concept
            # -- and only the wire speaks the operational one (eval-fairness
            # design, §5.3).
            "actionable_service": self.service_under_test,
            "hour": self.hour,
            "hours_remaining": self.hours_remaining,
            "issued_at": self.issuance.issued_at,
            "cones": {h: {"width_km": c.width_km, "center": c.center}
                      for h, c in self.issuance.horizons.items()},
            "exposure": self.exposure,
            "services": [
                {**dict(s), "actionable": True}
                if s["id"] == self.service_under_test else dict(s)
                for s in self.services],
            "spares_on_hand": self.spares_on_hand,
            "lead_time_hours": {
                lever: lead_time_hours_for(lever, self.lead_time_hours)
                for lever in LEAD_TIME_BY_LEVER},
            "damage_radius_km": self.damage_radius_km,
            "risk_group_ids": self.risk_group_ids,
            "iteration": self.iteration,
            "last_rejection": self.last_rejection,
            "actions_taken": [dict(a) for a in self.actions_taken],
            "spares_spent": self.spares_spent,
            "horizon_totals": self.horizon_totals,
            "restorable_groups": self.restorable_groups,
            "issuance_schedule": list(self.issuance_schedule),
            "deadline_hour": self.deadline_hour,
            "next_issuance": self.next_issuance,
            "standing_claim_priority": list(self.standing_claim_priority),
            # "current": whether the entry's risk_group_id is one of THIS
            # hour's own group ids (spec 4.4). Group ids embed issuance and
            # horizon (`rg_{id}_{issued}_{h}`), so a reissued cone -- even one
            # that barely moved -- mints a new id, and simple membership is
            # the correct staleness test: a superseded answer was correct for
            # the group it names, but that group is not this hour's.
            "probe_answers_this_episode": [
                {**dict(entry),
                 "current": entry["risk_group_id"] in set(
                     self.risk_group_ids.values())}
                for entry in self.probe_answers_this_episode],
            "hours_down_if_cut": self.hours_down_if_cut,
        }
        # Present only where the harness actually supplied them, so the
        # timing payload never carries an empty shell (see the field
        # comments above).
        if self.decided_this_hour is not None:
            payload["decided_this_hour"] = self.decided_this_hour
        if self.attempts_this_hour:
            payload["attempts_this_hour"] = [dict(a)
                                             for a in self.attempts_this_hour]
        if self.risk_group_assets:
            payload["risk_group_assets"] = [dict(h)
                                            for h in self.risk_group_assets]
        return payload


def outcome_rows_from_masks(
    masks: dict[str, np.ndarray | None], n: int, *,
    merge_below: float, ndigits: int | None,
) -> list[dict]:
    """The joint distribution of `masks` over their shared `n`-sample space,
    reduced to rows of {"down": [ids...], "p": ...} -- one row per DISTINCT
    combination that actually occurs, mutually exclusive and summing to 1.0
    (before any merge). A service absent from a row's `down` list was up in
    every sample of that row.

    `masks[s] is None` is treated as an all-False array (never cut) rather
    than excluded: it participates in the partition exactly like a real
    mask would if that service's own cut probability were 0.0 at this
    horizon, which is what a missing mask means (`Observation.cut_masks`'s
    own docstring).

    Rows with `p < merge_below` are folded into a single trailing
    `{"other": True, "p": <summed p>, "count": <n merged rows>}` row, so a
    long tail of rare combinations does not blow up the payload -- the same
    reasoning `agent.project_observation` already applies to a single
    service's `p_cut` (P_CUT_ENUMERATION_THRESHOLD).

    `ndigits=None` returns unrounded probabilities (the Step-1 tests'
    exactness proof); production calls (`cut_outcome_rows`) round to 3
    decimals, the same precision `p_cut` itself is rounded to."""
    ids = sorted(masks)
    matrix = np.stack([masks[s] if masks[s] is not None else np.zeros(n, bool)
                       for s in ids], axis=1)
    patterns, counts = np.unique(matrix, axis=0, return_counts=True)
    rnd = (lambda x: round(x, ndigits)) if ndigits is not None else (lambda x: x)
    rows, other_p, other_n = [], 0.0, 0
    for pattern, count in zip(patterns, counts):
        p = count / n
        if p < merge_below:
            other_p += p; other_n += 1; continue
        rows.append({"down": [s for s, bit in zip(ids, pattern) if bit], "p": rnd(p)})
    rows.sort(key=lambda r: (-r["p"], r["down"]))
    if other_n:
        rows.append({"other": True, "p": rnd(other_p), "count": other_n})
    return rows


def cut_outcome_rows(
    obs: "Observation", service_ids: Iterable[str], *,
    merge_below: float = P_CUT_ENUMERATION_THRESHOLD,
    ndigits: int | None = 3,
) -> dict[str, list[dict]]:
    """`outcome_rows_from_masks` per horizon of `obs.issuance`, over exactly
    the `service_ids` that appear in `obs.exposure` -- "shown" is decided by
    the caller (`agent.project_observation`'s eligibility/threshold filter),
    not here, so this function never re-derives that rule. `{}` when no
    named id has exposure data at all: an empty table is not shown on the
    wire (design spec §4.2, "absent while no service is shown"), and the
    caller's own emptiness check is what enforces that -- this function just
    makes the emptiness easy to detect."""
    ids = [s for s in service_ids if s in obs.exposure]
    if not ids:
        return {}
    return {h: outcome_rows_from_masks(
                {s: obs.cut_masks.get(s, {}).get(h) for s in ids},
                SAMPLE_COUNT, merge_below=merge_below, ndigits=ndigits)
            for h in obs.issuance.horizons}


def build_observation(
    scenario: ScenarioFile, hour: str, *,
    service_spans: dict[str, tuple[Segment, ...]],
    services: tuple[dict, ...],
    spares_on_hand: int,
    risk_group_ids: dict[str, str] | None = None,
    iteration: int = 0,
    last_rejection: dict | None = None,
    actions_taken: tuple[dict, ...] = (),
    spares_spent: int = 0,
    endpoint_sites: dict[str, tuple[str, str]] | None = None,
    depot_site: str | None = None,
    protection_spans: dict[str, tuple[Segment, ...]] | None = None,
    standing_claim_priority: tuple[str, ...] = (),
    decided_this_hour: dict | None = None,
    attempts_this_hour: tuple[dict, ...] = (),
    risk_group_assets: tuple[dict, ...] = (),
    probe_answers_this_episode: tuple[dict, ...] = (),
) -> Observation:
    """The observation for one hour. `service_spans` maps a service id to the
    spans of its working path that the event's own filter admits -- the
    runner derives these from the server's view of the path plus the local
    topology's coordinates and mount types; keeping them an argument is what
    lets this module stay pure and testable without a server, exactly as
    `service_points` did.

    `endpoint_sites` (service id -> (src_site, dst_site)) and `depot_site`
    drive `_restorable_groups`/`_horizon_totals`'s depot-eligibility check.
    Both default to None/empty so every existing caller that has no depot
    context to offer keeps constructing -- with no eligible service, every
    non-SUT service falls into `non_sut_ineligible_ecar_gbps` and
    `restorable_groups`/`largest_restorable_group_ecar_gbps` come back
    empty/zero, which is the honest answer when the caller cannot say what
    terminates where.

    `protection_spans` (service id -> storm-cuttable spans of that service's
    PROTECTION path) is what makes a protected service's `p_cut` the JOINT
    both-legs-cut probability (`cone.p_cut_service`) instead of the working
    leg's alone. It is the ONLY exposure number a protected service shows:
    the per-leg entry this row used to carry (`legs`, categorical
    `cuttable_spans`/`in_footprint`) was removed 2026-09-06 (T2/T3 probe
    redesign, §5.3) because a number that carries no information about the
    SUT's own risk answered T1 2/2 by a bare threshold. Defaulted to None so
    every existing caller with no protection geometry to offer keeps
    constructing exactly as before (`p_cut_service(spans, None, ...)` is
    `p_cut_region(spans, ...)`, unchanged).

    `standing_claim_priority`, `decided_this_hour`, `attempts_this_hour` and
    `risk_group_assets` are runner-supplied facts, not derived from the
    scenario -- `run_episode` tracks the standing ranking, this hour's
    decision/iteration history, and (on the iteration-loop call only) each
    horizon's risk-group contents, and passes them straight through so the
    constraints and objective steps of one iteration can see what the timing
    step (and any earlier iteration this hour) already decided."""
    issuance = latest_issuance(scenario, hour)
    hour_index = scenario.hours.index(hour)

    issuance_schedule = tuple(sorted(scenario.forecast, key=scenario.hours.index))
    # LATER THAN THE ISSUANCE IN FORCE, not later than `hour`: an issuance
    # published at this very hour is already read, and its own revision is
    # what waiting buys. At the last issuance this is None and the revision
    # band (revision.py) is withheld with it -- there is nothing left to
    # revise. Computed here, ABOVE the exposure loop, because the band
    # computed inside that loop needs to know whether one is still coming.
    in_force = scenario.hours.index(issuance.issued_at)
    upcoming = [h for h in issuance_schedule
                if scenario.hours.index(h) > in_force]
    next_issuance = {"hour": upcoming[0]} if upcoming else None

    exposure: dict[str, dict[str, dict[str, float]]] = {}
    cut_masks: dict[str, dict[str, np.ndarray]] = {}
    for svc in services:
        spans = service_spans.get(svc["id"])
        if not spans:
            # No storm-cuttable span: p_cut is exactly 0 at every horizon, and
            # an all-zero row would only invite an undefined `offset_km`. The
            # service is still counted in project_observation's
            # `omitted_services["count"]`, which is over the full roster.
            continue
        protected = protection_spans is not None and svc["id"] in protection_spans
        protection_leg = protection_spans.get(svc["id"]) if protected else None
        per_horizon: dict[str, dict[str, float]] = {}
        for horizon, cone in issuance.horizons.items():
            lat, lon = cone.center["lat"], cone.center["lon"]
            offset = nearest_span_offset_km(spans, lat, lon)
            mask = service_cut_mask(spans, protection_leg, lat, lon,
                                    cone.width_km, scenario.damage_radius_km)
            raw = float(mask.mean()) if mask is not None else 0.0
            p_cut = round(raw, 3)
            # Stored only where the service is EVER cut at this horizon
            # (design note, spec §4.2): a missing entry means "never cut",
            # not "not computed", and `cut_outcome_rows` relies on exactly
            # that distinction.
            if raw > 0.0:
                cut_masks.setdefault(svc["id"], {})[horizon] = mask
            entry = {
                "hours_ahead": scenario.hours.index(horizon) - hour_index,
                # Distance from the cone centre to the NEAREST CUTTABLE SPAN
                # OF THE WORKING PATH, not to a representative midpoint and
                # not to the joint region -- so `offset_km <= width_km / 2`
                # reads "a span this storm can cut is inside the footprint".
                "offset_km": round(offset, 1),
                "width_km": cone.width_km,
                "p_cut": p_cut,
                "demand_gbps": svc["demand_gbps"],
                # p_cut x demand_gbps -- "the single quantity every
                # gold.rationale is arithmetic over" (cone.py). Precomputed
                # because it IS arithmetic, and CLAUDE.md's doctrine puts
                # arithmetic in code and leaves the comparison to the agent
                # (remediation spec, W3.1).
                #
                # Derived from the p_cut one line above, ROUNDED TO 3
                # DECIMALS, NOT from the raw probability, so the payload is
                # internally consistent: a reader who multiplies the two
                # numbers shown gets the number shown. That makes it
                # deliberately NOT bit-identical to
                # derived.py's FlipScalars, which use the unrounded value and
                # feed the static suite assertions -- those must not move.
                "expected_capacity_at_risk_gbps": round(
                    expected_capacity_at_risk_gbps(
                        p_cut, svc["demand_gbps"]), 3),
            }
            # Shown only while another issuance is still scheduled: at the
            # last one there is nothing left to revise, and a band printed
            # there would read as a claim about the storm rather than about
            # the forecast. Computed for the ACTIONABLE service and every
            # depot-eligible service with real exposure -- the same set the
            # projection keeps -- so the payload never carries a band for a
            # row it does not show, and the twelve extra p_cut evaluations
            # are not paid for hundreds of background services.
            if next_issuance is not None and p_cut > 0.0 and (
                    svc["id"] == scenario.service_under_test
                    or depot_site in (endpoint_sites or {}).get(svc["id"], ())):
                entry["p_cut_if_track_revised"] = revision_band(
                    spans, protection_leg, lat, lon,
                    width_km=cone.width_km,
                    damage_radius_km=scenario.damage_radius_km,
                    radius_km=(scenario.track_revision_km_per_hour_ahead
                               * entry["hours_ahead"]))
            per_horizon[horizon] = entry
        exposure[svc["id"]] = per_horizon

    groups = _restorable_groups(
        exposure, endpoint_sites or {}, depot_site=depot_site,
        service_under_test=scenario.service_under_test)

    deadline_hour: dict[str, str | None] = {}
    if issuance.horizons:
        latest_horizon = max(issuance.horizons, key=scenario.hours.index)
        latest_index = scenario.hours.index(latest_horizon)
        for lever in LEAD_TIME_BY_LEVER:
            d = latest_index - lead_time_hours_for(lever, scenario.lead_time_hours)
            deadline_hour[lever] = scenario.hours[d] if d >= hour_index else None
    else:
        deadline_hour = {lever: None for lever in LEAD_TIME_BY_LEVER}

    # unrestored: hours from this horizon to the last episode hour,
    # INCLUSIVE. restored_after_cut: the optical lead time, capped at
    # unrestored (a cut in the episode's last hour or two leaves no room for
    # the full lead time to play out). Global Constraints formula, 2026-09-27
    # T2 correlated-claims plan.
    hours_down_if_cut: dict[str, dict[str, int]] = {}
    for horizon in issuance.horizons:
        unrestored = len(scenario.hours) - scenario.hours.index(horizon)
        hours_down_if_cut[horizon] = {
            "unrestored": unrestored,
            "restored_after_cut": min(scenario.lead_time_hours, unrestored),
        }

    return Observation(
        scenario_id=scenario.id,
        service_under_test=scenario.service_under_test,
        hour=hour,
        hour_index=hour_index,
        hours_remaining=len(scenario.hours) - 1 - hour_index,
        issuance=issuance,
        exposure=exposure,
        services=services,
        spares_on_hand=spares_on_hand,
        lead_time_hours=scenario.lead_time_hours,
        damage_radius_km=scenario.damage_radius_km,
        risk_group_ids=dict(risk_group_ids or {}),
        iteration=iteration,
        last_rejection=last_rejection,
        actions_taken=tuple(actions_taken),
        spares_spent=spares_spent,
        horizon_totals=_horizon_totals(
            exposure, endpoint_sites or {}, depot_site=depot_site,
            service_under_test=scenario.service_under_test, groups=groups),
        restorable_groups=groups,
        issuance_schedule=issuance_schedule,
        deadline_hour=deadline_hour,
        next_issuance=next_issuance,
        standing_claim_priority=tuple(standing_claim_priority),
        decided_this_hour=decided_this_hour,
        attempts_this_hour=tuple(attempts_this_hour),
        risk_group_assets=tuple(risk_group_assets),
        probe_answers_this_episode=tuple(probe_answers_this_episode),
        cut_masks=cut_masks,
        hours_down_if_cut=hours_down_if_cut,
    )
