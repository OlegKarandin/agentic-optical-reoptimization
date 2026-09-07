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
from typing import Any

from .cone import (
    Segment, expected_capacity_at_risk_gbps, nearest_span_offset_km,
    p_cut_service,
)
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
                       depot_site: str | None, service_under_test: str
                       ) -> dict[str, tuple[dict, ...]]:
    """Per horizon, the depot-eligible services grouped by what one new
    lightpath could restore.

    ONE spare transponder buys ONE lightpath. A set of services is jointly
    restored by it only if they CO-TERMINATE -- share both endpoints, so a
    single bidirectional lightpath serves all of them. Otherwise they COMPETE
    for the spare and only one is restored, which makes summing across groups
    an overstatement: the honest figure across groups is the MAXIMUM.

    Depot-eligible means terminating at `depot_site`. A service that does not
    may be badly exposed, but restoring it draws on ITS OWN sites' depots --
    a satna line card cannot restore kolkata <-> mumbai -- so it is no part
    of this contest. The service under test is excluded: it is the claim
    being weighed, not a claim against itself."""
    groups: dict[str, dict[tuple[str, str], dict]] = {}
    for svc_id, per_horizon in exposure.items():
        if svc_id == service_under_test:
            continue
        sites = endpoint_sites.get(svc_id)
        if sites is None or depot_site not in sites:
            continue
        key = tuple(sorted(sites))
        for horizon, entry in per_horizon.items():
            slot = groups.setdefault(horizon, {}).setdefault(
                key, {"endpoints": key, "members": [], "ecar_gbps": 0.0})
            slot["members"].append(svc_id)
            slot["ecar_gbps"] += entry["expected_capacity_at_risk_gbps"]
    return {horizon: tuple(
                sorted(({**g, "members": tuple(sorted(g["members"])),
                         "ecar_gbps": round(g["ecar_gbps"], 3)}
                        for g in by_key.values()),
                       key=lambda g: (-g["ecar_gbps"], g["endpoints"])))
            for horizon, by_key in groups.items()}


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
    #                               "expected_capacity_at_risk_gbps"}
    # `offset_km` is the distance from the cone centre to the NEAREST point
    # of the service's storm-cuttable span union (0.0 if the centre lies on
    # one), and `p_cut` is the probability that union is cut -- not a
    # representative-midpoint reading of either. A service with no
    # storm-cuttable span on its working path has no key here at all: see
    # `build_observation`.
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

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, for the trace and for step 6's prompt."""
        return {
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
        }


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
    `p_cut_region(spans, ...)`, unchanged)."""
    issuance = latest_issuance(scenario, hour)
    hour_index = scenario.hours.index(hour)

    exposure: dict[str, dict[str, dict[str, float]]] = {}
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
            p_cut = round(p_cut_service(spans, protection_leg, lat, lon,
                                        cone.width_km,
                                        scenario.damage_radius_km), 3)
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
            per_horizon[horizon] = entry
        exposure[svc["id"]] = per_horizon

    groups = _restorable_groups(
        exposure, endpoint_sites or {}, depot_site=depot_site,
        service_under_test=scenario.service_under_test)

    issuance_schedule = tuple(sorted(scenario.forecast, key=scenario.hours.index))
    deadline_hour: dict[str, str | None] = {}
    if issuance.horizons:
        latest_horizon = max(issuance.horizons, key=scenario.hours.index)
        latest_index = scenario.hours.index(latest_horizon)
        for lever in LEAD_TIME_BY_LEVER:
            d = latest_index - lead_time_hours_for(lever, scenario.lead_time_hours)
            deadline_hour[lever] = scenario.hours[d] if d >= hour_index else None
    else:
        deadline_hour = {lever: None for lever in LEAD_TIME_BY_LEVER}

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
    )
