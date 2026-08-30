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
    p_cut_region,
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


def _horizon_totals(exposure: dict, service_under_test: str
                    ) -> dict[str, dict[str, float]]:
    """Per horizon, the two operands every gold rationale in the suite
    compares: the service under test's own expected capacity at risk, and the
    SUMMED expected capacity at risk of everyone else competing for the same
    spare pair (remediation spec, W3.2).

    Summed over EVERY service with a representative point, not merely the
    ones a projection later chooses to list -- the aggregate is a fact about
    the network, and a total that silently covered only the visible rows
    would be worse than no total at all.

    Deliberately NOT shared with derived._ecar_at_cone, which computes the
    same shape from the UNROUNDED cut probability. That one feeds FLIP_VARS
    and assert_no_global_policy_solves_the_suite, whose interleave margins
    are under 1 G; this one is agent-facing and must agree with the rounded
    per-row numbers printed beside it. Two readers, two roundings, one
    quantity -- keep them apart."""
    totals: dict[str, dict[str, float]] = {}
    for svc_id, per_horizon in exposure.items():
        key = ("sut_ecar_gbps" if svc_id == service_under_test
               else "non_sut_total_ecar_gbps")
        for horizon, entry in per_horizon.items():
            slot = totals.setdefault(
                horizon, {"sut_ecar_gbps": 0.0,
                          "non_sut_total_ecar_gbps": 0.0})
            slot[key] += entry["expected_capacity_at_risk_gbps"]
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
    # horizon hour -> {"sut_ecar_gbps", "non_sut_total_ecar_gbps"}: the two
    # operands of the comparison every gold rationale makes. Present on every
    # Observation and in the trace; whether the AGENT is shown it is
    # project_observation's decision, gated per arm (remediation spec, W3.2).
    # Defaulted so the hand-built Observations in tests/eval/test_agent.py and
    # tests/eval/test_baseline.py keep constructing.
    horizon_totals: dict[str, dict[str, float]] = field(default_factory=dict)

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
            "risk_group_ids": self.risk_group_ids,
            "iteration": self.iteration,
            "last_rejection": self.last_rejection,
            "actions_taken": [dict(a) for a in self.actions_taken],
            "spares_spent": self.spares_spent,
            "horizon_totals": self.horizon_totals,
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
) -> Observation:
    """The observation for one hour. `service_spans` maps a service id to the
    spans of its working path that the event's own filter admits -- the
    runner derives these from the server's view of the path plus the local
    topology's coordinates and mount types; keeping them an argument is what
    lets this module stay pure and testable without a server, exactly as
    `service_points` did."""
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
        per_horizon: dict[str, dict[str, float]] = {}
        for horizon, cone in issuance.horizons.items():
            lat, lon = cone.center["lat"], cone.center["lon"]
            offset = nearest_span_offset_km(spans, lat, lon)
            p_cut = round(p_cut_region(spans, lat, lon, cone.width_km,
                                       scenario.damage_radius_km), 4)
            per_horizon[horizon] = {
                "hours_ahead": scenario.hours.index(horizon) - hour_index,
                # Distance from the cone centre to the NEAREST CUTTABLE SPAN,
                # not to a representative midpoint -- so
                # `offset_km <= width_km / 2` reads "a span this storm can cut
                # is inside the footprint".
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
                # Derived from the ROUNDED p_cut one line above, NOT from the
                # raw probability, so the payload is internally consistent: a
                # reader who multiplies the two numbers shown gets the number
                # shown. That makes it deliberately NOT bit-identical to
                # derived.py's FlipScalars, which use the unrounded value and
                # feed the static suite assertions -- those must not move.
                "expected_capacity_at_risk_gbps": round(
                    expected_capacity_at_risk_gbps(
                        p_cut, svc["demand_gbps"]), 3),
            }
        exposure[svc["id"]] = per_horizon

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
        risk_group_ids=dict(risk_group_ids or {}),
        iteration=iteration,
        last_rejection=last_rejection,
        actions_taken=tuple(actions_taken),
        spares_spent=spares_spent,
        horizon_totals=_horizon_totals(exposure, scenario.service_under_test),
    )
