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

from .cone import cut_probability, radial_offset_km
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
    #                               "width_km", "demand_gbps"}
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

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, for the trace and for step 6's prompt."""
        return {
            "scenario_id": self.scenario_id,
            "service_under_test": self.service_under_test,
            "hour": self.hour,
            "hours_remaining": self.hours_remaining,
            "issued_at": self.issuance.issued_at,
            "cones": {h: {"width_km": c.width_km, "center": c.center}
                      for h, c in self.issuance.horizons.items()},
            "exposure": self.exposure,
            "services": [dict(s) for s in self.services],
            "spares_on_hand": self.spares_on_hand,
            "lead_time_hours": {
                lever: lead_time_hours_for(lever, self.lead_time_hours)
                for lever in LEAD_TIME_BY_LEVER},
            "risk_group_ids": self.risk_group_ids,
            "iteration": self.iteration,
            "last_rejection": self.last_rejection,
            "actions_taken": [dict(a) for a in self.actions_taken],
            "spares_spent": self.spares_spent,
        }


def build_observation(
    scenario: ScenarioFile, hour: str, *,
    service_points: dict[str, tuple[float, float]],
    services: tuple[dict, ...],
    spares_on_hand: int,
    risk_group_ids: dict[str, str] | None = None,
    iteration: int = 0,
    last_rejection: dict | None = None,
    actions_taken: tuple[dict, ...] = (),
    spares_spent: int = 0,
) -> Observation:
    """The observation for one hour. `service_points` maps a service id to a
    representative (lat, lon) for its footprint -- the runner derives these
    from the topology; keeping them an argument is what lets this module stay
    pure and testable without a server."""
    issuance = latest_issuance(scenario, hour)
    hour_index = scenario.hours.index(hour)

    exposure: dict[str, dict[str, dict[str, float]]] = {}
    for svc in services:
        point = service_points.get(svc["id"])
        if point is None:
            continue
        lat, lon = point
        per_horizon: dict[str, dict[str, float]] = {}
        for horizon, cone in issuance.horizons.items():
            offset = radial_offset_km(cone.center["lat"], cone.center["lon"],
                                      lat, lon)
            per_horizon[horizon] = {
                "hours_ahead": scenario.hours.index(horizon) - hour_index,
                "offset_km": round(offset, 1),
                "width_km": cone.width_km,
                "p_cut": round(cut_probability(
                    offset, cone.width_km, scenario.damage_radius_km), 4),
                "demand_gbps": svc["demand_gbps"],
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
    )
