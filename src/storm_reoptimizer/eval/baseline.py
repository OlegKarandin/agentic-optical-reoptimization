# src/storm_reoptimizer/eval/baseline.py
"""The forecast-blind fixed policy (eval design spec, "Baseline"), and a
scripted replay decider the pre-flight assertions need.

Same harness, same loop, same tools, same candidate menus. No hand-rolled
solver: the baseline calls route_service and validate_plan exactly as the
agent does, so any difference in outcome is attributable to the three
decisions and nothing else. A k-shortest-path baseline would reimplement what
the server owns (CLAUDE.md, "Explicitly out of scope") and would be a less
fair comparison, not a more honest one.

"Forecast-blind" is precise here: both variants read only the NEAREST exposed
horizon of the current issuance -- am I inside the cone, and how many hours
until then. Neither reads revisions, competing claimants, or expected
capacity at risk. That is what makes each variant emit the SAME answer to
both halves of a twin pair whose enumerated scalars are held equal, which is
in turn what makes the exactly-50% property arithmetic rather than tuning."""
from __future__ import annotations

from .decisions import (
    ConstraintDecision, ObjectiveDecision, TimingDecision, rank_by_priority,
)
from .observation import Observation, lead_time_hours_for

BASELINE_VARIANTS = ("immediate", "at_deadline")

# The (service_class -> priority ordering) dict CLAUDE.md names as the thing
# the agent must beat. Deliberately reasonable, not hobbled.
SERVICE_CLASS_PRIORITY = {
    "premium": ("dropped_traffic", "services_at_risk", "total_margin",
                "transponders", "added_latency", "spectrum_used", "max_util"),
    "standard": ("transponders", "spectrum_used", "dropped_traffic",
                 "max_util", "added_latency", "services_at_risk",
                 "total_margin"),
}
PREMIUM_GBPS = 300.0


def service_class(demand_gbps: float) -> str:
    return "premium" if demand_gbps >= PREMIUM_GBPS else "standard"


def _nearest_exposed_horizon(obs: Observation) -> tuple[str, dict] | None:
    """The soonest horizon at which the service under test lies INSIDE the
    cone -- offset within the cone's own half-width. A purely geometric test,
    with no probability reasoning, which is the whole point."""
    per_horizon = obs.exposure.get(obs.service_under_test, {})
    inside = [(h, e) for h, e in per_horizon.items()
              if e["offset_km"] <= e["width_km"] / 2.0]
    if not inside:
        return None
    return min(inside, key=lambda item: item[1]["hours_ahead"])


class ForecastBlindBaseline:
    """Timing from a fixed rule, constraints pinned to current exposure,
    objective from a fixed (service_class -> priority) table."""

    def __init__(self, variant: str) -> None:
        if variant not in BASELINE_VARIANTS:
            raise ValueError(
                f"unknown baseline variant {variant!r} "
                f"(known: {list(BASELINE_VARIANTS)})")
        self.variant = variant
        self.name = f"baseline:{variant}"

    def timing(self, obs: Observation) -> TimingDecision:
        exposed = _nearest_exposed_horizon(obs)
        if exposed is None:
            return TimingDecision(
                "wait", f"{self.name}: service under test is outside every "
                        f"cone of the {obs.issuance.issued_at} issuance")
        horizon, entry = exposed
        if self.variant == "immediate":
            return TimingDecision(
                "act", f"{self.name}: exposed at {horizon}; act on any exposure")
        deadline = lead_time_hours_for("optical_reroute", obs.lead_time_hours)
        if entry["hours_ahead"] <= deadline:
            return TimingDecision(
                "act", f"{self.name}: {entry['hours_ahead']}h to exposure at "
                       f"{horizon} meets the {deadline}h lead time")
        return TimingDecision(
            "wait", f"{self.name}: {entry['hours_ahead']}h to exposure at "
                    f"{horizon} still exceeds the {deadline}h lead time")

    def constraints(self, obs, unconstrained_menu=None):
        # Accepted and IGNORED. This policy is forecast-blind by definition
        # (see the class docstring) and Claim 1's exactly-50% arithmetic
        # depends on it staying that way.
        exposed = _nearest_exposed_horizon(obs)
        risk_groups = ([obs.risk_group_ids[exposed[0]]]
                       if exposed and exposed[0] in obs.risk_group_ids else [])
        return ConstraintDecision(
            avoid={"risk_groups": risk_groups},
            reasoning=f"{self.name}: avoid the currently-exposed risk group "
                      f"only; protection posture pinned. protected=False/"
                      f"basis=physical/level=link: this topology has no "
                      f"static SRLGs (basis='srlg' is a no-op on it), and "
                      f"protected=True would populate route_service's "
                      f"`pairs` menu, which plan_from_candidate has no path "
                      f"to translate -- it only builds a single working-leg "
                      f"reroute op.",
            protected=False, best_effort=False, basis="physical", level="link")

    def objective(self, obs: Observation, menu: dict) -> ObjectiveDecision:
        candidates = menu.get("candidates") or []
        if not candidates:
            return ObjectiveDecision(
                "infeasible", None,
                f"{self.name}: menu is empty (status={menu.get('status')})")
        svc = next((s for s in obs.services
                    if s["id"] == obs.service_under_test), None)
        demand = svc["demand_gbps"] if svc else PREMIUM_GBPS
        priority = SERVICE_CLASS_PRIORITY[service_class(demand)]
        best = rank_by_priority(candidates, priority)[0]
        return ObjectiveDecision(
            f"candidate_{best}", priority,
            f"{self.name}: fixed {service_class(demand)} priority ordering")


class ScriptedDecider:
    """Replays decisions supplied up front. Used by the pre-flight assertion
    that the two decisions a pair is NOT testing are non-binding: replay each
    half with plausible alternatives for those decisions and require the
    outcome not to move."""

    def __init__(self, name: str, *,
                 timing_by_hour: dict[str, TimingDecision] | None = None,
                 constraints_by_hour: dict[str, ConstraintDecision] | None = None,
                 objective_by_hour: dict[str, ObjectiveDecision] | None = None,
                 default_timing: TimingDecision | None = None,
                 default_constraints: ConstraintDecision | None = None,
                 default_objective: ObjectiveDecision | None = None) -> None:
        self.name = name
        self._timing = timing_by_hour or {}
        self._constraints = constraints_by_hour or {}
        self._objective = objective_by_hour or {}
        self._default_timing = default_timing or TimingDecision(
            "wait", "scripted default")
        self._default_constraints = default_constraints or ConstraintDecision(
            avoid={}, reasoning="scripted default")
        self._default_objective = default_objective or ObjectiveDecision(
            "candidate_0", None, "scripted default")

    def timing(self, obs: Observation) -> TimingDecision:
        return self._timing.get(obs.hour, self._default_timing)

    def constraints(self, obs: Observation,
                    unconstrained_menu: dict | None = None
                    ) -> ConstraintDecision:
        return self._constraints.get(obs.hour, self._default_constraints)

    def objective(self, obs: Observation, menu: dict) -> ObjectiveDecision:
        return self._objective.get(obs.hour, self._default_objective)
