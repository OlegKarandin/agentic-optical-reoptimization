# src/storm_reoptimizer/eval/derived.py
"""Scalars DERIVED from an episode's real forecast geometry, as opposed to
the ones its author typed into `metadata`.

Why this module exists (whole-branch review, 2026-08-23, finding C2). Every
confound check the harness had -- `assert_shared_scalars_equal` and
`rules.py`'s `OBSERVABLE_VARS` -- read only author-DECLARED `metadata`
fields. Nothing ever cross-checked those declarations against the `forecast`
block they claim to describe, and nothing ever enumerated a quantity that is
computable from the forecast but was never declared. Two real confounds hid
in that gap for three rounds of review of T1 alone:

  * `p_cut(service_under_test)` at the exposure horizon: 0.1349 in T1a vs
    0.8834 in T1b. A bare threshold on that one number -- no comparison to
    the competing claimant, no reasoning of any kind -- answers T1 2/2. T1's
    own rehearsal claim that "no rule beats it" was only ever true of the
    scalars `rules.py` happened to enumerate.
  * WITHIN-issuance cone motion: the decision-hour issuance's own horizons
    move 126.1 km/h apart in T1a and 0.0 km/h in T1b. `cone_motion_kmh` is
    declared equal in both halves, but it describes the t0->t1 REVISION
    motion -- a different quantity entirely, which is why holding it equal
    told nobody anything about this one.

Both are pure functions of data the agent can see, so both are exploitable,
so both have to be held equal across a pair's halves. The rule is the same
one `assert_shared_scalars_equal` already states for declared scalars: the
flip must live in a RELATION between two services, never in a bare scalar
about the cone or about the service under test.

Only ONE of these -- `sut_p_cut_at_exposure_horizon` -- is fed to `rules.py`
as a thresholdable observable. The motion scalar is checked for
equality-across-halves only, deliberately: its "km/h" is measured over the
hour-LABEL index gap (`hours: [t0, t1, t2, t6]` makes t2->t6 one index step,
not four hours), which is fine for an equality test but too muddy a unit to
drive an automated threshold sweep. Equality across the halves already
implies no threshold can separate them, so nothing is lost.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..mcp_client import call_tool_json
from .cone import cut_probability, expected_capacity_at_risk_gbps, radial_offset_km
from .observation import latest_issuance
from .scenario_file import Issuance, ScenarioFile

if TYPE_CHECKING:                    # pragma: no cover
    from mcp.client import Client

# NOTE: `runner.service_points` is imported lazily inside `derived_geometry`,
# not at module scope. rules.py imports this module for DERIVED_VARS, and
# rules.py's whole selling point is that it is static -- "no server, no
# rollout, no agent". Importing the runner at module scope would drag the MCP
# client into that supposedly-pure import graph for no reason.

# The derived scalars `rules.py` threshold-sweeps, alongside its declared
# OBSERVABLE_VARS. See the module docstring on why the motion scalar is not
# in here.
DERIVED_VARS = ("sut_p_cut_at_exposure_horizon",)

# Two halves' derived geometry must agree to this many km / probability
# units. Not a modelling tolerance -- the halves are supposed to be EXACTLY
# equal on every one of these, so this is only float slack. It is
# deliberately tight enough to catch T2's documented +0.2 km width epsilon,
# which propagates into a 4.4e-4 p_cut difference.
DERIVED_TOLERANCE = 1e-6


class DerivedGeometryError(ValueError):
    """An episode whose declared metadata cannot be reconciled with its own
    forecast block -- e.g. an `exposure_horizon_hours` that names no hour."""


@dataclass(frozen=True)
class DerivedGeometry:
    """Everything this module computes for one episode, in one object so a
    failure message can name the inputs as well as the mismatched output."""
    scenario_id: str
    sut_point: tuple[float, float]
    decision_hour: str
    exposure_horizon: str
    sut_offset_km: float
    sut_p_cut_at_exposure_horizon: float
    within_issuance_cone_motion_kmh: float
    horizon_widths_km: dict[str, float]
    # (offset_km, p_cut) for the SUT at EVERY horizon the decision-hour
    # issuance publishes, not just `exposure_horizon`. Added 2026-08-23
    # (scoped re-review, Important #3): `sut_p_cut_at_exposure_horizon` alone
    # only holds the SUT's exposure equal at the ONE horizon `metadata.
    # exposure_horizon_hours` names -- a pair whose flip moved a DIFFERENT
    # horizon (the way T2's near horizon does) could in principle change the
    # SUT's exposure THERE without this check ever looking. The shipped
    # suite happens to keep every horizon's SUT exposure equal already (T2/T3
    # because the far horizon IS the SUT's own point in both halves), but
    # that was never checked, only true by construction of these three
    # specific pairs.
    sut_exposure_by_horizon: dict[str, tuple[float, float]]

    def scalars(self) -> dict[str, float]:
        """The flat name -> number view `rules.py` and the pair assertion
        both consume."""
        return {
            "sut_p_cut_at_exposure_horizon": self.sut_p_cut_at_exposure_horizon,
            "within_issuance_cone_motion_kmh": self.within_issuance_cone_motion_kmh,
        }


def decision_issuance(scenario: ScenarioFile) -> Issuance:
    """The issuance the decider actually reads at the decision hour -- the
    one whose geometry the pair's flip is supposed to live in."""
    return latest_issuance(scenario, scenario.decision_hour)


def exposure_horizon_hour(scenario: ScenarioFile) -> str:
    """The hour label `metadata.exposure_horizon_hours` names.

    `exposure_horizon_hours` counts HOUR-LABEL INDEX steps from the decision
    hour, not wall-clock hours -- the same convention `Observation.exposure`'s
    own `hours_ahead` uses (`scenario.hours.index(horizon) - hour_index`), and
    the one T2/T3's own metadata comments spell out ("t6 is hours_ahead=2 from
    t1" on a `hours: [t0, t1, t2, t6]` timeline)."""
    hours = scenario.hours
    index = hours.index(scenario.decision_hour) + int(
        scenario.metadata["exposure_horizon_hours"])
    if not 0 <= index < len(hours):
        raise DerivedGeometryError(
            f"{scenario.id}: exposure_horizon_hours="
            f"{scenario.metadata['exposure_horizon_hours']} from decision hour "
            f"{scenario.decision_hour!r} names no hour in {list(hours)}")
    horizon = hours[index]
    if horizon not in decision_issuance(scenario).horizons:
        raise DerivedGeometryError(
            f"{scenario.id}: the {decision_issuance(scenario).issued_at!r} "
            f"issuance publishes no cone at {horizon!r}, the hour "
            f"exposure_horizon_hours names "
            f"(it has {sorted(decision_issuance(scenario).horizons)})")
    return horizon


def within_issuance_cone_motion_kmh(scenario: ScenarioFile) -> float:
    """How far the decision-hour issuance's OWN cone centre travels between
    its own consecutive horizons, per hour-label index step.

    This is NOT `metadata.cone_motion_kmh`, which (in T1 and T3) describes
    the t0 -> decision-hour REVISION of a single horizon's centre. This one
    is entirely internal to one issuance: it is the "is the storm stalling or
    running?" signal a decider reads straight off the cones in front of it,
    and it was never declared and never checked until 2026-08-23. An issuance
    with a single horizon has no motion and returns 0.0."""
    issuance = decision_issuance(scenario)
    ordered = sorted(issuance.horizons.items(),
                     key=lambda item: scenario.hours.index(item[0]))
    if len(ordered) < 2:
        return 0.0
    distance = sum(
        radial_offset_km(first.center["lat"], first.center["lon"],
                         second.center["lat"], second.center["lon"])
        for (_, first), (_, second) in zip(ordered, ordered[1:]))
    span = (scenario.hours.index(ordered[-1][0])
            - scenario.hours.index(ordered[0][0]))
    return distance / span


def horizon_widths_km(scenario: ScenarioFile) -> dict[str, float]:
    """Every cone width the decision-hour issuance publishes, by horizon.
    Held equal across a pair's halves for the same reason the declared
    `cone_width_km` is: a bare width difference is a threshold, and T2's
    halves differ in NOTHING ELSE (a documented +0.2 km epsilon added only to
    satisfy the "issuances must differ at d" text requirement)."""
    return {h: c.width_km for h, c in decision_issuance(scenario).horizons.items()}


def sut_p_cut_at_exposure_horizon(
    scenario: ScenarioFile, sut_point: tuple[float, float]
) -> tuple[float, float]:
    """`(offset_km, p_cut)` for the service under test against the
    decision-hour issuance's cone at the exposure horizon.

    Exactly the arithmetic `build_observation` already puts in front of the
    decider (`radial_offset_km` -> `cut_probability`), on exactly the same
    representative point (`runner.service_points`) -- which is the point: this
    is a number the agent can read, so it is a number a one-line rule can key
    on. Unrounded, unlike the observation's 4-decimal display copy, so the
    equality check across halves is not papering over a real difference."""
    horizon = exposure_horizon_hour(scenario)
    cone = decision_issuance(scenario).horizons[horizon]
    offset = radial_offset_km(cone.center["lat"], cone.center["lon"], *sut_point)
    return offset, cut_probability(offset, cone.width_km,
                                   scenario.damage_radius_km)


def sut_exposure_by_horizon(
    scenario: ScenarioFile, sut_point: tuple[float, float]
) -> dict[str, tuple[float, float]]:
    """`(offset_km, p_cut)` for the service under test at EVERY horizon the
    decision-hour issuance publishes -- not just the one `exposure_horizon_
    hours` names. Same `radial_offset_km` -> `cut_probability` chain as
    `sut_p_cut_at_exposure_horizon`, run once per horizon, so a pair whose
    flip lives in a horizon OTHER than the declared exposure one still has
    the SUT's own exposure there held to account."""
    issuance = decision_issuance(scenario)
    out: dict[str, tuple[float, float]] = {}
    for horizon, cone in issuance.horizons.items():
        offset = radial_offset_km(cone.center["lat"], cone.center["lon"], *sut_point)
        out[horizon] = (offset, cut_probability(offset, cone.width_km,
                                                 scenario.damage_radius_km))
    return out


# The claimant-side scalars W1.2's WHOLE-SUITE check sweeps. Deliberately
# NOT in DERIVED_VARS and NOT in DerivedGeometry.scalars(): see this module's
# docstring on the two-check doctrine. Order is the sweep order.
FLIP_VARS = ("claimant_ecar_at_exposure_horizon",
             "claimant_ecar_before_exposure_horizon",
             "claimant_ecar_peak_over_horizons")


@dataclass(frozen=True)
class FlipScalars:
    """The claimant side of the comparison every gold rationale makes.

    Every gold rationale decides ONE comparison: the service under test's own
    expected capacity at risk against the aggregate expected capacity at risk
    of the other services competing for the one spare pair. The SUT side is
    held equal within each pair by construction; the claimant side IS the
    flip. Nothing enumerated it until this class existed, and a single global
    threshold on it answered all six halves (remediation spec, finding F1)."""
    scenario_id: str
    exposure_horizon: str
    earlier_horizons: tuple[str, ...]
    claimant_ecar_at_exposure_horizon: float
    claimant_ecar_before_exposure_horizon: float
    claimant_ecar_peak_over_horizons: float
    # SUT-own expected capacity at risk at EVERY horizon of the decision-hour
    # issuance. Not a claimant quantity -- it is here because W1.5's
    # flip-dominance check needs both sides of the same comparison and
    # recomputing it from DerivedGeometry.sut_exposure_by_horizon would mean
    # threading the SUT's demand a second time.
    sut_ecar_by_horizon: dict[str, float]

    def values(self) -> dict[str, float]:
        """The flat name -> number view W1.2's sweep consumes. Exactly
        FLIP_VARS, and nothing the per-pair enumeration is allowed to see."""
        return {name: getattr(self, name) for name in FLIP_VARS}


def _ecar_at_cone(scenario: ScenarioFile, cone, points, demands_gbps, *,
                  exclude: str | None) -> float:
    """Summed `p_cut x demand_gbps` over every service with a known point and
    a known demand, optionally excluding one (the SUT)."""
    total = 0.0
    for service_id, point in points.items():
        if service_id == exclude:
            continue
        demand = demands_gbps.get(service_id)
        if demand is None:
            continue
        offset = radial_offset_km(cone.center["lat"], cone.center["lon"],
                                  *point)
        total += expected_capacity_at_risk_gbps(
            cut_probability(offset, cone.width_km, scenario.damage_radius_km),
            float(demand))
    return total


def flip_scalars_from_points(
    scenario: ScenarioFile, *,
    points: dict[str, tuple[float, float]],
    demands_gbps: dict[str, float],
) -> FlipScalars:
    """The pure half of W1.1: the claimant aggregates, given every service's
    representative point and demand. Split out from `flip_scalars_for` for the
    same reason `derived_geometry_from_point` is -- so the arithmetic is
    unit-testable without a server."""
    issuance = decision_issuance(scenario)
    exposure_horizon = exposure_horizon_hour(scenario)
    exposure_index = scenario.hours.index(exposure_horizon)
    sut = scenario.service_under_test

    earlier = tuple(sorted(
        (h for h in issuance.horizons
         if scenario.hours.index(h) < exposure_index),
        key=scenario.hours.index))

    per_horizon = {
        horizon: _ecar_at_cone(scenario, cone, points, demands_gbps,
                               exclude=sut)
        for horizon, cone in issuance.horizons.items()}

    # Fail loud, exactly like `derived_geometry` does for the identical gap:
    # a missing SUT point must not silently zero `sut_ecar_by_horizon`, which
    # is precisely the side of the comparison W1.5's `assert_flip_dominates`
    # reads -- a silent zero there could make that check draw a wrong
    # conclusion with no error signal at all.
    sut_point = points.get(sut)
    if sut_point is None:
        raise DerivedGeometryError(
            f"{scenario.id}: the server reports no working-path coordinates "
            f"for service_under_test {sut!r}; its own expected capacity at "
            f"risk cannot be derived")

    sut_demand = float(demands_gbps.get(sut, 0.0))
    sut_ecar = {}
    for horizon, cone in issuance.horizons.items():
        offset = radial_offset_km(cone.center["lat"], cone.center["lon"],
                                  *sut_point)
        sut_ecar[horizon] = expected_capacity_at_risk_gbps(
            cut_probability(offset, cone.width_km, scenario.damage_radius_km),
            sut_demand)

    return FlipScalars(
        scenario_id=scenario.id,
        exposure_horizon=exposure_horizon,
        earlier_horizons=earlier,
        claimant_ecar_at_exposure_horizon=per_horizon[exposure_horizon],
        claimant_ecar_before_exposure_horizon=sum(
            per_horizon[h] for h in earlier),
        claimant_ecar_peak_over_horizons=max(per_horizon.values(),
                                             default=0.0),
        sut_ecar_by_horizon=sut_ecar)


async def flip_scalars_for(client: "Client", scenarios: list[ScenarioFile],
                           *, topology_path: str | Path
                           ) -> dict[str, FlipScalars]:
    """`{scenario_id: FlipScalars}` for a whole suite off ONE server
    connection, the same shape and the same one-state-file contract
    `derived_scalars_for` has."""
    from .runner import service_points          # lazy: see the module note

    state_files = {s.state_file for s in scenarios}
    if len(state_files) > 1:
        raise DerivedGeometryError(
            f"flip_scalars_for got episodes across {len(state_files)} "
            f"different state files ({sorted(state_files)}) but only one "
            f"server connection; call it once per state file")
    points = await service_points(client, topology_path)
    services = await call_tool_json(client, "get_services")
    demands = {s["id"]: float(s["demand_gbps"]) for s in services["services"]}
    return {s.id: flip_scalars_from_points(s, points=points,
                                           demands_gbps=demands)
            for s in scenarios}


def derived_geometry_from_point(
    scenario: ScenarioFile, sut_point: tuple[float, float]
) -> DerivedGeometry:
    """The pure half: everything above, given the service under test's
    representative point. Split out from `derived_geometry` so the geometry
    is unit-testable without a server."""
    offset, p_cut = sut_p_cut_at_exposure_horizon(scenario, sut_point)
    return DerivedGeometry(
        scenario_id=scenario.id,
        sut_point=sut_point,
        decision_hour=scenario.decision_hour,
        exposure_horizon=exposure_horizon_hour(scenario),
        sut_offset_km=offset,
        sut_p_cut_at_exposure_horizon=p_cut,
        within_issuance_cone_motion_kmh=within_issuance_cone_motion_kmh(scenario),
        horizon_widths_km=horizon_widths_km(scenario),
        sut_exposure_by_horizon=sut_exposure_by_horizon(scenario, sut_point))


async def derived_geometry(client: "Client", scenario: ScenarioFile, *,
                           topology_path: str | Path) -> DerivedGeometry:
    """MCP-backed: read the service under test's REAL representative point off
    a live server loaded with this half's own state file, then derive.

    This is deliberately not computed from the state JSON directly. The
    representative point is the midpoint of the service's WORKING PATH node
    coordinates, and reconstructing a working path from the persisted model
    would reimplement what the server owns (CLAUDE.md's hard seam). It is also
    why this function is async and takes a client, the same shape
    `assert_menus_identical` and `assert_each_baseline_variant_ties` already
    have."""
    from .runner import service_points          # lazy: see the module note

    points = await service_points(client, topology_path)
    point = points.get(scenario.service_under_test)
    if point is None:
        raise DerivedGeometryError(
            f"{scenario.id}: the server reports no working-path coordinates "
            f"for service_under_test {scenario.service_under_test!r}; its "
            f"exposure geometry cannot be derived")
    return derived_geometry_from_point(scenario, point)


async def derived_scalars_for(client: "Client", scenarios: list[ScenarioFile],
                              *, topology_path: str | Path
                              ) -> dict[str, dict[str, float]]:
    """`{scenario_id: DerivedGeometry.scalars()}` for a whole suite, off ONE
    server connection -- the shape `rules.observables` and
    `assertions.assert_no_single_variable_rule_solves` consume.

    Every shipped episode names the same `state_file`, so the service under
    test's representative point is the same in all of them and reading it once
    is not a shortcut, it is the same number seven times. That is asserted
    rather than assumed: a future episode on a different state file would
    silently get the wrong coordinates, which is precisely the class of
    error this module exists to make impossible."""
    from .runner import service_points          # lazy: see the module note

    state_files = {s.state_file for s in scenarios}
    if len(state_files) > 1:
        raise DerivedGeometryError(
            f"derived_scalars_for got episodes across {len(state_files)} "
            f"different state files ({sorted(state_files)}) but only one "
            f"server connection; call it once per state file")
    points = await service_points(client, topology_path)
    out: dict[str, dict[str, float]] = {}
    for scenario in scenarios:
        point = points.get(scenario.service_under_test)
        if point is None:
            raise DerivedGeometryError(
                f"{scenario.id}: the server reports no working-path "
                f"coordinates for service_under_test "
                f"{scenario.service_under_test!r}")
        out[scenario.id] = derived_geometry_from_point(scenario, point).scalars()
    return out
