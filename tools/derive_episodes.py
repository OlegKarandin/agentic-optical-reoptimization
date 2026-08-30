# tools/derive_episodes.py
"""Task 13 (exposure-and-depot plan): derive candidate cone geometry for the
four shipped episode pairs (D1, T1, T2, T3) against the REAL satna-homed
claimant family (`claimant-satna-jabalpur-fwd`/`-rev`, pinned in Task 7 --
see docs/superpowers/plans/notes/2026-08-30-claimant-family.md), and solve
the anti-threshold tuning JOINTLY across all three twin pairs at once.

Runs in THIS repo's env (storm-reoptimizer) against a LIVE
multilayer-optical-mcp server loaded with the real eval state
(`eval/states/loaded-s17.json`) -- it is NOT the offline-build exception and
imports nothing from multilayer_optical_network. Not a pytest suite: this is
exploratory authoring tooling in the same spirit as tools/probe_claimants.py
and tools/probe_episode.py (design doc's own precedent) -- the "test" is
running it and reading the printed report.

**Why this exists (Task 12's own finding, carried forward).** The scenario
YAMLs shipped today still name the OLD placeholder claimants
(T1: d0029/d0348; T2: eight kanpur/agra services; T3: d0462/d0212/d0363) --
none of which terminate at `satna` (the depot site) under the corrected,
per-site-ledger model, so none of them can honestly compete for this
episode's one spare transponder pair. The REAL satna-homed claimants are
`claimant-satna-jabalpur-fwd`/`-rev` (satna<->jabalpur, 100 Gbps each,
co-terminating -- one new lightpath restores both directions). This tool
finds cone geometry that genuinely exposes THAT corridor at the right
horizon for each pair, re-derives every FLIP_VARS member against it, and
sweeps `assertions.assert_no_global_policy_solves_the_suite` (plus a
from-scratch exhaustive best-achievable-score sweep, since that function
only tells you PASS/FAIL, not the score) over all three pairs TOGETHER.
Task 14 freezes whatever this tool settles on into the real scenario YAMLs;
this tool does not touch them.

**Frame correctness (do not reintroduce either historical bug --
derived.py's own module docstring, "The one binding condition").** A cone
centre at bearing `b` (compass degrees, clockwise from north -- 0=N, 90=E,
matching events.geo.circle_polygon's own `theta` convention) and radial
distance `d` km from an origin (lat0, lon0) is placed CLOSED FORM, in the
offset function's own frame:

    lat = lat0 + degrees(d * cos(b) / R)
    lon = lon0 + degrees(d * sin(b) / (R * cos(radians(lat))))   -- note: the
                                                    JUST-COMPUTED lat, not lat0
    R = events.geo.EARTH_RADIUS_KM = 6371.0                       -- not 6371.0088

Verified against the two already-shipped T2 near centres (300.0 km at
bearings 290.0/296.0 deg from storm-svc-1's own point): this formula
reproduces both to the full float, matching T2a.yaml/T2b.yaml's own comments
bit for bit.

**Why "same radius, different bearing" (T2's proven trick) does NOT
transfer to T1 unchanged.** T2/T3's near-cone construction holds
storm-svc-1's own exposure equal across halves FOR FREE, because both
bearings sit ~250-300 km from storm-svc-1's own point and its real
storm-cuttable span (`satna<->rewa`, ~47 km long) is far enough away that
`p_cut` rounds to 0.0 (well under DERIVED_TOLERANCE) at EVERY bearing tried
-- confirmed empirically below, not assumed. T1 needs the OPPOSITE: a
non-trivial, EQUAL, NONZERO p_cut at the flip horizon. Measured directly
(see `_investigate_t1_bearing_sensitivity` below, or just rerun this module
with --explore): at a FIXED radius from storm-svc-1's own averaged point,
`nearest_span_offset_km` against the REAL span varies by up to ~90 km across
bearing alone, because the averaged point (which folds in allahabad, on the
buried leg) sits well off the actual aerial span's own line. So "same
radius, arbitrary bearing" does NOT hold p_cut equal for T1 the way it does
for T2/T3's near horizon -- equal RADIUS is not equal EXPOSURE here. T1's
two centres are therefore placed at DIFFERENT radii by construction: one
bearing is fixed pointing at the real claimant corridor, and the sibling
bearing (pointing away from it) has its radius solved by a plain grid+bisect
search so the REAL, server-derived SUT p_cut matches the first half's to
below 1e-9 -- comfortably inside DERIVED_TOLERANCE (1e-6) -- while the
claimant corridor's own exposure stays near zero. The "closed form, not an
iterative solver" instruction is about the FRAME (never approximate the
frame with a Newton loop that absorbs the wrong Earth radius, per derived.py's
own cautionary history) -- solving for which RADIUS along an already-correct
frame hits a target p_cut is a different, harmless kind of search, done in
the metric the model actually uses (real p_cut off the real span), not in a
proxy that can drift from it.

Run (from repo root, this repo's own env):

    C:/Users/olegk/miniconda3/envs/storm-reoptimizer/python.exe tools/derive_episodes.py \\
        --topology src/storm_reoptimizer/data/toy_india_topology.json \\
        --state eval/states/loaded-s17.json \\
        --server-command '["C:/Users/olegk/miniconda3/envs/multilayer-optical-mcp/python.exe", "-c", "import sys; sys.argv = [\\'multilayer-optical-mcp\\', *sys.argv[1:]]; from multilayer_optical_mcp.server import main; main()"]'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from storm_reoptimizer.eval.cone import (               # noqa: E402
    Segment, cone_polygon, nearest_span_offset_km, p_cut_region,
)
from storm_reoptimizer.eval.derived import (             # noqa: E402
    FLIP_VARS, DerivedGeometry, FlipScalars, _ecar_at_cone,
    derived_geometry_from_spans, flip_scalars_from_spans,
)
from storm_reoptimizer.eval.rules import split_points     # noqa: E402
from storm_reoptimizer.eval.runner import ServiceGeometry, service_geometry  # noqa: E402
from storm_reoptimizer.eval.scenario_file import (        # noqa: E402
    ConeAtHorizon, Gold, Issuance, ScenarioFile,
)
from storm_reoptimizer.events.geo import EARTH_RADIUS_KM  # noqa: E402
from storm_reoptimizer.mcp_client import call_tool_json, connect_server  # noqa: E402

SUT = "storm-svc-1"
CLAIMANT_FWD = "claimant-satna-jabalpur-fwd"
CLAIMANT_REV = "claimant-satna-jabalpur-rev"

DEFAULT_TOPOLOGY = "src/storm_reoptimizer/data/toy_india_topology.json"
DEFAULT_STATE = "eval/states/loaded-s17.json"
# This workspace's own workaround (tests/conftest.py's local_server_command):
# the sibling server repo's console script never gets a .exe here (Cyrillic
# path trips the editable-install bug), so the server is launched by
# invoking its main() directly via `python -c`, in the SIBLING conda env.
DEFAULT_SERVER_PYTHON = (
    r"C:\Users\olegk\miniconda3\envs\multilayer-optical-mcp\python.exe")
DEFAULT_SERVER_COMMAND = [
    DEFAULT_SERVER_PYTHON, "-c",
    "import sys; sys.argv = ['multilayer-optical-mcp', *sys.argv[1:]]; "
    "from multilayer_optical_mcp.server import main; main()",
]

SPARE_ACTIONS = ("spend", "conserve")


# --------------------------------------------------------------------------
# Frame: the closed-form centre placement, in the offset function's own frame
# --------------------------------------------------------------------------

def place_centre(origin_lat: float, origin_lon: float, bearing_deg: float,
                 d_km: float, *, R: float = EARTH_RADIUS_KM) -> tuple[float, float]:
    """The one closed form this whole tool is built on. `bearing_deg` is
    compass bearing (0=N, 90=E, clockwise), matching events.geo.circle_
    polygon's own `theta` convention (verified: `radial_offset_km` of the
    result back to the origin reproduces `d_km` to ~1e-13, and this
    reproduces T2a/T2b's own shipped near centres bit for bit at bearings
    290.0/296.0, d=300.0)."""
    b = math.radians(bearing_deg)
    lat = origin_lat + math.degrees(d_km * math.cos(b) / R)
    lon = origin_lon + math.degrees(d_km * math.sin(b) / (R * math.cos(math.radians(lat))))
    return lat, lon


def radial_offset_km(lat0: float, lon0: float, lat1: float, lon1: float,
                     *, R: float = EARTH_RADIUS_KM) -> float:
    dlat = math.radians(lat1 - lat0)
    dlon = math.radians(lon1 - lon0) * math.cos(math.radians(lat0))
    return R * math.hypot(dlat, dlon)


def solve_radius_for_target_pcut(
    origin_lat: float, origin_lon: float, bearing_deg: float,
    target_pcut: float, spans: tuple[Segment, ...], width_km: float,
    damage_radius_km: float, *, lo_km: float = 1.0, hi_km: float = 500.0,
    coarse_step_km: float = 1.0,
) -> float:
    """Grid-then-refine search (NOT a Newton loop absorbing a wrong
    constant -- this only searches the free radius parameter along an
    already-correct closed-form frame) for the radius along `bearing_deg`
    from (origin_lat, origin_lon) whose REAL, server-derived p_cut against
    `spans` matches `target_pcut`. p_cut is not perfectly monotonic in
    radius in general (radial_offset_km to the segment can wobble slightly
    off-axis), so this does a coarse linear scan for the sign change nearest
    the target, then bisects that bracket -- robust without assuming
    monotonicity globally."""
    def pcut_at(d: float) -> float:
        lat, lon = place_centre(origin_lat, origin_lon, bearing_deg, d)
        return p_cut_region(spans, lat, lon, width_km, damage_radius_km)

    n_steps = int((hi_km - lo_km) / coarse_step_km)
    prev_d = lo_km
    prev_f = pcut_at(lo_km) - target_pcut
    bracket = None
    for i in range(1, n_steps + 1):
        d = lo_km + i * coarse_step_km
        f = pcut_at(d) - target_pcut
        if prev_f == 0.0:
            return prev_d
        if (prev_f > 0) != (f > 0):
            bracket = (prev_d, d)
            break
        prev_d, prev_f = d, f
    if bracket is None:
        raise ValueError(
            f"solve_radius_for_target_pcut: no sign change for target "
            f"p_cut={target_pcut} along bearing {bearing_deg} in "
            f"[{lo_km}, {hi_km}] km (p_cut range observed does not bracket "
            f"the target -- widen the search range or pick another bearing)")
    a, b_ = bracket
    fa = pcut_at(a) - target_pcut
    for _ in range(80):
        mid = (a + b_) / 2.0
        fm = pcut_at(mid) - target_pcut
        if abs(fm) < 1e-12:
            return mid
        if (fa > 0) == (fm > 0):
            a, fa = mid, fm
        else:
            b_ = mid
    return (a + b_) / 2.0


# --------------------------------------------------------------------------
# Candidate cone specs -> in-memory ScenarioFile (never written to disk)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ConeSpec:
    """One horizon's cone, expressed either as an absolute (lat, lon) --
    for horizons that must stay BYTE-IDENTICAL to the shipped YAML (T2/T3's
    far horizon) -- or as (bearing_deg, radius_km) from a named origin,
    resolved against the live SUT point at evaluation time."""
    width_km: float
    lat: float | None = None
    lon: float | None = None
    bearing_deg: float | None = None
    radius_km: float | None = None
    origin: str = "sut"     # "sut" -- the only origin this tool needs


@dataclass(frozen=True)
class IssuanceSpec:
    horizons: dict[str, ConeSpec]


@dataclass(frozen=True)
class HalfSpec:
    id: str
    pair: str | None
    hours: tuple[str, ...]
    decision_hour: str
    exposure_horizon_hours: int
    damage_radius_km: float
    forecast: dict[str, IssuanceSpec]
    gold_spare_action: str


def resolve_cone(spec: ConeSpec, sut_point: tuple[float, float]) -> ConeAtHorizon:
    if spec.lat is not None:
        lat, lon = spec.lat, spec.lon
    else:
        origin_lat, origin_lon = sut_point
        lat, lon = place_centre(origin_lat, origin_lon, spec.bearing_deg,
                                spec.radius_km)
    return ConeAtHorizon(cone=cone_polygon(lat, lon, spec.width_km),
                         width_km=spec.width_km, center={"lat": lat, "lon": lon})


_DUMMY_GOLD = Gold(survived=(), max_spares_wasted=0, decision_at_t0="wait",
                   label="n/a", rationale="derive_episodes.py candidate; not gold")


def build_scenario(spec: HalfSpec, sut_point: tuple[float, float]) -> ScenarioFile:
    forecast = {
        issued_at: Issuance(
            issued_at=issued_at,
            horizons={h: resolve_cone(c, sut_point)
                     for h, c in issuance.horizons.items()})
        for issued_at, issuance in spec.forecast.items()
    }
    return ScenarioFile(
        id=spec.id, pair=spec.pair, seed=17, state_file=DEFAULT_STATE,
        service_under_test=SUT, track="hudhud", hours=spec.hours,
        decision_hour=spec.decision_hour, lead_time_hours=1, spares_on_hand=1,
        depot_site="satna", spare_inventory={"satna": 1},
        damage_radius_km=spec.damage_radius_km, reference_avoid={},
        forecast=forecast, realized={}, gold=_DUMMY_GOLD, flip_variable=(),
        metadata={"exposure_horizon_hours": spec.exposure_horizon_hours,
                 "gold_spare_action": spec.gold_spare_action,
                 "spares_on_hand": 1})


# --------------------------------------------------------------------------
# Evaluation: derived geometry + flip scalars + two extra summaries, per half
# --------------------------------------------------------------------------

@dataclass
class HalfResult:
    scenario: ScenarioFile
    derived: DerivedGeometry
    flip: FlipScalars
    per_horizon_network_sum: dict[str, float]
    claimant_group_p_cut_by_horizon: dict[str, float]
    extra_earliest_horizon: float
    extra_median: float


def evaluate_half(spec: HalfSpec, geometry: ServiceGeometry,
                  demands: dict[str, float]) -> HalfResult:
    scenario = build_scenario(spec, geometry.points[SUT])
    sut_spans = geometry.cuttable_spans[SUT]
    derived = derived_geometry_from_spans(scenario, sut_spans)
    flip = flip_scalars_from_spans(
        scenario, spans=geometry.cuttable_spans, demands_gbps=demands,
        endpoint_sites=geometry.endpoint_sites)

    issuance = scenario.forecast[scenario.decision_hour]
    per_horizon = {h: _ecar_at_cone(scenario, cone, geometry.cuttable_spans,
                                    demands, exclude=SUT)
                   for h, cone in issuance.horizons.items()}
    claim_spans = geometry.cuttable_spans[CLAIMANT_FWD]
    claim_p_cut = {h: p_cut_region(claim_spans, cone.center["lat"],
                                   cone.center["lon"], cone.width_km,
                                   scenario.damage_radius_km)
                  for h, cone in issuance.horizons.items()}

    hours = scenario.hours
    earliest_horizon = min(per_horizon, key=hours.index)
    return HalfResult(
        scenario=scenario, derived=derived, flip=flip,
        per_horizon_network_sum=per_horizon,
        claimant_group_p_cut_by_horizon=claim_p_cut,
        extra_earliest_horizon=per_horizon[earliest_horizon],
        extra_median=statistics.median(per_horizon.values()))


EXTRA_SUMMARY_NAMES = ("claimant_ecar_at_earliest_horizon",
                       "claimant_ecar_median_over_horizons")


def all_flip_values(result: HalfResult) -> dict[str, float]:
    values = dict(result.flip.values())
    values["claimant_ecar_at_earliest_horizon"] = result.extra_earliest_horizon
    values["claimant_ecar_median_over_horizons"] = result.extra_median
    return values


ALL_SWEPT_VARS = FLIP_VARS + EXTRA_SUMMARY_NAMES


# --------------------------------------------------------------------------
# Scoring: exhaustive threshold sweep, both orientations, plus tie/interleave
# diagnosis -- mirrors assertions.assert_no_global_policy_solves_the_suite's
# own logic (same split_points, same two orientations) but reports a SCORE
# and a witness instead of only raising on a full 6/6 solve.
# --------------------------------------------------------------------------

@dataclass
class SweepReport:
    var: str
    values: dict[str, float]          # half id -> value
    actions: dict[str, str]           # half id -> "spend"/"conserve"
    best_score: float
    best_threshold: float | None
    best_lo_action: str | None
    misclassified: tuple[str, ...]
    ties: tuple[tuple[str, str, float], ...]   # (id_a, id_b, |delta|)
    sorted_order: tuple[tuple[str, float, str], ...]  # (id, value, action)


def find_ties(values: dict[str, float], actions: dict[str, str],
             tol: float = 1e-6) -> tuple[tuple[str, str, float], ...]:
    ids = sorted(values)
    ties = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if actions[a] != actions[b] and abs(values[a] - values[b]) <= tol:
                ties.append((a, b, abs(values[a] - values[b])))
    return tuple(ties)


def sweep_variable(var: str, values: dict[str, float],
                   actions: dict[str, str]) -> SweepReport:
    ids = list(values)
    thresholds = split_points([values[i] for i in ids]) or [
        sum(values.values()) / len(values) if values else 0.0]
    best_score, best_threshold, best_lo, best_mis = -1.0, None, None, ()
    n = len(ids)
    for threshold in thresholds:
        for lo_action in SPARE_ACTIONS:
            hi_action = "conserve" if lo_action == "spend" else "spend"
            predicted = {i: (lo_action if values[i] < threshold else hi_action)
                        for i in ids}
            mis = tuple(sorted(i for i in ids if predicted[i] != actions[i]))
            score = (n - len(mis)) / n if n else 0.0
            if score > best_score:
                best_score, best_threshold, best_lo, best_mis = (
                    score, threshold, lo_action, mis)
    sorted_order = tuple(sorted(((i, values[i], actions[i]) for i in ids),
                                key=lambda t: t[1]))
    return SweepReport(
        var=var, values=dict(values), actions=dict(actions),
        best_score=best_score, best_threshold=best_threshold,
        best_lo_action=best_lo, misclassified=best_mis,
        ties=find_ties(values, actions), sorted_order=sorted_order)


def transitions(sorted_order: tuple[tuple[str, float, str], ...]) -> list[
        tuple[str, float, str, str, float, str]]:
    """Every adjacent pair (by value) whose action differs -- a perfectly
    separable column has exactly one; more than one is what "interleaved"
    means, mechanically. Diagnostic display ONLY: the smallest gap in this
    list is NOT, in general, the real binding edge (see `bisect_binding_
    edge`'s docstring) -- two values can swap position on the SAME side of
    the actual separating boundary without restoring separability at all,
    so their adjacent gap can understate the true margin substantially.
    Reported here purely so a reader can see the interleave shape; the
    actual margin figure this module reports is computed by `bisect_
    binding_edge`, against the real scoring rule, not read off this list."""
    out = []
    for (id_a, v_a, act_a), (id_b, v_b, act_b) in zip(
            sorted_order, sorted_order[1:]):
        if act_a != act_b:
            out.append((id_a, v_a, act_a, id_b, v_b, act_b))
    return out


def _is_solved_by_some_threshold(values: dict[str, float],
                                 actions: dict[str, str]) -> bool:
    """True iff some global threshold, under some orientation, predicts
    every half's action correctly -- exactly `sweep_variable`'s own scoring
    rule (same `rules.split_points`, same two orientations), exposed as a
    boolean oracle so `bisect_binding_edge` can bisect against it for a
    variable `assertions.assert_no_global_policy_solves_the_suite` does not
    itself enumerate (the two extra summaries, which are not in
    `derived.FLIP_VARS`)."""
    ids = list(values)
    n = len(ids)
    if n == 0:
        return False
    for threshold in split_points([values[i] for i in ids]):
        for lo_action in SPARE_ACTIONS:
            hi_action = "conserve" if lo_action == "spend" else "spend"
            predicted = {i: (lo_action if values[i] < threshold else hi_action)
                        for i in ids}
            if all(predicted[i] == actions[i] for i in ids):
                return True
    return False


def arithmetic_overlap(values: dict[str, float],
                       actions: dict[str, str]) -> tuple[float, float]:
    """The two candidate overlaps -- `(lo=spend overlap, lo=conserve
    overlap)` -- computed in closed form, no bisection: `max(spend values)
    - min(conserve values)` and its mirror. Whichever is smaller (and
    positive) is the orientation actually closest to separable, and its
    value is the TRUE binding margin for a column with exactly one
    out-of-place value on that side. Used here ONLY as an independent
    cross-check against the live-bisected margin below -- not as a
    replacement for it, since this formula assumes there is exactly one
    value causing the overlap, which `bisect_binding_edge`'s live oracle
    call does not need to assume."""
    spend_vals = [v for h, v in values.items() if actions[h] == "spend"]
    conserve_vals = [v for h, v in values.items() if actions[h] == "conserve"]
    overlap_lo_spend = max(spend_vals) - min(conserve_vals)
    overlap_lo_conserve = max(conserve_vals) - min(spend_vals)
    return overlap_lo_spend, overlap_lo_conserve


def bisect_binding_edge(values: dict[str, float], actions: dict[str, str],
                        solved_fn, *, perturb_id: str) -> dict:
    """The REAL binding edge for an interleaved variable, found by bisecting
    `solved_fn` (a callable `(values_dict, actions) -> bool`, True iff some
    global threshold/orientation solves the whole column) -- NOT read off
    the smallest adjacent gap in sorted order (`transitions()`), which a
    live review found to be WRONG for this exact geometry (up to 2.2x off):
    moving the misclassified half's value by the smallest adjacent gap does
    not, in general, restore separability, because the value it actually
    needs to clear is the GLOBAL extreme of the opposite-action set, not
    merely its nearest sorted neighbour.

    `perturb_id` is the ONE half to move -- normally `SweepReport.
    misclassified[0]`, the single half the exhaustive sweep already
    identified as the one no split point classifies correctly. Its
    direction is inferred from its OWN declared action: a 'spend' half
    sitting too high is shrunk down; a 'conserve' half sitting too low is
    grown up. (A symmetric "move the opposite extreme instead" variant was
    tried and abandoned: for `peak_over_horizons`/`claimant_ecar_at_
    earliest_horizon` in this suite, TWO conserve values sit below the
    offending spend value, so growing only the smaller one does not restore
    separability at all -- confirmed live, the growing bisection never
    converges to "solved" no matter how far it is pushed. Perturbing the
    actual misclassified half, in its own natural direction, is the one
    operation guaranteed to work regardless of how many other values are on
    the "wrong" side of it.)

    Cross-checked, not just trusted: the caller should also compute
    `arithmetic_overlap` on the SAME (values, actions) and confirm it
    agrees with the bisected margin to close tolerance -- see the printed
    "cross-check" line in `run()`."""
    real_action = actions[perturb_id]
    direction = -1 if real_action == "spend" else 1
    base = values[perturb_id]

    def solved_with(v: float) -> bool:
        vv = dict(values)
        vv[perturb_id] = v
        return solved_fn(vv, actions)

    assert not solved_with(base), (
        f"baseline value for {perturb_id!r} already solves the suite -- not "
        f"a valid starting point for a binding-edge search")
    bound = ((min(values.values()) - abs(base) - 1.0) if direction < 0
            else (max(values.values()) + abs(base) + 1.0))
    assert solved_with(bound), (
        f"{bound} does not solve the suite either -- widen the bound")
    lo, hi = (bound, base) if direction < 0 else (base, bound)
    # invariant: solved_with(lo if direction<0 else hi) is always the "far"
    # (solved) end; narrow until lo/hi converge on the flip point.
    for _ in range(200):
        mid = (lo + hi) / 2.0
        mid_solved = solved_with(mid)
        if direction < 0:
            if mid_solved:
                lo = mid
            else:
                hi = mid
        else:
            if mid_solved:
                hi = mid
            else:
                lo = mid
    flip = (lo + hi) / 2.0
    margin = abs(base - flip)
    return {"perturb_id": perturb_id, "direction": direction,
           "base_value": base, "flip_value": flip, "margin": margin}


def blocking_kind(report: SweepReport) -> str:
    if report.ties:
        return "TIE"
    if report.best_score < 1.0:
        return "INTERLEAVE"
    return "NOT BLOCKED -- a global threshold solves this variable"


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_half_summary(result: HalfResult) -> None:
    s = result.scenario
    d = result.derived
    print(f"\n=== {s.id} (pair={s.pair}) ===")
    print(f"  decision_hour={s.decision_hour} exposure_horizon="
          f"{d.exposure_horizon} gold_spare_action="
          f"{s.metadata['gold_spare_action']}")
    print(f"  SUT offset_km={d.sut_offset_km:.6f} "
          f"p_cut={d.sut_p_cut_at_exposure_horizon:.8f} -> ecar="
          f"{d.sut_p_cut_at_exposure_horizon * 300.0:.3f} G (300G demand)")
    issuance = s.forecast[s.decision_hour]
    for h, (off, pc) in sorted(d.sut_exposure_by_horizon.items(),
                               key=lambda kv: s.hours.index(kv[0])):
        cone = issuance.horizons[h]
        print(f"    SUT @ {h}: centre=({cone.center['lat']:.6f},"
              f"{cone.center['lon']:.6f}) width={cone.width_km:.1f} "
              f"offset={off:9.3f} km  p_cut={pc:.8f}")
    for h, pc in sorted(result.claimant_group_p_cut_by_horizon.items(),
                        key=lambda kv: s.hours.index(kv[0])):
        group = 2 * pc * 100.0    # fwd + rev, 100G each, identical p_cut
        print(f"    claimant-satna-jabalpur @ {h}: p_cut={pc:.8f} "
              f"group_ecar={group:.3f} G")
    print(f"  FlipScalars: at_exposure={result.flip.claimant_ecar_at_exposure_horizon:.3f} "
          f"before={result.flip.claimant_ecar_before_exposure_horizon:.3f} "
          f"peak={result.flip.claimant_ecar_peak_over_horizons:.3f} "
          f"min={result.flip.claimant_ecar_min_over_horizons:.3f} "
          f"largest_group={result.flip.largest_restorable_group_ecar_gbps:.3f}")
    print(f"  extras: earliest_horizon={result.extra_earliest_horizon:.3f} "
          f"median={result.extra_median:.3f}")


def print_sweep_report(report: SweepReport) -> None:
    kind = blocking_kind(report)
    print(f"\n--- {report.var} ---  [{kind}]")
    for id_, value, action in report.sorted_order:
        print(f"    {id_:6s} {value:12.4f}  {action:9s}")
    print(f"  best achievable score: {report.best_score:.3f} "
          f"({int(round(report.best_score * len(report.values)))}/"
          f"{len(report.values)}) at threshold={report.best_threshold} "
          f"lo->{report.best_lo_action}")
    if report.misclassified:
        print(f"  misclassified at best split: {list(report.misclassified)}")
    if report.ties:
        for a, b, delta in report.ties:
            print(f"  TIE: {a} == {b} (|delta|={delta:.3g}), opposite actions "
                  f"-- no threshold can separate them")
    trans = transitions(report.sorted_order)
    if len(trans) > 1:
        print(f"  {len(trans)} action-transitions in sorted order (>1 means "
              f"interleaved, no single split works). SHAPE ONLY -- the "
              f"'adjacent_gap' column below is NOT the binding margin (see "
              f"bisect_binding_edge's docstring); the real margin is printed "
              f"separately, below the sweep, via a live bisection:")
        for id_a, v_a, act_a, id_b, v_b, act_b in trans:
            print(f"    {id_a}({act_a}, {v_a:.3f}) | adjacent_gap="
                  f"{v_b - v_a:.3f} | {id_b}({act_b}, {v_b:.3f})")


# --------------------------------------------------------------------------
# Candidate geometry. Edit these constants to iterate; re-run the module.
# --------------------------------------------------------------------------

# Bearing (compass, from storm-svc-1's own averaged point) toward the real
# claimant corridor's own representative point (measured directly: distance
# 144.72 km, bearing 221.12 deg -- see this module's --explore output). Used
# as the AIM DIRECTION for every "toward the claimant" cone below; the exact
# bearing used per pair is tuned within a neighbourhood of this value against
# each pair's own width/radius so the corridor's OWN p_cut lands where each
# pair's rationale needs it, not to hit this exact number.
CLAIMANT_BEARING_APPROX = 221.12

# D1 -- UNCHANGED from the shipped YAML (Step 2: confirm only, no retune).
D1_SPEC = HalfSpec(
    id="D1", pair=None, hours=("t0", "t1"), decision_hour="t0",
    exposure_horizon_hours=1, damage_radius_km=74.0,
    forecast={"t0": IssuanceSpec({"t1": ConeSpec(
        width_km=15.0, lat=24.58333, lon=80.83333)})},   # satna itself
    gold_spare_action="spend")

# T1 -- single t+3 horizon at the decision hour. Bearing A points at the real
# claimant corridor; bearing B points away from it, at a radius solved (see
# solve_radius_for_target_pcut) so the SUT's own p_cut matches bearing A's
# exactly. Both use width=90 (kept from the discarded T1 draft's own choice
# -- irrelevant to the retune, held equal across the pair by construction
# since it's a single shared constant below).
T1_WIDTH_KM = 90.0
T1_BEARING_TOWARD = 195.0     # near the claimant corridor's own bearing
T1_RADIUS_TOWARD = 150.0      # km from storm-svc-1's own point
T1_BEARING_AWAY = 40.0        # roughly opposite; claimant p_cut ~0 there

T1A_SPEC = HalfSpec(
    id="T1a", pair="T1", hours=("t0", "t1", "t2", "t3"), decision_hour="t1",
    exposure_horizon_hours=2, damage_radius_km=74.0,
    forecast={
        "t0": IssuanceSpec({"t3": ConeSpec(width_km=90.0, lat=25.4428,
                                           lon=81.32778)}),
        "t1": IssuanceSpec({"t3": ConeSpec(
            width_km=T1_WIDTH_KM, bearing_deg=T1_BEARING_TOWARD,
            radius_km=T1_RADIUS_TOWARD)}),
    },
    gold_spare_action="conserve")

# T1b's radius is solved at runtime (needs live spans); see build_t1b_spec().


# T2 -- far horizon BYTE-IDENTICAL to the shipped T2a/T2b (storm-svc-1's own
# point, width 200). Near horizon re-aimed at the claimant corridor, same
# radius both halves (300 km reused from the shipped construction -- at this
# radius the SUT's own p_cut is ~0 in EVERY bearing tried, confirmed below,
# so equality is free the same way it was for the old kanpur/agra cluster).
T2_FAR_LAT, T2_FAR_LON, T2_FAR_WIDTH = 24.855553333333333, 81.32777666666667, 200.0
T2_NEAR_WIDTH_KM = 60.0
T2_NEAR_RADIUS_KM = 250.0
T2A_NEAR_BEARING = 202.0     # moderate claimant exposure -> small near claim
T2B_NEAR_BEARING = 213.0     # heavy claimant exposure -> large near claim

T2A_SPEC = HalfSpec(
    id="T2a", pair="T2", hours=("t0", "t1", "t2", "t6"), decision_hour="t1",
    exposure_horizon_hours=2, damage_radius_km=74.0,
    forecast={
        "t0": IssuanceSpec({"t6": ConeSpec(width_km=90.0, lat=23.0, lon=79.6)}),
        "t1": IssuanceSpec({
            "t2": ConeSpec(width_km=T2_NEAR_WIDTH_KM,
                          bearing_deg=T2A_NEAR_BEARING,
                          radius_km=T2_NEAR_RADIUS_KM),
            "t6": ConeSpec(width_km=T2_FAR_WIDTH, lat=T2_FAR_LAT, lon=T2_FAR_LON),
        }),
    },
    gold_spare_action="spend")

T2B_SPEC = HalfSpec(
    id="T2b", pair="T2", hours=("t0", "t1", "t2", "t6"), decision_hour="t1",
    exposure_horizon_hours=2, damage_radius_km=74.0,
    forecast={
        "t0": IssuanceSpec({"t6": ConeSpec(width_km=90.0, lat=23.0, lon=79.6)}),
        "t1": IssuanceSpec({
            "t2": ConeSpec(width_km=T2_NEAR_WIDTH_KM,
                          bearing_deg=T2B_NEAR_BEARING,
                          radius_km=T2_NEAR_RADIUS_KM),
            "t6": ConeSpec(width_km=T2_FAR_WIDTH, lat=T2_FAR_LAT, lon=T2_FAR_LON),
        }),
    },
    gold_spare_action="conserve")


# T3 -- far horizon BYTE-IDENTICAL to the shipped T3a/T3b (storm-svc-1's own
# point, width 320). Near horizon (claimant) re-aimed; the shipped
# construction held storm-svc-1's OWN offset from the near cone equal across
# T3's halves too (228.075 km) -- reused directly here (both T3 halves'
# near cones sit at exactly that radius from storm-svc-1's own point), so
# both this pair's SUT-side invariants stay intact for free.
T3_FAR_LAT, T3_FAR_LON, T3_FAR_WIDTH = 24.855553333333333, 81.32777666666667, 320.0
T3_NEAR_WIDTH_KM = 90.0
# NOT 228.0754620624 (the shipped T3a/T3b's own radius): at that radius the
# REAL span-based SUT p_cut is small but NOT exactly 0 (3.05e-5 at bearing
# 196.4 deg vs 2.14e-4 at 211.0 deg -- a |delta| of 1.8e-4, over 100x
# DERIVED_TOLERANCE), so "same radius, either bearing" does not actually hold
# SUT exposure equal at THIS near horizon the way it does at T2's -- the
# same lesson T1 forced, just less severely. 290.0 km is the smallest radius
# at which every bearing in the 185-217 deg neighbourhood used below lands
# past `p_cut_region`'s own NEGLIGIBLE_SIGMAS fast path (offset >
# damage_radius_km + 6*sigma = 74 + 6*30.64 = 257.9 km at width_km=90), so
# BOTH halves' near-horizon SUT p_cut is the EXACT SAME FLOAT (0.0), not
# merely close -- confirmed live (see derive_episodes' own run log).
T3_NEAR_RADIUS_KM = 290.0
T3A_NEAR_BEARING = 204.0     # off the corridor -> small near claim
T3B_NEAR_BEARING = 213.0     # on the corridor -> large near claim

T3A_SPEC = HalfSpec(
    id="T3a", pair="T3", hours=("t0", "t1", "t2", "t6"), decision_hour="t1",
    exposure_horizon_hours=2, damage_radius_km=74.0,
    forecast={
        "t0": IssuanceSpec({"t6": ConeSpec(width_km=90.0, lat=23.0, lon=79.6)}),
        "t1": IssuanceSpec({
            "t2": ConeSpec(width_km=T3_NEAR_WIDTH_KM,
                          bearing_deg=T3A_NEAR_BEARING,
                          radius_km=T3_NEAR_RADIUS_KM),
            "t6": ConeSpec(width_km=T3_FAR_WIDTH, lat=T3_FAR_LAT, lon=T3_FAR_LON),
        }),
    },
    gold_spare_action="spend")

T3B_SPEC = HalfSpec(
    id="T3b", pair="T3", hours=("t0", "t1", "t2", "t6"), decision_hour="t1",
    exposure_horizon_hours=2, damage_radius_km=74.0,
    forecast={
        "t0": IssuanceSpec({"t6": ConeSpec(width_km=90.0, lat=23.0, lon=79.6)}),
        "t1": IssuanceSpec({
            "t2": ConeSpec(width_km=T3_NEAR_WIDTH_KM,
                          bearing_deg=T3B_NEAR_BEARING,
                          radius_km=T3_NEAR_RADIUS_KM),
            "t6": ConeSpec(width_km=T3_FAR_WIDTH, lat=T3_FAR_LAT, lon=T3_FAR_LON),
        }),
    },
    gold_spare_action="conserve")


def build_t1b_spec(geometry: ServiceGeometry) -> HalfSpec:
    """T1b's radius is solved against the LIVE SUT spans so its p_cut
    matches T1a's exactly (see this module's docstring on why "same radius"
    does not transfer to T1 from T2/T3's near-zero case)."""
    sut_point = geometry.points[SUT]
    sut_spans = geometry.cuttable_spans[SUT]
    target_lat, target_lon = place_centre(
        *sut_point, T1_BEARING_TOWARD, T1_RADIUS_TOWARD)
    target_pcut = p_cut_region(sut_spans, target_lat, target_lon,
                               T1_WIDTH_KM, 74.0)
    radius_away = solve_radius_for_target_pcut(
        *sut_point, T1_BEARING_AWAY, target_pcut, sut_spans, T1_WIDTH_KM, 74.0)
    return HalfSpec(
        id="T1b", pair="T1", hours=("t0", "t1", "t2", "t3"), decision_hour="t1",
        exposure_horizon_hours=2, damage_radius_km=74.0,
        forecast={
            "t0": IssuanceSpec({"t3": ConeSpec(width_km=90.0, lat=25.4428,
                                               lon=81.32778)}),
            "t1": IssuanceSpec({"t3": ConeSpec(
                width_km=T1_WIDTH_KM, bearing_deg=T1_BEARING_AWAY,
                radius_km=radius_away)}),
        },
        gold_spare_action="spend")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

async def run(topology: str, state: str, server_command: list[str]) -> None:
    async with connect_server(topology, server_command=server_command,
                              env=dict(os.environ),
                              extra_args=["--state", state]) as client:
        geometry = await service_geometry(client, topology)
        services = await call_tool_json(client, "get_services")
        demands = {s["id"]: float(s["demand_gbps"]) for s in services["services"]}

        print("SUT point:", geometry.points[SUT])
        print("SUT cuttable spans:", geometry.cuttable_spans[SUT])
        print("claimant fwd spans:", geometry.cuttable_spans[CLAIMANT_FWD])
        print("claimant fwd endpoints:", geometry.endpoint_sites[CLAIMANT_FWD])
        satna_terminating = [sid for sid, sites in geometry.endpoint_sites.items()
                             if "satna" in sites]
        print("satna-terminating services:", sorted(satna_terminating))

        t1b_spec = build_t1b_spec(geometry)
        specs = [D1_SPEC, T1A_SPEC, t1b_spec, T2A_SPEC, T2B_SPEC,
                T3A_SPEC, T3B_SPEC]
        results = {spec.id: evaluate_half(spec, geometry, demands)
                  for spec in specs}

        for spec in specs:
            print_half_summary(results[spec.id])

        # ---- explicit derived-geometry equality check, per pair ----------
        # The same pure function assertions.assert_pair_derived_geometry_is_
        # equal calls (assertions._derived_mismatches), run directly against
        # this tool's own DerivedGeometry objects -- no server round trip
        # needed since DerivedGeometry is already computed. This is the
        # check that would catch a "same radius, different bearing" near
        # cone whose real span-based SUT p_cut looks equal at 8 decimals
        # printed above but differs by more than DERIVED_TOLERANCE (the T3
        # near-horizon lesson this module's own docstring records).
        from storm_reoptimizer.eval.assertions import _derived_mismatches
        print("\n=== pairwise derived-geometry equality (DERIVED_TOLERANCE"
              "=1e-6) ===")
        for pair_id, (a_id, b_id) in (
            ("T1", ("T1a", "T1b")), ("T2", ("T2a", "T2b")),
            ("T3", ("T3a", "T3b")),
        ):
            problems = _derived_mismatches(
                results[a_id].scenario, results[a_id].derived,
                results[b_id].scenario, results[b_id].derived)
            if problems:
                print(f"  {pair_id}: MISMATCH --")
                for p in problems:
                    print(f"    - {p}")
            else:
                print(f"  {pair_id}: OK -- every derived scalar equal to "
                      f"tolerance")

        # ---- Step 2 (D1 smoke test): confirm the brief's specific claims ----
        d1 = results["D1"]
        print("\n=== D1 smoke-test assertions ===")
        old_model_pcut = 0.9761
        new_model_pcut = d1.derived.sut_p_cut_at_exposure_horizon
        print(f"  old-model p_cut (declared in D1.yaml prose): {old_model_pcut}")
        print(f"  new-model p_cut (this tool, live server):    {new_model_pcut:.6f}")
        print(f"  n_future_claimants (network sum ex-SUT, at exposure horizon): "
              f"{d1.flip.claimant_ecar_at_exposure_horizon:.6f} G "
              f"(0 claimants iff this and largest_group both ~0)")
        print(f"  largest_restorable_group_ecar_gbps: "
              f"{d1.flip.largest_restorable_group_ecar_gbps:.6f}")
        old_offset_km = 58.4457
        new_offset_km = d1.derived.sut_offset_km
        cone_half_width = 15.0 / 2.0
        print(f"  old offset_km (averaged-midpoint model): {old_offset_km} "
              f"-> {'INSIDE' if old_offset_km <= cone_half_width else 'OUTSIDE'} "
              f"the cone (half-width {cone_half_width})")
        print(f"  new offset_km (nearest-span model):      {new_offset_km:.6f} "
              f"-> {'INSIDE' if new_offset_km <= cone_half_width else 'OUTSIDE'} "
              f"the cone (half-width {cone_half_width})")

        # ---- Step 3/4: joint sweep over the three pairs' six halves --------
        pair_halves = ["T1a", "T1b", "T2a", "T2b", "T3a", "T3b"]
        actions = {h: results[h].scenario.metadata["gold_spare_action"]
                  for h in pair_halves}
        print("\n=== gold_spare_action per half (3 pairs, 6 halves; D1 has "
              "no pair and is excluded from this sweep) ===")
        for h in pair_halves:
            print(f"  {h}: {actions[h]}")

        print("\n=== per-variable sweep (5 FLIP_VARS + 2 extra summaries) ===")
        reports = {}
        for var in ALL_SWEPT_VARS:
            values = {h: all_flip_values(results[h])[var] for h in pair_halves}
            report = sweep_variable(var, values, actions)
            reports[var] = report
            print_sweep_report(report)

        blocked = [v for v in ALL_SWEPT_VARS
                  if blocking_kind(reports[v]) != "NOT BLOCKED -- a global "
                  "threshold solves this variable"]
        print(f"\n=== SUMMARY: {len(blocked)}/{len(ALL_SWEPT_VARS)} variables "
              f"blocked (no global threshold solves them) ===")
        for var in ALL_SWEPT_VARS:
            print(f"  {var}: {blocking_kind(reports[var])}  "
                  f"best={reports[var].best_score:.3f}")

        # ---- also exercise the REAL gate, as a cross-check --------------
        from storm_reoptimizer.eval.assertions import (
            PairInvalid, assert_no_global_policy_solves_the_suite)
        flip_values = {h: all_flip_values(results[h]) for h in pair_halves}
        # assert_no_global_policy_solves_the_suite only enumerates FLIP_VARS
        # (not the two extra summaries -- those are this tool's own insurance
        # sweep, matching the brief's step 1d).
        flip_values_only = {h: {v: flip_values[h][v] for v in FLIP_VARS}
                            for h in pair_halves}
        try:
            assert_no_global_policy_solves_the_suite(
                [results[h].scenario for h in pair_halves], flip_values_only)
            print("\nassert_no_global_policy_solves_the_suite: PASSED "
                  "(no threshold on any FLIP_VARS member solves the suite)")
        except PairInvalid as exc:
            print(f"\nassert_no_global_policy_solves_the_suite: FAILED -- "
                  f"{exc}")

        # ---- REAL binding-edge, per interleaved variable, via bisection --
        # A live review of this tool's first cut found the "smallest
        # adjacent gap in sorted order" (transitions(), above) is NOT the
        # same quantity as the real binding edge, and was wrong here by up
        # to 2.2x: two values can swap position on the SAME side of the
        # actual separating boundary without restoring separability, so
        # their gap understates the true margin. This section finds the
        # REAL edge by bisecting a live oracle -- the actual
        # `assert_no_global_policy_solves_the_suite` for variables it
        # enumerates (members of `derived.FLIP_VARS`), or this tool's own
        # verified-equivalent `_is_solved_by_some_threshold` for the two
        # extra summaries that assertion does not enumerate at all -- from
        # BOTH ends of the overlap, and cross-checks the two margins agree.
        print("\n=== REAL binding edge per interleaved variable (bisected "
              "against a live oracle, not read off sorted-order gaps) ===")
        scenarios_list = [results[h].scenario for h in pair_halves]

        def real_assertion_oracle(var: str):
            def solved_fn(values: dict[str, float], acts: dict[str, str]) -> bool:
                vv = {h: dict(flip_values_only[h]) for h in pair_halves}
                for h in pair_halves:
                    vv[h][var] = values[h]
                try:
                    # RAISES iff a threshold SOLVES the suite (bad); returns
                    # silently iff BLOCKED (good). So "no exception" means
                    # NOT solved -- the polarity a first draft of this
                    # function got backwards, caught by re-running against
                    # the tool's own already-confirmed-PASSED baseline
                    # before trusting any bisected number.
                    assert_no_global_policy_solves_the_suite(scenarios_list, vv)
                    return False
                except PairInvalid:
                    return True
            return solved_fn

        for var in ALL_SWEPT_VARS:
            report = reports[var]
            if blocking_kind(report) != "INTERLEAVE":
                continue    # only interleave-blocked variables have a real
                            # edge to bisect; a tied variable's bound is the
                            # tie itself, already reported above
            values = report.values
            assert len(report.misclassified) == 1, (
                f"{var}: expected exactly one misclassified half at the "
                f"best split, got {report.misclassified!r} -- the "
                f"single-perturbation binding-edge search assumes exactly "
                f"one out-of-place half and needs a different approach here")
            perturb_id = report.misclassified[0]
            if var in FLIP_VARS:
                oracle = real_assertion_oracle(var)
                oracle_name = "assert_no_global_policy_solves_the_suite (real)"
            else:
                oracle = _is_solved_by_some_threshold
                oracle_name = "_is_solved_by_some_threshold (verified-equivalent local oracle)"
            edge = bisect_binding_edge(values, actions, oracle,
                                       perturb_id=perturb_id)
            overlap_lo_spend, overlap_lo_conserve = arithmetic_overlap(
                values, actions)
            arithmetic_margin = (overlap_lo_spend if edge["direction"] < 0
                                 else overlap_lo_conserve)
            consistent = abs(edge["margin"] - arithmetic_margin) < 1e-4
            direction_word = "shrinking" if edge["direction"] < 0 else "growing"
            print(f"\n  {var}  [oracle: {oracle_name}]")
            print(f"    {direction_word} {perturb_id} ({edge['base_value']:.4f}"
                  f", {actions[perturb_id]}) -> flips (starts solving the "
                  f"suite) at {edge['flip_value']:.4f}")
            print(f"    BISECTED margin: {edge['margin']:.4f} G")
            print(f"    arithmetic cross-check (max/min over the two action "
                  f"groups, no bisection): {arithmetic_margin:.4f} G "
                  f"(lo=spend overlap {overlap_lo_spend:.4f}, lo=conserve "
                  f"overlap {overlap_lo_conserve:.4f})")
            print(f"    cross-check: {'CONSISTENT' if consistent else 'MISMATCH'} "
                  f"(|{edge['margin']:.4f} - {arithmetic_margin:.4f}| = "
                  f"{abs(edge['margin'] - arithmetic_margin):.6g})")
            print(f"    REAL BINDING EDGE: {perturb_id}, margin = "
                  f"{edge['margin']:.3f} G")


def main() -> None:
    parser = argparse.ArgumentParser(prog="derive_episodes")
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--server-command", default=None,
                        help="JSON list; defaults to this workspace's "
                        "multilayer-optical-mcp conda-env workaround")
    args = parser.parse_args()
    server_command = (json.loads(args.server_command) if args.server_command
                      else DEFAULT_SERVER_COMMAND)
    asyncio.run(run(args.topology, args.state, server_command))


if __name__ == "__main__":
    main()
