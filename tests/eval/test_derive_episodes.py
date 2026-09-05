"""tools/derive_episodes.py::solve_radius_for_target_pcut, generalised (Task
14, T1 spend-or-hold redesign plan) to accept an optional `pcut_fn(lat, lon)
-> float` callable in place of its own internal `cone.p_cut_region(spans,
...)` computation -- so a caller (tools/derive_t1.py) can target the JOINT
`cone.p_cut_service` (working AND protection) this whole plan is about,
without hardcoding to a single-leg `spans` argument.

Pure/offline: no server, no real span geometry -- a synthetic, monotone
`pcut_fn` built from `radial_offset_km` alone, so this test exercises the
bisection machinery itself (does the generalised hook actually get called,
does bisection converge to the target), not any real geometry model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import derive_episodes as de  # noqa: E402


def test_solve_radius_for_target_pcut_with_pcut_fn_hits_target():
    """A monotone synthetic pcut_fn (linearly decreasing in radial offset
    from the origin) stands in for cone.p_cut_region entirely when passed as
    `pcut_fn` -- `spans` is irrelevant and passed empty. The target (0.5) is
    hit, BY THIS FUNCTION'S OWN CONSTRUCTION, at offset == 100.0 km exactly,
    so this also pins down that `pcut_at(d)` calls `place_centre(...)` to
    convert the search radius `d` into (lat, lon) before invoking `pcut_fn`
    -- not `pcut_fn(d)` directly on the raw radius."""
    origin_lat, origin_lon = 24.0, 81.0
    bearing = 120.0

    def pcut_fn(lat: float, lon: float) -> float:
        offset = de.radial_offset_km(origin_lat, origin_lon, lat, lon)
        return max(0.0, 1.0 - offset / 200.0)

    target = 0.5
    radius = de.solve_radius_for_target_pcut(
        origin_lat, origin_lon, bearing, target,
        spans=(), width_km=90.0, damage_radius_km=74.0, pcut_fn=pcut_fn)

    lat, lon = de.place_centre(origin_lat, origin_lon, bearing, radius)
    assert abs(pcut_fn(lat, lon) - target) < 1e-9
    # Close to, but not required to be exactly, 100.0 km: place_centre's own
    # frame (lat computed first, then lon adjusted by the NEW lat's cosine)
    # and radial_offset_km's flat-plane inverse are each internally
    # consistent but not perfect round-trip inverses of one another at this
    # distance -- the target-hit assertion above is the one this generalised
    # parameter is actually responsible for.
    assert abs(radius - 100.0) < 1.0


def test_solve_radius_for_target_pcut_without_pcut_fn_is_unchanged():
    """The `pcut_fn=None` (default) path must be BIT-IDENTICAL to the
    function's pre-existing behaviour -- the existing `spans`-based
    computation, untouched by the generalisation. Uses a single degenerate
    "span" (a zero-length segment, i.e. a point) so `cone.p_cut_region`
    reduces to the closed-form `cone.p_cut_point` this suite already trusts,
    giving an independent, non-tautological target to solve for."""
    origin_lat, origin_lon = 24.0, 81.0
    bearing = 45.0
    spans = (((24.5, 81.5), (24.5, 81.5)),)   # one zero-length span == a point
    width_km = 90.0
    damage_radius_km = 74.0

    # Pick an achievable target: p_cut at some real offset along this bearing.
    probe_lat, probe_lon = de.place_centre(origin_lat, origin_lon, bearing, 120.0)
    target = de.p_cut_region(spans, probe_lat, probe_lon, width_km,
                             damage_radius_km)

    radius = de.solve_radius_for_target_pcut(
        origin_lat, origin_lon, bearing, target, spans, width_km,
        damage_radius_km)

    lat, lon = de.place_centre(origin_lat, origin_lon, bearing, radius)
    got = de.p_cut_region(spans, lat, lon, width_km, damage_radius_km)
    assert abs(got - target) < 1e-9
