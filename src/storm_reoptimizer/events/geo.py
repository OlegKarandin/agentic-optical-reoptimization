# src/storm_reoptimizer/events/geo.py
"""Plain-trig circle-polygon approximation. Deliberately no Shapely here --
that's the geo mapper's dependency (CLAUDE.md build order step 3), not
needed just to author event geometry in step 2."""
from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0


def displace_km(lat: float, lon: float, radius_km: float,
                bearing_deg: float) -> tuple[float, float]:
    """(lat, lon) moved `radius_km` along `bearing_deg` (0 = north, 90 =
    east), under the same equirectangular flat-earth approximation
    `circle_polygon` draws with and `eval.cone.radial_offset_km` measures
    against.

    Extracted from `circle_polygon`'s own per-point formula -- which is
    exactly this displacement swept over a full turn -- so a point the
    revision band displaces to and a point the drawn cone would contain are
    computed by one piece of arithmetic, not two that agree by inspection."""
    theta = math.radians(bearing_deg)
    lat_rad = math.radians(lat)
    dlat = (radius_km / EARTH_RADIUS_KM) * math.cos(theta)
    dlon = (radius_km / (EARTH_RADIUS_KM * math.cos(lat_rad))) * math.sin(theta)
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def circle_polygon(lat: float, lon: float, radius_km: float, n_points: int = 24) -> dict:
    """A GeoJSON Polygon approximating a circle of `radius_km` around
    (lat, lon), via an equirectangular flat-earth approximation local to the
    center. Adequate at these radii (tens to a few hundred km) for a demo
    hazard footprint -- not navigation-grade geometry."""
    if radius_km <= 0:
        raise ValueError("radius_km must be > 0")
    coords = []
    for i in range(n_points + 1):  # +1 to close the ring
        plat, plon = displace_km(lat, lon, radius_km, 360.0 * i / n_points)
        coords.append([round(plon, 5), round(plat, 5)])
    return {"type": "Polygon", "coordinates": [coords]}


def damage_footprint_radius_km(width_km: float, damage_radius_km: float) -> float:
    """The single source of truth for the hazard-footprint radius: the
    Minkowski-sum radius of the track-containment disc (`width_km / 2`) and
    the damage disc (`damage_radius_km`). Shared by `damage_footprint` below
    and by `eval.baseline._nearest_exposed_horizon`, which must test
    exposure against the SAME radius `damage_footprint` builds its polygon
    from -- see `damage_footprint`'s docstring for why the two must agree."""
    return width_km / 2.0 + damage_radius_km


def damage_footprint(center_lat: float, center_lon: float, width_km: float,
                     damage_radius_km: float, n_points: int = 24) -> dict:
    """The assets this storm can cut IF ITS TRACK VERIFIES ANYWHERE INSIDE
    THE PUBLISHED CONE -- the Minkowski sum of the track-containment disc
    (radius `width_km / 2`) with the damage disc (radius
    `damage_radius_km`), which for two discs is just a disc of the summed
    radius.

    Two different questions, kept apart deliberately:

      * `circle_polygon(lat, lon, width_km / 2)` -- and its `cone.
        cone_polygon` alias -- is WHERE THE STORM CENTRE PROBABLY GOES. It is
        what a scenario file's `cone` field stores and what the viewer draws
        as the track cone.
      * this function is WHAT THE STORM BREAKS ONCE IT GETS THERE. It is what
        `map_geo_event_to_assets` must be intersected against to build a risk
        group, and what a fixed operational policy would key its containment
        test on.

    Confusing the two is the seam defect the 2026-08-31 hazard-footprint spec
    documents: the GIS side intersected the containment circle alone while
    `cone._region_in_sigmas` -- which performs this SAME Minkowski sum on the
    probability side, buffering the spans by `damage_radius_km` before
    integrating -- had been doing it correctly all along. Both modules were
    right against their own docstrings and wrong at the join.

    NOT a `p_cut >= threshold` contour, though that would be tighter and
    would agree with the exposure numbers by construction. It would introduce
    a tunable and make a GEOMETRIC object depend on the probability model.
    Geometry stays geometric; the probability model already has its own
    consumer.

    LIMITATION: exact only for CIRCULAR cones. Every hazard footprint this
    project emits is a circle (`circle_polygon`, and `cone.py`'s own module
    docstring says so), so it is exact today. A non-circular cone would need
    a buffer taken in the projected km frame about the centre and projected
    back -- a degree-space `.buffer()` is anisotropic and wrong at these
    latitudes. Noted, not built.
    """
    if width_km <= 0:
        raise ValueError(f"width_km must be > 0, got {width_km!r}")
    if damage_radius_km <= 0:
        raise ValueError(
            f"damage_radius_km must be > 0, got {damage_radius_km!r}")
    return circle_polygon(center_lat, center_lon,
                          damage_footprint_radius_km(width_km, damage_radius_km),
                          n_points)
