# src/storm_reoptimizer/events/geo.py
"""Plain-trig circle-polygon approximation. Deliberately no Shapely here --
that's the geo mapper's dependency (CLAUDE.md build order step 3), not
needed just to author event geometry in step 2."""
from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0


def circle_polygon(lat: float, lon: float, radius_km: float, n_points: int = 24) -> dict:
    """A GeoJSON Polygon approximating a circle of `radius_km` around
    (lat, lon), via an equirectangular flat-earth approximation local to the
    center. Adequate at these radii (tens to a few hundred km) for a demo
    hazard footprint -- not navigation-grade geometry."""
    if radius_km <= 0:
        raise ValueError("radius_km must be > 0")
    lat_rad = math.radians(lat)
    coords = []
    for i in range(n_points + 1):  # +1 to close the ring
        theta = 2 * math.pi * i / n_points
        dlat = (radius_km / EARTH_RADIUS_KM) * math.cos(theta)
        dlon = (radius_km / (EARTH_RADIUS_KM * math.cos(lat_rad))) * math.sin(theta)
        coords.append([
            round(lon + math.degrees(dlon), 5),
            round(lat + math.degrees(dlat), 5),
        ])
    return {"type": "Polygon", "coordinates": [coords]}
