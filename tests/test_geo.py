"""events/geo.py: the two hazard geometries and the relationship between
them. `circle_polygon` had no tests of its own; the properties asserted here
are the ones `damage_footprint` inherits from it."""
import math

import pytest
from shapely.geometry import shape

from storm_reoptimizer.events.geo import (
    EARTH_RADIUS_KM, circle_polygon, damage_footprint,
)

CENTRE = (25.0, 81.0)   # (lat, lon), in the toy topology's region


def _north_vertex_offset_km(polygon: dict, center_lat: float) -> float:
    """The i=0 vertex of circle_polygon's ring is due north of the centre
    (theta=0 -> dlat = r/R, dlon = 0), so its latitude difference times the
    earth radius recovers the radius the polygon was built with."""
    lon, lat = polygon["coordinates"][0][0]
    return math.radians(lat - center_lat) * EARTH_RADIUS_KM


def test_the_footprint_radius_is_the_half_width_plus_the_damage_radius():
    lat, lon = CENTRE
    footprint = damage_footprint(lat, lon, width_km=90.0,
                                 damage_radius_km=74.0)
    assert _north_vertex_offset_km(footprint, lat) == pytest.approx(
        45.0 + 74.0, abs=0.05)


def test_the_ring_is_closed_and_in_lon_lat_order():
    lat, lon = CENTRE
    ring = damage_footprint(lat, lon, width_km=90.0,
                            damage_radius_km=74.0)["coordinates"][0]
    assert ring[0] == ring[-1]
    # [lon, lat], the order circle_polygon and geo_mapper.load_edges use.
    assert ring[0][0] == pytest.approx(lon, abs=1.0)
    assert ring[0][1] == pytest.approx(lat, abs=2.0)


def test_the_footprint_strictly_contains_the_track_containment_circle():
    """The Minkowski-sum semantics, stated as geometry: every point the
    published cone covers is a point the storm could cut from."""
    lat, lon = CENTRE
    containment = shape(circle_polygon(lat, lon, 90.0 / 2.0))
    footprint = shape(damage_footprint(lat, lon, width_km=90.0,
                                       damage_radius_km=74.0))
    assert footprint.contains(containment)
    assert not containment.contains(footprint)


def test_a_nonpositive_width_or_damage_radius_is_an_error():
    lat, lon = CENTRE
    with pytest.raises(ValueError):
        damage_footprint(lat, lon, width_km=0.0, damage_radius_km=74.0)
    with pytest.raises(ValueError):
        damage_footprint(lat, lon, width_km=90.0, damage_radius_km=0.0)
