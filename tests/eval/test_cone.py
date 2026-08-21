"""Cone containment semantics (eval design spec, "Cone semantics"). Pure
maths -- no server, no state file."""
import pytest

from storm_reoptimizer.eval.cone import (
    CONTAINMENT_P, cone_polygon, cross_track_sigma_km, cut_probability,
    expected_capacity_at_risk_gbps, radial_offset_km,
)


def test_sigma_is_calibrated_so_the_cone_half_width_captures_P():
    # The defining property: an asset whose damaging-wind radius exactly
    # equals the cone's half-width, sitting on the axis, is cut with
    # probability CONTAINMENT_P. Everything else follows from this.
    assert cut_probability(0.0, 90.0, 45.0) == pytest.approx(CONTAINMENT_P, abs=1e-9)


def test_cut_probability_falls_off_with_offset_from_the_axis():
    centre = cut_probability(0.0, 90.0, 60.0)
    edge = cut_probability(45.0, 90.0, 60.0)
    far = cut_probability(200.0, 90.0, 60.0)
    assert centre > edge > far
    assert 0.0 <= far < 0.1


def test_a_wider_cone_is_a_less_certain_forecast_on_the_axis():
    # More cross-track spread => the track is less likely to pass right over
    # an on-axis asset. This is what makes "the cone tightens" informative.
    assert cut_probability(0.0, 60.0, 40.0) > cut_probability(0.0, 200.0, 40.0)


def test_width_must_be_positive():
    with pytest.raises(ValueError):
        cross_track_sigma_km(0.0)


def test_radial_offset_is_zero_at_the_centre_and_grows_with_distance():
    assert radial_offset_km(25.0, 81.5, 25.0, 81.5) == pytest.approx(0.0, abs=1e-6)
    one_degree_north = radial_offset_km(25.0, 81.5, 26.0, 81.5)
    assert one_degree_north == pytest.approx(111.2, rel=0.02)


def test_expected_capacity_at_risk_is_the_product():
    assert expected_capacity_at_risk_gbps(0.2, 100.0) == pytest.approx(20.0)


def test_cone_polygon_is_a_closed_geojson_polygon_of_the_stated_width():
    poly = cone_polygon(25.0, 81.5, 90.0)
    assert poly["type"] == "Polygon"
    ring = poly["coordinates"][0]
    assert ring[0] == ring[-1]
    lon, lat = ring[0]
    assert radial_offset_km(25.0, 81.5, lat, lon) == pytest.approx(45.0, rel=0.02)
