"""Cone containment semantics (eval design spec, "Cone semantics"). Pure
maths -- no server, no state file."""
import math

import numpy as np
import pytest
from scipy.stats import ncx2

from storm_reoptimizer.eval.cone import (
    CONTAINMENT_P, RAYLEIGH_DIVISOR, SAMPLE_COUNT, cone_polygon,
    cross_track_sigma_km, expected_capacity_at_risk_gbps, p_cut_point,
    p_cut_region, p_cut_region_joint, p_cut_service, radial_offset_km,
    service_cut_mask,
)


def test_sigma_is_the_two_dimensional_rayleigh_calibration():
    # The cone circle of radius R contains the storm CENTRE with probability
    # 0.66 -- a statement about a disc in the PLANE. The distance from the
    # forecast position to an isotropic 2-D Gaussian's realization is
    # Rayleigh, not normal, so the quantile is sqrt(-2 ln(1-P)) = 1.46888...,
    # not Phi^-1((1+P)/2) = 0.9542. The shipped sigma was 54% too large.
    assert RAYLEIGH_DIVISOR == pytest.approx(
        math.sqrt(-2.0 * math.log(1.0 - CONTAINMENT_P)), abs=1e-15)
    assert cross_track_sigma_km(90.0) == pytest.approx(30.6355, abs=1e-4)
    assert cross_track_sigma_km(320.0) == pytest.approx(108.9263, abs=1e-4)


def test_the_calibration_round_trips_through_the_point_model():
    # A target ON the axis with a damage radius equal to the cone's own
    # half-width must be cut with probability exactly CONTAINMENT_P. At
    # offset 0 the noncentral chi-square degenerates to the central one,
    # whose CDF is the Rayleigh CDF the calibration was solved from -- so
    # this asserts the two steps agree rather than merely coexist.
    assert p_cut_point(0.0, 90.0, 45.0) == pytest.approx(CONTAINMENT_P, abs=1e-12)


def test_the_point_model_is_the_noncentral_chi_square_cdf():
    sigma = cross_track_sigma_km(90.0)
    expected = float(ncx2.cdf((74.0 / sigma) ** 2, df=2,
                              nc=(120.6 / sigma) ** 2))
    assert p_cut_point(120.6, 90.0, 74.0) == pytest.approx(expected, abs=1e-15)
    assert expected == pytest.approx(0.046013, abs=1e-6)


def test_the_point_model_saturates_and_vanishes_at_the_limits():
    assert p_cut_point(0.0, 15.0, 74.0) == pytest.approx(1.0, abs=1e-9)
    assert p_cut_point(5000.0, 90.0, 74.0) == pytest.approx(0.0, abs=1e-12)
    assert p_cut_point(120.0, 90.0, 1e5) == pytest.approx(1.0, abs=1e-9)


def test_cut_probability_falls_off_with_offset_from_the_axis():
    centre = p_cut_point(0.0, 90.0, 60.0)
    edge = p_cut_point(45.0, 90.0, 60.0)
    far = p_cut_point(200.0, 90.0, 60.0)
    assert centre > edge > far
    assert 0.0 <= far < 0.1


def test_a_wider_cone_is_a_less_certain_forecast_on_the_axis():
    # More cross-track spread => the track is less likely to pass right over
    # an on-axis asset. This is what makes "the cone tightens" informative.
    assert p_cut_point(0.0, 60.0, 40.0) > p_cut_point(0.0, 200.0, 40.0)


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


def test_joint_equals_union_when_both_legs_are_the_same_span():
    span = ((24.6, 80.8), (24.5, 81.3))
    args = (24.55, 81.0, 90.0, 74.0)
    assert p_cut_region_joint((span,), (span,), *args) == p_cut_region((span,), *args)


def test_joint_is_zero_when_legs_are_far_apart():
    near = ((24.6, 80.8), (24.5, 81.3))
    far = ((12.0, 77.0), (12.1, 77.5))
    assert p_cut_region_joint((near,), (far,), 24.55, 81.0, 90.0, 74.0) == 0.0


def test_joint_never_exceeds_either_marginal():
    a = ((24.6, 80.8), (24.5, 81.3)); b = ((24.6, 80.8), (24.9, 80.6))
    args = (24.6, 80.8, 120.0, 74.0)
    j = p_cut_region_joint((a,), (b,), *args)
    assert j <= min(p_cut_region((a,), *args), p_cut_region((b,), *args))


def test_service_p_cut_is_joint_for_protected_and_zero_for_uncuttable_protection():
    a = ((24.6, 80.8), (24.5, 81.3))
    args = (24.55, 81.0, 90.0, 74.0)
    assert p_cut_service((a,), None, *args) == p_cut_region((a,), *args)
    assert p_cut_service((a,), (), *args) == 0.0


def test_sample_count_is_the_standard_normal_points_array_length():
    # SAMPLE_COUNT is what a caller stacking masks across services (a joint
    # cut table) uses as the denominator, and it must be the same length
    # every p_cut_* scalar already divides by.
    a = ((24.6, 80.8), (24.5, 81.3))
    mask = service_cut_mask((a,), None, 24.55, 81.0, 90.0, 74.0)
    assert SAMPLE_COUNT == len(mask)


def test_service_cut_mask_means_are_bit_identical_to_the_scalar_p_cut_functions():
    # The proof of bit-identity for the p_cut_region/p_cut_region_joint/
    # p_cut_service refactor onto service_cut_mask: the same span sets, run
    # through the mask path and averaged by hand, must equal the scalar
    # functions' own answers exactly -- not approximately.
    a = ((24.6, 80.8), (24.5, 81.3))
    b = ((24.6, 80.8), (24.9, 80.6))
    args = (24.6, 80.8, 120.0, 74.0)

    mask_unprotected = service_cut_mask((a,), None, *args)
    assert float(mask_unprotected.mean()) == p_cut_region((a,), *args)

    mask_joint = service_cut_mask((a,), (b,), *args)
    assert float(mask_joint.mean()) == p_cut_region_joint((a,), (b,), *args)
    assert float(mask_joint.mean()) == p_cut_service((a,), (b,), *args)


def test_service_cut_mask_is_none_exactly_where_the_scalars_are_zero_by_prefilter():
    # No cuttable span, the far-field prefilter, and an empty intersection
    # region all make the scalar functions return 0.0 WITHOUT computing a
    # mask. service_cut_mask must return None, not an all-False array, at
    # each of those -- brief: "returns None whenever the existing prefilters
    # ... return 0.0 today".
    args = (24.55, 81.0, 90.0, 74.0)
    assert service_cut_mask((), None, *args) is None
    assert p_cut_region((), *args) == 0.0

    near = ((24.6, 80.8), (24.5, 81.3))
    far = ((12.0, 77.0), (12.1, 77.5))
    assert service_cut_mask((near,), (far,), 24.55, 81.0, 90.0, 74.0) is None
    assert p_cut_region_joint((near,), (far,), 24.55, 81.0, 90.0, 74.0) == 0.0

    # A protected leg with no cuttable span at all: "not working_spans or
    # not protection_spans" branch, distinct from the far-field one above.
    assert service_cut_mask((near,), (), *args) is None
    assert p_cut_service((near,), (), *args) == 0.0


def test_service_cut_mask_dtype_is_boolean():
    a = ((24.6, 80.8), (24.5, 81.3))
    mask = service_cut_mask((a,), None, 24.55, 81.0, 90.0, 74.0)
    assert mask.dtype == np.bool_
