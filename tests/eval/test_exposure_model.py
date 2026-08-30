"""The exposure model's correctness argument (exposure-and-depot design, §2.5).

It does NOT rest on the algebra in the spec. The general pipeline -- calibrate
sigma, inverse-transform Sobol points onto a 2-D Gaussian, buffer the spans,
count what lands inside -- is validated against an INDEPENDENT authority by
degenerating its input: shrink a span to zero length and the region becomes a
disc, for which scipy.stats.ncx2 gives the exact closed form. One test then
exercises the calibration, the transform, the scaling, the region construction
and the membership test together."""
import math

import numpy as np
import pytest
from scipy.stats import norm

from storm_reoptimizer.eval.cone import (
    SOBOL_M, cross_track_sigma_km, nearest_span_offset_km, p_cut_point,
    p_cut_region, project_to_km, spans_to_km,
)

# A cone centre in the region the suite actually uses.
CENTRE = (24.855553333333333, 81.32777666666667)

# Sampling error is measured, not assumed: 2.56e-4 is the worst deviation
# over 45 (offset, width) cases at SOBOL_M=16 and BUFFER_QUAD_SEGS=64. The
# spec's §2.5 table quotes 1e-6; that was NOT reproduced, and this bound is
# the honest one. At shapely's DEFAULT quad_segs=8 the same sweep is off by
# 7.99e-3 -- larger than the suite's own decision margins.
ORACLE_TOLERANCE = 5e-4


def _point_span(offset_km, centre=CENTRE):
    """A degenerate (1 metre) span `offset_km` due east of the centre, so the
    buffered region is a disc and p_cut_point is exact for it."""
    lat, lon = centre
    dlon = math.degrees(offset_km / (6371.0 * math.cos(math.radians(lat))))
    a = (lat, lon + dlon)
    b = (lat, lon + dlon + 1e-8)
    return (a, b)


@pytest.mark.parametrize("offset_km,width_km", [
    (0.0, 15.0), (45.0, 90.0), (58.4, 60.0), (120.6, 90.0), (250.0, 320.0),
])
def test_the_pipeline_reproduces_the_closed_form_on_a_degenerate_span(
        offset_km, width_km):
    sampled = p_cut_region([_point_span(offset_km)], *CENTRE, width_km, 74.0)
    exact = p_cut_point(offset_km, width_km, 74.0)
    assert sampled == pytest.approx(exact, abs=ORACLE_TOLERANCE)


def test_the_union_is_not_the_independent_product():
    # All spans are cut by the SAME single random storm centre, so the events
    # are strongly positively correlated and 1 - prod(1 - p_i) overstates the
    # answer. Two nearby parallel spans make the gap visible.
    lat, lon = CENTRE
    a = ((lat, lon), (lat + 0.4, lon))
    b = ((lat, lon + 0.2), (lat + 0.4, lon + 0.2))
    union = p_cut_region([a, b], lat, lon, 90.0, 74.0)
    pa = p_cut_region([a], lat, lon, 90.0, 74.0)
    pb = p_cut_region([b], lat, lon, 90.0, 74.0)
    assert union < 1.0 - (1.0 - pa) * (1.0 - pb)
    assert union >= max(pa, pb) - ORACLE_TOLERANCE


def test_the_union_matches_a_million_plain_monte_carlo_draws():
    lat, lon = CENTRE
    spans = [((lat, lon), (lat + 0.4, lon + 0.3)),
             ((lat + 0.4, lon + 0.3), (lat + 0.9, lon + 1.1))]
    sampled = p_cut_region(spans, lat, lon, 90.0, 74.0)

    from shapely import contains, points
    from shapely.geometry import MultiLineString
    sigma = cross_track_sigma_km(90.0)
    region = MultiLineString(spans_to_km(spans, lat, lon)).buffer(
        74.0, quad_segs=64)
    rng = np.random.default_rng(20260830)
    xy = rng.normal(0.0, sigma, size=(1_000_000, 2))
    reference = float(contains(region, points(xy[:, 0], xy[:, 1])).mean())
    assert sampled == pytest.approx(reference, abs=2e-3)


def test_the_point_set_is_deterministic_with_no_seed_anywhere():
    # assertions.SHARED_SCALARS and the T1/T2/T3 constructions compare floats
    # bitwise, and a random-number STREAM is not guaranteed stable across
    # library versions. scramble=False + random_base2 gives published
    # direction numbers and a fixed point set: same N, same points, every
    # machine, every scipy build.
    spans = [((24.0, 81.0), (24.5, 81.5))]
    first = p_cut_region(spans, *CENTRE, 90.0, 74.0)
    second = p_cut_region(spans, *CENTRE, 90.0, 74.0)
    assert first == second        # bitwise, not approx
    assert SOBOL_M == 16


def test_nearest_span_offset_is_zero_when_the_centre_lies_on_a_span():
    lat, lon = CENTRE
    on_it = ((lat, lon), (lat + 0.5, lon))
    assert nearest_span_offset_km([on_it], lat, lon) == pytest.approx(0.0, abs=1e-9)
    away = _point_span(120.6)
    assert nearest_span_offset_km([away], lat, lon) == pytest.approx(120.6, rel=1e-3)


def test_a_service_with_no_cuttable_span_is_certainly_not_cut():
    assert p_cut_region([], *CENTRE, 90.0, 74.0) == 0.0
    assert nearest_span_offset_km([], *CENTRE) == float("inf")


def test_projection_agrees_with_the_radial_offset_helper():
    from storm_reoptimizer.eval.cone import radial_offset_km
    x, y = project_to_km(*CENTRE, 25.5, 80.4)
    assert math.hypot(x, y) == pytest.approx(
        radial_offset_km(*CENTRE, 25.5, 80.4), abs=1e-9)


def test_the_far_field_prefilter_changes_no_answer_it_is_allowed_to_change():
    # A pre-filter that returns 0.0 early is only honest if the number it
    # skips computing is below anything the suite can resolve. Sweep a span
    # outward and require the filtered and unfiltered answers to agree to well
    # inside the sampling error everywhere, including right at the boundary.
    from storm_reoptimizer.eval.cone import NEGLIGIBLE_SIGMAS, _p_cut_region_unfiltered
    lat, lon = CENTRE
    sigma = cross_track_sigma_km(90.0)
    for k in (5.0, NEGLIGIBLE_SIGMAS - 0.01, NEGLIGIBLE_SIGMAS,
              NEGLIGIBLE_SIGMAS + 0.01, 8.0):
        span = _point_span(74.0 + k * sigma)
        filtered = p_cut_region([span], lat, lon, 90.0, 74.0)
        unfiltered = _p_cut_region_unfiltered([span], lat, lon, 90.0, 74.0)
        assert filtered == pytest.approx(unfiltered, abs=1e-6), k
        assert p_cut_point(74.0 + k * sigma, 90.0, 74.0) < 1e-6, k
