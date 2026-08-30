# src/storm_reoptimizer/eval/cone.py
"""Cone containment semantics (eval design spec, "Cone semantics", build
order item 1). This is the module that makes every gold timing label
DERIVABLE rather than asserted: P(cut) x outage_cost against
spare_opportunity_cost, recorded in each scenario's gold.rationale.

The model, stated once so nobody has to reverse-engineer it:

  * A forecast cone at horizon h is published with a cross-track diameter
    `width_km` and the NHC convention that it contains the realized track
    with probability P = 0.66.
  * The track's position error is modelled as an isotropic 2-D Gaussian
    about the forecast centre, with sigma (per coordinate) calibrated so a
    disc of radius width_km/2 captures CONTAINMENT_P of that distribution --
    the Rayleigh calibration below.
  * An asset sitting `offset_km` off the cone axis is cut if the realized
    CENTRE passes within `damage_radius_km` of it -- its cut probability is
    therefore the mass of that 2-D Gaussian inside the disc of radius
    `damage_radius_km` centred on the asset, not the mass of an interval.

`offset_km` is measured RADIALLY from the cone centre rather than strictly
cross-track. For this project's hazard footprints -- circles, per
events/geo.circle_polygon, which every scripted track already emits -- the
two are the same quantity, and distinguishing them would be false precision
on top of an equirectangular flat-earth approximation."""
from __future__ import annotations

import math

from scipy.stats import ncx2

from ..events.geo import EARTH_RADIUS_KM, circle_polygon

# NHC convention: the forecast cone contains the realized track ~2/3 of the
# time. Fixed by the spec, not a tunable.
CONTAINMENT_P = 0.66

# The cone circle of radius R contains the storm CENTRE with probability
# CONTAINMENT_P -- a statement about a DISC IN THE PLANE. Model the centre as
# an isotropic 2-D Gaussian about the forecast position; the distance from the
# forecast position to the realization is then RAYLEIGH, so
#   P(D <= R) = 1 - exp(-R^2 / (2 sigma^2)) = CONTAINMENT_P
#   =>  sigma = R / sqrt(-2 ln(1 - CONTAINMENT_P)).
# The shipped model converted the same statement with a ONE-dimensional
# quantile, sigma = R / Phi^-1((1+P)/2) = R / 0.9542, and so ran 54% wide at
# every width in the suite.
RAYLEIGH_DIVISOR = math.sqrt(-2.0 * math.log(1.0 - CONTAINMENT_P))


def cross_track_sigma_km(width_km: float) -> float:
    """Std dev of EACH COORDINATE of the storm centre's position error,
    implied by a cone of diameter `width_km` under the 2-D calibration
    above."""
    if width_km <= 0:
        raise ValueError(f"width_km must be > 0, got {width_km!r}")
    return (width_km / 2.0) / RAYLEIGH_DIVISOR


def p_cut_point(offset_km: float, width_km: float,
                damage_radius_km: float) -> float:
    """P(the storm centre lands within `damage_radius_km` of a POINT target
    `offset_km` away), exactly.

    Put the origin at the forecast centre and, using isotropy, the target at
    (d, 0). Then |X - target|^2 / sigma^2 = (Z1 - d/sigma)^2 + Z2^2 with
    Z1, Z2 ~ N(0,1) -- a sum of squares of unit-variance Gaussians with one
    nonzero mean, which IS a noncentral chi-square with df=2 and
    nc=(d/sigma)^2. So the answer is that distribution's CDF at
    (r/sigma)^2, with no integration to derive and none to get wrong.

    This is also the ORACLE the region sampler is validated against
    (tests/eval/test_exposure_model.py): shrink a span to zero length and the
    region degenerates to the disc this function integrates in closed form."""
    if damage_radius_km <= 0:
        raise ValueError(
            f"damage_radius_km must be > 0, got {damage_radius_km!r}")
    if offset_km < 0:
        raise ValueError(f"offset_km must be >= 0, got {offset_km!r}")
    sigma = cross_track_sigma_km(width_km)
    return float(ncx2.cdf((damage_radius_km / sigma) ** 2, df=2,
                          nc=(offset_km / sigma) ** 2))


from scipy.stats import norm, qmc
from shapely import contains, points as _points, prepare
from shapely.geometry import MultiLineString, Point

# ((lat, lon), (lat, lon)) -- one fibre span, in the same order the topology
# and events.geo use.
Segment = tuple[tuple[float, float], tuple[float, float]]

# 2^16 = 65,536 Sobol points, minus the corner one dropped below. Measured
# worst-case deviation from the ncx2 oracle at this budget: 2.56e-4, against
# suite decision margins of order 3e-3. Raising it is a contained change --
# every consumer goes through p_cut_region -- but it is a CONTRACT: two
# different values give two different (deterministic) answers, and the
# episodes' frozen scalars are computed at this one.
SOBOL_M = 16

# NOT shapely's default of 8. The end caps of a buffered segment are polygonal
# approximations of a half-disc, and at quad_segs=8 they under-approximate it
# enough to bias p_cut by up to 8.0e-3 -- LARGER than the decision margins this
# suite resolves, and a bias that would sit undetected inside a sampling error
# thirty times smaller. Measured worst deviation from the closed form over 45
# (offset, width) cases: 2.56e-4 at 64, 7.99e-3 at 8.
BUFFER_QUAD_SEGS = 64


def _standard_normal_points():
    """A deterministic shapely point array of standard-normal sample locations.

    Sobol supplies low-discrepancy positions (integration error ~ (log N)^2/N
    rather than 1/sqrt(N)); `norm.ppf` maps them from the unit square onto a
    standard 2-D Gaussian by inverse transform, which is valid coordinate-wise
    precisely because the two axes are independent.

    Row 0 of an UNSCRAMBLED Sobol sequence is the corner (0, 0), whose inverse
    transform is (-inf, -inf). It is dropped rather than nudged: a point at
    infinity is outside every bounded region, so keeping it would silently
    shave 1/N off every probability, and nudging it would put a sample where
    the sequence never placed one.

    Built ONCE, at import. That is only possible because p_cut_region scales
    the REGION into units of sigma rather than scaling these points into km --
    see its docstring."""
    unit = qmc.Sobol(d=2, scramble=False).random_base2(SOBOL_M)[1:]
    z = norm.ppf(unit)
    return _points(z[:, 0], z[:, 1])


_STANDARD_NORMAL_POINTS = _standard_normal_points()


def project_to_km(center_lat: float, center_lon: float,
                  lat: float, lon: float) -> tuple[float, float]:
    """(east_km, north_km) of a point relative to a cone centre, under the
    same equirectangular flat-earth approximation events.geo.circle_polygon
    draws the cone with and radial_offset_km measures against."""
    dlat = math.radians(lat - center_lat)
    dlon = math.radians(lon - center_lon) * math.cos(math.radians(center_lat))
    return EARTH_RADIUS_KM * dlon, EARTH_RADIUS_KM * dlat


def spans_to_km(spans, center_lat: float, center_lon: float) -> list[list[tuple[float, float]]]:
    """Every span rewritten into the local km frame about the cone centre."""
    return [[project_to_km(center_lat, center_lon, *end) for end in span]
            for span in spans]


def _region_in_sigmas(spans, center_lat, center_lon, damage_radius_km,
                      sigma):
    """The set of storm-centre positions that would cut at least one span --
    the Minkowski sum of the spans with a disc of radius `damage_radius_km` --
    expressed in units of sigma.

    shapely owns geometry ONLY. `.buffer()` builds the union of capsules --
    each a rectangle of length L and width 2r with two half-disc end caps --
    and merges overlapping ones on a bent path into a single polygon. It never
    sees a probability.

    Everything is divided by sigma so the region can be tested against a FIXED
    standard-normal point set. Uniform isotropic scaling commutes with
    buffering, so this is not an approximation: verified bit-identical against
    the scale-the-points form at widths 15, 60, 90 and 320."""
    if damage_radius_km <= 0:
        raise ValueError(
            f"damage_radius_km must be > 0, got {damage_radius_km!r}")
    scaled = [[(x / sigma, y / sigma) for x, y in span]
              for span in spans_to_km(spans, center_lat, center_lon)]
    region = MultiLineString(scaled).buffer(damage_radius_km / sigma,
                                            quad_segs=BUFFER_QUAD_SEGS)
    # 23x on the measured workload (234 ms -> 10.1 ms per service-horizon),
    # bit-identical results. shapely builds a spatial index over the region's
    # segments; without it every one of the 65,535 membership tests walks the
    # whole ring.
    prepare(region)
    return region


def p_cut_region(spans, center_lat: float, center_lon: float,
                 width_km: float, damage_radius_km: float) -> float:
    """P(at least one of `spans` is cut) for a cone of diameter `width_km`
    centred at (center_lat, center_lon).

    Change of viewpoint, which is what makes this tractable: rather than "a
    disc around the storm touches the span", ask "the storm centre lands in a
    region around the span". The two are the same event, and the second is a
    single random point against a single fixed region -- a probability
    integral, estimated as the fraction of Gaussian-drawn points inside it.

    Deliberately NOT 1 - prod(1 - p_i) over the spans: ONE storm centre cuts
    all of them, so the per-span events are strongly positively correlated and
    the product form overstates the union. Only the union region is correct.

    Only spans the event's own filter admits are passed in (the caller does
    that filtering, so this module stays free of event vocabulary): buried
    conduit contributes nothing to a storm, and a service with none is
    certainly not cut."""
    if not spans:
        return 0.0
    sigma = cross_track_sigma_km(width_km)
    region = _region_in_sigmas(spans, center_lat, center_lon,
                               damage_radius_km, sigma)
    return float(contains(region, _STANDARD_NORMAL_POINTS).mean())


def nearest_span_offset_km(spans, center_lat: float,
                           center_lon: float) -> float:
    """Distance from the cone centre to the nearest point of any span, 0.0
    when the centre lies on one and `inf` when there are no spans.

    This is what the exposure row's `offset_km` now means, and what
    baseline._nearest_exposed_horizon's containment test now reads: `offset_km
    <= width_km / 2` asks "is a cuttable span of this service inside the cone
    footprint", where it used to ask the strictly weaker "is the midpoint of
    its whole path inside"."""
    if not spans:
        return float("inf")
    return float(MultiLineString(
        spans_to_km(spans, center_lat, center_lon)).distance(Point(0.0, 0.0)))


def radial_offset_km(center_lat: float, center_lon: float,
                     lat: float, lon: float) -> float:
    """Distance from a cone's centre to a point, under the same
    equirectangular flat-earth approximation events/geo.circle_polygon uses
    to draw the cone -- so an asset the polygon contains is exactly an asset
    whose offset is under the cone's half-width."""
    dlat = math.radians(lat - center_lat)
    dlon = math.radians(lon - center_lon) * math.cos(math.radians(center_lat))
    return EARTH_RADIUS_KM * math.hypot(dlat, dlon)


def expected_capacity_at_risk_gbps(p_cut: float, demand_gbps: float) -> float:
    """The single quantity every gold.rationale is arithmetic over."""
    return p_cut * demand_gbps


def cone_polygon(lat: float, lon: float, width_km: float) -> dict:
    """The GeoJSON footprint of a cone of cross-track diameter `width_km`
    centred at (lat, lon). Deliberately the same circle_polygon the scripted
    tracks already emit, so a cone hour and a track hour are the same kind of
    object to the geo mapper."""
    if width_km <= 0:
        raise ValueError(f"width_km must be > 0, got {width_km!r}")
    return circle_polygon(lat, lon, width_km / 2.0)
