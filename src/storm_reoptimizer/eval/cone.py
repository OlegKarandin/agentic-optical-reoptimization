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
