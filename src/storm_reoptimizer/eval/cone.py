# src/storm_reoptimizer/eval/cone.py
"""Cone containment semantics (eval design spec, "Cone semantics", build
order item 1). This is the module that makes every gold timing label
DERIVABLE rather than asserted: P(cut) x outage_cost against
spare_opportunity_cost, recorded in each scenario's gold.rationale.

The model, stated once so nobody has to reverse-engineer it:

  * A forecast cone at horizon h is published with a cross-track diameter
    `width_km` and the NHC convention that it contains the realized track
    with probability P = 0.66.
  * So the track's cross-track error is modelled as N(0, sigma) with sigma
    calibrated to make P(|x| <= width_km/2) = 0.66.
  * An asset sitting `offset_km` off the cone axis is cut if the realized
    track passes within `damage_radius_km` of it -- its cut probability is
    therefore the mass of that Gaussian over [offset - r, offset + r].

`offset_km` is measured RADIALLY from the cone centre rather than strictly
cross-track. For this project's hazard footprints -- circles, per
events/geo.circle_polygon, which every scripted track already emits -- the
two are the same quantity, and distinguishing them would be false precision
on top of an equirectangular flat-earth approximation."""
from __future__ import annotations

import math
from statistics import NormalDist

from ..events.geo import EARTH_RADIUS_KM, circle_polygon

# NHC convention: the forecast cone contains the realized track ~2/3 of the
# time. Fixed by the spec, not a tunable.
CONTAINMENT_P = 0.66

_STD_NORMAL = NormalDist()
# z such that P(|x| <= z) = CONTAINMENT_P for a standard normal.
_Z_FOR_CONTAINMENT = _STD_NORMAL.inv_cdf((1.0 + CONTAINMENT_P) / 2.0)


def cross_track_sigma_km(width_km: float) -> float:
    """Std dev of the track's cross-track error implied by a cone of
    diameter `width_km`, calibrated so the cone's own half-width captures
    CONTAINMENT_P of the distribution."""
    if width_km <= 0:
        raise ValueError(f"width_km must be > 0, got {width_km!r}")
    return (width_km / 2.0) / _Z_FOR_CONTAINMENT


def cut_probability(offset_km: float, width_km: float,
                    damage_radius_km: float) -> float:
    """P(an asset `offset_km` off the cone axis is cut), i.e. P(the realized
    track passes within `damage_radius_km` of it). Monotonically decreasing
    in `offset_km`, and on the axis also in `width_km`."""
    if damage_radius_km <= 0:
        raise ValueError(f"damage_radius_km must be > 0, got {damage_radius_km!r}")
    sigma = cross_track_sigma_km(width_km)
    hi = (offset_km + damage_radius_km) / sigma
    lo = (offset_km - damage_radius_km) / sigma
    return _STD_NORMAL.cdf(hi) - _STD_NORMAL.cdf(lo)


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
