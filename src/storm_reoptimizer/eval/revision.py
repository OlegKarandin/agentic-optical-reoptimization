# src/storm_reoptimizer/eval/revision.py
"""How much a row's `p_cut` can move when the next forecast issuance revises
the track (decider allocation redesign spec, 2026-09-10, section 5.1).

The 2026-09-09 run's timing reasoning read `p_cut` as a settled number:
"p_cut 0.914 is unlikely to fall" at T1's t0, where it read 0.117 an hour
later -- in BOTH halves, so the misreading was about the field and not about
the episode. `p_cut` says how likely the cut is IF THIS ISSUANCE IS RIGHT.
This module says how much that number can change when it is revised.

The model is deliberately the crudest thing that is honest: displace the
cone CENTRE by a declared radius in twelve directions and recompute the same
`p_cut_service` the row itself was computed with. It is not a second
probability model layered on the first -- same spans, same joint logic, same
width, same damage radius -- so `min`/`max`/`mean` are readings of the SAME
quantity under a moved centre, and the row's own `p_cut` is directly
comparable with them.

Two things it is NOT. It is not the raw cone geometry (`offset_km`,
`width_km`, the polygon), which was removed for being read as uncertainty
when it was corroborating detail behind a number already shown. And it is
not a prose warning, which cannot be compared against anything.

`revision_band` is `lru_cache`d because `runner.run_episode` rebuilds the
observation once per ITERATION (up to five per acting hour) with identical
geometry, and twelve `p_cut_service` evaluations per row at ~15 ms each is
the one place in this payload where that would be felt. Every argument is
hashable by construction: spans are tuples of coordinate pairs."""
from __future__ import annotations

from functools import lru_cache
from statistics import fmean

from ..events.geo import displace_km
from .cone import Segment, p_cut_service

# Twelve compass points, 30 degrees apart. Enough to bound the spread of a
# smooth function of a displaced centre without pretending to more angular
# resolution than an equirectangular flat-earth projection supports.
REVISION_BEARINGS = tuple(range(0, 360, 30))


@lru_cache(maxsize=4096)
def revision_band(working_spans: tuple[Segment, ...],
                  protection_spans: tuple[Segment, ...] | None,
                  center_lat: float, center_lon: float, *,
                  width_km: float, damage_radius_km: float,
                  radius_km: float) -> dict:
    """The smallest, largest and mean `p_cut` this service would show if the
    next issuance moved the cone centre by `radius_km` in any direction.

    Rounded to 3 decimals, matching the row's own `p_cut` and
    `derived.DERIVED_TOLERANCE` -- a band the twin-pair gate calls equal must
    not be shown to the agent at a precision that separates the halves."""
    if radius_km <= 0:
        base = round(p_cut_service(working_spans, protection_spans,
                                   center_lat, center_lon, width_km,
                                   damage_radius_km), 3)
        return {"revision_radius_km": float(radius_km),
                "min": base, "max": base, "mean": base}
    readings = []
    for bearing in REVISION_BEARINGS:
        lat, lon = displace_km(center_lat, center_lon, radius_km, bearing)
        readings.append(p_cut_service(working_spans, protection_spans, lat,
                                      lon, width_km, damage_radius_km))
    return {"revision_radius_km": float(radius_km),
            "min": round(min(readings), 3),
            "max": round(max(readings), 3),
            "mean": round(fmean(readings), 3)}
