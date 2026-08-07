# src/storm_reoptimizer/events/hudhud_track.py
"""Very Severe Cyclonic Storm HUDHUD (7-14 October 2014), CLAUDE.md build
order step 4's demo track (replaces Cyclone Amphan for the storm scenario --
see docs/superpowers/specs/2026-08-07-storm-scenario-design.md for why).

ANCHORS below are real: position/wind/grade at each point is sourced from
IMD RSMC New Delhi's "Very Severe Cyclonic Storm, HUDHUD over the Bay of
Bengal (07-14 October 2014): A Report" (Table 1, the official 3-hourly best
track). This module scripts a deliberately sparse subset of that table (ten
anchors spanning landfall through the last tracked inland point), the same
anchor density Amphan's own module uses -- linear interpolation between real
anchors, not independently sourced for every hour.

Unlike Amphan (brief landfall-and-dissipate), Hudhud's real track was
continuously tracked for ~54 hours after landfall, moving inland through
Chhattisgarh, Madhya Pradesh, and Uttar Pradesh before weakening to a
well-marked low near Lucknow -- this inland portion is what gives the storm
scenario genuine multi-hour, multi-edge real exposure against the toy
topology (see the design spec for the verified real numbers).

Radius modeling: the landfall anchor's radius (150km) is real, sourced to
the same report's Table 21 ("Verification of Gale Wind Forecast," the
0830 IST 12 Oct bulletin) -- its outer wind band (80-90 kmph) was forecast
over Koraput/Malkangiri districts, ~150-165km from the landfall point,
corroborated by the same report's Fig. 11 (storm damage photographed in
Jeypore town, Koraput district) -- the same "named district + independent
damage/observatory corroboration" evidentiary pattern Amphan's own landfall
anchor already used. EVERY anchor after landfall has NO published radius
figure: IMD's Hudhud report documents no grade-based radius convention (its
own methodology section, pp.52-53, references quadrant-wind-radius issuance
in principle but republishes none of it), and no post-landfall RMW/damaging-
wind-radius data exists in the report at all. Rather than inventing an
unsourced taper across intensity grades, every post-landfall anchor instead
reuses this project's own already-established 74km "damaging-wind-radius"
figure (amphan_track.py's module docstring, itself grounded in a real IMD
figure) flat across every remaining grade -- this is convention, explicitly
not a primary-sourced per-anchor figure, consistent with how this project
already treats convention differently from sourced data.

The track window is landfall through the last tracked inland point
(2014-10-12T06:30 through 2014-10-14T09:00, ~51 hours) -- Hudhud's
pre-landfall open-water approach (7-12 Oct) is real and documented but
contributes no real exposure against the toy topology (mirrors Amphan's own
windowing decision, see the geo-mapper plan's Outcome section) and is out of
this scenario's scripted window.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from .contract import HazardEvent
from .geo import circle_polygon

EVENT_TYPE = "storm"

ANCHORS = [
    dict(time="2014-10-12T06:30:00+00:00", lat=17.7, lon=83.3,
         category="very_severe_cyclonic_storm", wind_kmh=185, radius_km=150,
         source="IMD RSMC New Delhi HUDHUD report, Table 1 (landfall entry, "
                "Visakhapatnam, ~950 hPa, 100kt/185km/h 3-min sustained); "
                "radius from the same report's Table 21 gale-wind-forecast "
                "outer band (80-90 kmph over Koraput/Malkangiri districts, "
                "~150-165km out), corroborated by Fig. 11's storm-damage "
                "photo in Jeypore town, Koraput district."),
    dict(time="2014-10-12T12:00:00+00:00", lat=18.0, lon=82.7,
         category="severe_cyclonic_storm", wind_kmh=111, radius_km=74,
         source="IMD RSMC New Delhi HUDHUD report, Table 1. Radius: no "
                "post-landfall figure published; this project's established "
                "74km damaging-wind-radius convention (see module docstring)."),
    dict(time="2014-10-12T18:00:00+00:00", lat=18.7, lon=82.3,
         category="cyclonic_storm", wind_kmh=74, radius_km=74,
         source="IMD RSMC New Delhi HUDHUD report, Table 1. Radius: same "
                "convention as the prior anchor."),
    dict(time="2014-10-13T00:00:00+00:00", lat=19.5, lon=81.5,
         category="deep_depression", wind_kmh=56, radius_km=74,
         source="IMD RSMC New Delhi HUDHUD report, Table 1. Radius: same "
                "convention."),
    dict(time="2014-10-13T12:00:00+00:00", lat=21.3, lon=81.5,
         category="depression", wind_kmh=46, radius_km=74,
         source="IMD RSMC New Delhi HUDHUD report, Table 1 (~15km NW of "
                "Raipur). Radius: same convention."),
    dict(time="2014-10-13T18:00:00+00:00", lat=22.3, lon=81.5,
         category="depression", wind_kmh=46, radius_km=74,
         source="IMD RSMC New Delhi HUDHUD report, Table 1 (near Bilaspur, "
                "north Chhattisgarh). Radius: same convention."),
    dict(time="2014-10-14T00:00:00+00:00", lat=24.8, lon=81.5,
         category="depression", wind_kmh=46, radius_km=74,
         source="IMD RSMC New Delhi HUDHUD report, Table 1 (east Madhya "
                "Pradesh). Radius: same convention."),
    dict(time="2014-10-14T03:00:00+00:00", lat=25.1, lon=81.6,
         category="depression", wind_kmh=37, radius_km=74,
         source="IMD RSMC New Delhi HUDHUD report, Table 1 (~45km SW of "
                "Allahabad/Prayagraj). Radius: same convention."),
    dict(time="2014-10-14T06:00:00+00:00", lat=25.6, lon=81.7,
         category="depression", wind_kmh=37, radius_km=74,
         source="IMD RSMC New Delhi HUDHUD report, Table 1 (~23km NW of "
                "Allahabad/Prayagraj). Radius: same convention."),
    dict(time="2014-10-14T09:00:00+00:00", lat=26.3, lon=81.8,
         category="depression", wind_kmh=37, radius_km=74,
         source="IMD RSMC New Delhi HUDHUD report, Table 1 (east Uttar "
                "Pradesh, ~100km SE of Lucknow; weakened to a well-marked "
                "low pressure area at 1200 UTC, just after this point). "
                "Radius: same convention."),
]


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _interp(a: float, b: float, frac: float) -> float:
    return a + (b - a) * frac


def hudhud_track(interval_hours: float = 1.0) -> list[HazardEvent]:
    """Cyclone Hudhud as a sequence of HazardEvents, one per `interval_hours`
    from landfall to the last tracked inland point (inclusive). Each event's
    geometry is a circular hazard footprint with position, radius, and
    category linearly interpolated between the bracketing real anchors --
    same shape and interpolation approach as amphan_track()."""
    if interval_hours <= 0:
        raise ValueError("interval_hours must be > 0")
    start = _parse(ANCHORS[0]["time"])
    end = _parse(ANCHORS[-1]["time"])
    step = timedelta(hours=interval_hours)

    events: list[HazardEvent] = []
    t = start
    while t <= end:
        events.append(_event_at(t))
        t += step
    if events[-1].valid_at != ANCHORS[-1]["time"]:
        events.append(_event_at(end))
    return events


def _event_at(t: datetime) -> HazardEvent:
    lo, hi = ANCHORS[0], ANCHORS[-1]
    for a, b in zip(ANCHORS, ANCHORS[1:]):
        if _parse(a["time"]) <= t <= _parse(b["time"]):
            lo, hi = a, b
            break

    t0, t1 = _parse(lo["time"]), _parse(hi["time"])
    frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    frac = min(1.0, max(0.0, frac))

    lat = _interp(lo["lat"], hi["lat"], frac)
    lon = _interp(lo["lon"], hi["lon"], frac)
    radius_km = _interp(lo["radius_km"], hi["radius_km"], frac)
    category = lo["category"] if frac < 0.5 else hi["category"]

    return HazardEvent(
        geometry=circle_polygon(lat, lon, radius_km),
        event_type=EVENT_TYPE,
        valid_at=t.isoformat(),
        attributes={
            "category": category,
            "radius_km": round(radius_km, 1),
            "center": {"lat": round(lat, 4), "lon": round(lon, 4)},
        },
    )
