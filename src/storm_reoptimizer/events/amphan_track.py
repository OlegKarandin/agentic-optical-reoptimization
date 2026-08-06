# src/storm_reoptimizer/events/amphan_track.py
"""Cyclone Amphan (May 2020), scripted as CLAUDE.md's demo data source:
"Script a known historical track ... as a sequence of GeoJSON cone polygons
... deterministic, repeatable, no API keys."

ANCHORS below are real: each point's time/lat/lon/category is individually
sourced from an IMD or PIB (India's Press Information Bureau) bulletin --
see each anchor's `source`. amphan_track()'s hourly sequence is built by
LINEAR interpolation between these anchors, not independently sourced for
every hour; real storm bulletins are 3-hourly at best.

Each anchor's `radius_km` (the hazard-footprint radius drawn around its
position) is a documented, approximate model, not a precise reconstruction:
grounded in IMD's two published radius-of-maximum-wind figures (278km on 18
May, 37km on 19 May -- a real, large contraction consistent with news
reports of an eyewall replacement cycle near peak intensity) plus reasoned
bounds from IMD's reported ~74km damaging-wind radius and >1,110km
cloud-shield extent for the depression/cyclonic-storm-stage anchors, where
no storm-specific radius figure was found.

The track ends at landfall (20 May, Sundarbans near Bakkhali, West Bengal).
Amphan's documented post-landfall decay ("degenerated into a well-marked
low pressure area" by 21 May) has no sourced position, so it is not
modeled here rather than fabricating one -- landfall is also the
operationally relevant end of the window for this app's asset-exposure
question.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from .contract import HazardEvent
from .geo import circle_polygon

EVENT_TYPE = "storm"

# Peak intensity is recorded separately, not tied to a fabricated position:
# IMD's 1030 UTC 18 May peak-intensity bulletin reported wind/pressure but
# no position. The 18 May ANCHORS point below is real-positioned for 1800
# UTC, ~7.5h after this peak.
PEAK_INTENSITY = {
    "time": "2020-05-18T10:30:00+00:00",
    "wind_kmh": 240,
    "pressure_hpa": 920,
    "category": "super_cyclonic_storm",
    "source": "IMD peak-intensity report via PIB, 18 May 2020 1030 UTC",
}

ANCHORS = [
    dict(time="2020-05-16T12:00:00+00:00", lat=10.9, lon=86.3,
         category="depression", wind_kmh=None, radius_km=100,
         source="PIB bulletin, 16 May 2020 1730 IST"),
    dict(time="2020-05-17T12:00:00+00:00", lat=12.0, lon=86.0,
         category="cyclonic_storm", wind_kmh=None, radius_km=120,
         source="PIB bulletin, 17 May 2020 1730 IST"),
    dict(time="2020-05-17T15:00:00+00:00", lat=12.3, lon=86.4,
         category="cyclonic_storm", wind_kmh=None, radius_km=130,
         source="IMD bulletin, 17 May 2020 1500 UTC"),
    dict(time="2020-05-18T18:00:00+00:00", lat=14.9, lon=86.5,
         category="super_cyclonic_storm", wind_kmh=None, radius_km=278,
         source="IMD bulletin, 18 May 2020 2330 IST; radius of maximum "
                "wind ~150nm/278km per IMD's 18 May report"),
    dict(time="2020-05-19T09:00:00+00:00", lat=16.5, lon=86.8,
         category="extremely_severe_cyclonic_storm", wind_kmh=None, radius_km=37,
         source="IMD/NASA bulletin, 19 May 2020 0900 UTC; radius of "
                "maximum wind ~20nm/37km per IMD's 19 May report "
                "(post-eyewall-replacement contraction)"),
    dict(time="2020-05-20T10:00:00+00:00", lat=21.65, lon=88.3,
         category="very_severe_cyclonic_storm", wind_kmh=155, radius_km=90,
         source="IMD landfall report, near Bakkhali/Sundarbans, West "
                "Bengal, 20 May 2020"),
]


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _interp(a: float, b: float, frac: float) -> float:
    return a + (b - a) * frac


def amphan_track(interval_hours: float = 1.0) -> list[HazardEvent]:
    """Cyclone Amphan as a sequence of HazardEvents, one per `interval_hours`
    from the first to the last anchor (inclusive). Each event's geometry is
    a circular hazard footprint (see module docstring) with position,
    radius, and category linearly interpolated between the bracketing real
    anchors."""
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
