"""Cyclone Amphan (May 2020) scripted historical track -- CLAUDE.md build
order step 2. Anchor data is real (IMD/PIB bulletins, see amphan_track.py's
ANCHORS list for per-point sources); the hourly sequence is linear
interpolation between those anchors, not independently sourced per hour."""
import math
from datetime import datetime

from storm_reoptimizer.events.amphan_track import ANCHORS, amphan_track

KOLKATA_LAT, KOLKATA_LON = 22.5726, 88.3639


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def test_anchors_are_chronological_and_within_bay_of_bengal():
    times = [datetime.fromisoformat(a["time"]) for a in ANCHORS]
    assert times == sorted(times)
    assert len(set(times)) == len(times)
    for a in ANCHORS:
        assert 5.0 <= a["lat"] <= 25.0
        assert 80.0 <= a["lon"] <= 92.0


def test_amphan_track_returns_hourly_events_spanning_first_to_last_anchor():
    events = amphan_track(interval_hours=1.0)
    assert events[0].valid_at == ANCHORS[0]["time"]
    assert events[-1].valid_at == ANCHORS[-1]["time"]
    stamps = [datetime.fromisoformat(e.valid_at) for e in events]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_every_event_has_a_closed_polygon_geometry_and_storm_type():
    events = amphan_track(interval_hours=6.0)
    assert len(events) > 1
    for e in events:
        assert e.event_type == "storm"
        assert e.geometry["type"] == "Polygon"
        ring = e.geometry["coordinates"][0]
        assert len(ring) >= 5
        assert ring[0] == ring[-1]


def test_landfall_event_matches_known_landfall_position_and_category():
    events = amphan_track(interval_hours=1.0)
    landfall = events[-1]
    assert landfall.valid_at == "2020-05-20T10:00:00+00:00"
    assert landfall.attributes["category"] == "very_severe_cyclonic_storm"
    assert abs(landfall.attributes["center"]["lat"] - 21.65) < 0.01
    assert abs(landfall.attributes["center"]["lon"] - 88.3) < 0.01


def test_radius_matches_sourced_rmw_figures_at_their_bulletin_times():
    events = {e.valid_at: e for e in amphan_track(interval_hours=1.0)}
    # 18 May 2020 1800 UTC: IMD-reported radius of maximum wind ~278km.
    assert abs(events["2020-05-18T18:00:00+00:00"].attributes["radius_km"] - 278) < 1
    # 19 May 2020 0900 UTC: IMD-reported radius of maximum wind ~37km
    # (post-eyewall-replacement contraction).
    assert abs(events["2020-05-19T09:00:00+00:00"].attributes["radius_km"] - 37) < 1


def test_track_passes_near_kolkata():
    # Amphan's real landfall/post-landfall track devastated Kolkata -- this
    # is the connective thread to storm-reoptimizer's toy_india_topology.json,
    # which already has a real "kolkata" node.
    events = amphan_track(interval_hours=1.0)
    min_dist = min(
        _haversine_km(KOLKATA_LAT, KOLKATA_LON, e.attributes["center"]["lat"], e.attributes["center"]["lon"])
        for e in events
    )
    assert min_dist < 150.0


def test_landfall_radius_reflects_gale_wind_extent_not_just_rmw_core():
    # IMD landfall advisory (via PIB, 20 May 2020): gale winds of 110-120
    # kmph gusting to 130 kmph forecast over Kolkata, Hoogli, Howrah and
    # West Medinipur districts during landfall -- West Medinipur contains
    # Kharagpur, ~125km from the landfall point. A radius-of-maximum-wind
    # figure (the storm's tightest high-wind core) undercounts this real,
    # broader damaging-wind extent.
    events = amphan_track(interval_hours=1.0)
    landfall = events[-1]
    assert landfall.attributes["radius_km"] >= 128.0  # 125.5km is the real threshold at which
    # kharagpur-bhubaneshwar enters the hazard footprint (see test_geo_mapper.py's
    # test_amphan_landfall_exposes_the_real_aerial_edge_near_kolkata) -- keep headroom above it
    # so this guard actually protects that integration test rather than permitting a value
    # that would silently break it.


def test_interval_hours_must_be_positive():
    import pytest
    with pytest.raises(ValueError):
        amphan_track(interval_hours=0)
