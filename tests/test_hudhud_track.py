"""Cyclone Hudhud (Oct 2014), CLAUDE.md build order step 4's demo track --
replaces Amphan for the storm scenario (see
docs/superpowers/specs/2026-08-07-storm-scenario-design.md for why: Amphan's
real track only ever exposes one aerial edge at one hour against the real
toy topology, which cannot produce the canonical both-legs-exposed case)."""
from storm_reoptimizer.events.hudhud_track import hudhud_track, EVENT_TYPE


def test_track_starts_at_landfall_and_ends_at_the_last_tracked_inland_point():
    events = hudhud_track(interval_hours=1.0)
    assert events[0].valid_at == "2014-10-12T06:30:00+00:00"
    assert events[-1].valid_at == "2014-10-14T09:00:00+00:00"
    assert all(e.event_type == EVENT_TYPE for e in events)


def test_landfall_event_matches_the_real_imd_landfall_position_and_radius():
    landfall = hudhud_track(interval_hours=1.0)[0]
    assert landfall.attributes["center"]["lat"] == 17.7
    assert landfall.attributes["center"]["lon"] == 83.3
    assert landfall.attributes["category"] == "very_severe_cyclonic_storm"
    assert landfall.attributes["radius_km"] == 150.0


def test_final_event_matches_the_real_imd_final_tracked_inland_position():
    final = hudhud_track(interval_hours=1.0)[-1]
    assert final.attributes["center"]["lat"] == 26.3
    assert final.attributes["center"]["lon"] == 81.8
    assert final.attributes["category"] == "depression"


def test_post_landfall_anchors_use_the_projects_established_damaging_wind_radius():
    # Every anchor after landfall has no published radius figure (confirmed
    # during design: IMD's Hudhud report gives no grade-based radius
    # convention and no post-landfall RMW data). They use this project's
    # own already-established 74km damaging-wind-radius convention
    # (amphan_track.py's module docstring), applied flat, so every anchor
    # from the SECOND one onward is exactly 74 -- interpolation between two
    # equal values is that same value. Only the landfall-to-second-anchor
    # bracket blends from 150 down to 74, so this checks from the second
    # anchor's own timestamp onward, not from the track's second sample
    # (which can still land inside that first, blending bracket).
    events = hudhud_track(interval_hours=1.0)
    second_anchor_time = "2014-10-12T12:00:00+00:00"
    post_landfall_radii = {
        e.attributes["radius_km"] for e in events if e.valid_at >= second_anchor_time
    }
    assert post_landfall_radii == {74.0}


def test_interval_hours_must_be_positive():
    import pytest
    with pytest.raises(ValueError):
        hudhud_track(interval_hours=0)
