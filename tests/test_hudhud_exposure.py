"""Real-data proof that Cyclone Hudhud's track, run through the existing
geo-mapper, produces genuine multi-hour, multi-edge exposure -- unlike
Amphan's single edge/single hour (see
docs/superpowers/specs/2026-08-07-storm-scenario-design.md). Numbers below
were computed directly against real data while writing this plan; this test
locks them in as a regression guard, not a fresh derivation."""
from pathlib import Path

from storm_reoptimizer.events.filters import get_filter
from storm_reoptimizer.events.hudhud_track import hudhud_track
from storm_reoptimizer.geo_mapper import load_edges, map_geo_event_to_assets

TOPOLOGY_PATH = (
    Path(__file__).parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)


def _exposed_pairs_by_hour() -> dict[str, set[tuple[str, str]]]:
    edges = load_edges(TOPOLOGY_PATH)
    filter_fn = get_filter("storm")
    out = {}
    for event in hudhud_track(interval_hours=1.0):
        exposed = map_geo_event_to_assets(event.geometry, edges, filter_fn)
        out[event.valid_at] = {(e.src, e.dst) for e in exposed}
    return out


def test_no_exposure_before_the_storm_closes_on_the_corridor():
    by_hour = _exposed_pairs_by_hour()
    quiet_hours = [v for k, v in by_hour.items() if k < "2014-10-13T22:30"]
    assert quiet_hours and all(v == set() for v in quiet_hours)


def test_both_ring_halves_are_exposed_together_at_least_one_hour():
    # The canonical both-legs case needs both satna-jhansi and satna-rewa
    # exposed at some point the risk group's union will include -- this
    # confirms the real geo-mapper output actually delivers that, not just
    # the topology-structure argument in the design spec. satna-jabalpur is
    # also exposed the same hour: it's the third aerial direction added in
    # the exposure-and-depot design (§3.2, Option B) and it sits inside the
    # same storm cone as the other two satna legs.
    by_hour = _exposed_pairs_by_hour()
    assert by_hour["2014-10-13T23:30:00+00:00"] == {
        ("satna", "jhansi"), ("satna", "rewa"), ("satna", "jabalpur"),
    }


def test_satna_rewa_stays_exposed_across_five_consecutive_hours():
    by_hour = _exposed_pairs_by_hour()
    consecutive = [
        "2014-10-13T22:30:00+00:00", "2014-10-13T23:30:00+00:00",
        "2014-10-14T00:30:00+00:00", "2014-10-14T01:30:00+00:00",
        "2014-10-14T02:30:00+00:00",
    ]
    for hour in consecutive:
        assert ("satna", "rewa") in by_hour[hour]


def test_a_third_distinct_edge_is_exposed_later_in_the_track():
    # Proves this isn't just the satna/rewa/jhansi cluster -- a genuinely
    # different edge, hours later, is also real exposure.
    by_hour = _exposed_pairs_by_hour()
    assert ("hadiagarh", "sitapur") in by_hour["2014-10-14T09:00:00+00:00"]
    assert ("hadiagarh", "sitapur") not in by_hour["2014-10-13T23:30:00+00:00"]
