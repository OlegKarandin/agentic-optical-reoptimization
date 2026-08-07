"""Geo/asset mapper (CLAUDE.md build order step 3): pure Shapely + local
topology JSON, no MCP. See docs/superpowers/specs/2026-08-07-geo-mapper-design.md
for the Edge-vs-span-vs-OMS terminology this module deliberately uses."""
from pathlib import Path

from shapely.geometry import LineString, shape
from storm_reoptimizer.geo_mapper import Edge, load_edges, map_geo_event_to_assets
from storm_reoptimizer.events.amphan_track import amphan_track
from storm_reoptimizer.events.filters import get_filter

TOPOLOGY_PATH = (
    Path(__file__).parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)


def test_load_edges_returns_one_edge_per_topology_row():
    edges = load_edges(TOPOLOGY_PATH)
    assert len(edges) == 180  # 143 nodes / 180 edges, per step 1's topology

    by_pair = {(e.src, e.dst): e for e in edges}
    kolkata_kharagpur = by_pair[("kolkata", "kharagpur")]
    assert kolkata_kharagpur.mount_type == "buried"
    assert list(kolkata_kharagpur.geometry.coords) == [
        (88.36972, 22.56972),  # kolkata: lon, lat
        (87.33333, 22.33333),  # kharagpur: lon, lat
    ]


def test_map_geo_event_to_assets_requires_both_intersection_and_filter_match():
    intersecting_aerial = Edge(
        src="a", dst="b", mount_type="aerial",
        geometry=LineString([(0.0, 0.0), (2.0, 0.0)]),
    )
    intersecting_buried = Edge(
        src="c", dst="d", mount_type="buried",
        geometry=LineString([(0.0, 0.0), (2.0, 0.0)]),
    )
    distant_aerial = Edge(
        src="e", dst="f", mount_type="aerial",
        geometry=LineString([(10.0, 10.0), (12.0, 10.0)]),
    )
    edges = [intersecting_aerial, intersecting_buried, distant_aerial]
    hazard_geometry = {
        "type": "Polygon",
        "coordinates": [[[-1, -1], [3, -1], [3, 1], [-1, 1], [-1, -1]]],
    }

    result = map_geo_event_to_assets(
        hazard_geometry, edges, lambda e: e.mount_type == "aerial",
    )

    assert result == [intersecting_aerial]


def test_amphan_landfall_exposes_the_real_aerial_edge_near_kolkata():
    edges = load_edges(TOPOLOGY_PATH)
    landfall = amphan_track(interval_hours=1.0)[-1]

    exposed = map_geo_event_to_assets(
        landfall.geometry, edges, get_filter("storm"),
    )

    exposed_pairs = {(e.src, e.dst) for e in exposed}
    assert exposed_pairs == {("kharagpur", "bhubaneshwar")}
    hazard = shape(landfall.geometry)
    for e in exposed:
        assert e.mount_type == "aerial"
        assert hazard.intersects(e.geometry)


def test_amphan_landfall_intersects_buried_edges_the_storm_filter_excludes():
    # kolkata-kharagpur and kolkata-ranchi are both buried and both
    # genuinely inside the landfall hazard circle -- proving the filter is
    # doing real work, not just passing through everything that intersects.
    edges = load_edges(TOPOLOGY_PATH)
    landfall = amphan_track(interval_hours=1.0)[-1]
    hazard = shape(landfall.geometry)

    by_pair = {(e.src, e.dst): e for e in edges}
    kolkata_kharagpur = by_pair[("kolkata", "kharagpur")]
    kolkata_ranchi = by_pair[("kolkata", "ranchi")]
    assert hazard.intersects(kolkata_kharagpur.geometry)
    assert hazard.intersects(kolkata_ranchi.geometry)

    exposed_pairs = {
        (e.src, e.dst)
        for e in map_geo_event_to_assets(landfall.geometry, edges, get_filter("storm"))
    }
    assert ("kolkata", "kharagpur") not in exposed_pairs
    assert ("kolkata", "ranchi") not in exposed_pairs


def test_amphan_landfall_does_not_expose_a_distant_aerial_edge():
    # torangallu-bellary is aerial (so a mount_type-only bug wouldn't catch
    # this) but far south in Karnataka -- geometric distance, not the
    # filter, is what must exclude it.
    edges = load_edges(TOPOLOGY_PATH)
    landfall = amphan_track(interval_hours=1.0)[-1]

    exposed_pairs = {
        (e.src, e.dst)
        for e in map_geo_event_to_assets(landfall.geometry, edges, get_filter("storm"))
    }
    assert ("torangallu", "bellary") not in exposed_pairs
