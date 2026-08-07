"""Geo/asset mapper (CLAUDE.md build order step 3): pure Shapely + local
topology JSON, no MCP. See docs/superpowers/specs/2026-08-07-geo-mapper-design.md
for the Edge-vs-span-vs-OMS terminology this module deliberately uses."""
from pathlib import Path

from shapely.geometry import LineString
from storm_reoptimizer.geo_mapper import Edge, load_edges, map_geo_event_to_assets

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
