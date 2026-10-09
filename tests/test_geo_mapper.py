"""Geo/asset mapper (CLAUDE.md build order step 3): pure Shapely + local
topology JSON, no MCP. See docs/superpowers/specs/2026-08-07-geo-mapper-design.md
for the Edge-vs-span-vs-OMS terminology this module deliberately uses."""
from pathlib import Path

import pytest
from shapely.geometry import LineString as _LS
from shapely.geometry import LineString, shape
from storm_reoptimizer.geo_mapper import (
    Edge, amplifier_points, load_edges, map_geo_event_to_amplifiers,
    map_geo_event_to_assets,
)
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


def test_edge_crossing_clean_through_the_hazard_is_exposed():
    # Both ROADMs outside the footprint, cable passes straight through --
    # the case an endpoint-containment check would miss.
    crossing = Edge(src="x", dst="y", mount_type="aerial",
                     geometry=LineString([(-5.0, 0.0), (5.0, 0.0)]))
    hazard_geometry = {
        "type": "Polygon",
        "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]],
    }
    result = map_geo_event_to_assets(hazard_geometry, [crossing], lambda e: True)
    assert result == [crossing]


def test_edge_exactly_on_the_hazard_boundary_counts_as_exposed():
    # An edge lying exactly on the polygon's boundary (not crossing the interior)
    # still counts as exposed -- boundary contact is real physical exposure for a
    # storm, and Shapely's `intersects` (unlike `within`) includes boundary touching.
    on_boundary = Edge(src="p", dst="q", mount_type="aerial",
                        geometry=LineString([(-1.0, -1.0), (1.0, -1.0)]))
    hazard_geometry = {
        "type": "Polygon",
        "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]],
    }
    result = map_geo_event_to_assets(hazard_geometry, [on_boundary], lambda e: True)
    assert result == [on_boundary]


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


def _box(x0, x1, y0=-0.1, y1=0.1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1],
                                                [x0, y1], [x0, y0]]]}


def _oms(a, b, lengths):
    elements = [f"roadm_{a}", f"amp_{a}_{b}_booster"]
    for i, _ in enumerate(lengths):
        elements += [f"fiber_{a}_{b}_{i}", f"amp_{a}_{b}_{i}"]
    return ({"id": f"oms_{a}_{b}", "src_node_id": a, "dst_node_id": b,
             "elements": elements},
            {f"fiber_{a}_{b}_{i}": length for i, length in enumerate(lengths)})


EDGE = Edge(src="a", dst="b", mount_type="aerial", geometry=_LS([(0, 0), (2, 0)]))


def test_amp_is_placed_at_its_cumulative_distance_along_the_oms():
    fwd, lengths = _oms("a", "b", [30.0, 90.0])
    points = amplifier_points([fwd], lengths, [EDGE])
    assert points["amp_a_b_booster"].x == pytest.approx(0.0)
    assert points["amp_a_b_0"].x == pytest.approx(0.5)
    assert points["amp_a_b_1"].x == pytest.approx(2.0)


def test_orientation_follows_oms_node_order_and_directions_are_independent():
    fwd, lf = _oms("a", "b", [30.0, 90.0])
    rev, lr = _oms("b", "a", [30.0, 90.0])   # deliberately NOT mirrored
    hits = map_geo_event_to_amplifiers(_box(0.4, 0.6), [fwd, rev], {**lf, **lr},
                                       [EDGE], lambda e: True)
    assert hits == ("amp_a_b_0",)           # reverse amp_b_a_0 sits at x=1.5
    hits = map_geo_event_to_amplifiers(_box(1.4, 1.6), [fwd, rev], {**lf, **lr},
                                       [EDGE], lambda e: True)
    assert hits == ("amp_b_a_0",)


def test_co_located_huts_are_both_returned():
    fwd, lf = _oms("a", "b", [60.0, 60.0])
    rev, lr = _oms("b", "a", [60.0, 60.0])
    hits = map_geo_event_to_amplifiers(_box(0.9, 1.1), [fwd, rev], {**lf, **lr},
                                       [EDGE], lambda e: True)
    assert hits == ("amp_a_b_0", "amp_b_a_0")


def test_filter_excludes_amps_on_rejected_edges():
    buried = Edge(src="a", dst="b", mount_type="buried", geometry=_LS([(0, 0), (2, 0)]))
    fwd, lf = _oms("a", "b", [60.0, 60.0])
    assert map_geo_event_to_amplifiers(
        _box(0.9, 1.1), [fwd], lf, [buried],
        lambda e: e.mount_type == "aerial") == ()
