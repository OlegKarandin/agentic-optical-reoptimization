# src/storm_reoptimizer/geo_mapper.py
"""Geo/asset mapper (CLAUDE.md build order step 3): map_geo_event_to_assets
intersects a hazard event's geometry against the toy topology's real edges
and filters by physical vulnerability. Pure Shapely + local topology JSON --
no MCP calls. Edge is this module's unit: a ROADM-to-ROADM link (what the
server calls an OMS once loaded), not a "span" (the server's
amplifier-spaced fiber segments -- this local topology has no per-amplifier
geometry of its own; `amplifier_points` places amplifiers from the SERVER's
per-OMS fibre order and lengths (heat spec 2026-10-08 §3.2)). See
docs/superpowers/specs/2026-08-07-geo-mapper-design.md for the full
rationale."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from shapely.geometry import LineString, Point, shape


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    mount_type: str
    geometry: LineString


def load_edges(path: str | Path) -> list[Edge]:
    """Build one Edge per entry in the topology JSON's graph.edges, with
    geometry as a straight line between the two endpoint nodes' [lon, lat]
    coordinates (matching the coordinate order storm_reoptimizer.events.geo
    already uses for hazard geometry)."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    nodes = {n["id"]: (n["lon"], n["lat"]) for n in data["graph"]["nodes"]}
    return [
        Edge(
            src=e["src"],
            dst=e["dst"],
            mount_type=e["mount_type"],
            geometry=LineString([nodes[e["src"]], nodes[e["dst"]]]),
        )
        for e in data["graph"]["edges"]
    ]


def map_geo_event_to_assets(
    geometry: dict, edges: list[Edge], filter_fn: Callable[[Edge], bool],
) -> list[Edge]:
    """Return the edges that both intersect the hazard `geometry` (a
    GeoJSON geometry dict, e.g. a storm cone polygon) and satisfy
    `filter_fn` (a physical-vulnerability predicate, e.g. "is this edge
    aerial" for a storm -- see events/filters.py)."""
    hazard = shape(geometry)
    return [e for e in edges if hazard.intersects(e.geometry) and filter_fn(e)]


def amplifier_points(oms: Iterable[dict], fiber_length_km: Mapping[str, float],
                     edges: Iterable[Edge],
                     filter_fn: Callable[[Edge], bool] = lambda e: True,
                     ) -> dict[str, Point]:
    """Where every amplifier physically sits (heat spec §3.2, plan deviation
    on D8): amp `i` of directed OMS a->b sits at the cumulative length of the
    fibres before it, measured from `a` along the straight edge a->b; the
    booster sits at `a`. Fibre order and lengths come from the SERVER's
    `get_topology(layer="optical")` -- `oms[].elements` and
    `fibers[].length_km` -- never from a local re-implementation of the
    sibling library's span split."""
    by_pair = {frozenset((e.src, e.dst)): e for e in edges}
    out: dict[str, Point] = {}
    for o in oms:
        a, b = o["src_node_id"], o["dst_node_id"]
        edge = by_pair.get(frozenset((a, b)))
        if edge is None or not filter_fn(edge):
            continue
        line = (edge.geometry if (edge.src, edge.dst) == (a, b)
                else LineString(list(edge.geometry.coords)[::-1]))
        fibres = [el for el in o["elements"] if el.startswith("fiber_")]
        total = sum(fiber_length_km[f] for f in fibres)
        cumulative = 0.0
        for el in o["elements"]:
            if el.startswith("fiber_"):
                cumulative += fiber_length_km[el]
            elif el.startswith("amp_"):
                out[el] = line.interpolate(cumulative / total, normalized=True)
    return out


def map_geo_event_to_amplifiers(geometry: dict, oms: Iterable[dict],
                                fiber_length_km: Mapping[str, float],
                                edges: Iterable[Edge],
                                filter_fn: Callable[[Edge], bool]) -> tuple[str, ...]:
    """Every amplifier whose hut lies inside (or on the boundary of) the
    hazard geometry, on an edge the event filter admits. Sorted. Both
    directions are mapped independently; on this topology they share huts,
    so a polygon normally returns the co-located pair."""
    hazard = shape(geometry)
    return tuple(sorted(amp for amp, point in amplifier_points(
        oms, fiber_length_km, edges, filter_fn).items() if hazard.intersects(point)))
