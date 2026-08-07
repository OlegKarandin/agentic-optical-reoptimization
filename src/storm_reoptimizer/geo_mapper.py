# src/storm_reoptimizer/geo_mapper.py
"""Geo/asset mapper (CLAUDE.md build order step 3): map_geo_event_to_assets
intersects a hazard event's geometry against the toy topology's real edges
and filters by physical vulnerability. Pure Shapely + local topology JSON --
no MCP calls. Edge is this module's unit: a ROADM-to-ROADM link (what the
server calls an OMS once loaded), not a "span" (the server's
amplifier-spaced fiber segments -- this local topology has no per-amplifier
geometry to model that granularity). See
docs/superpowers/specs/2026-08-07-geo-mapper-design.md for the full
rationale."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from shapely.geometry import LineString, shape


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
