# src/storm_reoptimizer/scenario_storm.py
"""Storm scenario pipeline (CLAUDE.md build order step 4): the deterministic,
no-LLM call sequence (map -> define risk group -> audit exposure -> disjoint
replan -> QoT verify -> validate), scripted against a real
multilayer-optical-mcp server. Every server interaction goes through
mcp_client.call_tool_json against a real subprocess -- no server-internals
imports, per CLAUDE.md's hard seam between the two repos. See
docs/superpowers/specs/2026-08-07-storm-scenario-design.md."""
from __future__ import annotations

from typing import Callable

from mcp.client import Client

from .events.contract import HazardEvent
from .geo_mapper import Edge, map_geo_event_to_assets
from .mcp_client import call_tool_json


def resolve_exposed_assets(
    edges: list[Edge], event: HazardEvent, filter_fn: Callable[[Edge], bool],
) -> list[Edge]:
    """Local, no MCP call: which real Edges does this hour's hazard
    geometry expose, per the event-type's vulnerability filter."""
    return map_geo_event_to_assets(event.geometry, edges, filter_fn)


async def edges_to_fiber_ids(client: Client, exposed_edges: list[Edge]) -> list[str]:
    """Resolve exposed Edges to the real fiber_* ids under each edge's OMS,
    via the server's own topology view. Every fiber under a matching OMS is
    included, not just one representative fiber -- an exposed ROADM-to-ROADM
    run's whole span inherits the exposure (see the geo-mapper design spec's
    modeling note)."""
    topo = await call_tool_json(client, "get_topology", {"layer": "optical"})
    pairs = {(e.src, e.dst) for e in exposed_edges} | {(e.dst, e.src) for e in exposed_edges}
    fiber_ids: list[str] = []
    for oms in topo["oms"]:
        if (oms["src_node_id"], oms["dst_node_id"]) in pairs:
            fiber_ids.extend(a for a in oms["elements"] if a.startswith("fiber_"))
    return fiber_ids


async def service_leg_oms_sequence(client: Client, ip_path: list[str]) -> list[str]:
    """Resolve a service leg's IP-link-id sequence (Service.working_path/
    protection_path's real shape) to the underlying OMS-id sequence, via the
    server's IP topology (each link carries its bound lightpath_id) and
    lightpath views (each lightpath carries its oms_sequence)."""
    ip_topo = await call_tool_json(client, "get_topology", {"layer": "ip"})
    lightpath_id_by_link = {
        link["id"]: link["lightpath_id"] for link in ip_topo["ip_links"]
    }
    lightpaths = await call_tool_json(client, "get_lightpaths", expect_list=True)
    oms_sequence_by_lightpath = {lp["id"]: lp["oms_sequence"] for lp in lightpaths}

    oms_sequence: list[str] = []
    for link_id in ip_path:
        lp_id = lightpath_id_by_link[link_id]
        oms_sequence.extend(oms_sequence_by_lightpath[lp_id])
    return oms_sequence
