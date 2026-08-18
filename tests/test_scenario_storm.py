"""scenario_storm.py: the reusable step-4 pipeline (CLAUDE.md's "spine"),
each function a thin wrapper over real MCP tool calls. See
docs/superpowers/specs/2026-08-07-storm-scenario-design.md."""
import asyncio
from pathlib import Path

from storm_reoptimizer.events.filters import get_filter
from storm_reoptimizer.events.hudhud_track import hudhud_track
from storm_reoptimizer.geo_mapper import Edge, load_edges
from storm_reoptimizer.mcp_client import call_tool_json, connect_server
from storm_reoptimizer.scenario_storm import (
    edges_to_fiber_ids, resolve_exposed_assets, service_leg_oms_sequence,
)

TOPOLOGY_PATH = (
    Path(__file__).parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)


def test_resolve_exposed_assets_delegates_to_the_real_geo_mapper():
    edges = load_edges(TOPOLOGY_PATH)
    both_ring_halves_hour = next(
        e for e in hudhud_track(interval_hours=1.0)
        if e.valid_at == "2014-10-13T23:30:00+00:00"
    )
    exposed = resolve_exposed_assets(edges, both_ring_halves_hour, get_filter("storm"))
    assert {(e.src, e.dst) for e in exposed} == {
        ("satna", "jhansi"), ("satna", "rewa"),
    }


async def _run_edges_to_fiber_ids(local_server_command, local_server_env):
    edges = [
        Edge(src="satna", dst="jhansi", mount_type="aerial", geometry=None),
        Edge(src="satna", dst="rewa", mount_type="aerial", geometry=None),
    ]
    async with connect_server(
        TOPOLOGY_PATH, server_command=local_server_command, env=local_server_env,
    ) as client:
        return await edges_to_fiber_ids(client, edges)


def test_edges_to_fiber_ids_resolves_every_fiber_under_each_matching_oms(
    local_server_command, local_server_env,
):
    fiber_ids = asyncio.run(_run_edges_to_fiber_ids(local_server_command, local_server_env))
    assert fiber_ids  # non-empty: the real OMSes exist and carry real fibers
    assert all(f.startswith("fiber_") for f in fiber_ids)
    assert any("satna_jhansi" in f or "jhansi_satna" in f for f in fiber_ids)
    assert any("satna_rewa" in f or "rewa_satna" in f for f in fiber_ids)


async def _run_service_leg_oms_sequence(
    state_path, local_server_command, local_server_env,
):
    async with connect_server(
        TOPOLOGY_PATH, server_command=local_server_command, env=local_server_env,
        extra_args=["--state", str(state_path)],
    ) as client:
        services = await call_tool_json(client, "get_services")
        svc = next(s for s in services["services"] if s["id"] == "storm-svc-1")
        working_oms = await service_leg_oms_sequence(client, svc["working_path"])
        protection_oms = await service_leg_oms_sequence(client, svc["protection_path"])
        return working_oms, protection_oms


def test_service_leg_oms_sequence_resolves_ip_links_to_real_oms_ids(
    storm_state_path, local_server_command, local_server_env,
):
    working_oms, protection_oms = asyncio.run(
        _run_service_leg_oms_sequence(
            storm_state_path, local_server_command, local_server_env
        )
    )
    both = {*working_oms, *protection_oms}
    assert {"oms_satna_jhansi", "oms_jhansi_allahabad"} <= both \
        and {"oms_satna_rewa", "oms_rewa_allahabad"} <= both
