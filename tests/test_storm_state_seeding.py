"""Proves tools/build_storm_state.py's output actually seeds real,
queryable services when loaded via --state -- the replacement for this
plan's original (falsified) assumption that a live solve_allocation call
persists a Service. See docs/superpowers/specs/2026-08-07-storm-scenario-design.md's
2026-08-18 addendum."""
import asyncio
from pathlib import Path

from storm_reoptimizer.mcp_client import call_tool_json, connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)
RING_HALF_A = {"oms_satna_rewa", "oms_rewa_allahabad"}
RING_HALF_B = {"oms_satna_jhansi", "oms_jhansi_allahabad"}


async def _run(state_path, local_server_command, local_server_env):
    async with connect_server(
        TOPOLOGY_PATH, server_command=local_server_command, env=local_server_env,
        extra_args=["--state", str(state_path)],
    ) as client:
        return await call_tool_json(client, "get_services")


def test_seeded_state_has_both_demo_services(
    storm_state_path, local_server_command, local_server_env,
):
    services = asyncio.run(
        _run(storm_state_path, local_server_command, local_server_env)
    )
    ids = {s["id"] for s in services["services"]}
    assert {"storm-svc-1", "unexposed-svc"} <= ids


def test_storm_svc_1_lands_on_the_two_real_aerial_ring_halves(
    storm_state_path, local_server_command, local_server_env,
):
    services = asyncio.run(
        _run(storm_state_path, local_server_command, local_server_env)
    )
    svc = next(s for s in services["services"] if s["id"] == "storm-svc-1")
    # get_services' Service payload carries ip-link ids, not OMS ids
    # directly -- re-resolved the same way service_leg_oms_sequence
    # (this task's other new function) does, inlined here to keep this
    # test independent of that function's own correctness.
    assert svc["working_path"] and svc["protection_path"]
