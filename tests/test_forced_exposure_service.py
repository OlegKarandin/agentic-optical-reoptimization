"""Proves the satna<->allahabad demand (design spec: a real 4-node topology
ring, both halves aerial) lands its working and protection legs on the two
real ring-half OMSes -- not asserted from the direct-model investigation
done while writing the plan, re-proven here against the real MCP tool
surface, over the offline-seeded state (tools/build_storm_state.py)."""
import asyncio
from pathlib import Path

from storm_reoptimizer.mcp_client import call_tool_json, connect_server
from storm_reoptimizer.scenario_storm import service_leg_oms_sequence

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
        services = await call_tool_json(client, "get_services")
        svc = next(s for s in services["services"] if s["id"] == "storm-svc-1")
        working_oms = set(await service_leg_oms_sequence(client, svc["working_path"]))
        protection_oms = set(await service_leg_oms_sequence(client, svc["protection_path"]))
        return working_oms, protection_oms


def test_satna_allahabad_demand_lands_on_the_two_real_aerial_ring_halves(
    storm_state_path, local_server_command, local_server_env,
):
    working_oms, protection_oms = asyncio.run(
        _run(storm_state_path, local_server_command, local_server_env)
    )

    # The builder's own choice of which leg is "working" vs "protection" is
    # not fixed by this test -- only that the PAIR is exactly the two real
    # ring halves, each leg getting a different one.
    assert {frozenset(working_oms), frozenset(protection_oms)} == {frozenset(RING_HALF_A), frozenset(RING_HALF_B)}
