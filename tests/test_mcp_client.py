import asyncio
from pathlib import Path

from storm_reoptimizer.mcp_client import call_tool_json, connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)


def test_connect_server_can_call_get_topology(local_server_command, local_server_env):
    async def run():
        async with connect_server(
            TOPOLOGY_PATH,
            server_command=local_server_command,
            env=local_server_env,
        ) as client:
            return await call_tool_json(client, "get_topology", {"layer": "optical"})

    data = asyncio.run(run())
    oms_ids = {o["id"] for o in data["oms"]}
    assert "oms_delhi_mathura" in oms_ids
    assert "oms_ratlam_ujjain" in oms_ids
    # 180 edges -> 360 directed OMS (one per direction), see
    # toy_india_topology.json / test_toy_india_topology.py.
    assert len(oms_ids) == 360
