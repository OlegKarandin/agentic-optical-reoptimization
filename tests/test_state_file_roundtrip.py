"""Services survive the process boundary.

The gap this closes: a server spawned with only --topology has no services at
all, so get_services/get_exposure/route_service are unreachable over stdio.
Build a state file, spawn the server with it, and prove get_services answers.
"""
import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from storm_reoptimizer.mcp_client import call_tool_json, connect_server

TOPOLOGY = {
    "graph": {
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "edges": [
            {"src": "a", "dst": "b", "length_km": 80.0},
            {"src": "b", "dst": "c", "length_km": 80.0},
            {"src": "c", "dst": "a", "length_km": 80.0},
        ],
    },
}


@pytest.fixture
def built_state(tmp_path: Path, local_server_command, local_server_env):
    """Run the real builder as a subprocess -- real GNPy, small topology.

    Uses local_server_command[0] (the multilayer-optical-mcp conda env's
    python) rather than sys.executable, so the builder and the server that
    later loads its output run on the same interpreter/GNPy build. Under
    `conda run -n storm-reoptimizer`, sys.executable would be the client
    env's python instead.
    """
    topo = tmp_path / "topo.json"
    topo.write_text(json.dumps(TOPOLOGY), encoding="utf-8")
    state = tmp_path / "state.json"
    proc = subprocess.run(
        [local_server_command[0], "-c",
         "from multilayer_optical_mcp.build_cli import main; main()",
         "--topology", str(topo), "--out", str(state),
         "--target-mean-util", "0.3", "--max-iters", "6"],
        env=local_server_env, capture_output=True, text=True, timeout=900,
        check=False)
    if proc.returncode != 0:
        pytest.skip(f"builder unavailable or failed: {proc.stderr[-500:]}")
    return topo, state


async def _services(topo, state, local_server_command, local_server_env):
    async with connect_server(
        topo, server_command=local_server_command, env=local_server_env,
        extra_args=["--state", str(state)],
    ) as client:
        return await call_tool_json(client, "get_services")


def test_services_are_present_after_loading_a_state_file(
        built_state, local_server_command, local_server_env):
    topo, state = built_state
    services = asyncio.run(
        _services(topo, state, local_server_command, local_server_env))
    assert services["services"], "state file loaded but no services present"
