"""Services survive the process boundary.

The gap this closes: a server spawned with only --topology has no services at
all, so get_services/get_exposure/route_service are unreachable over stdio.
Build a state file, spawn the server with it, and prove get_services answers.
"""
import asyncio
import json
import subprocess

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


# Module-scoping `built_state` isn't an option: it would need to depend on
# local_server_command/local_server_env (conftest.py), which are function
# -scoped, and pytest raises ScopeMismatch for a broader-scoped fixture
# depending on a narrower one. Cache the build result at module level
# instead, keyed off nothing but presence -- the fixture itself stays
# function-scoped (so it still composes with conftest.py's fixtures) but
# the real subprocess build only ever runs once per test session, and the
# two tests below share its output.
_build_cache: dict = {}


@pytest.fixture
def built_state(tmp_path_factory, local_server_command, local_server_env):
    """Run the real builder as a subprocess -- real GNPy, small topology.

    Uses local_server_command[0] (the multilayer-optical-mcp conda env's
    python) rather than sys.executable, so the builder and the server that
    later loads its output run on the same interpreter/GNPy build. Under
    `conda run -n storm-reoptimizer`, sys.executable would be the client
    env's python instead.

    Cached in `_build_cache` (see module comment above) so the two tests
    below -- the positive and negative case for the same question -- share
    one real build instead of running it twice.
    """
    if "result" in _build_cache:
        return _build_cache["result"]
    tmp_path = tmp_path_factory.mktemp("state_file_roundtrip")
    topo = tmp_path / "topo.json"
    topo.write_text(json.dumps(TOPOLOGY), encoding="utf-8")
    state = tmp_path / "state.json"
    proc = subprocess.run(
        [local_server_command[0], "-c",
         "from multilayer_optical_network.build_cli import main; main()",
         "--topology", str(topo), "--out", str(state),
         "--target-mean-util", "0.3", "--max-iters", "6"],
        env=local_server_env, capture_output=True, text=True, timeout=900,
        check=False)
    if proc.returncode != 0:
        pytest.fail(
            f"builder failed (rc={proc.returncode}): "
            f"stdout={proc.stdout[-500:]!r} stderr={proc.stderr[-500:]!r}"
        )
    _build_cache["result"] = (topo, state)
    return topo, state


async def _services(topo, local_server_command, local_server_env, extra_args=None):
    async with connect_server(
        topo, server_command=local_server_command, env=local_server_env,
        extra_args=extra_args,
    ) as client:
        return await call_tool_json(client, "get_services")


def test_services_are_present_after_loading_a_state_file(
        built_state, local_server_command, local_server_env):
    topo, state = built_state
    services = asyncio.run(
        _services(topo, local_server_command, local_server_env,
                   extra_args=["--state", str(state)]))
    assert services["services"], "state file loaded but no services present"


def test_no_services_without_a_state_file(
        built_state, local_server_command, local_server_env):
    """Negative control for the above: a server spawned with --topology
    only (no --state) has no services at all -- the gap this whole test
    module exists to close. Reuses built_state's topology so no second
    build runs; the point here is the spawn, not the artifact."""
    topo, _state = built_state
    services = asyncio.run(
        _services(topo, local_server_command, local_server_env))
    assert services["services"] == [], (
        "server spawned without --state unexpectedly has services")
