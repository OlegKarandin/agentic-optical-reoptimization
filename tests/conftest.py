"""Shared fixtures land here as the suite grows."""
import json
import os
import subprocess
from pathlib import Path

import pytest

# Workaround for this local dev workspace only: the sibling server repo split
# (2026-08-18) into multilayer-optical-mcp-server (the MCP tool surface,
# package multilayer_optical_mcp) depending on multilayer-optical-network
# (the simulator library it used to contain). Both are now REGULAR (non -e)
# pip installs into the "multilayer-optical-mcp" conda env -- a real install
# from git, not this workspace's checkout -- which sidesteps the Cyrillic-path
# editable-install bug (...\Документы\... trips editable_wheel.py's
# _encode_pth -> UnicodeEncodeError, confirmed during step 1) for the package
# imports themselves. The bug still blocks the [project.scripts] console-
# script .exe from being generated on install, though, so tests still spawn
# the server by invoking its main() directly via `python -c` instead of by
# name. No PYTHONPATH override is needed any more: both packages already
# resolve from the env's site-packages.
_MULTILAYER_OPTICAL_MCP_ENV_PYTHON = Path(
    r"C:\Users\olegk\miniconda3\envs\multilayer-optical-mcp\python.exe"
)


@pytest.fixture
def local_server_command() -> list[str]:
    """server_command= override for connect_server(), valid only in this
    workspace. A real install (`multilayer-optical-mcp` on PATH) needs none
    of this -- see DEFAULT_SERVER_COMMAND in mcp_client.py."""
    if not _MULTILAYER_OPTICAL_MCP_ENV_PYTHON.exists():
        pytest.skip(
            f"multilayer-optical-mcp conda env python not found at "
            f"{_MULTILAYER_OPTICAL_MCP_ENV_PYTHON}; adjust conftest.py's "
            f"workaround paths for your machine"
        )
    return [
        str(_MULTILAYER_OPTICAL_MCP_ENV_PYTHON),
        "-c",
        "import sys; sys.argv = ['multilayer-optical-mcp', *sys.argv[1:]]; "
        "from multilayer_optical_mcp.server import main; main()",
    ]


@pytest.fixture
def local_server_env() -> dict[str, str]:
    """env= override for connect_server(), pairing with local_server_command:
    stdio_client's default subprocess env doesn't inherit the parent's
    environment at all, so this just forwards it (both packages are properly
    installed in the target conda env -- see local_server_command's
    docstring -- no PYTHONPATH override needed)."""
    return dict(os.environ)


_STORM_STATE_CACHE: dict = {}

TOY_INDIA_TOPOLOGY_PATH = (
    Path(__file__).parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)


@pytest.fixture
def storm_state_path(tmp_path_factory, local_server_command, local_server_env):
    """Offline-build this app's seeded demo services (storm-svc-1,
    unexposed-svc) into a state file for --state, once per test session
    (~2-3s real solve against the real 143-node toy topology -- see
    tools/build_storm_state.py). Cached at module level for the same
    ScopeMismatch reason test_state_file_roundtrip.py's built_state fixture
    documents. Returns the state file's Path; pair with
    TOY_INDIA_TOPOLOGY_PATH and connect_server(..., extra_args=["--state",
    str(path)])."""
    if "path" in _STORM_STATE_CACHE:
        return _STORM_STATE_CACHE["path"]
    tmp_path = tmp_path_factory.mktemp("storm_state")
    out = tmp_path / "toy_india_state.json"
    script = Path(__file__).parent.parent / "tools" / "build_storm_state.py"
    proc = subprocess.run(
        [local_server_command[0], str(script),
         "--topology", str(TOY_INDIA_TOPOLOGY_PATH), "--out", str(out)],
        env=local_server_env, capture_output=True, text=True, timeout=300,
        check=False)
    if proc.returncode != 0:
        pytest.fail(
            f"build_storm_state failed (rc={proc.returncode}): "
            f"stdout={proc.stdout[-500:]!r} stderr={proc.stderr[-500:]!r}"
        )
    _STORM_STATE_CACHE["path"] = out
    return out
