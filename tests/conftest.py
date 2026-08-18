"""Shared fixtures land here as the suite grows."""
import os
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
