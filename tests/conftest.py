"""Shared fixtures land here as the suite grows."""
import os
import sys
from pathlib import Path

import pytest

# Workaround for this local dev workspace only: `pip install -e` is broken
# for both repos here because they sit under a path with Cyrillic characters
# (...\Документы\...), which trips a real setuptools bug (editable_wheel.py's
# _encode_pth opens its wrapper file without an explicit encoding and falls
# back to the Windows ANSI codepage -> UnicodeEncodeError). Confirmed during
# Task 1 that neither PYTHONUTF8=1 nor --no-build-isolation fixes it cleanly.
# See docs/superpowers/plans/2026-08-06-step1-server-topology-roundtrip.md's
# Global Constraints for detail. Until that's sorted out for real (move the
# repos outside the Cyrillic path, or fix the pbr/pkg_resources issue that
# --no-build-isolation surfaces), multilayer-optical-mcp's console script is
# never installed, so tests spawn it by invoking its main() directly instead.
_MULTILAYER_OPTICAL_MCP_ENV_PYTHON = Path(
    r"C:\Users\olegk\miniconda3\envs\multilayer-optical-mcp\python.exe"
)
_MULTILAYER_OPTICAL_MCP_SRC = (
    Path(__file__).parent.parent.parent / "multilayer-optical-mcp" / "src"
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
    puts multilayer-optical-mcp's src/ on the spawned process's PYTHONPATH,
    since it isn't installed (see local_server_command's docstring) and
    stdio_client's default subprocess env doesn't inherit PYTHONPATH."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_MULTILAYER_OPTICAL_MCP_SRC)
    return env
