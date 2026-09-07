"""Shared fixtures land here as the suite grows."""
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from storm_reoptimizer.mcp_client import connect_server

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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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
    documents. Returns the state file's Path, for use with
    connect_server(..., extra_args=["--state", str(path)])."""
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


EVAL_STATES_DIR = Path(__file__).parent.parent / "eval" / "states"
EVAL_SEED = 17


@pytest.fixture(scope="session")
def loaded_state_path(local_server_command, local_server_env) -> Path:
    """The eval harness's loaded operating network (tools/build_eval_state.py).
    Built once and CACHED ON DISK at eval/states/loaded-s17.json (git-ignored),
    not merely for the session: the two-stage build binary-searches offered
    demand scale against the real 143-node topology and takes minutes, not
    seconds. Delete the file to force a rebuild."""
    out = EVAL_STATES_DIR / f"loaded-s{EVAL_SEED}.json"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).parent.parent / "tools" / "build_eval_state.py"
    proc = subprocess.run(
        [local_server_command[0], str(script),
         "--topology", str(TOY_INDIA_TOPOLOGY_PATH), "--out", str(out),
         "--seed", str(EVAL_SEED)],
        env=local_server_env, capture_output=True, text=True, timeout=1800,
        check=False)
    if proc.returncode != 0:
        pytest.fail(
            f"build_eval_state failed (rc={proc.returncode}): "
            f"stdout={proc.stdout[-500:]!r} stderr={proc.stderr[-1500:]!r}")
    return out


REPO_ROOT = Path(__file__).parent.parent
# scenario.state_file -> the build_eval_state.PIN_SETS name that produces it
# from loaded-s17.json (None for the base itself). T2/T3 probe redesign,
# spec §7: each pair has its own state file.
EVAL_STATE_PIN_SETS = {
    "eval/states/loaded-s17.json": None,
    "eval/states/t2-jalgaon-s17.json": "t2",
    "eval/states/t3-jalgaon-s17.json": "t3",
}


@pytest.fixture(scope="session")
def eval_state_paths(loaded_state_path, local_server_command,
                     local_server_env) -> dict[str, Path]:
    """Every state file a shipped scenario may name, built if missing or
    older than the base it is pinned onto (a rebuilt base invalidates the
    pinned states). Keyed by the scenario's own `state_file` string."""
    script = REPO_ROOT / "tools" / "build_eval_state.py"
    paths: dict[str, Path] = {}
    for state_file, pin_set in EVAL_STATE_PIN_SETS.items():
        out = REPO_ROOT / state_file
        if pin_set is None:
            assert out.resolve() == loaded_state_path.resolve()
        elif (not out.exists()
              or out.stat().st_mtime < loaded_state_path.stat().st_mtime):
            proc = subprocess.run(
                [local_server_command[0], str(script),
                 "--topology", str(TOY_INDIA_TOPOLOGY_PATH), "--out", str(out),
                 "--seed", str(EVAL_SEED), "--base-state", str(loaded_state_path),
                 "--pin-set", pin_set],
                env=local_server_env, capture_output=True, text=True,
                timeout=600, check=False)
            if proc.returncode != 0:
                pytest.fail(
                    f"build_eval_state --pin-set {pin_set} failed "
                    f"(rc={proc.returncode}): stdout={proc.stdout[-500:]!r} "
                    f"stderr={proc.stderr[-1500:]!r}")
        paths[state_file] = out
    return paths


@pytest.fixture(scope="session")
def connect_for(eval_state_paths, local_server_command, local_server_env):
    """`connect_for(state_file)()` -> a FRESH server connection against that
    state file. The one-arg shape suite.run_suite and the derived-scalar
    suite helpers take; each returned factory is the zero-arg shape every
    older assertion takes."""
    def _factory(state_file: str):
        @asynccontextmanager
        async def _connect():
            async with connect_server(
                TOY_INDIA_TOPOLOGY_PATH, server_command=local_server_command,
                env=local_server_env,
                extra_args=["--state", str(eval_state_paths[state_file])],
            ) as client:
                yield client
        return _connect
    return _factory
