# src/storm_reoptimizer/mcp_client.py
"""MCP stdio client wrapper: launches multilayer-optical-mcp as a real
subprocess (the production deployment shape -- two independent processes
talking MCP over stdio) rather than importing its Python internals, per
CLAUDE.md's hard rule that the two repos stay separate."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

# The registered console script (see multilayer-optical-mcp's pyproject.toml
# [project.scripts]). Override via STORM_REOPTIMIZER_MCP_SERVER_CMD (a JSON
# list) when the script isn't on PATH -- e.g. this workspace's repos sit
# under a path with Cyrillic characters, which trips a setuptools bug
# (`editable_wheel.py`'s `_encode_pth` uses the ANSI codepage) that blocks
# `pip install -e`, so the console script never gets installed here.
DEFAULT_SERVER_COMMAND = ["multilayer-optical-mcp"]


def _default_server_command() -> list[str]:
    override = os.environ.get("STORM_REOPTIMIZER_MCP_SERVER_CMD")
    return json.loads(override) if override else list(DEFAULT_SERVER_COMMAND)


@asynccontextmanager
async def connect_server(
    topology_path: str | Path,
    *,
    server_command: list[str] | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> AsyncIterator[Client]:
    """Launch multilayer-optical-mcp seeded with `topology_path` and yield a
    connected Client. `server_command` overrides how the server process is
    invoked (see DEFAULT_SERVER_COMMAND); `env` overrides the subprocess
    environment (stdio_client's default only inherits a safe subset of
    os.environ -- notably not PYTHONPATH, which the console-script-unavailable
    workaround needs). The subprocess is torn down on exit (stdio_client's own
    context-manager contract)."""
    command, *base_args = server_command or _default_server_command()
    params = StdioServerParameters(
        command=command,
        args=[*base_args, "--topology", str(topology_path), *(extra_args or [])],
        env=env,
    )
    async with Client(stdio_client(params)) as client:
        yield client


async def call_tool_json(
    client: Client, name: str, arguments: dict[str, Any] | None = None,
) -> Any:
    """Call an MCP tool and return its parsed JSON payload.

    These tools return plain dicts, not a typed/Pydantic output schema, so
    the installed mcp SDK (2.0.0) leaves `CallToolResult.structured_content`
    as None -- the actual payload is JSON text in `content[0].text`.
    Confirmed by round-tripping get_topology against a real server
    subprocess; do not read `.structured_content` for these tools."""
    result = await client.call_tool(name, arguments)
    if result.is_error:
        text = "".join(getattr(block, "text", "") for block in result.content)
        raise RuntimeError(f"tool {name!r} failed: {text}")
    return json.loads(result.content[0].text)
