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
    *, expect_list: bool = False,
) -> Any:
    """Call an MCP tool and return its parsed JSON payload.

    These tools return plain dicts, not a typed/Pydantic output schema, so
    the installed mcp SDK (2.0.0) leaves `CallToolResult.structured_content`
    as None -- the actual payload is JSON text in the content block(s).
    Confirmed by round-tripping get_topology against a real server
    subprocess; do not read `.structured_content` for these tools.

    Tools whose return type is a bare list (e.g. get_lightpaths) come back
    as *multiple* content blocks, one per list item, not a single block
    holding a JSON array -- confirmed against a real server subprocess
    (get_lightpaths on a 4-lightpath state returns 4 blocks). Tools
    returning an object (get_services, get_topology) come back as exactly
    one block.

    This is genuinely ambiguous at N=1: a single-object tool result and a
    list-returning tool that happens to produce exactly one item both
    serialize to exactly one content block on the wire -- block count alone
    can't tell them apart. Resolve it by caller declaration instead of
    inference: pass `expect_list=True` for any tool whose return type is a
    bare list (get_lightpaths and friends), so a single block is still
    wrapped in a 1-item list rather than returned as a bare dict. Every
    other (object-returning) call site leaves `expect_list` at its default
    `False` and is unaffected. The >1-block case needs no gating on the
    flag -- multiple blocks can only legitimately happen for a list-
    returning tool in the first place, regardless of what the caller
    passed."""
    result = await client.call_tool(name, arguments)
    if result.is_error:
        text = "".join(getattr(block, "text", "") for block in result.content)
        raise RuntimeError(f"tool {name!r} failed: {text}")
    if len(result.content) == 1 and not expect_list:
        return json.loads(result.content[0].text)
    return [json.loads(block.text) for block in result.content]
