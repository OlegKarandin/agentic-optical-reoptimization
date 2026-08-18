import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import mcp.types as types

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


def _single_block_result() -> types.CallToolResult:
    """A CallToolResult with exactly one content block -- the wire shape a
    list-returning tool (e.g. get_lightpaths) produces when its list
    happens to have exactly one item, indistinguishable on the wire from an
    object-returning tool's single-block result (see call_tool_json's
    docstring: this is the genuine N=1 ambiguity block-count alone can't
    resolve)."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text='{"id": "lp-only"}')],
    )


def test_call_tool_json_expect_list_wraps_a_single_block_in_a_list():
    """Regression test for the N=1 ambiguity: no real-server fixture in this
    suite naturally produces a 1-lightpath state, so this exercises
    call_tool_json's own parsing logic directly against a faked
    single-block CallToolResult instead. Without expect_list=True, this
    exact result would parse as a bare {"id": "lp-only"} dict -- silently
    wrong for a caller that expects a list of lightpaths (scenario_storm.py's
    `for lp in lightpaths`), reproducing the original bug's symptom for
    N=1 instead of N>1."""
    client = AsyncMock()
    client.call_tool.return_value = _single_block_result()

    result = asyncio.run(
        call_tool_json(client, "get_lightpaths", expect_list=True)
    )

    assert result == [{"id": "lp-only"}]


def test_call_tool_json_default_still_unwraps_a_single_block():
    """Contrast case: every existing (object-returning) call site relies on
    a single content block parsing to a bare value, not a 1-item list --
    confirms expect_list's default (False) leaves that behavior
    unchanged."""
    client = AsyncMock()
    client.call_tool.return_value = _single_block_result()

    result = asyncio.run(call_tool_json(client, "get_services"))

    assert result == {"id": "lp-only"}
