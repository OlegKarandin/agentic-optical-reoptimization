"""Reproduces CLAUDE.md's canonical case end-to-end against a real
multilayer-optical-mcp subprocess, over the real TataNld backbone topology:
a working and a protection path between two cities that are disjoint at
design time (share zero physical assets) but both happen to cross a
dynamically-defined storm risk group, because each route runs at least one
aerial span through the same region.

Nothing here is hardcoded to a specific route: the disjoint pair and each
path's exposed (aerial) span are discovered via real tool calls, the same
way an actual geo-mapper/agent would -- only the SELECTION RULE (take the
first aerial span found along each path) is fixed, so the test stays valid
if the solver's tie-breaking ever changes the exact route it returns."""
import asyncio
import json
from pathlib import Path

from storm_reoptimizer.mcp_client import call_tool_json, connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)
SRC, DST = "delhi", "nagpur"


def _mount_type_by_pair() -> dict[frozenset, str]:
    data = json.loads(TOPOLOGY_PATH.read_text())
    return {
        frozenset((e["src"], e["dst"])): e["mount_type"]
        for e in data["graph"]["edges"]
    }


def _first_aerial_oms(oms_sequence, oms_by_id, mount_type_by_pair) -> str:
    for oms_id in oms_sequence:
        oms = oms_by_id[oms_id]
        pair = frozenset((oms["src_node_id"], oms["dst_node_id"]))
        if mount_type_by_pair[pair] == "aerial":
            return oms_id
    raise AssertionError(f"no aerial span found in {oms_sequence}")


async def _run(local_server_command, local_server_env):
    mount_type_by_pair = _mount_type_by_pair()

    async with connect_server(
        TOPOLOGY_PATH, server_command=local_server_command, env=local_server_env,
    ) as client:
        # 1. Discover a genuinely disjoint working/protection pair -- not
        #    hand-picked, the server's own solver finds it.
        disjoint = await call_tool_json(
            client, "compute_disjoint_paths",
            {"src": SRC, "dst": DST, "basis": "physical", "level": "link"},
        )
        assert disjoint["status"] == "solution", disjoint
        working = disjoint["path_a"]["oms_sequence"]
        protection = disjoint["path_b"]["oms_sequence"]

        # 2. Confirm design-time disjointness holds with no risk group defined.
        physical = await call_tool_json(
            client, "check_disjointness",
            {"path_a": working, "path_b": protection, "basis": "physical"},
        )

        # 3. Find each path's exposed (aerial) span using the server's own
        #    topology view correlated against storm-reoptimizer's local
        #    geometry/mount_type data (the server carries no geometry, by
        #    design -- see CLAUDE.md's seam).
        topo = await call_tool_json(client, "get_topology", {"layer": "optical"})
        oms_by_id = {o["id"]: o for o in topo["oms"]}
        exposed_working_oms = _first_aerial_oms(working, oms_by_id, mount_type_by_pair)
        exposed_protection_oms = _first_aerial_oms(protection, oms_by_id, mount_type_by_pair)

        exposed_fiber_ids = [
            asset_id
            for oms_id in (exposed_working_oms, exposed_protection_oms)
            for asset_id in oms_by_id[oms_id]["elements"]
            if asset_id.startswith("fiber_")
        ]

        # 4. Define the storm's dynamic risk group and re-check.
        define_result = await call_tool_json(
            client, "define_risk_group",
            {"rg_id": "storm_rg", "asset_ids": exposed_fiber_ids,
             "metadata": {"event_type": "storm"}},
        )
        risk_group = await call_tool_json(
            client, "check_disjointness",
            {"path_a": working, "path_b": protection, "basis": "risk_group"},
        )

        return physical, define_result, risk_group, exposed_working_oms, exposed_protection_oms


def test_design_time_disjoint_but_storm_exposed_round_trip(local_server_command, local_server_env):
    physical, define_result, risk_group, exposed_working_oms, exposed_protection_oms = asyncio.run(
        _run(local_server_command, local_server_env)
    )

    assert physical["disjoint"] is True

    assert define_result["id"] == "storm_rg"
    assert len(define_result["asset_ids"]) > 0

    assert risk_group["disjoint"] is False
    assert "storm_rg" in risk_group["shared_groups"]

    # Sanity: the exposure is genuinely cross-route, not the same span twice.
    assert exposed_working_oms != exposed_protection_oms
