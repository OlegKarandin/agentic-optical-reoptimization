"""Candidate -> committable plan translation (eval design spec, build order
item 3). The pure translation is unit-tested against a hand-built index; the
round-trip (does validate_plan accept what we emit?) and the state-advancement
property (does a commit change the next hour's menu?) run against a real
server subprocess, per CLAUDE.md's hard seam."""
import asyncio
from pathlib import Path

import pytest

from storm_reoptimizer.eval.plans import (
    TopologyIndex, build_topology_index, commit_candidate,
    lam_to_center_freq_hz, plan_from_candidate, validate_candidate,
)
from storm_reoptimizer.mcp_client import call_tool_json, connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)

INDEX = TopologyIndex(
    oms_by_id={
        "oms_1": {"id": "oms_1", "src_node_id": "satna", "dst_node_id": "rewa"},
        "oms_2": {"id": "oms_2", "src_node_id": "allahabad", "dst_node_id": "rewa"},
    },
    router_by_site={"satna": "router_satna", "rewa": "router_rewa",
                    "allahabad": "router_allahabad"},
    ip_link_by_lightpath={
        "lp-existing": {"id": "ipl-existing", "a_router": "router_satna",
                        "z_router": "router_rewa", "lightpath_id": "lp-existing"},
    },
    service_by_id={
        "storm-svc-1": {"id": "storm-svc-1", "src_router": "router_satna",
                        "dst_router": "router_allahabad", "demand_gbps": 300.0,
                        "working_path": [], "protection_path": []},
    },
)


def test_lam_maps_to_the_servers_default_grid():
    assert lam_to_center_freq_hz(0) == pytest.approx(191.4e12)
    assert lam_to_center_freq_hz(3) == pytest.approx(191.7e12)


def test_an_optical_reroute_candidate_becomes_provision_ops_plus_a_reroute():
    candidate = {
        "lever": "optical_reroute", "reused_lightpaths": [],
        "new_lightpaths": [
            {"oms_sequence": ["oms_1", "oms_2"], "lam": 3,
             "mode_id": "300G@4.8dB", "gsnr_db": 12.0, "bitrate_gbps": 300.0},
        ],
        "restored_gbps": 300.0, "shortfall_gbps": 0.0, "cost_vector": {},
    }
    plan = plan_from_candidate(INDEX, candidate, "storm-svc-1", prefix="eval-t1")

    assert [op["op"] for op in plan["ops"]] == [
        "provision_lightpath", "reroute_service"]
    lp = plan["ops"][0]["lightpath"]
    assert lp["id"] == "lp-eval-t1-storm-svc-1-0"
    assert lp["oms_sequence"] == ["oms_1", "oms_2"]
    assert lp["center_freq_hz"] == pytest.approx(191.7e12)
    # oms_1 and oms_2 share `rewa`; the run therefore runs satna <-> allahabad.
    link = plan["ops"][0]["ip_link"]
    assert {link["a_router"], link["z_router"]} == {
        "router_satna", "router_allahabad"}
    assert plan["ops"][1] == {
        "op": "reroute_service", "service_id": "storm-svc-1",
        "ip_path": ["ipl-eval-t1-storm-svc-1-0"], "which": "working"}


def test_a_groom_that_cannot_reach_the_destination_fails_loudly():
    candidate = {
        "lever": "ip_reroute", "reused_lightpaths": ["lp-existing"],
        "new_lightpaths": [], "restored_gbps": 250.0, "shortfall_gbps": 50.0,
        "cost_vector": {},
    }
    # Endpoint mismatch is intentional here: the reused link stops at rewa,
    # so the stitched walk cannot reach allahabad and must say so loudly
    # rather than emit a truncated ip_path the server will reject obscurely.
    with pytest.raises(Exception):
        plan_from_candidate(INDEX, candidate, "storm-svc-1", prefix="eval-t1")


def test_a_single_oms_run_uses_that_oms_own_endpoints():
    candidate = {
        "lever": "optical_reroute", "reused_lightpaths": [],
        "new_lightpaths": [
            {"oms_sequence": ["oms_1"], "lam": 0, "mode_id": "300G@4.8dB",
             "gsnr_db": 12.0, "bitrate_gbps": 300.0}],
        "restored_gbps": 300.0, "shortfall_gbps": 0.0, "cost_vector": {},
    }
    index = TopologyIndex(
        oms_by_id=INDEX.oms_by_id, router_by_site=INDEX.router_by_site,
        ip_link_by_lightpath={},
        service_by_id={"svc-x": {"id": "svc-x", "src_router": "router_satna",
                                 "dst_router": "router_rewa",
                                 "demand_gbps": 100.0, "working_path": [],
                                 "protection_path": []}})
    plan = plan_from_candidate(index, candidate, "svc-x", prefix="p")
    link = plan["ops"][0]["ip_link"]
    assert {link["a_router"], link["z_router"]} == {"router_satna", "router_rewa"}


async def _round_trip(state_path, server_command, server_env):
    async with connect_server(
        TOPOLOGY_PATH, server_command=server_command, env=server_env,
        extra_args=["--state", str(state_path)],
    ) as client:
        index = await build_topology_index(client)
        menu = await call_tool_json(client, "route_service", {
            "service_id": "storm-svc-1", "protected": False})
        assert menu["candidates"], "loaded network produced an empty menu"
        chosen = menu["candidates"][0]
        plan = plan_from_candidate(index, chosen, "storm-svc-1", prefix="eval-t0")
        report = await validate_candidate(client, plan, basis="physical",
                                          level="link")
        commit = await commit_candidate(client, plan, basis="physical",
                                        level="link")
        after = await call_tool_json(client, "route_service", {
            "service_id": "storm-svc-1", "protected": False})
        return chosen, report, commit, menu, after


def test_plan_round_trips_and_the_commit_changes_the_next_hour_menu(
    loaded_state_path, local_server_command, local_server_env,
):
    chosen, report, commit, before, after = asyncio.run(
        _round_trip(loaded_state_path, local_server_command, local_server_env))

    assert report["ok"], report["violations"]
    assert commit["status"] == "committed", commit
    assert commit["intended_snapshot_id"]
    # Gap 2's fix, made observable: the committed lightpaths occupy spectrum
    # and change grooming residual, so the next menu is not the previous one.
    assert after["candidates"] != before["candidates"]
