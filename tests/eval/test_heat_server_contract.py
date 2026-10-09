# tests/eval/test_heat_server_contract.py
"""Heat episode Task 1 (spec 2026-10-08 §10 "Live, free"; plan Task 1): the
server facts the heat harness is built on, pinned live against loaded-s17.
If one fails, STOP and report -- the heat design depends on every one
(spec §12: "if it fails, stop and redesign -- do not author around it")."""
from __future__ import annotations

import asyncio
from pathlib import Path

from storm_reoptimizer.mcp_client import call_tool_json, connect_server

TOPOLOGY_PATH = (Path(__file__).parent.parent.parent / "src" / "storm_reoptimizer"
                 / "data" / "toy_india_topology.json")
SUT_LP = "lp-cand-d0463-0"
SUT_IPL = "ipl-cand-d0463-0"
SUT_SERVICE = "d0463"           # rides ONLY ipl-cand-d0463-0
AMP = "amp_wardha_chandrapur_0"  # mid-span hut of the aerial wardha-chandrapur edge
TOP, NEXT = "800G@15.1dB", "750G@14.1dB"


def _live(state, cmd, env, body):
    async def go():
        async with connect_server(TOPOLOGY_PATH, server_command=cmd, env=env,
                                  extra_args=["--state", str(state)]) as client:
            return await body(client)
    return asyncio.run(go())


async def _sut(client) -> dict:
    lps = await call_tool_json(client, "get_lightpaths", expect_list=True)
    return next(lp for lp in lps if lp["id"] == SUT_LP)


def _row(routing: dict, ipl: str) -> dict:
    return next(r for r in routing["utilizations"] if r["ip_link_id"] == ipl)


def test_optical_topology_orders_amps_and_carries_fiber_lengths(
        loaded_state_path, local_server_command, local_server_env):
    async def body(client):
        return await call_tool_json(client, "get_topology", {"layer": "optical"})
    optical = _live(loaded_state_path, local_server_command, local_server_env, body)
    oms = {o["id"]: o for o in optical["oms"]}["oms_wardha_chandrapur"]
    # If src_node_id comes back as "roadm_wardha", record it and make
    # geo_mapper.amplifier_points strip the "roadm_" prefix (Task 3).
    assert (oms["src_node_id"], oms["dst_node_id"]) == ("wardha", "chandrapur")
    assert oms["elements"] == [
        "roadm_wardha", "amp_wardha_chandrapur_booster",
        "fiber_wardha_chandrapur_0", "amp_wardha_chandrapur_0",
        "fiber_wardha_chandrapur_1", "amp_wardha_chandrapur_1"]
    lengths = {f["id"]: f["length_km"] for f in optical["fibers"]}
    assert abs(lengths["fiber_wardha_chandrapur_0"]
               + lengths["fiber_wardha_chandrapur_1"] - 113.97) < 0.05


def test_ip_topology_and_modes_carry_the_fields_the_heat_runner_reads(
        loaded_state_path, local_server_command, local_server_env):
    async def body(client):
        return (await call_tool_json(client, "get_ip_topology"),
                await call_tool_json(client, "get_transceiver_modes", expect_list=True))
    ip, modes = _live(loaded_state_path, local_server_command, local_server_env, body)
    link = next(l for l in ip["ip_links"] if l["lightpath_id"] == SUT_LP)
    assert link["id"] == SUT_IPL
    assert link["capacity_gbps"] == 800.0 and link["load_gbps"] == 800.0
    by_id = {m["id"]: m for m in modes}
    assert by_id[TOP]["bitrate_gbps"] == 800.0
    assert by_id[TOP]["required_gsnr_db"] == 15.1
    assert len(by_id) == 11


def test_snapshot_restore_returns_margins_exactly(
        loaded_state_path, local_server_command, local_server_env):
    async def body(client):
        before = (await _sut(client))["qot"]["margin_db"]
        base = (await call_tool_json(client, "snapshot_create"))["id"]
        await call_tool_json(client, "inject_degradation",
                             {"asset_id": AMP, "nf_delta": 3.0})
        await call_tool_json(client, "snapshot_restore", {"snapshot_id": base})
        return before, (await _sut(client))["qot"]["margin_db"]
    before, after = _live(loaded_state_path, local_server_command, local_server_env, body)
    assert abs(before - after) < 1e-9


def test_margin_falls_monotonically_with_nf_rise(
        loaded_state_path, local_server_command, local_server_env, capsys):
    grid = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    async def body(client):
        base = (await call_tool_json(client, "snapshot_create"))["id"]
        out = []
        for delta in grid:
            if delta:
                await call_tool_json(client, "inject_degradation",
                                     {"asset_id": AMP, "nf_delta": delta})
            out.append((await _sut(client))["qot"]["margin_db"])
            if delta:
                await call_tool_json(client, "snapshot_restore", {"snapshot_id": base})
        return out
    margins = _live(loaded_state_path, local_server_command, local_server_env, body)
    with capsys.disabled():   # recorded in the Task 1 note: sensitivity per dB NF
        print("\nΔNF -> SUT margin:", list(zip(grid, [round(m, 4) for m in margins])))
    assert all(b < a for a, b in zip(margins, margins[1:]))


def test_negative_margin_downs_the_link_and_drops_its_service(
        loaded_state_path, local_server_command, local_server_env):
    """The margin-feasibility gate is load-bearing (spec §12): a negative
    margin at the FROZEN mode takes the link down (capacity 0); the server
    never auto-downshifts, and the IP layer does not restore the service
    around it. If d0463 shows up in `restored`, hold is not dangerous and the
    newsvendor collapses -- STOP."""
    async def body(client):
        await call_tool_json(client, "inject_degradation",
                             {"asset_id": AMP, "nf_delta": 16.0})
        sut = await _sut(client)
        return sut, await call_tool_json(client, "simulate_ip_routing")
    sut, routing = _live(loaded_state_path, local_server_command, local_server_env, body)
    assert sut["qot"]["margin_db"] < 0
    assert sut["mode_id"] == TOP                    # frozen, not auto-downshifted
    row = _row(routing, SUT_IPL)
    assert row["down"] is True and row["capacity_gbps"] == 0.0
    assert SUT_IPL in routing["dropped"]["down_links"]
    assert SUT_SERVICE in {d["service_id"] for d in routing["dropped"]["services"]}
    assert SUT_SERVICE not in routing["restored"]


def test_downshift_adds_exactly_the_threshold_gap_and_congests_its_own_link(
        loaded_state_path, local_server_command, local_server_env):
    async def body(client):
        before = (await _sut(client))["qot"]["margin_db"]
        result = await call_tool_json(client, "set_modulation_format",
                                      {"lightpath_id": SUT_LP, "mode_id": NEXT})
        return before, result, await call_tool_json(client, "simulate_ip_routing")
    before, result, routing = _live(
        loaded_state_path, local_server_command, local_server_env, body)
    assert result.get("ok") is not False, result
    assert abs(result["margin_db"] - (before + (15.1 - 14.1))) < 1e-3
    row = _row(routing, SUT_IPL)
    assert row["down"] is False and row["capacity_gbps"] == 750.0
    assert row["offered_gbps"] == 800.0           # load does not move away


def test_validate_plan_blocks_a_full_load_downshift(
        loaded_state_path, local_server_command, local_server_env):
    """Why the heat lever is the direct tool, not a plan op (deviation 4)."""
    async def body(client):
        return await call_tool_json(client, "validate_plan", {
            "plan": {"ops": [{"op": "set_modulation_format",
                              "lightpath_id": SUT_LP, "mode_id": NEXT}]},
            "baseline": "standing"})
    report = _live(loaded_state_path, local_server_command, local_server_env, body)
    assert report["ok"] is False
    assert any(v["type"] == "ip_link_overload" and v["asset_id"] == SUT_IPL
               for v in report["violations"])


def test_upshift_into_negative_margin_is_allowed_and_downs_the_link(
        loaded_state_path, local_server_command, local_server_env):
    """Reversibility: the server never refuses an upshift for the lightpath's
    OWN infeasibility, so a too-early upshift is a real, losable decision."""
    async def body(client):
        await call_tool_json(client, "set_modulation_format",
                             {"lightpath_id": SUT_LP, "mode_id": NEXT})
        # push 800G negative while 750G stays feasible (headroom 1.0 dB)
        before = (await _sut(client))["qot"]["margin_db"]  # at 750G
        delta = 0.5
        while True:
            base = (await call_tool_json(client, "snapshot_create"))["id"]
            await call_tool_json(client, "inject_degradation",
                                 {"asset_id": AMP, "nf_delta": delta})
            m750 = (await _sut(client))["qot"]["margin_db"]
            if m750 - 1.0 < 0 <= m750:
                break
            await call_tool_json(client, "snapshot_restore", {"snapshot_id": base})
            delta *= 1.5
            assert delta < 64, "never found a ΔNF window with 800G<0<=750G"
        up = await call_tool_json(client, "set_modulation_format",
                                  {"lightpath_id": SUT_LP, "mode_id": TOP})
        return before, up, await call_tool_json(client, "simulate_ip_routing")
    _, up, routing = _live(loaded_state_path, local_server_command, local_server_env, body)
    assert up.get("ok") is not False and up["mode_feasible"] is False
    assert _row(routing, SUT_IPL)["down"] is True
