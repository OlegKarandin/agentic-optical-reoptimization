import pytest

from storm_reoptimizer.eval.decisions import HOLD_CHOICE, ModeDecision
from storm_reoptimizer.eval.heat_runner import (
    HeatEpisodeError, bystanders_of, check_heated_amplifiers, hour_loss,
    link_row, mode_change_rejected, target_mode,
)


def test_naming_the_current_mode_is_a_noop():                    # Review Focus 1
    assert target_mode(ModeDecision("800", "x"), "800") is None
    assert target_mode(ModeDecision(HOLD_CHOICE, "x"), "800") is None
    assert target_mode(ModeDecision("750", "x"), "800") == "750"


def test_refusal_is_detected_by_ok_false():                      # Review Focus 4
    assert mode_change_rejected({"ok": False, "violations": []})
    assert not mode_change_rejected({"lightpath_id": "lp", "margin_db": 0.3})


def test_amplifier_set_mismatch_refuses_the_episode():           # Review Focus 5
    check_heated_amplifiers(["b", "a"], ("a", "b"))
    with pytest.raises(HeatEpisodeError, match="thermal.amplifiers"):
        check_heated_amplifiers(["a"], ("a", "b"))


def test_hour_loss_reads_down_and_congestion():
    assert hour_loss({"down": True, "capacity_gbps": 0.0}, 800.0) == 800.0
    assert hour_loss({"down": False, "capacity_gbps": 750.0}, 800.0) == 50.0
    assert hour_loss({"down": False, "capacity_gbps": 800.0}, 800.0) == 0.0


def test_link_row_and_bystanders():
    routing = {"utilizations": [{"ip_link_id": "x"}, {"ip_link_id": "y", "down": False}]}
    assert link_row(routing, "y") == {"ip_link_id": "y", "down": False}
    lps = [{"id": "sut", "oms_sequence": ["oms_a_b"]},
           {"id": "rev", "oms_sequence": ["oms_b_a"]},
           {"id": "far", "oms_sequence": ["oms_c_d"]}]
    oms = [{"id": "oms_a_b", "elements": ["amp_a_b_0"]},
           {"id": "oms_b_a", "elements": ["amp_b_a_0"]},
           {"id": "oms_c_d", "elements": ["amp_c_d_0"]}]
    assert bystanders_of(lps, ("amp_a_b_0", "amp_b_a_0"), oms, "sut") == ["rev"]


import asyncio
import json

from shapely.geometry import Polygon

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))
import build_eval_state  # noqa: E402
from storm_reoptimizer.eval.heat_baseline import HeatBaseline  # noqa: E402
from storm_reoptimizer.eval.heat_oracle import OracleModeDecider  # noqa: E402
from storm_reoptimizer.eval.runner import run_episode  # noqa: E402
from storm_reoptimizer.eval.scenario_file import load_scenario  # noqa: E402
from storm_reoptimizer.geo_mapper import amplifier_points, load_edges  # noqa: E402
from storm_reoptimizer.mcp_client import call_tool_json, connect_server  # noqa: E402

TOPOLOGY_PATH = (Path(__file__).parent.parent.parent / "src" / "storm_reoptimizer"
                 / "data" / "toy_india_topology.json")


def _heat_yaml(tmp_path, eval_state_paths, cmd, env, realized, failed_at):
    """A 3-hour heat scenario on the Task 5 heat state, polygon = a small
    square around the mid-span hut of HEAT_TARGET_EDGE."""
    a, b = build_eval_state.HEAT_TARGET_EDGE
    pin = build_eval_state.HEAT_PINS[0]

    async def body(client):
        return await call_tool_json(client, "get_topology", {"layer": "optical"})

    async def go():
        async with connect_server(TOPOLOGY_PATH, server_command=cmd, env=env, extra_args=[
                "--state", str(eval_state_paths["eval/states/heat-s17.json"])]) as c:
            return await body(c)
    optical = asyncio.run(go())
    lengths = {f["id"]: f["length_km"] for f in optical["fibers"]}
    points = amplifier_points(optical["oms"], lengths, load_edges(TOPOLOGY_PATH))
    hut = points[f"amp_{a}_{b}_0"]
    square = Polygon.from_bounds(hut.x - 0.02, hut.y - 0.02, hut.x + 0.02, hut.y + 0.02)
    amps = sorted(k for k, p in points.items() if square.intersects(p))
    lp_id = f"lp-cand-{pin['id']}-0"
    doc = {
        "id": "HEAT_SMOKE", "event_type": "heat", "seed": 17,
        "state_file": "eval/states/heat-s17.json", "service_under_test": pin["id"],
        "lightpath_under_test": lp_id, "hours": ["t0", "t1", "t2"],
        "thermal": {"amplifiers": amps, "class": {
            "setpoint_c": 25.0, "coeff_db_per_c": 0.2, "k_cooled": 0.15,
            "k_uncooled": 1.0, "self_heating_c": 8.0, "p_cooling_failure": {
                "base": 0.02, "per_c_above_setpoint": 0.01, "per_hour_above_setpoint": 0.03}}},
        "heat_polygon": {"t0": json.loads(json.dumps(square.__geo_interface__))},
        "temperature_forecast": {"t0": {h: {"t_mean_c": t, "t_sigma_c": 2.0}
                                        for h, t in zip(["t0", "t1", "t2"], realized)}},
        "temperature_realized": dict(zip(["t0", "t1", "t2"], realized)),
        "cooling_failed_at": failed_at, "flip_variable": ["cooling"],
        "metadata": {"label_rule": "mode_by_hour"}}
    path = tmp_path / "HEAT_SMOKE.yaml"
    path.write_text(json.dumps(doc), encoding="utf-8")   # JSON is valid YAML
    return load_scenario(path)


def _rollout(scenario, decider, eval_state_paths, cmd, env):
    async def go():
        async with connect_server(TOPOLOGY_PATH, server_command=cmd, env=env, extra_args=[
                "--state", str(eval_state_paths[scenario.state_file])]) as client:
            return await run_episode(client, scenario, decider, topology_path=TOPOLOGY_PATH)
    return asyncio.run(go())


def test_heat_rollout_records_every_hour_and_recovers(
        tmp_path, eval_state_paths, local_server_command, local_server_env):
    # hot, failure at t1, then back below setpoint at t2: a NEGATIVE NF step
    s = _heat_yaml(tmp_path, eval_state_paths, local_server_command, local_server_env,
                   [32.0, 40.0, 20.0], "t1")
    trace = _rollout(s, HeatBaseline("never_downshift"), eval_state_paths,
                     local_server_command, local_server_env)
    assert [r["hour"] for r in trace.hours] == ["t0", "t1", "t2"]
    assert all(r["mode_change_outcome"] == "noop" for r in trace.hours)
    assert trace.hours[2]["nf_delta_applied_db"] < trace.hours[1]["nf_delta_applied_db"]
    offered = trace.hours[0]["observation"]["link"]["offered_gbps"]
    assert all(r["loss_gbps"] in (0.0, offered) for r in trace.hours)
    assert trace.hours[1]["observation"]["cooling_state"] == "holding"
    assert trace.hours[2]["observation"]["cooling_state"] == "failed"
    assert "internal" not in trace.hours[0]["projected"]


def test_oracle_rollout_is_deterministic_and_on_ladder(
        tmp_path, eval_state_paths, local_server_command, local_server_env):
    s = _heat_yaml(tmp_path, eval_state_paths, local_server_command, local_server_env,
                   [32.0, 40.0, 20.0], "t1")
    a = _rollout(s, OracleModeDecider(), eval_state_paths, local_server_command, local_server_env)
    b = _rollout(s, OracleModeDecider(), eval_state_paths, local_server_command, local_server_env)
    ladder = {m["mode_id"] for m in a.hours[0]["observation"]["mode_ladder"]}
    assert all(r["mode_in_force"] in ladder for r in a.hours)
    assert [r["mode_in_force"] for r in a.hours] == [r["mode_in_force"] for r in b.hours]
    assert [r["loss_gbps"] for r in a.hours] == [r["loss_gbps"] for r in b.hours]


def _leaf_exceptions(exc: BaseException):
    """mcp_client.connect_server's stdio transport wraps whatever the body
    raises in (here, empirically, two levels of) BaseExceptionGroup during
    its own cleanup -- mcp's ClientSession and anyio's stdio_client each run
    their own internal TaskGroup, and anyio always reports even a single
    exception through a TaskGroup as a group. That nesting is a property of
    this mcp/anyio version's transport, not of heat_runner.py, so unwrap
    recursively instead of asserting an exact depth."""
    if isinstance(exc, BaseExceptionGroup):
        for e in exc.exceptions:
            yield from _leaf_exceptions(e)
    else:
        yield exc


def test_a_storm_decider_cannot_run_a_heat_episode(
        tmp_path, eval_state_paths, local_server_command, local_server_env):
    from storm_reoptimizer.eval.baseline import ForecastBlindBaseline
    s = _heat_yaml(tmp_path, eval_state_paths, local_server_command, local_server_env,
                   [30.0, 30.0, 30.0], None)
    with pytest.raises(BaseException) as exc_info:
        _rollout(s, ForecastBlindBaseline("immediate"), eval_state_paths,
                 local_server_command, local_server_env)
    type_errors = [e for e in _leaf_exceptions(exc_info.value) if isinstance(e, TypeError)]
    assert type_errors, f"no TypeError among {list(_leaf_exceptions(exc_info.value))}"
    assert "decide_mode" in str(type_errors[0])
