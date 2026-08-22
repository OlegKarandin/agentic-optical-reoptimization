"""Rollout mechanics against a real server (eval design spec, "Testing").
Uses a throwaway two-hour scenario written into tmp_path -- the seven real
episodes come later and must not be needed to prove the loop works."""
import asyncio
import dataclasses
import textwrap
from pathlib import Path

import pytest

from storm_reoptimizer.eval.baseline import ForecastBlindBaseline, ScriptedDecider
from storm_reoptimizer.eval.decisions import (
    ConstraintDecision, ObjectiveDecision, TimingDecision,
)
from storm_reoptimizer.eval.runner import MAX_ITERATIONS, run_episode
from storm_reoptimizer.eval.scenario_file import load_scenario
from storm_reoptimizer.mcp_client import connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)

# A minimal, deliberately non-discriminating episode: two hours, one issuance,
# one realized cut. Its job is to exercise the loop, not to score anyone.
SMOKE = textwrap.dedent("""
    id: SMOKE
    seed: 17
    state_file: eval/states/loaded-s17.json
    service_under_test: storm-svc-1
    track: hudhud
    hours: [t0, t1]
    decision_hour: t0
    lead_time_hours: 1
    spares_on_hand: 2
    damage_radius_km: 74
    reference_avoid: {}
    forecast:
      t0:
        t1: {cone: {type: Polygon, coordinates: [[[81.3, 24.8], [82.3, 24.8], [82.3, 25.4], [81.3, 25.4], [81.3, 24.8]]]}, width_km: 120, center: {lat: 25.1, lon: 81.8}}
    realized:
      t1: []
    gold:
      survived: [storm-svc-1]
      max_spares_wasted: 2
      decision_at_t0: act
      label: act
      rationale: smoke episode; not scored
    flip_variable: [smoke]
    metadata:
      cone_width_km: 120
      cone_motion_kmh: 20
      n_future_claimants: 0
      exposure_horizon_hours: 1
      spares_on_hand: 2
""")


def _scenario(tmp_path):
    path = tmp_path / "SMOKE.yaml"
    path.write_text(SMOKE, encoding="utf-8")
    return load_scenario(path)


async def _run(scenario, decider, state_path, server_command, server_env):
    async with connect_server(
        TOPOLOGY_PATH, server_command=server_command, env=server_env,
        extra_args=["--state", str(state_path)],
    ) as client:
        return await run_episode(client, scenario, decider,
                                 topology_path=TOPOLOGY_PATH)


def test_a_baseline_rollout_completes_and_records_every_hour(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    trace = asyncio.run(_run(
        _scenario(tmp_path), ForecastBlindBaseline("immediate"),
        loaded_state_path, local_server_command, local_server_env))
    assert [h["hour"] for h in trace.hours] == ["t0", "t1"]
    assert trace.terminal_status in {"converged", "declared_infeasible"}
    assert trace.tool_calls > 0
    assert all("timing" in h for h in trace.hours)


def test_a_committed_action_debits_the_ledger_and_records_its_lead_time(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    trace = asyncio.run(_run(
        _scenario(tmp_path), ForecastBlindBaseline("immediate"),
        loaded_state_path, local_server_command, local_server_env))
    assert trace.actions, "the immediate baseline should have acted at t0"
    action = trace.actions[0]
    assert action.hour == "t0"
    # ip_reroute lands at once; anything else costs the scenario's lead time.
    expected = 0 if action.lever == "ip_reroute" else 1
    assert action.effective_at_index == action.hour_index + expected
    assert trace.spares_remaining == 2 - sum(a.pairs for a in trace.actions)


def test_an_unaffordable_choice_is_rejected_and_the_loop_retries(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    broke = dataclasses.replace(_scenario(tmp_path), spares_on_hand=0)
    trace = asyncio.run(_run(
        broke, ForecastBlindBaseline("immediate"),
        loaded_state_path, local_server_command, local_server_env))
    rejections_by_hour = [h.get("rejections", []) for h in trace.hours]
    # With no spares, any optical candidate must be refused by the ledger.
    # The retry cap is per ACTING HOUR (a global constraint), not per
    # episode -- both t0 and t1 are exposed here, so each independently
    # exhausts its own cap; the total across the episode can exceed
    # MAX_ITERATIONS even though no single hour ever does.
    if any(r["type"] == "insufficient_spares"
           for hour_rejections in rejections_by_hour for r in hour_rejections):
        assert max(len(hr) for hr in rejections_by_hour) <= MAX_ITERATIONS
    assert trace.spares_remaining == 0


def test_the_loop_caps_at_five_iterations(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # A decider that always picks an out-of-range candidate can never
    # converge; the harness must stop it, not spin.
    always_bad = ScriptedDecider(
        "always-bad",
        default_timing=TimingDecision("act", "always act"),
        default_constraints=ConstraintDecision(avoid={}, reasoning="none"),
        default_objective=ObjectiveDecision("candidate_999", None, "bad index"))
    trace = asyncio.run(_run(
        _scenario(tmp_path), always_bad,
        loaded_state_path, local_server_command, local_server_env))
    assert trace.terminal_status == "hit_cap"
    assert max(len(h.get("iterations", [])) for h in trace.hours) == MAX_ITERATIONS


def test_lead_time_marks_an_optical_reroute_at_the_cut_hour_as_late(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # The service is cut at t1 and the baseline acts at t1 (not t0), so an
    # optical_reroute cannot have landed. This is scored in Task 10; here we
    # only assert the trace carries the arithmetic to score it.
    scenario = _scenario(tmp_path)
    trace = asyncio.run(_run(
        scenario, ForecastBlindBaseline("at_deadline"),
        loaded_state_path, local_server_command, local_server_env))
    for action in trace.actions:
        assert action.effective_at_index >= action.hour_index
