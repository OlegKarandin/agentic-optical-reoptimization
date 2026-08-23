"""Suite orchestration and the results table (eval design spec, "Run budget",
"Scoring"). Baseline determinism is the load-bearing assertion here: "the
same seed and scenario produce byte-identical traces across runs"."""
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from storm_reoptimizer.eval.baseline import ForecastBlindBaseline
from storm_reoptimizer.eval.scenario_file import load_all_scenarios
from storm_reoptimizer.eval.suite import (
    RUNS_PER_EPISODE, render_results_table, run_suite,
)
from storm_reoptimizer.mcp_client import connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)


def test_the_budget_is_three_runs_per_episode():
    assert RUNS_PER_EPISODE == 3


def _connect_factory(state_path, server_command, server_env):
    @asynccontextmanager
    async def _connect():
        async with connect_server(
            TOPOLOGY_PATH, server_command=server_command, env=server_env,
            extra_args=["--state", str(state_path)],
        ) as client:
            yield client
    return _connect


def test_baseline_is_deterministic_across_runs(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    d1 = {"D1": load_all_scenarios()["D1"]}
    results = asyncio.run(run_suite(
        _connect_factory(loaded_state_path, local_server_command,
                         local_server_env),
        topology_path=TOPOLOGY_PATH,
        deciders=[ForecastBlindBaseline("immediate")],
        runs_per_episode=2, scenarios=d1, traces_dir=tmp_path,
        collapse_deterministic=False))
    runs = results["episodes"]["D1"]["baseline:immediate"]["runs"]
    # Same seed, same scenario, same decisions: identical decision records.
    assert len(runs) == 2
    assert runs[0]["hours"] == runs[1]["hours"]


def test_the_results_table_reports_both_claims_separately(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    results = asyncio.run(run_suite(
        _connect_factory(loaded_state_path, local_server_command,
                         local_server_env),
        topology_path=TOPOLOGY_PATH,
        deciders=[ForecastBlindBaseline(v) for v in ("immediate", "at_deadline")],
        runs_per_episode=1, traces_dir=tmp_path))
    table = render_results_table(results)
    assert "pair_solved" in table
    assert "Claim 1 (provable)" in table
    assert "single forecast variable" in table
    # Both fixed policies must land at exactly one half of each pair.
    for variant in ("baseline:immediate", "baseline:at_deadline"):
        assert results["pairs"][variant]["pair_solved"] == 0.0


def test_traces_land_on_disk(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    asyncio.run(run_suite(
        _connect_factory(loaded_state_path, local_server_command,
                         local_server_env),
        topology_path=TOPOLOGY_PATH,
        deciders=[ForecastBlindBaseline("immediate")],
        runs_per_episode=1,
        scenarios={"D1": load_all_scenarios()["D1"]}, traces_dir=tmp_path))
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert written == ["D1-baseline_immediate-0.json"]
    assert json.loads((tmp_path / written[0]).read_text())["scenario_id"] == "D1"
