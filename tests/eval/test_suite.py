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


def test_the_results_table_reports_regret_and_inert_commits():
    # A fabricated results dict -- no server, no rollout -- exercising only
    # render_results_table's own aggregation (Task 16): regret_gbps_h is the
    # MEAN over every run's episode_metrics entry, inert_commits is the SUM.
    results = {
        "episodes": {
            "T1a": {"baseline:immediate": {
                "runs": [], "label_correct_mean": 1.0,
                "label_correct_spread": 0.0,
                "metrics": [{"regret_gbps_h": 10.0, "inert_commits": 1},
                            {"regret_gbps_h": 20.0, "inert_commits": 3}]}},
            "T1b": {"baseline:immediate": {
                "runs": [], "label_correct_mean": 0.0,
                "label_correct_spread": 0.0,
                "metrics": [{"regret_gbps_h": 30.0, "inert_commits": 0}]}},
        },
        "pairs": {"baseline:immediate": {"pair_solved": 0.0, "detail": {}}},
        "budget": {"seeds": [17], "episodes": 2, "runs_per_episode": 1,
                  "rollouts": 2},
    }
    table = render_results_table(results)
    assert "regret_gbps_h" in table
    assert "inert_commits" in table
    # mean([10.0, 20.0, 30.0]) == 20.0; sum([1, 3, 0]) == 4
    header, _, row = table.splitlines()[:3]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3] == "20.0"
    assert cells[4] == "4"


def test_the_results_table_tolerates_episodes_with_no_regret_figure():
    # scoring.episode_metrics reports regret_gbps_h=None on any episode not
    # graded on `spare_action_by_deadline` (today: everything but T1a/T1b --
    # D1/T2/T3 have no gold.outcome_gbps_h to regret against). Found live
    # running the FULL 7-episode suite: statistics.fmean over a generator
    # that includes a None crashes with TypeError -- this test is the
    # regression lock for that fix, reproducing the exact shape (some runs
    # None, some real) rather than only the all-real fixture above.
    results = {
        "episodes": {
            "T1a": {"baseline:immediate": {
                "runs": [], "label_correct_mean": 1.0,
                "label_correct_spread": 0.0,
                "metrics": [{"regret_gbps_h": 10.0, "inert_commits": 1}]}},
            "D1": {"baseline:immediate": {
                "runs": [], "label_correct_mean": 1.0,
                "label_correct_spread": 0.0,
                "metrics": [{"regret_gbps_h": None, "inert_commits": 0}]}},
        },
        "pairs": {"baseline:immediate": {"pair_solved": 0.0, "detail": {}}},
        "budget": {"seeds": [17], "episodes": 2, "runs_per_episode": 1,
                  "rollouts": 2},
    }
    row = render_results_table(results).splitlines()[2]
    cells = [c.strip() for c in row.strip("|").split("|")]
    # mean over the ONE real value (10.0), the None is skipped, not averaged
    # in; inert_commits is a plain sum across both episodes (1 + 0).
    assert cells[3] == "10.0"
    assert cells[4] == "1"


def test_the_results_table_reports_n_a_when_no_run_has_a_regret_figure():
    results = {
        "episodes": {
            "D1": {"baseline:immediate": {
                "runs": [], "label_correct_mean": 1.0,
                "label_correct_spread": 0.0,
                "metrics": [{"regret_gbps_h": None, "inert_commits": 0}]}},
        },
        "pairs": {"baseline:immediate": {"pair_solved": 0.0, "detail": {}}},
        "budget": {"seeds": [17], "episodes": 1, "runs_per_episode": 1,
                  "rollouts": 1},
    }
    row = render_results_table(results).splitlines()[2]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3] == "n/a"


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
