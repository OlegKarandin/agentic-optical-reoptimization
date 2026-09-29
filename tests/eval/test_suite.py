"""Suite orchestration and the results table (eval design spec, "Run budget",
"Scoring"). Baseline determinism is the load-bearing assertion here: "the
same seed and scenario produce byte-identical traces across runs"."""
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import storm_reoptimizer.eval.suite as suite_module
from storm_reoptimizer.eval.baseline import ForecastBlindBaseline
from storm_reoptimizer.eval.jev import DEFAULT_JEV_MODEL, JevDecider
from storm_reoptimizer.eval.scenario_file import load_all_scenarios
from storm_reoptimizer.eval.suite import (
    RUNS_PER_EPISODE, build_arg_parser, build_deciders, render_results_table,
    run_suite, select_episodes,
)
from storm_reoptimizer.mcp_client import connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)


def test_the_budget_is_three_runs_per_episode():
    assert RUNS_PER_EPISODE == 3


def test_select_episodes_with_no_filter_returns_the_same_object():
    # main()'s whole-suite gates rely on getting the IDENTICAL full roster
    # back (not just an equal-by-value copy) when --only is absent, so a
    # plain suite run's behavior is provably unchanged by this flag's
    # existence.
    all_episodes = load_all_scenarios()
    assert select_episodes(all_episodes, None) is all_episodes
    assert select_episodes(all_episodes, "") is all_episodes


def test_select_episodes_filters_to_the_named_ids_in_the_given_order():
    all_episodes = load_all_scenarios()
    filtered = select_episodes(all_episodes, "T1b,T1a")
    assert list(filtered) == ["T1b", "T1a"]
    assert filtered["T1a"] is all_episodes["T1a"]


def test_select_episodes_rejects_an_unknown_id():
    all_episodes = load_all_scenarios()
    with pytest.raises(SystemExit, match="nonexistent-id"):
        select_episodes(all_episodes, "T1a,nonexistent-id")


def test_select_episodes_does_not_mutate_the_full_roster():
    # Regression lock for the bug this flag shipped with: `main()` used the
    # `--only`-filtered dict for `assert_no_global_policy_solves_the_suite`
    # too, and a 2-episode subset with different gold labels is ALWAYS
    # trivially separable by a threshold on the flip variable -- found live,
    # `--only T1a,T1b` crashed that gate with "a fixed global policy solves
    # the suite 2/2" before any agent call was made. The fix is that the
    # full roster passed in must come back unmodified and distinct from
    # whatever a caller does with the filtered result.
    all_episodes = load_all_scenarios()
    original_ids = set(all_episodes)
    select_episodes(all_episodes, "T1a,T1b")
    assert set(all_episodes) == original_ids


def _connect_for(state_paths, server_command, server_env):
    def _factory(state_file):
        @asynccontextmanager
        async def _connect():
            async with connect_server(
                TOPOLOGY_PATH, server_command=server_command, env=server_env,
                extra_args=["--state", str(state_paths[state_file])],
            ) as client:
                yield client
        return _connect
    return _factory


def test_baseline_is_deterministic_across_runs(
    tmp_path, eval_state_paths, local_server_command, local_server_env,
):
    d1 = {"D1": load_all_scenarios()["D1"]}
    results = asyncio.run(run_suite(
        _connect_for(eval_state_paths, local_server_command,
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
    tmp_path, eval_state_paths, local_server_command, local_server_env,
):
    results = asyncio.run(run_suite(
        _connect_for(eval_state_paths, local_server_command,
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


def test_only_narrowed_main_does_not_crash_the_whole_suite_gates(
    monkeypatch, capsys, eval_state_paths, local_server_command,
):
    """Integration-level regression lock for the `--only` bug (2026-09-06,
    T1 spend-or-hold redesign plan, Task 17): `main()` initially routed the
    `--only`-narrowed episode dict into the two whole-suite "no rule solves
    it" gates (`assert_no_single_variable_rule_solves`,
    `assert_no_global_policy_solves_the_suite`). A 2-episode subset with
    different gold labels is ALWAYS trivially separable by a threshold on
    whatever scalar differs between them -- that is what makes them a valid
    flipped pair at all -- so this crashed `main()` outright the first time
    `--only T1a,T1b` was run, before any agent call was made:
    `PairInvalid: a fixed global policy solves the suite 2/2`.

    The unit tests above (`test_select_episodes_*`) exercise the extracted
    `select_episodes()` helper's own mechanics, but the actual bug lived in
    `main()`'s OWN WIRING -- which dict gets passed to which gate call --
    and nothing short of running `main()` itself catches a future edit that
    swaps `all_episodes`/`episodes` back at one of those call sites (or at
    `_run_dimensional_coherence_invariants`'s). This test runs the REAL
    `main()`, scoped to `--only T1a,T1b` (the exact command line the bug was
    found under), with no `--include-agent` so it stays free -- only the two
    deterministic baselines roll out. `main()` itself reads
    `STORM_REOPTIMIZER_MCP_SERVER_CMD` via `os.environ.get(...)` and then
    forwards `dict(os.environ)` to the server subprocess (the same pattern
    `conftest.py`'s `local_server_env` fixture uses standalone), so setting
    the var via `monkeypatch` before calling `main()` is the whole setup
    needed -- no fixture-provided env dict has to be threaded through."""
    monkeypatch.setenv(
        "STORM_REOPTIMIZER_MCP_SERVER_CMD", json.dumps(local_server_command))

    from storm_reoptimizer.eval.suite import main

    main(["--only", "T1a,T1b", "--runs", "1"])  # must not raise

    out = capsys.readouterr().out
    assert "pair_solved" in out
    # Confirms `--only` genuinely scoped the ROLLOUT (2 episodes), which is
    # the one thing it's supposed to narrow -- distinguishing "ran narrowed
    # and passed" from some accidental full-suite fallback silently masking
    # the regression this test exists to catch.
    assert "x 2 episodes x" in out


def test_traces_land_on_disk(
    tmp_path, eval_state_paths, local_server_command, local_server_env,
):
    asyncio.run(run_suite(
        _connect_for(eval_state_paths, local_server_command,
                    local_server_env),
        topology_path=TOPOLOGY_PATH,
        deciders=[ForecastBlindBaseline("immediate")],
        runs_per_episode=1,
        scenarios={"D1": load_all_scenarios()["D1"]}, traces_dir=tmp_path))
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert written == ["D1-baseline_immediate-0-metrics.json",
                       "D1-baseline_immediate-0.json"]
    assert json.loads((tmp_path / written[1]).read_text())["scenario_id"] == "D1"


def test_a_metrics_sidecar_lands_beside_every_trace(
    tmp_path, eval_state_paths, local_server_command, local_server_env,
):
    """tools/build_viewer_data.py imports nothing that pulls in the MCP
    client, and scoring.py imports EpisodeTrace from runner, which does -- so
    decision_label, regret_gbps_h, acted_too_late and inert_commits are
    uncomputable in the viewer and on no trace (harness explainer, §11 item
    7's stated blocker)."""
    results = asyncio.run(run_suite(
        _connect_for(eval_state_paths, local_server_command,
                    local_server_env),
        topology_path=TOPOLOGY_PATH,
        deciders=[ForecastBlindBaseline("immediate")],
        runs_per_episode=1,
        scenarios={"D1": load_all_scenarios()["D1"]}, traces_dir=tmp_path))
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert written == ["D1-baseline_immediate-0-metrics.json",
                       "D1-baseline_immediate-0.json"]
    sidecar = json.loads(
        (tmp_path / "D1-baseline_immediate-0-metrics.json").read_text())
    assert sidecar == results["episodes"]["D1"]["baseline:immediate"]["metrics"][0]
    assert {"decision_label", "regret_gbps_h", "acted_too_late",
            "inert_commits"} <= set(sidecar)


def test_include_jev_appends_both_arms_without_the_extra_or_a_key():
    built = build_deciders(build_arg_parser().parse_args(["--include-jev"]))
    jevs = [d for d in built if isinstance(d, JevDecider)]
    assert [d.name for d in jevs] == [f"jev:{DEFAULT_JEV_MODEL}",
                                      f"jev:{DEFAULT_JEV_MODEL}+totals"]
    assert [d.include_totals for d in jevs] == [False, True]
    assert all(d._client is None for d in jevs)
    assert all(d._audit_path.name == "jev-calls.jsonl" for d in jevs)
    assert all(d._audit_path.parent.name == "traces" for d in jevs)


def test_jev_model_is_overridable_and_off_by_default():
    assert not any(isinstance(d, JevDecider)
                   for d in build_deciders(build_arg_parser().parse_args([])))
    built = build_deciders(build_arg_parser().parse_args(
        ["--include-jev", "--jev-model", "jev-9"]))
    assert [d.name for d in built[-2:]] == ["jev:jev-9", "jev:jev-9+totals"]


def test_build_deciders_tolerates_a_namespace_without_jev_fields():
    # test_agent.py builds args as SimpleNamespace(include_agent, agent_model,
    # agent_effort) with no jev fields; that must keep working.
    built = build_deciders(SimpleNamespace(
        include_agent=False, agent_model="x", agent_effort=None))
    assert len(built) == 2
    assert not any(isinstance(d, JevDecider) for d in built)


def test_preflight_names_the_missing_sdk(monkeypatch):
    monkeypatch.setattr(suite_module.importlib.util, "find_spec",
                        lambda name: None)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    with pytest.raises(SystemExit) as exc:
        suite_module.preflight_jev(SimpleNamespace(include_jev=True))
    assert "typesafe_sdk" in str(exc.value)
    assert 'pip install -e ".[jev]"' in str(exc.value)


def test_preflight_names_the_missing_key(monkeypatch):
    monkeypatch.setattr(suite_module.importlib.util, "find_spec",
                        lambda name: object())
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        suite_module.preflight_jev(SimpleNamespace(include_jev=True))
    assert "TYPESAFE_API_KEY" in str(exc.value)


def test_preflight_is_skipped_without_include_jev(monkeypatch):
    monkeypatch.setattr(suite_module.importlib.util, "find_spec",
                        lambda name: None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    suite_module.preflight_jev(SimpleNamespace(include_jev=False))
    suite_module.preflight_jev(SimpleNamespace())


def test_main_runs_the_jev_preflight_before_any_work(monkeypatch):
    def refuse(args):
        raise SystemExit("preflight ran")
    monkeypatch.setattr(suite_module, "preflight_jev", refuse)
    with pytest.raises(SystemExit, match="preflight ran"):
        suite_module.main(["--include-jev"])


def test_results_table_notes_jev_rows_and_the_citation_footnote():
    results = {"pairs": {"jev:jev-1.13.0": {"pair_solved": 0.0},
                         "jev:jev-1.13.0+totals": {"pair_solved": 0.0}},
               "episodes": {}, "budget": {"seeds": [17], "episodes": 0,
                                          "runs_per_episode": 1, "rollouts": 0}}
    table = render_results_table(results)
    assert "System-1 arm" in table and "horizon_totals" in table
    assert "cites_flip_variable_frac" in table
    # Only the +totals row names horizon_totals; the raw arm's row must not.
    rows = {line.split("|")[1].strip(): line for line in table.splitlines()
            if line.startswith("| jev:")}
    assert "System-1 arm" in rows["jev:jev-1.13.0"]
    assert "horizon_totals" not in rows["jev:jev-1.13.0"]
    assert "horizon_totals" in rows["jev:jev-1.13.0+totals"]
