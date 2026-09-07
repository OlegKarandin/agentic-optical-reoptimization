# src/storm_reoptimizer/eval/suite.py
"""Suite orchestration and the README's results table (eval design spec,
"Run budget", "Scoring").

The first cut is ONE seed (17) x 7 episodes x N=3 runs = 21 rollouts (6 twin
halves + 1 diagnostic). N=3 covers agent nondeterminism; both baseline
variants are deterministic and run once per episode. Additional seeds are a
later extension: three seeds would triple the budget for a robustness signal
that only becomes interesting once the episodes are known to discriminate at
all.

Note pair_solved's granularity honestly wherever it is reported: over three
pairs it takes values in {0, 1/3, 2/3, 1}. That is enough to tell a working
harness from a broken one and NOT enough to separate luck from skill."""
from __future__ import annotations

import argparse
import asyncio
import copy
import statistics
from pathlib import Path

from .agent import ClaudeDecider, DEFAULT_MODEL
from .assertions import (assert_claim_is_one_lightpath,
                         assert_claimants_depot_eligible,
                         assert_claimants_have_filterable_exposure,
                         assert_depot_is_the_binding_site,
                         assert_escape_route_survives,
                         assert_group_fits_one_lightpath,
                         assert_no_global_policy_solves_the_suite,
                         assert_no_single_variable_rule_solves,
                         assert_realized_cuts_pass_the_event_filter,
                         assert_risk_group_covers_measurable_exposure,
                         assert_sampling_error_within_margin)
from .baseline import ForecastBlindBaseline
from .runner import run_episode, service_geometry
from .scenario_file import ScenarioFile, load_all_scenarios
from .scoring import cross_twin_metrics, episode_metrics

RUNS_PER_EPISODE = 3
# src/storm_reoptimizer/eval/suite.py -> eval -> storm_reoptimizer -> src -> repo root
REPO_ROOT = Path(__file__).parent.parent.parent.parent
TRACES_DIR = REPO_ROOT / "eval" / "traces"
AGENT_AUDIT_PATH = TRACES_DIR / "agent-calls.jsonl"


def _redact_volatile_ids(trace_dict: dict) -> dict:
    """Strip the one class of field in an episode trace that is guaranteed to
    differ across separately-launched rollouts EVEN WHEN every decision is
    identical: snapshot ids the server mints via uuid.uuid4().hex
    (multilayer_optical_network's SnapshotStore.put(), confirmed by reading
    it directly -- snapshot_create() and a live commit_plan()'s
    intended_snapshot_id both go through it). These are real, meaningful ids
    for reconcile()/snapshot_restore() within ONE run, but they carry no
    decision content across runs, so comparing them defeats the point of
    "same seed, same scenario, same decisions -> identical decision records"
    (this module's determinism claim, tests/eval/test_suite.py's load-bearing
    assertion). Only the copy stored under results[...]["runs"] is redacted;
    the on-disk trace file runner.py writes via trace_path keeps the real
    ids -- that file is telemetry, not a determinism check."""
    redacted = copy.deepcopy(trace_dict)
    for hour in redacted.get("hours") or ():
        hour.pop("snapshot_id", None)
        for iteration in hour.get("iterations") or ():
            iteration.pop("intended_snapshot_id", None)
    return redacted


async def run_suite(connect_for, *, topology_path, deciders,
                    runs_per_episode: int = RUNS_PER_EPISODE,
                    scenarios: dict[str, ScenarioFile] | None = None,
                    traces_dir: Path | None = None,
                    collapse_deterministic: bool = True) -> dict:
    """`connect_for(state_file)` returns a zero-arg async context manager
    factory yielding a fresh connection against that scenario's own state
    file -- one per rollout, because a rollout mutates state.

    `collapse_deterministic` runs a deterministic decider once instead of
    `runs_per_episode` times; set it False to prove the determinism rather
    than assume it (tests/eval/test_suite.py does exactly that)."""
    episodes = scenarios if scenarios is not None else load_all_scenarios()
    traces_dir = Path(traces_dir) if traces_dir is not None else TRACES_DIR

    results: dict = {"episodes": {}, "pairs": {}, "budget": {
        "seeds": sorted({e.seed for e in episodes.values()}),
        "episodes": len(episodes), "runs_per_episode": runs_per_episode,
        "rollouts": len(episodes) * runs_per_episode * len(deciders)}}

    traces_by_decider: dict[str, dict[str, list]] = {}
    for scenario_id, scenario in sorted(episodes.items()):
        results["episodes"][scenario_id] = {}
        for decider in deciders:
            runs, metrics = [], []
            # A deterministic decider needs only one rollout; running three
            # identical ones buys nothing and costs three server launches.
            n = (1 if collapse_deterministic
                 and isinstance(decider, ForecastBlindBaseline)
                 else runs_per_episode)
            for run_index in range(n):
                safe = decider.name.replace(":", "_")
                async with connect_for(scenario.state_file)() as client:
                    trace = await run_episode(
                        client, scenario, decider, topology_path=topology_path,
                        run_index=run_index,
                        trace_path=traces_dir / f"{scenario_id}-{safe}-{run_index}.json")
                runs.append(_redact_volatile_ids(trace.to_dict()))
                metrics.append(episode_metrics(scenario, trace))
                traces_by_decider.setdefault(decider.name, {}).setdefault(
                    scenario_id, []).append(trace)
            results["episodes"][scenario_id][decider.name] = {
                "runs": runs, "metrics": metrics,
                "label_correct_mean": statistics.fmean(
                    float(m["label_correct"]) for m in metrics),
                "label_correct_spread": (
                    max(float(m["label_correct"]) for m in metrics)
                    - min(float(m["label_correct"]) for m in metrics)),
            }

    pairs = sorted({e.pair for e in episodes.values() if e.pair})
    for decider in deciders:
        solved, details = [], {}
        for pair in pairs:
            a = episodes[f"{pair}a"]
            b = episodes[f"{pair}b"]
            cross = cross_twin_metrics(
                a, traces_by_decider[decider.name][a.id][0],
                b, traces_by_decider[decider.name][b.id][0])
            details[pair] = cross
            solved.append(float(cross["pair_solved"]))
        results["pairs"][decider.name] = {
            "pair_solved": statistics.fmean(solved) if solved else 0.0,
            "detail": details}
    return results


def _decider_metrics(results: dict, name: str) -> list[dict]:
    """Every per-run `episode_metrics` dict recorded for `name`, across every
    episode -- what the table's regret/inert_commits columns aggregate over.
    Not every decider necessarily ran every episode (e.g. a scenarios= subset
    passed to run_suite), so this skips episodes `name` is absent from rather
    than assuming a uniform roster."""
    return [m for ep in results["episodes"].values() if name in ep
           for m in ep[name]["metrics"]]


def render_results_table(results: dict) -> str:
    """The README's table, plus the two claims stated separately -- they carry
    different weight and must not be blurred.

    `regret_gbps_h` and `inert_commits` (Task 8/6's `episode_metrics` keys)
    are surfaced here as one number per decider: the mean over every
    recorded run for `regret_gbps_h` (a per-episode outcome gap, so mean is
    the natural rollup), and the plain sum for `inert_commits` (a count of
    free no-op commits -- T1's finding that a candidate changing no path and
    spending no spare is otherwise indistinguishable from a real action).

    `regret_gbps_h` is `None` on any run whose episode is not graded on the
    `spare_action_by_deadline` label rule (scoring.episode_metrics: today
    that's every episode but T1a/T1b -- D1/T2/T3 have no `gold.
    outcome_gbps_h` to regret against). The mean is over the runs that DO
    carry a real value; a decider with none at all reports "n/a" rather than
    crashing statistics.fmean on a generator of Nones (found live running
    the full 7-episode suite -- --scenario-scoped tests only ever saw T1a/
    T1b and never hit this)."""
    lines = ["| decider | pair_solved | episodes correct | regret_gbps_h | "
             "inert_commits | notes |",
             "|---|---|---|---|---|---|"]
    for name, summary in sorted(results["pairs"].items()):
        correct = sum(
            1 for ep in results["episodes"].values()
            if name in ep and ep[name]["label_correct_mean"] >= 0.5)
        metrics = _decider_metrics(results, name)
        regrets = [m["regret_gbps_h"] for m in metrics
                  if m["regret_gbps_h"] is not None]
        regret_str = (f"{statistics.fmean(regrets):.1f}" if regrets
                     else "n/a")
        inert_total = sum(m["inert_commits"] for m in metrics)
        if name.startswith("baseline:"):
            note = ("fixed policy: same input in both halves, so exactly one "
                    "half per pair")
        elif name.startswith("agent:"):
            note = "agent arm: shown the per-horizon rival/actionable ECAR totals"
        else:
            note = ""
        lines.append(f"| {name} | {summary['pair_solved']:.2f} | "
                     f"{correct}/{len(results['episodes'])} | "
                     f"{regret_str} | {inert_total} | {note} |")
    budget = results["budget"]
    lines += [
        "",
        f"Budget: seed(s) {budget['seeds']} x {budget['episodes']} episodes "
        f"x N={budget['runs_per_episode']} = {budget['rollouts']} rollouts.",
        "",
        "**Claim 1 (provable).** The agent beats every fixed policy that does "
        "not read the forecast: the twins' menus and observables are identical "
        "by construction, so such a policy emits the same answer twice and "
        "scores exactly 50%.",
        "**Claim 2 (asserted at build time).** No rule keyed on any single "
        "forecast variable -- and no parameter-free greedy policy -- solves "
        "the suite; checked over the gold labels before any rollout runs.",
        "",
        "`pair_solved` over three pairs takes values in {0, 1/3, 2/3, 1}: "
        "enough to tell a working harness from a broken one, not enough to "
        "separate luck from skill. Held-out seeds are the path to power.",
        "",
        "The flip-variable citation metric is a NECESSARY, NOT SUFFICIENT "
        "filter for \"right answer, absent reason\". It is entity matching, "
        "not reasoning verification.",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="storm_reoptimizer.eval.suite",
        description="Run every eval episode against the deciders and print "
                    "the results table.")
    p.add_argument(
        "--include-agent", action="store_true",
        help="Also run the LLM decider. Requires the `agent` extra "
             "(pip install -e '.[agent]') and API credentials in the "
             "environment; costs money. Off by default so a plain run stays "
             "free and keyless.")
    p.add_argument(
        "--agent-model", default=DEFAULT_MODEL,
        help=f"Model for --include-agent (default: {DEFAULT_MODEL}).")
    p.add_argument(
        "--only", default=None,
        help="Comma-separated scenario ids to run (e.g. T1a,T1b). Restricts "
             "both the pre-flight gates and the rollouts to this subset. "
             "Default: every scenario `load_all_scenarios` finds.")
    p.add_argument(
        "--runs", type=int, default=RUNS_PER_EPISODE, dest="runs",
        help=f"Rollouts per episode per (non-collapsed) decider (default: "
             f"{RUNS_PER_EPISODE}). Scoped low for a paid --include-agent "
             f"run; a deterministic baseline still collapses to one "
             f"rollout regardless (see collapse_deterministic).")
    return p


def build_deciders(args: argparse.Namespace) -> list:
    """The decider list one suite run drives. Split out of main() so the flag
    is testable without a server, a network, or an API key -- main() itself
    launches the real MCP subprocess."""
    deciders = [ForecastBlindBaseline("immediate"),
                ForecastBlindBaseline("at_deadline")]
    if args.include_agent:
        deciders.append(ClaudeDecider(
            model=args.agent_model, audit_path=AGENT_AUDIT_PATH))
    return deciders


def select_episodes(all_episodes: dict[str, ScenarioFile],
                    only: str | None) -> dict[str, ScenarioFile]:
    """`--only`'s own filter, pulled out of `main()` so it is unit-testable
    without a live server: a comma-separated allowlist of scenario ids, or
    `all_episodes` itself unchanged when `only` is falsy. Never mutates
    `all_episodes` -- `main()` relies on that to keep passing the FULL
    roster to the two whole-suite gates
    (`assert_no_single_variable_rule_solves`,
    `assert_no_global_policy_solves_the_suite`) regardless of this filter;
    see `main()`'s own comment on why those two must never see a narrowed
    set (a two-episode subset with different labels is trivially separable
    by construction, so scoping them down fails the build for a reason that
    has nothing to do with a real confound -- found live the first time
    `--only T1a,T1b` was run)."""
    if not only:
        return all_episodes
    wanted = [s.strip() for s in only.split(",") if s.strip()]
    missing = [s for s in wanted if s not in all_episodes]
    if missing:
        raise SystemExit(
            f"--only names scenario id(s) not found by load_all_scenarios: "
            f"{missing}. Known ids: {sorted(all_episodes)}")
    return {sid: all_episodes[sid] for sid in wanted}


async def _groups_for(client, scenario: ScenarioFile, *,
                      topology_path: Path) -> dict[str, tuple[dict, ...]]:
    """`observation._restorable_groups`'s own output for `scenario`, at its
    decision hour, computed fresh against the live server -- what
    invariants 4/5 (Task 12, exposure-and-depot plan) check against.

    Built through `observation.build_observation` -- the SAME function the
    real rollout calls -- rather than reimplementing the grouping, so this
    check and the payload the agent is actually shown can never disagree
    (the same discipline `assert_pair_derived_geometry_is_equal` states for
    derived geometry)."""
    from ..mcp_client import call_tool_json
    from .observation import build_observation

    geometry = await service_geometry(client, topology_path)
    services = (await call_tool_json(client, "get_services"))["services"]
    obs = build_observation(
        scenario, scenario.decision_hour, service_spans=geometry.cuttable_spans,
        services=services, spares_on_hand=scenario.spares_on_hand,
        endpoint_sites=geometry.endpoint_sites, depot_site=scenario.depot_site)
    return obs.restorable_groups


async def _run_dimensional_coherence_invariants(
    connect_for, episodes: dict[str, ScenarioFile], *, topology_path: Path,
) -> None:
    """Task 12 (exposure-and-depot plan, §6 invariants 1-8): the
    dimensional-coherence class, checked over the FULL episode set before
    any rollout runs -- a build-time gate, exactly like the two checks
    above it in `main()`. `connect_for` is the same `connect_for(state_file)`
    factory `run_suite` takes; each scenario gets its own fresh connection
    (the same contract every client-taking check in assertions.py already
    has), except the one topology read shared across all seven episodes
    (the OMS list is a property of the topology file, identical across
    state files).

    Invariant 8 (`assert_sampling_error_within_margin`) is NOT called here:
    it takes the whole suite's `FlipScalars.values()` map directly (the same
    shape `assert_no_global_policy_solves_the_suite` consumes), which
    `main()` below already computes once for that other check -- so `main()`
    calls invariant 8 alongside it rather than this function re-fetching the
    same data a second time.

    Invariants 6/7 (`assert_depot_is_the_binding_site`/
    `assert_escape_route_survives`) are SKIPPED for any scenario whose
    `metadata.stale_invariants` is true (2026-09-05 plan, Task 11): D1
    was built and frozen against the OLD working-leg-only exposure numbers,
    and Tasks 3/4 of that same plan moved to a joint (working-AND-protection)
    probability for protected services, so those two invariants are
    known-stale there pending D1's own redesign turn -- see the ruling
    in `tests/eval/test_episodes.py`'s `PAIRS`/`STALE_PAIRS` comment."""
    async with connect_for(next(iter(episodes.values())).state_file)() as client:
        from ..mcp_client import call_tool_json
        oms_by_id_list = (await call_tool_json(
            client, "get_topology", {"layer": "optical"}))["oms"]
        oms_by_id = {o["id"]: o for o in oms_by_id_list}

    for scenario in episodes.values():
        # Sync, no client -- checked first and separately so a violation
        # here (T1a's buried realized cut, today) is reported without
        # needing a second connection.
        assert_realized_cuts_pass_the_event_filter(
            scenario, topology_path=topology_path, oms_by_id=oms_by_id)

        # The batching below (one connection for invariants 2/3/6/7 plus the
        # groups read) is justified ONLY because no shipped episode realizes
        # a cut strictly before its own decision hour: `assert_depot_is_the_
        # binding_site`'s `menu_at_decision_hour` conditionally calls
        # `inject_failure` for exactly that case, which would mutate the
        # connection's network state midway through the batch and make
        # `assert_escape_route_survives` (called after it, on the SAME
        # connection) silently see a POST-cut network while the checks
        # before it saw the PRE-cut one. Asserted here, not merely assumed,
        # for the identical reason `assert_wait_gold_has_no_free_escape`
        # asserts its own version of this same hazard: "so a future
        # violation fails loudly instead of silently misreading" (2026-08-30
        # review fix, finding 3).
        d = scenario.hours.index(scenario.decision_hour)
        assert not any(scenario.realized.get(hour) for hour in scenario.hours[:d]), (
            f"{scenario.id}: realizes a cut strictly before its own decision "
            f"hour {scenario.decision_hour!r}; batching invariants 2/3/6/7 "
            f"onto one connection assumes this never happens -- split this "
            f"scenario's checks across separate connections (one per side of "
            f"the replay) instead of relying on that assumption")

        async with connect_for(scenario.state_file)() as client:
            await assert_claimants_have_filterable_exposure(
                client, scenario, topology_path=topology_path)
            await assert_claimants_depot_eligible(
                client, scenario, topology_path=topology_path)
            await assert_risk_group_covers_measurable_exposure(
                client, scenario, topology_path=topology_path, oms=oms_by_id_list)
            groups = await _groups_for(
                client, scenario, topology_path=topology_path)
            # stale_invariants (2026-09-05 plan, Task 11): D1's
            # `assert_depot_is_the_binding_site`/`assert_escape_route_survives`
            # were built and frozen against the OLD working-leg-only exposure
            # numbers. Tasks 3/4 of this same plan moved to a joint
            # (working-AND-protection) probability for protected services,
            # so these two invariants are known-stale for that episode --
            # ruling: "don't worry about old tests or T2/T3, we'll update
            # those later". Declared per-episode via
            # `metadata.stale_invariants: true` (D1 only) rather
            # than skipped by id here, so the flag travels with the scenario
            # file that earns it.
            if not scenario.metadata.get("stale_invariants"):
                await assert_depot_is_the_binding_site(
                    client, scenario, topology_path=topology_path)
                await assert_escape_route_survives(client, scenario)

        assert_claim_is_one_lightpath(scenario, groups)
        assert_group_fits_one_lightpath(scenario, groups)


def main(argv: list[str] | None = None) -> None:
    import os
    from contextlib import asynccontextmanager

    from .derived import (
        FlipScalars, derived_scalars_for_suite, flip_scalars_for_suite,
    )
    from ..mcp_client import connect_server

    args = build_arg_parser().parse_args(argv)

    topology = (Path(__file__).parent.parent / "data"
                / "toy_india_topology.json")
    # `all_episodes` is the FULL roster and is never narrowed by `--only`:
    # every build-time gate below (the two whole-suite checks,
    # `assert_no_single_variable_rule_solves` and `assert_no_global_policy_
    # solves_the_suite`, AND `_run_dimensional_coherence_invariants`) runs
    # against it, not against `--only`'s subset. For the two whole-suite
    # checks this is load-bearing correctness, not just consistency: they
    # ask whether any rule separates every episode's gold label, and that
    # question is only meaningful over the full suite -- any two-episode
    # subset with different labels is TRIVIALLY separable by a threshold on
    # whatever scalar happens to differ between them (that's what makes
    # them a valid flipped pair at all), so scoping these two checks down to
    # `--only` would make them fail by construction, not find a real
    # confound. Found live: `--only T1a,T1b` crashed
    # `assert_no_global_policy_solves_the_suite` with "a fixed global policy
    # solves the suite 2/2" the first time this flag was wired up, before
    # this comment existed. `_run_dimensional_coherence_invariants` is
    # widened here for a different reason -- consistency with its own
    # docstring's promise of a full-suite build-time gate, not because a
    # narrowed run would spuriously fail the way the other two would (its
    # checks are per-episode, not cross-episode separability). `--only`
    # narrows exactly one thing: the actual rollout (`episodes`, below) --
    # never any of these three validation gates.
    all_episodes = load_all_scenarios()
    episodes = select_episodes(all_episodes, args.only)

    # A real install puts `multilayer-optical-mcp` on PATH and
    # connect_server()'s own default handles it. This workspace's sibling
    # server repo isn't on PATH (see mcp_client.DEFAULT_SERVER_COMMAND's
    # docstring on the Cyrillic-path editable-install bug), so
    # STORM_REOPTIMIZER_MCP_SERVER_CMD -- already read by
    # mcp_client._default_server_command() -- is how this workspace points
    # at the sibling conda env's python. stdio_client's own default only
    # inherits a restricted subset of the environment, which is too narrow
    # for that subprocess to start at all, so forward the full parent
    # environment here the same way tests/conftest.py's local_server_env
    # fixture does.
    env = dict(os.environ) if os.environ.get(
        "STORM_REOPTIMIZER_MCP_SERVER_CMD") else None

    def _connect_for(state_file: str):
        # state_file is repo-relative (scenario_file.py's module docstring);
        # resolve against REPO_ROOT so the module runs from any cwd.
        state = REPO_ROOT / state_file

        @asynccontextmanager
        async def _connect():
            async with connect_server(
                topology, env=env, extra_args=["--state", str(state)]) as client:
                yield client
        return _connect

    async def _derived_scalars() -> dict[str, dict[str, float]]:
        return await derived_scalars_for_suite(
            _connect_for, list(all_episodes.values()), topology_path=topology)

    async def _flip_scalars() -> dict[str, FlipScalars]:
        return await flip_scalars_for_suite(
            _connect_for, list(all_episodes.values()), topology_path=topology)

    # Widened with DERIVED (not just author-declared) geometry -- the
    # declared-only check is exactly what missed T1's p_cut confound for
    # three rounds of review (whole-branch review, finding C2). The entry
    # point that prints Claim 2 below must be backed by the same enumeration
    # that closed the finding, not the weaker one that missed it.
    #
    # Both this and the next check run over `all_episodes`, NOT the
    # `--only`-filtered `episodes` -- see all_episodes' own comment above.
    assert_no_single_variable_rule_solves(
        list(all_episodes.values()), asyncio.run(_derived_scalars()))

    # ...alongside the existing per-pair check. Two checks, two questions:
    # the per-pair one asks whether any scalar the halves SHARE differs; this
    # one asks whether one fixed threshold answers every half. See rules.py's
    # module docstring on why the flip scalar is in exactly one of them.
    flip_values = {sid: f.values()
                  for sid, f in asyncio.run(_flip_scalars()).items()}
    assert_no_global_policy_solves_the_suite(
        list(all_episodes.values()), flip_values)

    # Invariant 8 (Task 12's own numbering; wired here in Task 14 against the
    # real, frozen per-episode numbers, per the exposure-and-depot plan) --
    # the suite's smallest flip margin must clear cone.py's own measured
    # Sobol sampling noise floor (tests/eval/test_exposure_model.py's bound)
    # by at least 10x, so the sampler's own noise could never flip a gold
    # label. Reuses the SAME flip_values just computed for the check above --
    # one server round trip, two questions asked of the same numbers.
    assert_sampling_error_within_margin(flip_values, measured_error=2.56e-4)

    # Task 12 (exposure-and-depot plan): the eight dimensional-coherence
    # invariants, over the full episode set. Same build-time-gate contract
    # as the two checks above -- they fire before tokens are spent. Also
    # over `all_episodes`, NOT the `--only`-filtered `episodes`: this
    # function's own docstring promises "the FULL episode set... a
    # build-time gate, exactly like the two checks above it in main()", and
    # `--only` exists to scope the PAID rollout below, not to silently skip
    # validating invariants 1-8 for episodes outside the subset (found in
    # review: the first fix here only widened the two checks immediately
    # above, leaving this third one still narrowed).
    asyncio.run(_run_dimensional_coherence_invariants(
        _connect_for, all_episodes, topology_path=topology))

    results = asyncio.run(run_suite(
        _connect_for, topology_path=topology, deciders=build_deciders(args),
        scenarios=episodes, runs_per_episode=args.runs))
    print(render_results_table(results))


if __name__ == "__main__":
    main()
