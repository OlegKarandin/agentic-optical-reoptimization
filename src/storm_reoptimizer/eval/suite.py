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
                         assert_realized_cuts_pass_the_event_filter)
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


async def run_suite(connect, *, topology_path, deciders,
                    runs_per_episode: int = RUNS_PER_EPISODE,
                    scenarios: dict[str, ScenarioFile] | None = None,
                    traces_dir: Path | None = None,
                    collapse_deterministic: bool = True) -> dict:
    """`connect` is a zero-arg async context manager factory yielding a fresh
    server connection -- one per rollout, because a rollout mutates state.

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
                async with connect() as client:
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


def render_results_table(results: dict) -> str:
    """The README's table, plus the two claims stated separately -- they carry
    different weight and must not be blurred."""
    lines = ["| decider | pair_solved | episodes correct | notes |",
             "|---|---|---|---|"]
    for name, summary in sorted(results["pairs"].items()):
        correct = sum(
            1 for ep in results["episodes"].values()
            if name in ep and ep[name]["label_correct_mean"] >= 0.5)
        if name.startswith("baseline:"):
            note = ("fixed policy: same input in both halves, so exactly one "
                    "half per pair")
        elif name.startswith("agent:"):
            note = "agent arm: shown the per-horizon rival/actionable ECAR totals"
        else:
            note = ""
        lines.append(f"| {name} | {summary['pair_solved']:.2f} | "
                     f"{correct}/{len(results['episodes'])} | {note} |")
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
    connect, episodes: dict[str, ScenarioFile], *, topology_path: Path,
) -> None:
    """Task 12 (exposure-and-depot plan, §6 invariants 1-8): the
    dimensional-coherence class, checked over the FULL episode set before
    any rollout runs -- a build-time gate, exactly like the two checks
    above it in `main()`. `connect` is the same zero-arg async context
    manager factory `run_suite` takes; each scenario gets its own fresh
    connection (the same contract every client-taking check in
    assertions.py already has), except the one topology read shared across
    all seven episodes, since they share one state file.

    Invariant 8 (`assert_sampling_error_within_margin`) is NOT called here:
    it needs the real, frozen per-episode flip values Task 14 produces, and
    is exercised only against constructed values until then (see its own
    docstring)."""
    async with connect() as client:
        from ..mcp_client import call_tool_json
        oms_by_id = {
            o["id"]: o
            for o in (await call_tool_json(
                client, "get_topology", {"layer": "optical"}))["oms"]}

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

        async with connect() as client:
            await assert_claimants_have_filterable_exposure(
                client, scenario, topology_path=topology_path)
            await assert_claimants_depot_eligible(
                client, scenario, topology_path=topology_path)
            groups = await _groups_for(
                client, scenario, topology_path=topology_path)
            await assert_depot_is_the_binding_site(
                client, scenario, topology_path=topology_path)
            await assert_escape_route_survives(client, scenario)

        assert_claim_is_one_lightpath(scenario, groups)
        assert_group_fits_one_lightpath(scenario, groups)


def main(argv: list[str] | None = None) -> None:
    import os
    from contextlib import asynccontextmanager

    from .derived import FlipScalars, derived_scalars_for, flip_scalars_for
    from ..mcp_client import connect_server

    args = build_arg_parser().parse_args(argv)

    topology = (Path(__file__).parent.parent / "data"
                / "toy_india_topology.json")
    episodes = load_all_scenarios()
    # state_file is recorded relative to the repo root (see scenario_file.py's
    # module docstring and every scenario YAML's "eval/states/..." value), not
    # relative to the current working directory -- resolve it against
    # REPO_ROOT so `python -m storm_reoptimizer.eval.suite` works regardless
    # of the caller's cwd.
    state = REPO_ROOT / episodes["D1"].state_file

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

    @asynccontextmanager
    async def _connect():
        async with connect_server(
            topology, env=env, extra_args=["--state", str(state)]) as client:
            yield client

    async def _derived_scalars() -> dict[str, dict[str, float]]:
        async with _connect() as client:
            return await derived_scalars_for(
                client, list(episodes.values()), topology_path=topology)

    async def _flip_scalars() -> dict[str, FlipScalars]:
        async with _connect() as client:
            return await flip_scalars_for(
                client, list(episodes.values()), topology_path=topology)

    # Widened with DERIVED (not just author-declared) geometry -- the
    # declared-only check is exactly what missed T1's p_cut confound for
    # three rounds of review (whole-branch review, finding C2). The entry
    # point that prints Claim 2 below must be backed by the same enumeration
    # that closed the finding, not the weaker one that missed it.
    assert_no_single_variable_rule_solves(
        list(episodes.values()), asyncio.run(_derived_scalars()))

    # ...alongside the existing per-pair check. Two checks, two questions:
    # the per-pair one asks whether any scalar the halves SHARE differs; this
    # one asks whether one fixed threshold answers every half. See rules.py's
    # module docstring on why the flip scalar is in exactly one of them.
    assert_no_global_policy_solves_the_suite(
        list(episodes.values()),
        {sid: f.values() for sid, f in asyncio.run(_flip_scalars()).items()})

    # Task 12 (exposure-and-depot plan): the eight dimensional-coherence
    # invariants, over the full episode set. Same build-time-gate contract
    # as the two checks above -- they fire before tokens are spent.
    asyncio.run(_run_dimensional_coherence_invariants(
        _connect, episodes, topology_path=topology))

    results = asyncio.run(run_suite(
        _connect, topology_path=topology, deciders=build_deciders(args)))
    print(render_results_table(results))


if __name__ == "__main__":
    main()
