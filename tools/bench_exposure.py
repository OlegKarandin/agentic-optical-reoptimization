# tools/bench_exposure.py
"""Throughput benchmark for cone.py's pre-filters (exposure-and-depot plan,
Task 4). build_observation loops over every service in the roster, once per
horizon of the current issuance; most services either have no storm-cuttable
span at all (observation.py's own `if not spans: continue`) or sit far enough
from the storm that cone.py's NEGLIGIBLE_SIGMAS distance pre-filter answers
0.0 without ever running the Sobol sampler. This script measures how much of
the roster still reaches the sampler, and how fast build_observation runs
against the real 573-service state, for one hour of scenario T3a.

NOT a test: it prints, it does not assert. Run it after any change to
cone.py's pre-filters or observation.py's exposure loop, and paste the
printed numbers into the commit message.

Needs a live multilayer-optical-mcp server and the eval harness's built
state (eval/states/loaded-s17.json, built once by tools/build_eval_state.py
-- see tests/conftest.py's loaded_state_path fixture). This script is not a
pytest fixture consumer, so it replicates conftest.py's local_server_command/
local_server_env workaround directly (this workspace's console script is
never actually installed -- a Cyrillic path trips a pip install -e bug, see
conftest.py's docstring); a real install needs none of this --
mcp_client.DEFAULT_SERVER_COMMAND is what that uses.

Run: C:/Users/olegk/miniconda3/envs/storm-reoptimizer/python.exe tools/bench_exposure.py
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from storm_reoptimizer.eval import cone as cone_module
from storm_reoptimizer.eval import observation as observation_module
from storm_reoptimizer.eval.observation import build_observation
from storm_reoptimizer.eval.runner import service_geometry
from storm_reoptimizer.eval.scenario_file import load_all_scenarios
from storm_reoptimizer.mcp_client import call_tool_json, connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)
STATE_PATH = (
    Path(__file__).parent.parent / "eval" / "states" / "loaded-s17.json"
)

# Mirrors tests/conftest.py's local_server_command/local_server_env fixtures.
_MULTILAYER_OPTICAL_MCP_ENV_PYTHON = Path(
    r"C:\Users\olegk\miniconda3\envs\multilayer-optical-mcp\python.exe"
)


def _local_server_command() -> list[str]:
    if not _MULTILAYER_OPTICAL_MCP_ENV_PYTHON.exists():
        raise SystemExit(
            f"multilayer-optical-mcp conda env python not found at "
            f"{_MULTILAYER_OPTICAL_MCP_ENV_PYTHON}; adjust this script's "
            f"hardcoded path for your machine (see tests/conftest.py's "
            f"local_server_command fixture)."
        )
    return [
        str(_MULTILAYER_OPTICAL_MCP_ENV_PYTHON),
        "-c",
        "import sys; sys.argv = ['multilayer-optical-mcp', *sys.argv[1:]]; "
        "from multilayer_optical_mcp.server import main; main()",
    ]


def _local_server_env() -> dict[str, str]:
    return dict(os.environ)


async def _fetch_geometry_and_services():
    """The two server reads build_observation needs, off the real loaded-s17
    state -- 573 services against the 143-node toy topology."""
    async with connect_server(
        TOPOLOGY_PATH, server_command=_local_server_command(),
        env=_local_server_env(),
        extra_args=["--state", str(STATE_PATH)],
    ) as client:
        geometry = await service_geometry(client, TOPOLOGY_PATH)
        services = tuple(
            (await call_tool_json(client, "get_services"))["services"])
    return geometry, services


class _Counters:
    """Call counts, keyed back to service ids by object identity: within one
    build_observation call, `service_spans.get(svc['id'])` returns the SAME
    tuple object for every horizon of that service, so `id(spans)` is a
    stable key for the duration of one benchmark run."""

    def __init__(self, spans_by_service: dict[str, tuple]):
        self.spans_id_to_service = {
            id(spans): svc_id for svc_id, spans in spans_by_service.items()
            if spans
        }
        self.total_calls = 0
        self.sampler_calls = 0
        self.services_seen: set[str] = set()
        self.services_reaching_sampler: set[str] = set()

    def note_call(self, spans) -> None:
        self.total_calls += 1
        svc_id = self.spans_id_to_service.get(id(spans))
        if svc_id is not None:
            self.services_seen.add(svc_id)

    def note_sampler(self, spans) -> None:
        self.sampler_calls += 1
        svc_id = self.spans_id_to_service.get(id(spans))
        if svc_id is not None:
            self.services_reaching_sampler.add(svc_id)


def _instrumented(counters: _Counters):
    """Patch cone.py's two entry points so every call build_observation makes
    is counted, without changing cone.py or observation.py themselves. The
    real p_cut_region/`_p_cut_region_unfiltered` are called through
    unchanged -- this only observes."""
    real_p_cut_region = cone_module.p_cut_region
    real_unfiltered = cone_module._p_cut_region_unfiltered

    def counting_p_cut_region(spans, *args, **kwargs):
        counters.note_call(spans)
        return real_p_cut_region(spans, *args, **kwargs)

    def counting_unfiltered(spans, *args, **kwargs):
        counters.note_sampler(spans)
        return real_unfiltered(spans, *args, **kwargs)

    return counting_p_cut_region, counting_unfiltered, real_p_cut_region, real_unfiltered


def main() -> None:
    scenario = load_all_scenarios()["T3a"]
    hour = scenario.decision_hour
    print(f"Fetching service geometry + roster off a live server "
          f"(state={STATE_PATH.name}) ...")
    t_fetch0 = time.perf_counter()
    geometry, services = asyncio.run(_fetch_geometry_and_services())
    t_fetch1 = time.perf_counter()
    print(f"  {len(services)} services, "
          f"{t_fetch1 - t_fetch0:.2f}s to fetch (server round trips, not "
          f"part of the exposure timing below)")

    with_spans = sum(1 for s in geometry.cuttable_spans.values() if s)
    without_spans = len(services) - with_spans
    print(f"\n{with_spans} of {len(services)} services have at least one "
          f"storm-cuttable span; {without_spans} are filtered out before "
          f"ever calling p_cut_region (observation.py's own "
          f"`if not spans: continue`).")

    # observation.py's module-level `p_cut_region` name is bound to the real
    # function object at import time, so the wrapper has to replace THAT
    # name (not cone_module.p_cut_region) to see every call build_observation
    # makes; the wrapper still calls through cone_module's real p_cut_region,
    # which resolves `_p_cut_region_unfiltered` as a global in cone's own
    # namespace -- patching cone_module._p_cut_region_unfiltered there is
    # enough to see whether the distance pre-filter let a call through.
    counters = _Counters(geometry.cuttable_spans)
    (counting_p_cut_region, counting_unfiltered,
     real_p_cut_region, real_unfiltered) = _instrumented(counters)
    observation_module.p_cut_region = counting_p_cut_region
    cone_module._p_cut_region_unfiltered = counting_unfiltered
    try:
        obs = build_observation(
            scenario, hour, service_spans=geometry.cuttable_spans,
            services=services, spares_on_hand=scenario.spares_on_hand)
    finally:
        observation_module.p_cut_region = real_p_cut_region
        cone_module._p_cut_region_unfiltered = real_unfiltered

    horizons_at_hour = len(obs.issuance.horizons)
    print(f"\n=== hour {hour} of {scenario.id} "
          f"({horizons_at_hour} horizon(s) in its issuance) ===")
    print(f"  (service, horizon) pairs evaluated: {counters.total_calls}")
    print(f"  of those, reached the Sobol sampler: {counters.sampler_calls} "
          f"({counters.sampler_calls / max(counters.total_calls, 1):.1%})")
    print(f"  short-circuited by the distance pre-filter: "
          f"{counters.total_calls - counters.sampler_calls}")
    print(f"  distinct services with a cuttable span, evaluated this hour: "
          f"{len(counters.services_seen)}")
    print(f"  distinct services that reached the sampler at least once: "
          f"{len(counters.services_reaching_sampler)}")
    print(f"  distinct services short-circuited for every horizon this "
          f"hour: "
          f"{len(counters.services_seen) - len(counters.services_reaching_sampler)}")

    # Timing: build_observation itself is pure Python/CPU (no server calls),
    # so time it in isolation, repeated for a stable reading, WITH the
    # pre-filters live (i.e. cone.py's real, unpatched p_cut_region).
    repeats = 5
    t0 = time.perf_counter()
    for _ in range(repeats):
        build_observation(
            scenario, hour, service_spans=geometry.cuttable_spans,
            services=services, spares_on_hand=scenario.spares_on_hand)
    t1 = time.perf_counter()
    per_call_s = (t1 - t0) / repeats
    print(f"\nbuild_observation({hour}), pre-filters ON: "
          f"{per_call_s * 1000:.2f} ms/call, averaged over {repeats} calls")

    # One full pass over every hour of the episode (no per-hour iteration
    # loop -- run_episode calls build_observation more than once on an
    # "act" hour, so this is a lower bound on a real episode's exposure
    # cost, not an exact replay).
    t0 = time.perf_counter()
    for h in scenario.hours:
        build_observation(
            scenario, h, service_spans=geometry.cuttable_spans,
            services=services, spares_on_hand=scenario.spares_on_hand)
    t1 = time.perf_counter()
    per_episode_s = t1 - t0
    print(f"one pass over all {len(scenario.hours)} hours of {scenario.id} "
          f"({', '.join(scenario.hours)}), pre-filters ON: "
          f"{per_episode_s:.3f}s total")
    if per_episode_s > 30.0:
        print("  WARNING: exceeds the ~30s/episode budget even with the "
              "pre-filters -- see plan Task 4's exact-quadrature fallback "
              "(design spec §2.4).")
    else:
        print("  well under the ~30s/episode budget.")

    # For contrast: the same one-hour timing with BOTH pre-filters bypassed
    # (every call forced through the Sobol sampler), to show what the
    # pre-filter actually buys.
    def always_sample(spans, *args, **kwargs):
        if not spans:
            return 0.0
        return real_unfiltered(spans, *args, **kwargs)

    observation_module.p_cut_region = always_sample
    try:
        t0 = time.perf_counter()
        build_observation(
            scenario, hour, service_spans=geometry.cuttable_spans,
            services=services, spares_on_hand=scenario.spares_on_hand)
        t1 = time.perf_counter()
    finally:
        observation_module.p_cut_region = real_p_cut_region
    print(f"\nfor contrast, same hour with the distance pre-filter forced "
          f"OFF (every call reaches the sampler): {(t1 - t0) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
