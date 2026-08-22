"""The shipped episodes must pass every pre-flight assertion before any
rollout is scored (eval design spec, "Twin-pair discipline"). A pair failing
these is cut, not shipped."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from storm_reoptimizer.eval.assertions import (
    assert_each_baseline_variant_ties, assert_gold_choices_differ,
    assert_issuance_prefix_shared, assert_menus_identical,
    assert_shared_scalars_equal,
)
from storm_reoptimizer.eval.scenario_file import load_all_scenarios
from storm_reoptimizer.mcp_client import connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)

# Extended by Tasks 15-17 to ("T1", "T2", "T3").
PAIRS = ("T1",)


@pytest.mark.parametrize("pair", PAIRS)
def test_pair_passes_the_static_assertions(pair):
    episodes = load_all_scenarios()
    a = episodes[f"{pair}a"]
    b = episodes[f"{pair}b"]
    assert_gold_choices_differ(a, b)
    assert_issuance_prefix_shared(a, b)
    assert_shared_scalars_equal(a, b)


async def _menus(a, b, state_path, server_command, server_env):
    @asynccontextmanager
    async def _connect():
        async with connect_server(
            TOPOLOGY_PATH, server_command=server_command, env=server_env,
            extra_args=["--state", str(state_path)],
        ) as client:
            yield client

    async with _connect() as client_a, _connect() as client_b:
        await assert_menus_identical(client_a, client_b, a, b)


@pytest.mark.parametrize("pair", PAIRS)
def test_pair_menus_are_identical_under_reference_avoid(
    pair, loaded_state_path, local_server_command, local_server_env,
):
    episodes = load_all_scenarios()
    asyncio.run(_menus(episodes[f"{pair}a"], episodes[f"{pair}b"],
                       loaded_state_path, local_server_command,
                       local_server_env))


@pytest.mark.parametrize("pair", PAIRS)
def test_each_baseline_variant_scores_exactly_one_half(
    pair, loaded_state_path, local_server_command, local_server_env,
):
    episodes = load_all_scenarios()
    a, b = episodes[f"{pair}a"], episodes[f"{pair}b"]

    async def _run():
        @asynccontextmanager
        async def _connect():
            async with connect_server(
                TOPOLOGY_PATH, server_command=local_server_command,
                env=local_server_env,
                extra_args=["--state", str(loaded_state_path)],
            ) as client:
                yield client

        # _connect is already a zero-arg async-context-manager factory --
        # matches assert_each_baseline_variant_ties's client_factory_a/
        # client_factory_b signature directly (fixed Task 11 this session:
        # reusing one connection across both baseline-variant replays
        # corrupted the second replay with state the first had mutated).
        await assert_each_baseline_variant_ties(
            _connect, _connect, a, b, topology_path=TOPOLOGY_PATH)

    asyncio.run(_run())
