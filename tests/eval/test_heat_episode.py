# tests/eval/test_heat_episode.py
"""H1, live and free (heat spec §10): assertions green, gold re-enumerated,
every baseline separated from gold, cover_p95 included."""
import asyncio

import pytest

from storm_reoptimizer.eval.heat_assertions import (
    assert_bystanders_never_cross, assert_heat_episode_discriminates,
    assert_heat_gold_matches_oracle, assert_heat_mapping,
)
from storm_reoptimizer.eval.heat_baseline import HEAT_BASELINE_VARIANTS, HeatBaseline
from storm_reoptimizer.eval.heat_gold import oracle_trace
from storm_reoptimizer.eval.heat_margin import MarginCurve
from storm_reoptimizer.eval.runner import run_episode
from storm_reoptimizer.eval.scenario_file import load_all_scenarios
from storm_reoptimizer.geo_mapper import load_edges
from pathlib import Path

from storm_reoptimizer.mcp_client import call_tool_json

TOPOLOGY_PATH = (Path(__file__).parent.parent.parent / "src" / "storm_reoptimizer"
                 / "data" / "toy_india_topology.json")


@pytest.fixture(scope="module")
def h1():
    return load_all_scenarios(event_type="heat")["H1"]


def _run(connect_for, scenario, decider):
    async def go():
        async with connect_for(scenario.state_file)() as client:
            return await run_episode(client, scenario, decider, topology_path=TOPOLOGY_PATH)
    return asyncio.run(go())


def test_h1_mapping(connect_for, h1):
    async def go():
        async with connect_for(h1.state_file)() as client:
            return await call_tool_json(client, "get_topology", {"layer": "optical"})
    assert_heat_mapping(h1, asyncio.run(go()), load_edges(TOPOLOGY_PATH))


def test_h1_gold_and_discrimination(connect_for, h1):
    oracle = asyncio.run(oracle_trace(connect_for(h1.state_file), h1,
                                      topology_path=TOPOLOGY_PATH))
    assert_heat_gold_matches_oracle(h1, oracle)
    assert_bystanders_never_cross(MarginCurve.from_dict(
        oracle.hours[0]["observation"]["internal"]["margin_curve"]))
    baselines = {v: _run(connect_for, h1, HeatBaseline(v)) for v in HEAT_BASELINE_VARIANTS}
    assert_heat_episode_discriminates(h1, oracle, baselines)
