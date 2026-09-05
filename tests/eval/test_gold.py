"""The gold enumerator (T1 spend-or-hold redesign, Task 10): turning real
oracle rollouts into a measured `Gold`, rather than trusting an author's
guess. `gold_from_outcomes` is pure -- exercised here against a fabricated
outcomes dict, no server. `enumerate_outcomes` is live: it runs
`oracle.spend_decider`/`hold_decider` through the REAL harness
(`runner.run_episode`), the same way `test_episodes.py`'s own scenarios are
run, so it is exercised against a real, already-shipped, already-verified
scenario (T1a) rather than a hand-built fixture that might not survive a
live server round trip."""
import asyncio
import dataclasses
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from storm_reoptimizer.eval.gold import enumerate_outcomes, gold_from_outcomes
from storm_reoptimizer.eval.scenario_file import load_all_scenarios, load_scenario
from storm_reoptimizer.mcp_client import connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)


# -- gold_from_outcomes (pure) --------------------------------------------

def test_gold_from_outcomes_picks_lower_loss_and_names_claimants(
        write_scenario, example_scenario_yaml):
    ok = example_scenario_yaml.replace(
        "claimant_services: []",
        "claimant_services: [claim-a, claim-b]\n"
        "  claimed_competing_ecar_gbps: 50.0\n"
        "  claimed_competing_ecar_at: t3")
    scenario = load_scenario(write_scenario(ok))

    outcomes = {
        "spend": {"gbps_hours_lost": {"claim-a": 100.0, "claim-b": 100.0},
                  "total": 200.0, "label": "spend"},
        "hold": {"gbps_hours_lost": {"storm-svc-1": 30.0},
                 "total": 30.0, "label": "hold"},
    }
    gold = gold_from_outcomes(scenario, outcomes)

    assert gold.label == "hold"
    assert gold.outcome_gbps_h == {"spend": 200.0, "hold": 30.0}
    assert gold.min_margin_gbps_h == pytest.approx(0.25 * 30.0)
    # storm-svc-1 (the SUT) lost 30G under the gold (hold) choice, so it did
    # NOT survive; the two claimants have no entry in hold's own losses, so
    # they read zero loss and DID survive.
    assert gold.survived == ("claim-a", "claim-b")
    assert gold.max_spares_wasted == 0
    assert gold.decision_at_t0 == "wait"
    # assertions.claimant_service_ids requires every metadata.claimant_services
    # id to appear literally in the rationale text.
    assert "claim-a" in gold.rationale
    assert "claim-b" in gold.rationale


def test_gold_from_outcomes_margin_floor_is_at_least_one_gbps_hour(
        write_scenario, example_scenario_yaml):
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    outcomes = {
        "spend": {"gbps_hours_lost": {}, "total": 0.1, "label": "spend"},
        "hold": {"gbps_hours_lost": {}, "total": 2.0, "label": "hold"},
    }
    gold = gold_from_outcomes(scenario, outcomes)
    # 0.25 * 0.1 = 0.025, well under the 1.0 floor.
    assert gold.min_margin_gbps_h == 1.0


# -- enumerate_outcomes (live) ---------------------------------------------

def test_enumerate_outcomes_runs_both_rollouts_end_to_end(
        loaded_state_path, local_server_command, local_server_env):
    """Runs T1a (a real, already-shipped, already-verified scenario) twice
    through the real harness -- once under spend_decider, once under
    hold_decider -- with `metadata.label_rule` overridden to
    `spare_action_by_deadline` so `decision_label`/`gbps_hours_lost` (which
    both require it) apply. Not a claim that T1a's OWN gold answer is
    correct under this rule -- T1a's `gold.label` here is still the
    unrelated `timing_at_decision_hour` answer; a later task rebuilds T1
    entirely on this rule. This only checks that enumerate_outcomes runs
    both rollouts end to end and returns the documented shape."""
    episodes = load_all_scenarios()
    t1a = episodes["T1a"]
    scenario = dataclasses.replace(
        t1a, metadata={**t1a.metadata,
                      "label_rule": "spare_action_by_deadline"})

    @asynccontextmanager
    async def _connect():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            yield client

    outcomes = asyncio.run(
        enumerate_outcomes(_connect, scenario, topology_path=TOPOLOGY_PATH))

    assert set(outcomes) == {"spend", "hold"}
    for data in outcomes.values():
        assert isinstance(data["gbps_hours_lost"], dict)
        assert isinstance(data["total"], float)
        assert data["total"] >= 0.0
    # hold_decider never spends the depot's pair on the SUT, so the
    # deadline rule always reads "hold" for it, regardless of what the
    # storm actually did this rollout.
    assert outcomes["hold"]["label"] == "hold"

    gold = gold_from_outcomes(scenario, outcomes)
    assert gold.label in outcomes
    assert gold.outcome_gbps_h == {c: d["total"] for c, d in outcomes.items()}
    for claimant in scenario.metadata["claimant_services"]:
        assert claimant in gold.rationale
