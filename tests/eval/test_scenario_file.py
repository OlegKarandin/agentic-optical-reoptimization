"""Scenario file contract (eval design spec, "Scenario file contract").
Fixture-driven: no server, no state file."""
import pytest

from storm_reoptimizer.eval.scenario_file import (
    ScenarioFileError, load_scenario,
)


def test_loads_every_field(write_scenario, example_scenario_yaml):
    s = load_scenario(write_scenario(example_scenario_yaml))
    assert s.id == "EXAMPLE_A"
    assert s.pair == "EXAMPLE"
    assert s.hours == ("t0", "t1", "t2", "t3")
    assert s.decision_hour == "t1"
    assert s.spares_on_hand == 1
    assert s.reference_avoid == {"risk_groups": ["rg_ref"]}
    assert s.gold.decision_at_t0 == "wait"
    assert s.gold.label == "wait"
    assert s.realized["t3"] == ("fiber_004", "fiber_005")
    assert s.metadata["cone_width_km"] == 90


def test_issuances_are_keyed_by_issue_hour_and_carry_horizons(
    write_scenario, example_scenario_yaml,
):
    s = load_scenario(write_scenario(example_scenario_yaml))
    assert sorted(s.forecast) == ["t0", "t1"]
    assert s.forecast["t0"].issued_at == "t0"
    assert s.forecast["t0"].horizons["t3"].width_km == 90
    assert s.forecast["t0"].horizons["t3"].center == {"lat": 25.0, "lon": 81.0}


def test_unknown_top_level_key_is_rejected(write_scenario, example_scenario_yaml):
    bad = example_scenario_yaml + "\nspares_on_hand_typo: 2\n"
    with pytest.raises(ScenarioFileError, match="unknown key"):
        load_scenario(write_scenario(bad))


def test_missing_required_key_is_rejected(write_scenario, example_scenario_yaml):
    bad = example_scenario_yaml.replace("decision_hour: t1\n", "")
    with pytest.raises(ScenarioFileError, match="decision_hour"):
        load_scenario(write_scenario(bad))


def test_issue_hour_not_in_hours_is_rejected(write_scenario, example_scenario_yaml):
    bad = example_scenario_yaml.replace(
        "  t1:\n    t3:", "  t9:\n    t3:")
    with pytest.raises(ScenarioFileError, match="t9"):
        load_scenario(write_scenario(bad))


def test_horizon_at_or_before_its_own_issue_hour_is_rejected(
    write_scenario, example_scenario_yaml,
):
    # An issuance can only forecast the future.
    bad = example_scenario_yaml.replace(
        "  t1:\n    t3:", "  t1:\n    t0:")
    with pytest.raises(ScenarioFileError, match="horizon"):
        load_scenario(write_scenario(bad))
