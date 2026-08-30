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


def test_the_depot_keys_are_required(write_scenario, example_scenario_yaml):
    # scenario_file.py's docstring is explicit: a misspelled key is a silently
    # different episode. `spare_inventory_` typed for `spare_inventory` would
    # run green and quietly restore a global depot.
    without = example_scenario_yaml.replace("depot_site: satna\n", "")
    with pytest.raises(ScenarioFileError, match="depot_site"):
        load_scenario(write_scenario(without))


def test_the_depot_site_must_appear_in_the_inventory(write_scenario,
                                                     example_scenario_yaml):
    bad = example_scenario_yaml.replace("depot_site: satna", "depot_site: rewa")
    with pytest.raises(ScenarioFileError, match="depot_site 'rewa'"):
        load_scenario(write_scenario(bad))


def test_the_declared_scalar_matches_the_depot_sites_inventory(
        write_scenario, example_scenario_yaml):
    # metadata.spares_on_hand survives as the declared scalar FOR THE DEPOT
    # SITE, which is what lets rules.OBSERVABLE_VARS and
    # assertions.SHARED_SCALARS stay unchanged. If the two ever disagree, the
    # one-variable check is enumerating a number the harness does not use.
    bad = example_scenario_yaml.replace("spare_inventory: {satna: 1}",
                                        "spare_inventory: {satna: 3}")
    with pytest.raises(ScenarioFileError, match="spares_on_hand"):
        load_scenario(write_scenario(bad))


# Task 12 (exposure-and-depot plan): metadata.claimant_services, a strict
# required key (Task 9's own depot_site/spare_inventory treatment above),
# and metadata.claimed_competing_ecar_gbps/_at -- required TOGETHER
# whenever claimant_services is non-empty (2026-08-30 review fix, finding
# 2: an unconditionally-optional claim value left invariant 4 permanently
# dead, since every episode with real claimants already has a real claimed
# figure sitting in its own gold.rationale prose today).
def test_claimant_services_is_required(write_scenario, example_scenario_yaml):
    without = example_scenario_yaml.replace("  claimant_services: []\n", "")
    with pytest.raises(ScenarioFileError, match="claimant_services"):
        load_scenario(write_scenario(without))


def test_an_empty_claimant_list_needs_no_claim_value(
        write_scenario, example_scenario_yaml):
    # example_scenario_yaml already declares claimant_services: [] and no
    # claimed_competing_ecar_gbps/_at -- test_loads_every_field above
    # already proves this loads; this test pins the SPECIFIC behaviour
    # (empty list -> no claim keys required) as its own regression.
    s = load_scenario(write_scenario(example_scenario_yaml))
    assert s.metadata["claimant_services"] == []
    assert "claimed_competing_ecar_gbps" not in s.metadata


def test_claimed_competing_ecar_gbps_is_required_once_claimants_are_named(
        write_scenario, example_scenario_yaml):
    bad = example_scenario_yaml.replace(
        "claimant_services: []", "claimant_services: [d0001]")
    with pytest.raises(ScenarioFileError, match="claimed_competing_ecar_gbps"):
        load_scenario(write_scenario(bad))


def test_claimed_competing_ecar_at_is_required_once_claimants_are_named(
        write_scenario, example_scenario_yaml):
    bad = example_scenario_yaml.replace(
        "claimant_services: []",
        "claimant_services: [d0001]\n  claimed_competing_ecar_gbps: 10.0")
    with pytest.raises(ScenarioFileError, match="claimed_competing_ecar_at"):
        load_scenario(write_scenario(bad))


def test_claimed_competing_ecar_at_must_name_a_real_hour(
        write_scenario, example_scenario_yaml):
    bad = example_scenario_yaml.replace(
        "claimant_services: []",
        "claimant_services: [d0001]\n  claimed_competing_ecar_gbps: 10.0"
        "\n  claimed_competing_ecar_at: t9")
    with pytest.raises(ScenarioFileError, match="claimed_competing_ecar_at"):
        load_scenario(write_scenario(bad))


def test_a_valid_non_empty_claimant_list_with_both_claim_keys_loads(
        write_scenario, example_scenario_yaml):
    ok = example_scenario_yaml.replace(
        "claimant_services: []",
        "claimant_services: [d0001]\n  claimed_competing_ecar_gbps: 10.0"
        "\n  claimed_competing_ecar_at: t3")
    s = load_scenario(write_scenario(ok))
    assert s.metadata["claimed_competing_ecar_gbps"] == 10.0
    assert s.metadata["claimed_competing_ecar_at"] == "t3"
