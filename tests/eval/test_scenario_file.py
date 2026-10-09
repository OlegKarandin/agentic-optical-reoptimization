"""Scenario file contract (eval design spec, "Scenario file contract").
Fixture-driven: no server, no state file."""
import dataclasses
from pathlib import Path

import pytest
import yaml

from storm_reoptimizer.eval.scenario_file import (
    SCENARIOS_DIR, HeatGold, ScenarioFileError, dump_scenario,
    expand_realized_spans, load_all_scenarios, load_scenario, span_mate,
)
from storm_reoptimizer.eval.thermal import HourForecast


# Task 1 (2026-09-27 plan): realized cuts denote whole SPANS -- the storm
# damage model (cone.py) always cuts both directions of a physical span, so
# a `realized` block naming only one direction is authoring a cut the
# forecast gives probability 0 of happening alone (spec §3.4, §4.1).
def test_a_fibre_expands_to_its_reverse_direction_mate():
    assert span_mate("fiber_a_b_0") == "fiber_b_a_0"
    assert expand_realized_spans(["fiber_a_b_0"]) == ("fiber_a_b_0", "fiber_b_a_0")


def test_expansion_is_idempotent_and_deduplicated():
    assert expand_realized_spans(["fiber_a_b_1", "fiber_b_a_1"]) == \
        ("fiber_a_b_1", "fiber_b_a_1")


def test_an_underscored_node_name_is_split_on_the_topology():
    # kot_kapura is a real toy-topology node; its actual neighbours are
    # amritsar and talwandi_bahi (bathinda is not a node in this topology at
    # all) -- confirmed directly against toy_india_topology.json. Only the
    # kot_kapura/talwandi_bahi split names two real nodes.
    mate = span_mate("fiber_kot_kapura_talwandi_bahi_0")
    assert mate == "fiber_talwandi_bahi_kot_kapura_0"


def test_a_non_span_id_is_an_authoring_error():
    with pytest.raises(ScenarioFileError):
        span_mate("fiber_004")


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
    assert s.realized["t3"] == ("fiber_rewa_satna_0", "fiber_satna_rewa_0")
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


def test_horizons_within_one_issuance_must_be_chronologically_ordered(
    write_scenario, example_scenario_yaml,
):
    """t0's own issuance publishes t1 THEN t3 (ascending `hours` order) --
    swapping the two lines makes it publish t3 before t1, which must be
    rejected at load time. `oracle.latest_horizon` (and every caller of it:
    `escape_objective`, `spend_decider`, `assertions.
    assert_both_legs_exposed`/`assert_spend_is_real`) picks "the latest
    horizon" as `next(reversed(issuance.horizons))` -- the LAST dict key --
    which only means anything if `load_scenario` guarantees ascending order
    here, once, rather than leaving it a silent YAML-authoring convention."""
    t1_line = ("    t1: {cone: {type: Polygon, coordinates: "
               "[[[81.0, 25.0], [81.1, 25.0], [81.1, 25.1], [81.0, 25.0]]]}, "
               "width_km: 190, center: {lat: 25.0, lon: 81.0}}\n")
    t3_line = ("    t3: {cone: {type: Polygon, coordinates: "
               "[[[81.0, 25.0], [81.1, 25.0], [81.1, 25.1], [81.0, 25.0]]]}, "
               "width_km: 90, center: {lat: 25.0, lon: 81.0}}\n")
    assert t1_line in example_scenario_yaml
    assert t3_line in example_scenario_yaml
    bad = example_scenario_yaml.replace(t1_line + t3_line, t3_line + t1_line)
    with pytest.raises(ScenarioFileError, match="chronological order"):
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


# T1 spend-or-hold redesign (Task 8): gold.outcome_gbps_h/min_margin_gbps_h
# are the oracle enumerator's own OUTCOME figures for the two candidate
# decisions -- optional, since only spare_action_by_deadline episodes
# populate them, and every other episode's gold stays exactly as strict as
# it always was.
def test_gold_outcome_gbps_h_and_min_margin_load_when_present(
        write_scenario, example_scenario_yaml):
    # NOT 6-space indent: after textwrap.dedent, conftest's `gold:` sub-keys
    # (survived, rationale, ...) sit at 2-space indent, with the `rationale:
    # |` block's own content one level deeper at 4. A line inserted at 4
    # spaces or deeper, immediately after that block, is read as MORE of the
    # literal block rather than a new gold key -- exactly the bug this
    # comment is here to stop a future edit from reintroducing.
    ok = example_scenario_yaml.replace(
        "\nflip_variable:",
        "\n  outcome_gbps_h: {spend: 1, hold: 2}"
        "\n  min_margin_gbps_h: 0.5"
        "\nflip_variable:")
    s = load_scenario(write_scenario(ok))
    assert s.gold.outcome_gbps_h == {"spend": 1.0, "hold": 2.0}
    assert s.gold.min_margin_gbps_h == 0.5


def test_gold_outcome_gbps_h_defaults_to_none_when_absent(
        write_scenario, example_scenario_yaml):
    s = load_scenario(write_scenario(example_scenario_yaml))
    assert s.gold.outcome_gbps_h is None
    assert s.gold.min_margin_gbps_h is None


# Task 10 (T1 spend-or-hold redesign): dump_scenario is the reverse of
# load_scenario -- tools/compute_gold.py needs it to rewrite a scenario's
# gold: block without a human hand-editing YAML. Round-tripped over EVERY
# shipped scenario, not just one fixture, so a field this writer forgets
# shows up as a real mismatch on a real file rather than a fixture that
# happens not to exercise it.
def test_dump_load_round_trip(tmp_path):
    scenarios = load_all_scenarios()
    assert scenarios, "expected at least one shipped scenario to round-trip"
    for scenario_id, scenario in scenarios.items():
        path = tmp_path / f"{scenario_id}.yaml"
        path.write_text(dump_scenario(scenario), encoding="utf-8")
        assert load_scenario(path) == scenario


def test_dump_scenario_omits_pair_for_a_singleton_episode(write_scenario,
                                                           example_scenario_yaml):
    # D1 (a singleton, no twin) omits `pair` entirely rather than writing
    # `pair: null` -- dump_scenario must reproduce that, not merely round
    # trip through a value load_scenario happens to tolerate either way.
    without_pair = example_scenario_yaml.replace("pair: EXAMPLE\n", "")
    scenario = load_scenario(write_scenario(without_pair))
    assert scenario.pair is None
    dumped = dump_scenario(scenario)
    assert "pair:" not in dumped


def test_the_track_revision_scale_is_required_and_parsed(
        example_scenario_yaml, write_scenario):
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    assert scenario.track_revision_km_per_hour_ahead == 30.0


def test_a_scenario_without_the_track_revision_scale_is_rejected(
        example_scenario_yaml, write_scenario):
    text = "\n".join(
        line for line in example_scenario_yaml.splitlines()
        if "track_revision_km_per_hour_ahead" not in line)
    with pytest.raises(ScenarioFileError,
                       match="track_revision_km_per_hour_ahead"):
        load_scenario(write_scenario(text))


def test_the_scenario_round_trips_through_dump(
        example_scenario_yaml, write_scenario):
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    again = load_scenario(write_scenario(dump_scenario(scenario), "RT.yaml"))
    assert again == scenario


FROZEN = Path(__file__).parent / "fixtures" / "storm_dumps"


@pytest.mark.parametrize("name", sorted(p.name for p in FROZEN.glob("*.yaml")))
def test_every_storm_yaml_dumps_byte_identically(name):
    assert dump_scenario(load_scenario(SCENARIOS_DIR / name)) == (
        FROZEN / name).read_text(encoding="utf-8")


def test_storm_files_load_as_storm():
    assert {s.event_type for s in load_all_scenarios().values()} == {"storm"}


def test_heat_loads(write_scenario, heat_example_yaml):
    s = load_scenario(write_scenario(heat_example_yaml, name="HEAT_EXAMPLE.yaml"))
    assert s.event_type == "heat" and s.decision_hour is None and s.gold is None
    assert s.thermal.amplifiers == ("amp_a_b_0", "amp_b_a_0")
    assert s.thermal.heat_class.setpoint_c == 25.0
    assert s.temperature_forecast["t0"]["t1"] == HourForecast(36.0, 2.0)
    assert s.temperature_realized == {"t0": 30.5, "t1": 37.0, "t2": 32.0}
    assert s.cooling_failed_at == "t1"


@pytest.mark.parametrize("key,value", [
    ("track", "amphan"), ("decision_hour", "t0"), ("lead_time_hours", 2),
    ("spares_on_hand", 1), ("depot_site", "x"), ("spare_inventory", {"x": 1}),
    ("damage_radius_km", 10.0), ("track_revision_km_per_hour_ahead", 30),
    ("reference_avoid", {}), ("forecast", {}), ("realized", {})])
def test_heat_rejects_every_storm_key(write_scenario, heat_example_yaml, key, value):
    raw = yaml.safe_load(heat_example_yaml)
    raw[key] = value
    with pytest.raises(ScenarioFileError, match="storm-only"):
        load_scenario(write_scenario(yaml.safe_dump(raw), name="H.yaml"))


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r.pop("thermal"), "missing"),
    (lambda r: r.pop("cooling_failed_at"), "missing"),
    (lambda r: r["temperature_realized"].pop("t2"), "temperature_realized"),
    (lambda r: r.update(cooling_failed_at="t9"), "cooling_failed_at"),
    (lambda r: r["temperature_forecast"].update(t1={"t1": {"t_mean_c": 1, "t_sigma_c": 1}}), "heat_polygon"),
    (lambda r: r["temperature_forecast"]["t0"].pop("t1"), "no forecast"),
    (lambda r: r["metadata"].update(label_rule="spare_action_by_deadline"), "mode_by_hour"),
    (lambda r: r["thermal"].update(amplifiers=[]), "amplifiers"),
])
def test_heat_rejects_malformed(write_scenario, heat_example_yaml, mutate, match):
    raw = yaml.safe_load(heat_example_yaml)
    mutate(raw)
    with pytest.raises(ScenarioFileError, match=match):
        load_scenario(write_scenario(yaml.safe_dump(raw), name="H.yaml"))


def test_heat_dump_round_trips_with_gold(write_scenario, heat_example_yaml, tmp_path):
    s = load_scenario(write_scenario(heat_example_yaml, name="HEAT_EXAMPLE.yaml"))
    s = dataclasses.replace(s, gold=HeatGold(
        mode_by_hour={"t0": "800G@15.1dB", "t1": "700G@13.3dB", "t2": "750G@14.1dB"},
        graded_hours=("t1",), outcome_gbps_h=150.0, rationale="table\n"))
    path = tmp_path / "rt.yaml"
    path.write_text(dump_scenario(s), encoding="utf-8")
    assert load_scenario(path) == s


def test_load_all_scenarios_filters_by_event_type(write_scenario, heat_example_yaml, tmp_path):
    write_scenario(heat_example_yaml, name="HEAT_EXAMPLE.yaml")
    root = next(tmp_path.rglob("HEAT_EXAMPLE.yaml")).parent
    assert set(load_all_scenarios(root)) == set()
    assert set(load_all_scenarios(root, event_type="heat")) == {"HEAT_EXAMPLE"}
    assert set(load_all_scenarios(root, event_type=None)) == {"HEAT_EXAMPLE"}
