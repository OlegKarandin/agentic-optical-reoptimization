# tests/eval/test_heat_assertions.py
"""Pure tests for heat_assertions.py (Task 10, step 1): one passing and one
failing case per assertion. No server -- fake `optical`/edges and
hand-built traces/curves only."""
from __future__ import annotations

import dataclasses

import pytest
from shapely.geometry import LineString as _LS

from storm_reoptimizer.eval.heat_assertions import (
    HeatEpisodeInvalid, assert_bystanders_never_cross,
    assert_heat_episode_discriminates, assert_heat_gold_matches_oracle,
    assert_heat_mapping,
)
from storm_reoptimizer.eval.heat_margin import MarginCurve
from storm_reoptimizer.eval.runner import EpisodeTrace
from storm_reoptimizer.eval.scenario_file import HeatGold, load_scenario
from storm_reoptimizer.eval.thermal import NfMixture
from storm_reoptimizer.geo_mapper import Edge

# ---------------------------------------------------------------------------
# assert_heat_mapping -- Task 3's synthetic oms/edge pattern
# (tests/test_geo_mapper.py), reused here rather than imported (that module
# carries no __init__.py / package exports of its own helpers).

EDGE = Edge(src="a", dst="b", mount_type="aerial", geometry=_LS([(0, 0), (2, 0)]))


def _oms(a, b, lengths):
    elements = [f"roadm_{a}", f"amp_{a}_{b}_booster"]
    for i, _ in enumerate(lengths):
        elements += [f"fiber_{a}_{b}_{i}", f"amp_{a}_{b}_{i}"]
    return ({"id": f"oms_{a}_{b}", "src_node_id": a, "dst_node_id": b,
             "elements": elements},
            {f"fiber_{a}_{b}_{i}": length for i, length in enumerate(lengths)})


def _box(x0, x1, y0=-0.1, y1=0.1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1],
                                                [x0, y1], [x0, y0]]]}


def _mapping_scenario(write_scenario, heat_example_yaml, amps):
    s = load_scenario(write_scenario(heat_example_yaml, name="HEAT_MAP.yaml"))
    return dataclasses.replace(
        s, thermal=dataclasses.replace(s.thermal, amplifiers=tuple(amps)),
        heat_polygon={"t0": _box(0.9, 1.1)})


def test_heat_mapping_matches_when_amplifiers_agree(write_scenario, heat_example_yaml):
    fwd, lengths = _oms("a", "b", [60.0, 60.0])
    rev, lr = _oms("b", "a", [60.0, 60.0])
    optical = {"oms": [fwd, rev], "fibers": [
        {"id": k, "length_km": v} for k, v in {**lengths, **lr}.items()]}
    scenario = _mapping_scenario(write_scenario, heat_example_yaml,
                                 ["amp_a_b_0", "amp_b_a_0"])
    assert_heat_mapping(scenario, optical, [EDGE])   # no raise


def test_heat_mapping_raises_when_declared_amplifiers_disagree(
        write_scenario, heat_example_yaml):
    fwd, lengths = _oms("a", "b", [60.0, 60.0])
    rev, lr = _oms("b", "a", [60.0, 60.0])
    optical = {"oms": [fwd, rev], "fibers": [
        {"id": k, "length_km": v} for k, v in {**lengths, **lr}.items()]}
    scenario = _mapping_scenario(write_scenario, heat_example_yaml, ["amp_a_b_0"])
    with pytest.raises(HeatEpisodeInvalid, match="polygon maps to"):
        assert_heat_mapping(scenario, optical, [EDGE])


# ---------------------------------------------------------------------------
# assert_bystanders_never_cross

def test_bystanders_never_cross_passes_when_every_column_stays_positive():
    curve = MarginCurve((0.0, 1.0, 2.0), (0.4, -0.6, -1.6),
                        {"lp-ok": (2.0, 1.5, 1.0)})
    assert_bystanders_never_cross(curve)   # no raise


def test_bystanders_never_cross_raises_when_a_column_dips_below_zero():
    curve = MarginCurve((0.0, 1.0, 2.0), (0.4, -0.6, -1.6),
                        {"lp-ok": (2.0, 1.5, 1.0), "lp-bad": (0.5, -0.1, -0.8)})
    with pytest.raises(HeatEpisodeInvalid, match="lp-bad"):
        assert_bystanders_never_cross(curve)


# ---------------------------------------------------------------------------
# assert_heat_gold_matches_oracle -- the LADDER and CURVE match
# tests/eval/test_heat_gold.py's own fixtures (Task 9's _record helper
# style).

LADDER = [{"mode_id": "800", "line_rate_gbps": 800.0, "threshold_db": 15.1},
          {"mode_id": "750", "line_rate_gbps": 750.0, "threshold_db": 14.1},
          {"mode_id": "700", "line_rate_gbps": 700.0, "threshold_db": 13.3}]
CURVE = MarginCurve((0.0, 1.0, 2.0), (0.4, -0.6, -1.6))


def _record(hour, p_fail, mode, loss):
    return {"hour": hour, "mode_in_force": mode, "loss_gbps": loss,
            "observation": {"mode_ladder": LADDER, "link": {"offered_gbps": 800.0},
                            "internal": {"margin_curve": CURVE.to_dict(), "mixtures": {
                                hour: dataclasses.asdict(NfMixture(p_fail, 0.6, 0.0, 1.5, 0.0))}}}}


def _gold_scenario(write_scenario, heat_example_yaml, gold):
    s = load_scenario(write_scenario(heat_example_yaml, name="HEAT_GOLD.yaml"))
    return dataclasses.replace(s, gold=gold)


def test_heat_gold_matches_a_fresh_oracle_rollout(write_scenario, heat_example_yaml):
    trace = EpisodeTrace("HEAT_EXAMPLE", "oracle:mode", 0, (
        _record("t0", 0.02, "750", 50.0), _record("t1", 0.102, "700", 100.0),
        _record("t2", 0.0667, "700", 50.0)), (), (), 0, "converged", {}, {}, 1, 1.0)
    gold = HeatGold(mode_by_hour={"t0": "750", "t1": "700", "t2": "700"},
                    graded_hours=("t0", "t1"), outcome_gbps_h=200.0, rationale="r")
    scenario = _gold_scenario(write_scenario, heat_example_yaml, gold)
    assert_heat_gold_matches_oracle(scenario, trace)   # no raise


def test_heat_gold_raises_when_the_shipped_gold_is_stale(write_scenario, heat_example_yaml):
    trace = EpisodeTrace("HEAT_EXAMPLE", "oracle:mode", 0, (
        _record("t0", 0.02, "750", 50.0), _record("t1", 0.102, "700", 100.0),
        _record("t2", 0.0667, "700", 50.0)), (), (), 0, "converged", {}, {}, 1, 1.0)
    # The real outcome is 200.0 Gbps*h -- claim 999.0 instead.
    stale = HeatGold(mode_by_hour={"t0": "750", "t1": "700", "t2": "700"},
                     graded_hours=("t0", "t1"), outcome_gbps_h=999.0, rationale="r")
    scenario = _gold_scenario(write_scenario, heat_example_yaml, stale)
    with pytest.raises(HeatEpisodeInvalid, match="shipped gold"):
        assert_heat_gold_matches_oracle(scenario, trace)


def test_heat_gold_raises_when_no_gold_is_shipped(write_scenario, heat_example_yaml):
    trace = EpisodeTrace("HEAT_EXAMPLE", "oracle:mode", 0, (
        _record("t0", 0.02, "750", 50.0),), (), (), 0, "converged", {}, {}, 1, 1.0)
    scenario = _gold_scenario(write_scenario, heat_example_yaml, None)
    with pytest.raises(HeatEpisodeInvalid, match="no gold"):
        assert_heat_gold_matches_oracle(scenario, trace)


# ---------------------------------------------------------------------------
# assert_heat_episode_discriminates

def _disc_scenario(write_scenario, heat_example_yaml, *, gold, failed_at="t1"):
    s = load_scenario(write_scenario(heat_example_yaml, name="HEAT_DISC.yaml"))
    return dataclasses.replace(s, gold=gold, cooling_failed_at=failed_at,
                               hours=("t0", "t1", "t2"))


def _oracle_trace_for_ladder():
    return EpisodeTrace("HEAT_DISC", "oracle:mode", 0, (
        {"hour": "t0", "observation": {"mode_ladder": LADDER}},
        {"hour": "t1", "observation": {"mode_ladder": LADDER}},
        {"hour": "t2", "observation": {"mode_ladder": LADDER}},
    ), (), (), 0, "converged", {}, {}, 1, 1.0)


def _baseline_trace(modes: dict, losses: dict):
    return EpisodeTrace("HEAT_DISC", "baseline", 0, tuple(
        {"hour": h, "mode_in_force": modes[h], "loss_gbps": losses[h]} for h in modes
    ), (), (), 0, "converged", {}, {}, 1, 1.0)


def test_discriminates_passes_with_three_steps_and_a_losing_baseline(
        write_scenario, heat_example_yaml):
    gold = HeatGold(mode_by_hour={"t0": "800", "t1": "750", "t2": "700"},
                    graded_hours=("t0", "t1", "t2"), outcome_gbps_h=50.0, rationale="r")
    scenario = _disc_scenario(write_scenario, heat_example_yaml, gold=gold)
    oracle = _oracle_trace_for_ladder()
    baselines = {"never_downshift": _baseline_trace(
        {"t0": "800", "t1": "800", "t2": "800"}, {"t0": 0.0, "t1": 200.0, "t2": 200.0})}
    assert_heat_episode_discriminates(scenario, oracle, baselines)   # no raise


def test_discriminates_raises_when_a_baseline_matches_gold_at_every_graded_hour(
        write_scenario, heat_example_yaml):
    gold = HeatGold(mode_by_hour={"t0": "800", "t1": "750", "t2": "700"},
                    graded_hours=("t0", "t1", "t2"), outcome_gbps_h=50.0, rationale="r")
    scenario = _disc_scenario(write_scenario, heat_example_yaml, gold=gold)
    oracle = _oracle_trace_for_ladder()
    baselines = {"copycat": _baseline_trace(
        {"t0": "800", "t1": "750", "t2": "700"}, {"t0": 0.0, "t1": 0.0, "t2": 0.0})}
    with pytest.raises(HeatEpisodeInvalid, match="matches gold"):
        assert_heat_episode_discriminates(scenario, oracle, baselines)


def test_discriminates_raises_when_fewer_than_three_steps(write_scenario, heat_example_yaml):
    gold = HeatGold(mode_by_hour={"t0": "800", "t1": "800", "t2": "750"},
                    graded_hours=("t0", "t1", "t2"), outcome_gbps_h=50.0, rationale="r")
    scenario = _disc_scenario(write_scenario, heat_example_yaml, gold=gold)
    oracle = _oracle_trace_for_ladder()
    baselines = {"never_downshift": _baseline_trace(
        {"t0": "800", "t1": "800", "t2": "800"}, {"t0": 0.0, "t1": 0.0, "t2": 200.0})}
    with pytest.raises(HeatEpisodeInvalid, match="gold step counts"):
        assert_heat_episode_discriminates(scenario, oracle, baselines)


def test_discriminates_raises_when_no_cooling_failure_is_realized(
        write_scenario, heat_example_yaml):
    gold = HeatGold(mode_by_hour={"t0": "800", "t1": "750", "t2": "700"},
                    graded_hours=("t0", "t1", "t2"), outcome_gbps_h=50.0, rationale="r")
    scenario = _disc_scenario(write_scenario, heat_example_yaml, gold=gold, failed_at=None)
    oracle = _oracle_trace_for_ladder()
    baselines = {"never_downshift": _baseline_trace(
        {"t0": "800", "t1": "800", "t2": "800"}, {"t0": 0.0, "t1": 200.0, "t2": 200.0})}
    with pytest.raises(HeatEpisodeInvalid, match="no cooling failure"):
        assert_heat_episode_discriminates(scenario, oracle, baselines)


def test_discriminates_raises_when_a_baseline_does_not_lose_more_than_gold(
        write_scenario, heat_example_yaml):
    gold = HeatGold(mode_by_hour={"t0": "800", "t1": "750", "t2": "700"},
                    graded_hours=("t0", "t1", "t2"), outcome_gbps_h=50.0, rationale="r")
    scenario = _disc_scenario(write_scenario, heat_example_yaml, gold=gold)
    oracle = _oracle_trace_for_ladder()
    # Differs from gold at every hour (so the "matches gold" check passes
    # through), but loses LESS than gold's own 50.0 -- should still raise.
    baselines = {"lucky": _baseline_trace(
        {"t0": "700", "t1": "700", "t2": "750"}, {"t0": 0.0, "t1": 0.0, "t2": 0.0})}
    with pytest.raises(HeatEpisodeInvalid, match="realized"):
        assert_heat_episode_discriminates(scenario, oracle, baselines)


def test_discriminates_raises_when_too_few_hours_are_graded(write_scenario, heat_example_yaml):
    gold = HeatGold(mode_by_hour={"t0": "800", "t1": "750", "t2": "700"},
                    graded_hours=("t0",), outcome_gbps_h=50.0, rationale="r")
    scenario = _disc_scenario(write_scenario, heat_example_yaml, gold=gold)
    oracle = _oracle_trace_for_ladder()
    baselines = {"never_downshift": _baseline_trace(
        {"t0": "800", "t1": "800", "t2": "800"}, {"t0": 0.0, "t1": 200.0, "t2": 200.0})}
    with pytest.raises(HeatEpisodeInvalid, match="graded hours"):
        assert_heat_episode_discriminates(scenario, oracle, baselines)
