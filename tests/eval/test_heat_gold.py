# tests/eval/test_heat_gold.py
from storm_reoptimizer.eval.heat_gold import expected_losses_from_record, heat_gold_from_trace
from storm_reoptimizer.eval.heat_margin import MarginCurve
from storm_reoptimizer.eval.runner import EpisodeTrace
from storm_reoptimizer.eval.thermal import NfMixture
from dataclasses import asdict

import pytest

LADDER = [{"mode_id": "800", "line_rate_gbps": 800.0, "threshold_db": 15.1},
          {"mode_id": "750", "line_rate_gbps": 750.0, "threshold_db": 14.1},
          {"mode_id": "700", "line_rate_gbps": 700.0, "threshold_db": 13.3}]
CURVE = MarginCurve((0.0, 1.0, 2.0), (0.4, -0.6, -1.6))


def _record(hour, p_fail, mode, loss):
    return {"hour": hour, "mode_in_force": mode, "loss_gbps": loss,
            "observation": {"mode_ladder": LADDER, "link": {"offered_gbps": 800.0},
                            "internal": {"margin_curve": CURVE.to_dict(), "mixtures": {
                                hour: asdict(NfMixture(p_fail, 0.6, 0.0, 1.5, 0.0))}}}}


def test_gold_from_the_oracle_trace(write_scenario, heat_example_yaml):
    from storm_reoptimizer.eval.scenario_file import load_scenario
    s = load_scenario(write_scenario(heat_example_yaml, name="HEAT_EXAMPLE.yaml"))
    # E800 = 800, E750 = 50 + 750p, E700 = 100 (Task 7's regression curve):
    # p=0.02 -> 750 by 35 (graded); p=0.102 -> 700 by 26.5 (graded -- picked
    # a shade above the nominal "0.10 -> gap 25" breakeven so the compared
    # gap clears the >= 25 floor outright rather than landing on it: at
    # p=0.10 the gap computes as 24.999999999999986 in float64, just under
    # the floor, from the curve's own linear interpolation -- confirmed
    # live against heat_oracle.decision_gap);
    # p=0.0667 -> breakeven is p=1/15=0.0667 exactly (50+750p==100); 0.0667
    # sits a hair ABOVE 1/15, so 700 (the constant-E side) wins by 0.025 --
    # within 0.03 of a dead tie (a near-tie: NOT graded) but NOT a tie, so
    # the winner is 700, not the tie-break's "higher mode" (confirmed live
    # against heat_oracle.expected_loss_by_mode/best_mode: {'750':
    # 100.025, '700': 100.0} -> best_mode == '700').
    trace = EpisodeTrace("HEAT_EXAMPLE", "oracle:mode", 0, (
        _record("t0", 0.02, "750", 50.0), _record("t1", 0.102, "700", 100.0),
        _record("t2", 0.0667, "700", 50.0)), (), (), 0, "converged", {}, {}, 1, 1.0)
    assert expected_losses_from_record(trace.hours[0])["750"] == pytest.approx(50 + 750 * 0.02)
    gold = heat_gold_from_trace(s, trace)
    assert gold.mode_by_hour == {"t0": "750", "t1": "700", "t2": "700"}
    assert gold.graded_hours == ("t0", "t1")        # t2 sits at the 1/15 near-tie
    assert gold.outcome_gbps_h == 200.0
    assert "t1" in gold.rationale
