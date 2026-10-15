# tests/eval/test_heat_oracle.py
import pytest

from storm_reoptimizer.eval.heat_margin import MarginCurve
from storm_reoptimizer.eval.heat_oracle import (
    GRADED_FLOOR_GBPS_H, best_mode, decision_gap, expected_loss_by_mode,
)
from storm_reoptimizer.eval.thermal import NfMixture

LADDER = ({"mode_id": "800", "line_rate_gbps": 800.0, "threshold_db": 15.1},
          {"mode_id": "750", "line_rate_gbps": 750.0, "threshold_db": 14.1},
          {"mode_id": "700", "line_rate_gbps": 700.0, "threshold_db": 13.3})
CURVE = MarginCurve((0.0, 1.0, 2.0), (0.4, -0.6, -1.6))


def _losses(p_fail):
    # cooled point mass at ΔNF 0.6 always downs 800; failed mass at 1.5 downs
    # 750 too; 700 (headroom 1.8) never goes down inside the curve.
    return expected_loss_by_mode(CURVE, NfMixture(p_fail, 0.6, 0.0, 1.5, 0.0), LADDER, 800.0)


def test_regression_flip_between_one_and_two_steps():
    # E800 = 800; E750 = 50 + 750 p; E700 = 100  ->  flip at p = 1/15
    low, high = _losses(0.05), _losses(0.10)
    assert low["750"] == pytest.approx(50 + 750 * 0.05)
    assert best_mode(low, LADDER) == "750"
    assert best_mode(high, LADDER) == "700"


def test_tie_goes_to_the_higher_mode():
    assert best_mode({"800": 100.0, "750": 100.0, "700": 120.0}, LADDER) == "800"


def test_graded_floor_fires_on_a_near_tie():
    assert decision_gap({"800": 100.0, "750": 110.0, "700": 400.0}) == 10.0
    assert decision_gap({"800": 100.0, "750": 110.0}) < GRADED_FLOOR_GBPS_H
