# tests/eval/test_heat_baseline.py
import asyncio

import pytest

from storm_reoptimizer.eval.decisions import HOLD_CHOICE
from storm_reoptimizer.eval.heat_baseline import HEAT_BASELINE_VARIANTS, HeatBaseline


class _Obs:
    def __init__(self, hour_index, current, p95):
        self.hour, self.hour_index, self.current_mode = "t0", hour_index, current
        self.mode_ladder = ({"mode_id": "800", "line_rate_gbps": 800.0, "threshold_db": 15.1},
                            {"mode_id": "750", "line_rate_gbps": 750.0, "threshold_db": 14.1},
                            {"mode_id": "700", "line_rate_gbps": 700.0, "threshold_db": 13.3})
        self.forecast = {"t0": {"margin_at_current_mode_db": {"p95": p95}}}


def _mode(variant, obs):
    return asyncio.run(HeatBaseline(variant).decide_mode(obs)).mode


def test_variants_and_names():
    assert HEAT_BASELINE_VARIANTS == ("never_downshift", "always_one_step", "cover_p95")
    assert HeatBaseline("cover_p95").name == "heat-baseline:cover_p95"
    with pytest.raises(ValueError):
        HeatBaseline("nope")


def test_never_and_always_one_step():
    assert _mode("never_downshift", _Obs(0, "800", -5.0)) == HOLD_CHOICE
    assert _mode("always_one_step", _Obs(0, "800", 1.0)) == "750"
    assert _mode("always_one_step", _Obs(3, "750", 1.0)) == HOLD_CHOICE


def test_cover_p95_picks_the_highest_mode_covering_the_pessimistic_margin():
    assert _mode("cover_p95", _Obs(0, "800", 0.2)) == HOLD_CHOICE
    assert _mode("cover_p95", _Obs(0, "800", -0.5)) == "750"      # 1.0 dB headroom
    assert _mode("cover_p95", _Obs(0, "800", -1.5)) == "700"      # 1.8 dB headroom
    assert _mode("cover_p95", _Obs(0, "800", -9.0)) == "700"      # lowest if none covers
    assert _mode("cover_p95", _Obs(0, "700", 1.5)) == "750"       # upshifts back: 1.5-0.8>=0
