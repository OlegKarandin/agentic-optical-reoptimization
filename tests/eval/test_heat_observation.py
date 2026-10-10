import pytest

from storm_reoptimizer.eval.heat_margin import MarginCurve
from storm_reoptimizer.eval.heat_observation import (
    HeatObservationError, build_heat_observation, cooling_failed_before,
    forecast_mixtures, latest_heat_issue, mode_ladder, project_heat_observation,
    realized_nf_delta_at,
)
from storm_reoptimizer.eval.scenario_file import load_scenario

MODES = [{"id": f"{r}G@{t}dB", "bitrate_gbps": float(r), "required_gsnr_db": t}
         for r, t in [(300, 4.8), (700, 13.3), (750, 14.1), (800, 15.1)]]
LADDER = mode_ladder(MODES, "750G@14.1dB")
CURVE = MarginCurve((0.0, 1.0, 2.0), (0.4, -0.6, -1.6))


@pytest.fixture
def heat(write_scenario, heat_example_yaml):
    return load_scenario(write_scenario(heat_example_yaml, name="HEAT_EXAMPLE.yaml"))


def test_ladder_is_at_or_below_provisioned_highest_first():
    assert [m["mode_id"] for m in LADDER] == ["750G@14.1dB", "700G@13.3dB", "300G@4.8dB"]
    with pytest.raises(HeatObservationError):
        mode_ladder(MODES, "999G@1dB")


def test_cooling_state_and_realized_delta(heat):
    assert latest_heat_issue(heat, "t2") == "t0"
    assert not cooling_failed_before(heat, "t1") and cooling_failed_before(heat, "t2")
    assert realized_nf_delta_at(heat, "t0") == pytest.approx(0.02 * 5.5 * 0.15)
    assert realized_nf_delta_at(heat, "t1") == pytest.approx(0.02 * 20.0)   # failed at t1


def test_mixtures_carry_history_and_known_failure(heat):
    at_t0 = forecast_mixtures(heat, "t0")
    assert set(at_t0) == {"t0", "t1", "t2"}
    assert at_t0["t1"].p_fail > at_t0["t0"].p_fail          # hotter, and t0 above setpoint
    assert all(m.p_fail == 1.0 for m in forecast_mixtures(heat, "t2").values())


def test_observation_projects_rounded_and_hides_internals(heat):
    obs = build_heat_observation(
        heat, "t0", current_mode_id="750G@14.1dB", current_margin_db=0.4,
        ladder=LADDER, link={"ip_link_id": "ipl", "offered_gbps": 750.0, "capacity_gbps": 750.0},
        bystanders=(), curve=CURVE)
    assert obs.cooling_state == "holding" and obs.hours_remaining == 3
    entry = obs.forecast["t1"]
    assert set(entry["margin_at_current_mode_db"]) == {"p05", "p25", "p50", "p75", "p95"}
    p = project_heat_observation(obs)
    assert "internal" not in p and "margin_curve" not in p and "mixtures" not in p
    text = repr(p).lower()
    for banned in ("cover", "newsvendor", "expected_loss", "gold", "oracle"):
        assert banned not in text
    assert p["forecast"]["t1"]["margin_at_current_mode_db"]["p50"] == round(
        entry["margin_at_current_mode_db"]["p50"], 3)
    assert obs.to_dict()["internal"]["margin_curve"]["sut_margin_db"] == [0.4, -0.6, -1.6]


def test_headroom_shifts_quantile_margins_by_the_threshold_gap(heat):
    kw = dict(current_margin_db=0.0, ladder=LADDER, bystanders=(), curve=CURVE,
              link={"ip_link_id": "i", "offered_gbps": 750.0, "capacity_gbps": 750.0})
    top = build_heat_observation(heat, "t0", current_mode_id="750G@14.1dB", **kw)
    low = build_heat_observation(heat, "t0", current_mode_id="700G@13.3dB", **kw)
    a = top.forecast["t0"]["margin_at_current_mode_db"]["p50"]
    b = low.forecast["t0"]["margin_at_current_mode_db"]["p50"]
    assert b - a == pytest.approx(14.1 - 13.3)
