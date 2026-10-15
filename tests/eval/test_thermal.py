import pytest

from storm_reoptimizer.eval.thermal import (
    CoolingFailureModel, HourForecast, NfMixture, ThermalClass, ThermalError,
    hours_above_setpoint, nf_mixture, p_cooling_failure, realized_nf_delta,
    thermal_class_from_dict, thermal_class_to_dict,
)

RAW = {"setpoint_c": 25.0, "coeff_db_per_c": 0.02, "k_cooled": 0.15,
       "k_uncooled": 1.0, "self_heating_c": 8.0,
       "p_cooling_failure": {"base": 0.02, "per_c_above_setpoint": 0.01,
                             "per_hour_above_setpoint": 0.03}}
TC = thermal_class_from_dict(RAW)


def test_round_trip_and_strict_keys():
    assert thermal_class_to_dict(TC) == RAW
    with pytest.raises(ThermalError, match="keys"):
        thermal_class_from_dict({**RAW, "extra": 1})
    with pytest.raises(ThermalError, match="keys"):
        thermal_class_from_dict({**RAW, "p_cooling_failure": {"base": 0.1}})


def test_zero_rise_at_or_below_setpoint_with_no_failure_mass():
    tc = thermal_class_from_dict({**RAW, "p_cooling_failure": {
        "base": 0.0, "per_c_above_setpoint": 0.0, "per_hour_above_setpoint": 0.0}})
    mix = nf_mixture(tc, HourForecast(20.0, 0.0), hours_above=0, cooling_failed=False)
    assert mix.p_fail == 0.0
    assert mix.quantile(0.5) == pytest.approx(0.0, abs=1e-9)
    assert mix.p_exceed(0.0) == 0.0


def test_mixture_quantiles_bracket_both_modes():
    # p_fail = 0.02 + 0.01*10 = 0.12; cooled N(0.03, 0.003); failed N(0.36, 0.02)
    mix = nf_mixture(TC, HourForecast(35.0, 1.0), hours_above=0, cooling_failed=False)
    assert mix.p_fail == pytest.approx(0.12)
    assert mix.quantile(0.5) == pytest.approx(0.0305, abs=1e-3)
    assert mix.quantile(0.95) == pytest.approx(0.3642, abs=1e-3)
    assert mix.tail() == pytest.approx(0.36 + 3.0902 * 0.02, abs=1e-3)


def test_p_exceed_is_monotone_non_increasing():
    mix = nf_mixture(TC, HourForecast(38.0, 2.0), hours_above=2, cooling_failed=False)
    xs = [i * 0.02 for i in range(40)]
    ps = [mix.p_exceed(x) for x in xs]
    assert all(b <= a + 1e-15 for a, b in zip(ps, ps[1:]))
    assert ps[0] > 0.99 and ps[-1] < 0.01


def test_hours_above_accumulates_and_raises_failure_probability():
    assert hours_above_setpoint(TC, [20.0, 30.0, 31.0, 24.0, 25.0]) == 2
    assert p_cooling_failure(TC, 30.0, 0) < p_cooling_failure(TC, 30.0, 3)


def test_failure_probability_is_capped_and_known_failure_is_certain():
    assert p_cooling_failure(TC, 200.0, 50) == 1.0
    mix = nf_mixture(TC, HourForecast(30.0, 1.0), hours_above=0, cooling_failed=True)
    assert mix.p_fail == 1.0


def test_realized_delta_never_negative_and_failure_adds_self_heating():
    assert realized_nf_delta(TC, 10.0, cooling_failed=False) == 0.0
    assert realized_nf_delta(TC, 10.0, cooling_failed=True) == 0.0   # clamp (deviation 9)
    assert realized_nf_delta(TC, 35.0, cooling_failed=False) == pytest.approx(0.03)
    assert realized_nf_delta(TC, 35.0, cooling_failed=True) == pytest.approx(0.36)


def test_quantile_rejects_out_of_range():
    with pytest.raises(ThermalError):
        NfMixture(0.0, 0.0, 0.0, 0.0, 0.0).quantile(1.0)
