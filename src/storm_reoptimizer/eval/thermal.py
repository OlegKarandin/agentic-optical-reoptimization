"""Heat episode, stage 1 (spec 2026-10-08 §3.3): an hour's temperature
forecast -> the heated amplifier's noise-figure rise, as a two-component
Gaussian mixture (cooling holding / cooling failed). Pure; no server; every
constant comes from the scenario YAML's `thermal.class` block -- none is a
code constant (spec §12: "nothing in code depends on their values").

Composition per forecast hour (spec §3.3, with plan deviations 8 and 9):
    ΔT_cooled(t) = max(0, t − setpoint) · k_cooled
    ΔT_failed(t) = max(0, t + self_heating − setpoint) · k_uncooled
    p_fail       = 1 if cooling already failed, else
                   min(1, base + per_c·max(0, t_mean − setpoint) + per_hour·hours_above)
    ΔNF ~ (1 − p_fail)·N(coeff·ΔT_cooled(t_mean), coeff·k_cooled·t_sigma)
        +      p_fail ·N(coeff·ΔT_failed(t_mean), coeff·k_uncooled·t_sigma)"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

_Z_999 = 3.090232306167813   # standard-normal 0.999 quantile
_CLASS_KEYS = {"setpoint_c", "coeff_db_per_c", "k_cooled", "k_uncooled",
               "self_heating_c", "p_cooling_failure"}
_FAILURE_KEYS = {"base", "per_c_above_setpoint", "per_hour_above_setpoint"}


class ThermalError(ValueError):
    """A thermal block or forecast that cannot be composed."""


@dataclass(frozen=True)
class CoolingFailureModel:
    base: float
    per_c_above_setpoint: float
    per_hour_above_setpoint: float


@dataclass(frozen=True)
class ThermalClass:
    setpoint_c: float
    coeff_db_per_c: float
    k_cooled: float
    k_uncooled: float
    self_heating_c: float
    p_cooling_failure: CoolingFailureModel


@dataclass(frozen=True)
class HourForecast:
    t_mean_c: float
    t_sigma_c: float


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0.0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


@dataclass(frozen=True)
class NfMixture:
    p_fail: float
    mu_cooled: float
    sigma_cooled: float
    mu_failed: float
    sigma_failed: float

    def cdf(self, x: float) -> float:
        return ((1.0 - self.p_fail) * _normal_cdf(x, self.mu_cooled, self.sigma_cooled)
                + self.p_fail * _normal_cdf(x, self.mu_failed, self.sigma_failed))

    def p_exceed(self, x: float) -> float:
        """P(ΔNF > x)."""
        return 1.0 - self.cdf(x)

    def quantile(self, q: float) -> float:
        """inf{x : cdf(x) >= q}, by bisection (no scipy)."""
        if not 0.0 < q < 1.0:
            raise ThermalError(f"quantile must be in (0, 1), got {q}")
        spread = max(self.sigma_cooled, self.sigma_failed, 1e-9)
        lo = min(self.mu_cooled, self.mu_failed) - 12.0 * spread
        hi = max(self.mu_cooled, self.mu_failed) + 12.0 * spread
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.cdf(mid) >= q:
                hi = mid
            else:
                lo = mid
        return hi

    def tail(self) -> float:
        """Each component's 0.999 point, the larger of the two: an upper
        bound on any ΔNF the observation or the oracle reads, whatever the
        mixing weight (heat_margin.curve_grid)."""
        return max(self.mu_cooled + _Z_999 * self.sigma_cooled,
                   self.mu_failed + _Z_999 * self.sigma_failed)


def thermal_class_from_dict(raw: dict) -> ThermalClass:
    if not isinstance(raw, dict) or set(raw) != _CLASS_KEYS:
        got = sorted(raw) if isinstance(raw, dict) else raw
        raise ThermalError(f"thermal class keys must be exactly "
                           f"{sorted(_CLASS_KEYS)}, got {got!r}")
    pf = raw["p_cooling_failure"]
    if not isinstance(pf, dict) or set(pf) != _FAILURE_KEYS:
        got = sorted(pf) if isinstance(pf, dict) else pf
        raise ThermalError(f"p_cooling_failure keys must be exactly "
                           f"{sorted(_FAILURE_KEYS)}, got {got!r}")
    return ThermalClass(
        setpoint_c=float(raw["setpoint_c"]),
        coeff_db_per_c=float(raw["coeff_db_per_c"]),
        k_cooled=float(raw["k_cooled"]), k_uncooled=float(raw["k_uncooled"]),
        self_heating_c=float(raw["self_heating_c"]),
        p_cooling_failure=CoolingFailureModel(**{k: float(pf[k]) for k in _FAILURE_KEYS}))


def thermal_class_to_dict(tc: ThermalClass) -> dict:
    return asdict(tc)


def delta_t_cooled(tc: ThermalClass, t_c: float) -> float:
    return max(0.0, t_c - tc.setpoint_c) * tc.k_cooled


def delta_t_failed(tc: ThermalClass, t_c: float) -> float:
    return max(0.0, t_c + tc.self_heating_c - tc.setpoint_c) * tc.k_uncooled


def p_cooling_failure(tc: ThermalClass, t_mean_c: float, hours_above: int) -> float:
    m = tc.p_cooling_failure
    return min(1.0, m.base
               + m.per_c_above_setpoint * max(0.0, t_mean_c - tc.setpoint_c)
               + m.per_hour_above_setpoint * hours_above)


def hours_above_setpoint(tc: ThermalClass, temps: Iterable[float]) -> int:
    return sum(1 for t in temps if t > tc.setpoint_c)


def nf_mixture(tc: ThermalClass, fc: HourForecast, *, hours_above: int,
               cooling_failed: bool) -> NfMixture:
    p = 1.0 if cooling_failed else p_cooling_failure(tc, fc.t_mean_c, hours_above)
    c = tc.coeff_db_per_c
    return NfMixture(
        p_fail=p,
        mu_cooled=c * delta_t_cooled(tc, fc.t_mean_c),
        sigma_cooled=c * tc.k_cooled * fc.t_sigma_c,
        mu_failed=c * delta_t_failed(tc, fc.t_mean_c),
        sigma_failed=c * tc.k_uncooled * fc.t_sigma_c)


def realized_nf_delta(tc: ThermalClass, t_c: float, *, cooling_failed: bool) -> float:
    rise = delta_t_failed(tc, t_c) if cooling_failed else delta_t_cooled(tc, t_c)
    return tc.coeff_db_per_c * rise
