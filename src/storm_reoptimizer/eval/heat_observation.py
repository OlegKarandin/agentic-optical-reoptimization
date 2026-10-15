# src/storm_reoptimizer/eval/heat_observation.py
"""What a heat decider reads (heat spec §5, plan deviations 1, 6, 8, 11).
Deliberately absent (spec D6): expected loss per mode, the oracle's
quantile, and any word naming a decision rule. `margin_curve`/`mixtures` are
harness-internal (compare=False, never projected) -- the oracle reads them,
the agent never does, the same pattern as Observation.cut_masks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Iterable

from .heat_margin import MARGIN_QUANTILES, MarginCurve, quantile_key
from .thermal import (NfMixture, hours_above_setpoint, nf_mixture,
                      realized_nf_delta, thermal_class_to_dict)

_INTERNAL = ("margin_curve", "mixtures")


class HeatObservationError(ValueError):
    """An observation that cannot be built from the inputs given."""


def mode_ladder(modes: Iterable[dict], provisioned_mode_id: str) -> tuple[dict, ...]:
    rows = {m["id"]: m for m in modes}
    if provisioned_mode_id not in rows:
        raise HeatObservationError(f"provisioned mode {provisioned_mode_id!r} unknown")
    top = rows[provisioned_mode_id]["bitrate_gbps"]
    ladder = [{"mode_id": m["id"], "line_rate_gbps": float(m["bitrate_gbps"]),
               "threshold_db": float(m["required_gsnr_db"])}
              for m in rows.values() if m["bitrate_gbps"] <= top]
    return tuple(sorted(ladder, key=lambda r: (-r["line_rate_gbps"], r["threshold_db"])))


def _index(scenario, hour: str) -> int:
    return scenario.hours.index(hour)


def latest_heat_issue(scenario, hour: str) -> str:
    issued = [i for i in scenario.temperature_forecast
              if _index(scenario, i) <= _index(scenario, hour)]
    return max(issued, key=lambda i: _index(scenario, i))


def cooling_failed_before(scenario, hour: str) -> bool:
    f = scenario.cooling_failed_at
    return f is not None and _index(scenario, f) < _index(scenario, hour)


def cooling_failed_by(scenario, hour: str) -> bool:
    f = scenario.cooling_failed_at
    return f is not None and _index(scenario, f) <= _index(scenario, hour)


def realized_nf_delta_at(scenario, hour: str) -> float:
    return realized_nf_delta(scenario.thermal.heat_class,
                             scenario.temperature_realized[hour],
                             cooling_failed=cooling_failed_by(scenario, hour))


def forecast_mixtures(scenario, decision_hour: str) -> dict[str, NfMixture]:
    """Every forecast hour >= decision_hour in the latest issuance, composed
    with realized history before decision_hour (deviation 8) and with a known
    failure as certain (deviation 6)."""
    tc = scenario.thermal.heat_class
    issue = latest_heat_issue(scenario, decision_hour)
    issuance = scenario.temperature_forecast[issue]
    d = _index(scenario, decision_hour)
    past = hours_above_setpoint(tc, (scenario.temperature_realized[h]
                                     for h in scenario.hours[:d]))
    failed = cooling_failed_before(scenario, decision_hour)
    out: dict[str, NfMixture] = {}
    for hour, fc in issuance.items():
        k = _index(scenario, hour)
        if k < d:
            continue
        ahead = hours_above_setpoint(tc, (issuance[h].t_mean_c for h in issuance
                                          if d <= _index(scenario, h) < k))
        out[hour] = nf_mixture(tc, fc, hours_above=past + ahead, cooling_failed=failed)
    return out


def all_forecast_mixtures(scenario) -> list[NfMixture]:
    return [m for hour in scenario.hours for m in forecast_mixtures(scenario, hour).values()]


@dataclass(frozen=True)
class HeatObservation:
    scenario_id: str
    hour: str
    hour_index: int
    hours_remaining: int
    lightpath_id: str
    service_id: str
    current_mode: str
    current_margin_db: float
    mode_ladder: tuple[dict, ...]
    link: dict
    thermal_class: dict
    cooling_state: str                 # "holding" | "failed"
    forecast: dict[str, dict]
    bystanders: tuple[dict, ...]
    actions_taken: tuple[dict, ...] = ()
    last_rejection: dict | None = None
    margin_curve: MarginCurve | None = field(default=None, compare=False, repr=False)
    mixtures: dict[str, NfMixture] = field(default_factory=dict, compare=False, repr=False)

    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in _INTERNAL}
        out["mode_ladder"] = list(self.mode_ladder)
        out["bystanders"] = list(self.bystanders)
        out["actions_taken"] = list(self.actions_taken)
        out["internal"] = {
            "margin_curve": self.margin_curve.to_dict() if self.margin_curve else None,
            "mixtures": {h: asdict(m) for h, m in self.mixtures.items()}}
        return out


def build_heat_observation(scenario, hour: str, *, current_mode_id: str,
                           current_margin_db: float, ladder: tuple[dict, ...],
                           link: dict, bystanders: tuple[dict, ...],
                           actions_taken: tuple[dict, ...] = (),
                           last_rejection: dict | None = None,
                           curve: MarginCurve) -> HeatObservation:
    by_id = {m["mode_id"]: m for m in ladder}
    if current_mode_id not in by_id:
        raise HeatObservationError(f"current mode {current_mode_id!r} not on the ladder")
    headroom = ladder[0]["threshold_db"] - by_id[current_mode_id]["threshold_db"]
    mixtures = forecast_mixtures(scenario, hour)
    issuance = scenario.temperature_forecast[latest_heat_issue(scenario, hour)]
    forecast = {
        h: {"t_mean_c": issuance[h].t_mean_c, "t_sigma_c": issuance[h].t_sigma_c,
            "p_cooling_failure": mix.p_fail,
            "margin_at_current_mode_db": {
                quantile_key(q): curve.margin_at(mix.quantile(q)) + headroom
                for q in MARGIN_QUANTILES}}
        for h, mix in mixtures.items()}
    i = _index(scenario, hour)
    return HeatObservation(
        scenario_id=scenario.id, hour=hour, hour_index=i,
        hours_remaining=len(scenario.hours) - i,
        lightpath_id=scenario.lightpath_under_test,
        service_id=scenario.service_under_test,
        current_mode=current_mode_id, current_margin_db=current_margin_db,
        mode_ladder=tuple(ladder), link=dict(link),
        thermal_class=thermal_class_to_dict(scenario.thermal.heat_class),
        cooling_state="failed" if cooling_failed_before(scenario, hour) else "holding",
        forecast=forecast, bystanders=tuple(bystanders),
        actions_taken=tuple(actions_taken), last_rejection=last_rejection,
        margin_curve=curve, mixtures=mixtures)


def _round(value, ndigits: int = 3):
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {k: _round(v, ndigits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, ndigits) for v in value]
    return value


def project_heat_observation(obs: HeatObservation) -> dict:
    """The single rendering both ClaudeDecider and the trace record (spec §5):
    every float to 3 decimals (DERIVED_TOLERANCE), internals dropped."""
    out = obs.to_dict()
    out.pop("internal")
    return _round(out)
