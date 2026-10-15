# src/storm_reoptimizer/eval/heat_assertions.py
"""H1's build-time invariants (heat spec §8, §10, §12 and plan Task 10's
acceptance criteria). Each raises HeatEpisodeInvalid with the reason."""
from __future__ import annotations

import math

from ..events.filters import get_filter
from ..geo_mapper import map_geo_event_to_amplifiers
from .heat_gold import heat_gold_from_trace
from .heat_margin import MarginCurve


class HeatEpisodeInvalid(AssertionError):
    """A heat episode that does not test what it claims to."""


def assert_heat_mapping(scenario, optical: dict, edges) -> None:
    lengths = {f["id"]: f["length_km"] for f in optical["fibers"]}
    for issue, polygon in scenario.heat_polygon.items():
        mapped = map_geo_event_to_amplifiers(polygon, optical["oms"], lengths,
                                             edges, get_filter("heat"))
        if set(mapped) != set(scenario.thermal.amplifiers):
            raise HeatEpisodeInvalid(f"{scenario.id} {issue}: polygon maps to {mapped}, "
                                     f"thermal.amplifiers is {scenario.thermal.amplifiers}")


def assert_bystanders_never_cross(curve: MarginCurve) -> None:
    for lp, margins in curve.bystander_margin_db.items():
        if min(margins) < 0.0:
            raise HeatEpisodeInvalid(f"bystander {lp} crosses zero inside the curve "
                                     f"(min {min(margins):.3f} dB)")


def assert_heat_gold_matches_oracle(scenario, oracle_trace) -> None:
    fresh = heat_gold_from_trace(scenario, oracle_trace)
    g = scenario.gold
    if g is None:
        raise HeatEpisodeInvalid(f"{scenario.id}: no gold; run tools/compute_gold.py --write")
    if (fresh.mode_by_hour != g.mode_by_hour or fresh.graded_hours != g.graded_hours
            or abs(fresh.outcome_gbps_h - g.outcome_gbps_h) > 1e-3):
        raise HeatEpisodeInvalid(f"{scenario.id}: shipped gold != a fresh oracle "
                                 f"rollout\n{fresh.rationale}")


def assert_heat_episode_discriminates(scenario, oracle_trace, baseline_traces: dict) -> None:
    gold = scenario.gold
    ladder = [m["mode_id"] for m in oracle_trace.hours[0]["observation"]["mode_ladder"]]
    steps = {ladder.index(m) for m in gold.mode_by_hour.values()}
    if scenario.cooling_failed_at is None:
        raise HeatEpisodeInvalid(f"{scenario.id}: no cooling failure is realized")
    if len(gold.graded_hours) < math.ceil(len(scenario.hours) / 2):
        raise HeatEpisodeInvalid(f"{scenario.id}: only {len(gold.graded_hours)} graded hours")
    if 0 not in steps or len(steps) < 3:
        raise HeatEpisodeInvalid(f"{scenario.id}: gold step counts {sorted(steps)} -- "
                                 f"need >= 3 distinct, including 0 (spec §12)")
    for name, trace in baseline_traces.items():
        modes = {r["hour"]: r["mode_in_force"] for r in trace.hours}
        if all(modes[h] == gold.mode_by_hour[h] for h in gold.graded_hours):
            raise HeatEpisodeInvalid(f"{scenario.id}: {name} matches gold at every graded hour")
        lost = sum(float(r["loss_gbps"]) for r in trace.hours)
        if lost <= gold.outcome_gbps_h:
            raise HeatEpisodeInvalid(f"{scenario.id}: {name} realized {lost} <= gold "
                                     f"{gold.outcome_gbps_h}")
