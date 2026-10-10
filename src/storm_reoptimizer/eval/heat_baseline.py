# src/storm_reoptimizer/eval/heat_baseline.py
"""Heat spec §6 baselines. `cover_p95` is CLAUDE.md's decision-tree test for
this episode: it matches gold when the right quantile is ~0.95 and fails
when the cooling-failure mass or the loss ratio moves it."""
from __future__ import annotations

from .decisions import HOLD_CHOICE, ModeDecision
from .heat_margin import quantile_key

HEAT_BASELINE_VARIANTS = ("never_downshift", "always_one_step", "cover_p95")


class HeatBaseline:
    event_types = frozenset({"heat"})
    deterministic = True

    def __init__(self, variant: str) -> None:
        if variant not in HEAT_BASELINE_VARIANTS:
            raise ValueError(f"unknown heat baseline {variant!r}; "
                             f"known: {HEAT_BASELINE_VARIANTS}")
        self.variant = variant
        self.name = f"heat-baseline:{variant}"

    async def decide_mode(self, obs) -> ModeDecision:
        if self.variant == "never_downshift":
            return ModeDecision(HOLD_CHOICE, "fixed policy: never downshift")
        if self.variant == "always_one_step":
            if obs.hour_index == 0 and len(obs.mode_ladder) > 1:
                return ModeDecision(obs.mode_ladder[1]["mode_id"],
                                    "fixed policy: one step down at the first hour")
            return ModeDecision(HOLD_CHOICE, "fixed policy: hold after the first hour")
        pessimistic = obs.forecast[obs.hour]["margin_at_current_mode_db"][quantile_key(0.95)]
        current = next(m for m in obs.mode_ladder if m["mode_id"] == obs.current_mode)
        chosen = next((m for m in obs.mode_ladder
                       if pessimistic + current["threshold_db"] - m["threshold_db"] >= 0.0),
                      obs.mode_ladder[-1])
        return ModeDecision(
            HOLD_CHOICE if chosen["mode_id"] == obs.current_mode else chosen["mode_id"],
            f"fixed policy: highest mode clear of the 95th-percentile NF rise "
            f"this hour ({pessimistic:.3f} dB at the current mode)")
