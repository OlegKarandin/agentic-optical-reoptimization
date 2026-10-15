# src/storm_reoptimizer/eval/heat_oracle.py
"""Heat spec §7.1 with plan deviation 2: this hour's expected loss per mode,
argmin, ties to the higher mode. Separable because the lever is reversible
with no lead time, so the myopic argmin IS the optimal policy."""
from __future__ import annotations

import json
from typing import Sequence

from .decisions import HOLD_CHOICE, ModeDecision
from .heat_margin import MarginCurve, p_link_down
from .thermal import NfMixture

GRADED_FLOOR_GBPS_H = 25.0   # 2026-09-27 spec's floor, applied per hour


def expected_loss_by_mode(curve: MarginCurve, mixture: NfMixture,
                          ladder: Sequence[dict], offered_gbps: float) -> dict[str, float]:
    top = ladder[0]["threshold_db"]     # the provisioned mode, where the curve was measured
    out: dict[str, float] = {}
    for m in ladder:
        p = p_link_down(curve, mixture, top - m["threshold_db"])
        out[m["mode_id"]] = ((1.0 - p) * max(0.0, offered_gbps - m["line_rate_gbps"])
                             + p * offered_gbps)
    return out


def best_mode(losses: dict[str, float], ladder: Sequence[dict]) -> str:
    rate = {m["mode_id"]: m["line_rate_gbps"] for m in ladder}
    return min(losses, key=lambda j: (losses[j], -rate[j]))


def decision_gap(losses: dict[str, float]) -> float:
    ordered = sorted(losses.values())
    return ordered[1] - ordered[0] if len(ordered) > 1 else float("inf")


def expected_loss_for(obs) -> dict[str, float]:
    return expected_loss_by_mode(obs.margin_curve, obs.mixtures[obs.hour],
                                 obs.mode_ladder, obs.link["offered_gbps"])


class OracleModeDecider:
    name = "oracle:mode"
    event_types = frozenset({"heat"})
    deterministic = True

    async def decide_mode(self, obs) -> ModeDecision:
        losses = expected_loss_for(obs)
        chosen = best_mode(losses, obs.mode_ladder)
        return ModeDecision(
            mode=HOLD_CHOICE if chosen == obs.current_mode else chosen,
            reasoning="argmin expected loss this hour: " + json.dumps(
                {k: round(v, 3) for k, v in losses.items()}, sort_keys=True))
