# src/storm_reoptimizer/eval/heat_gold.py
"""Gold for a heat episode = the oracle's own rollout (heat spec §7.2):
enumerated, never hand-edited. Expected losses are recomputed from what the
trace recorded (the curve and this hour's mixture), so the gold table is
reproducible from the trace alone."""
from __future__ import annotations

from .heat_margin import MarginCurve
from .heat_oracle import (GRADED_FLOOR_GBPS_H, OracleModeDecider, best_mode,
                          decision_gap, expected_loss_by_mode)
from .runner import EpisodeTrace, run_episode
from .scenario_file import HeatGold
from .thermal import NfMixture


def expected_losses_from_record(record: dict) -> dict[str, float]:
    obs = record["observation"]
    internal = obs["internal"]
    return expected_loss_by_mode(
        MarginCurve.from_dict(internal["margin_curve"]),
        NfMixture(**internal["mixtures"][record["hour"]]),
        obs["mode_ladder"], obs["link"]["offered_gbps"])


def heat_gold_from_trace(scenario, trace: EpisodeTrace, *,
                         floor: float = GRADED_FLOOR_GBPS_H) -> HeatGold:
    lines = [f"{'hour':5} {'gold':>14} {'E[best]':>9} {'E[2nd]':>9} graded"]
    graded, mode_by_hour = [], {}
    for record in trace.hours:
        hour = record["hour"]
        losses = expected_losses_from_record(record)
        ladder = record["observation"]["mode_ladder"]
        best = best_mode(losses, ladder)
        if best != record["mode_in_force"]:
            raise ValueError(f"{scenario.id} {hour}: oracle chose {best} but "
                             f"{record['mode_in_force']} was in force (a refused "
                             f"mode change?) -- gold would not be the argmin")
        mode_by_hour[hour] = best
        gap = decision_gap(losses)
        if gap >= floor:
            graded.append(hour)
        second = sorted(losses.values())[1] if len(losses) > 1 else float("nan")
        lines.append(f"{hour:5} {best:>14} {losses[best]:9.3f} {second:9.3f} "
                     f"{'yes' if gap >= floor else 'no'}")
    outcome = sum(float(r["loss_gbps"]) for r in trace.hours)
    lines.append(f"realized loss of the gold sequence: {outcome:.3f} Gbps·h")
    return HeatGold(mode_by_hour=mode_by_hour, graded_hours=tuple(graded),
                    outcome_gbps_h=outcome, rationale="\n".join(lines) + "\n")


async def oracle_trace(connect, scenario, *, topology_path) -> EpisodeTrace:
    async with connect() as client:
        return await run_episode(client, scenario, OracleModeDecider(),
                                 topology_path=topology_path)
