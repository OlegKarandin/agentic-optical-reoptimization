# src/storm_reoptimizer/eval/heat_scoring.py
"""Heat spec §7.2 (label_rule `mode_by_hour`, plan deviation 12)."""
from __future__ import annotations

from .runner import EpisodeTrace


def _cites_frac(texts, tokens) -> float:
    if not tokens:
        return 1.0
    joined = " ".join(texts).lower()
    return sum(1 for t in tokens if t.lower() in joined) / len(tokens)


def heat_episode_metrics(scenario, trace: EpisodeTrace) -> dict:
    gold = scenario.gold
    if gold is None:
        raise ValueError(f"{scenario.id}: no gold yet -- run tools/compute_gold.py "
                         f"--scenario <path> --write")
    modes = {r["hour"]: r["mode_in_force"] for r in trace.hours}
    match = {h: modes[h] == gold.mode_by_hour[h] for h in gold.graded_hours}
    lost = sum(float(r["loss_gbps"]) for r in trace.hours)
    outcomes = [r["mode_change_outcome"] for r in trace.hours]
    return {
        "scenario_id": scenario.id, "decider": trace.decider_name,
        "decision_label": ",".join(modes[h] for h in scenario.hours),
        "label_correct": all(match.values()), "mode_match_by_hour": match,
        "graded_hours": list(gold.graded_hours),
        "mode_changes": outcomes.count("committed"),
        "rejected_mode_changes": outcomes.count("rejected"),
        "gbps_hours_lost_total": lost,
        # Can be NEGATIVE: gold minimises EXPECTED loss (plan deviation 12).
        "regret_gbps_h": lost - gold.outcome_gbps_h,
        "inert_commits": 0, "terminal_status": trace.terminal_status,
        "tool_calls": trace.tool_calls, "wall_clock_s": trace.wall_clock_s,
        # Standing caveat: entity matching, necessary not sufficient.
        "cites_flip_variable_frac": _cites_frac(
            [r["mode_decision"]["reasoning"] for r in trace.hours], scenario.flip_variable),
    }
