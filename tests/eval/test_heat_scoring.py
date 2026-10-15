# tests/eval/test_heat_scoring.py
from storm_reoptimizer.eval.heat_scoring import heat_episode_metrics
from storm_reoptimizer.eval.runner import EpisodeTrace
from storm_reoptimizer.eval.scenario_file import HeatGold, load_scenario
from storm_reoptimizer.eval.scoring import LABEL_RULES, decision_label, episode_metrics
import dataclasses


def _trace(modes, losses, reasonings=None):
    hours = tuple({"hour": f"t{i}", "mode_in_force": m, "loss_gbps": l,
                   "mode_change_outcome": "noop",
                   "mode_decision": {"mode": "hold", "reasoning": (reasonings or ["x"] * 3)[i]}}
                  for i, (m, l) in enumerate(zip(modes, losses)))
    return EpisodeTrace("HEAT_EXAMPLE", "d", 0, hours, (), (), 0, "converged", {}, {}, 10, 1.0)


def _scenario(write_scenario, heat_example_yaml):
    s = load_scenario(write_scenario(heat_example_yaml, name="HEAT_EXAMPLE.yaml"))
    return dataclasses.replace(s, gold=HeatGold(
        mode_by_hour={"t0": "800", "t1": "700", "t2": "750"},
        graded_hours=("t1", "t2"), outcome_gbps_h=150.0, rationale="r"))


def test_match_vector_regret_and_dispatch(write_scenario, heat_example_yaml):
    s = _scenario(write_scenario, heat_example_yaml)
    t = _trace(["750", "700", "750"], [50.0, 100.0, 50.0], ["the cooling may fail", "margin", "x"])
    m = episode_metrics(s, t)                     # dispatches to heat
    assert m == heat_episode_metrics(s, t)
    assert m["mode_match_by_hour"] == {"t1": True, "t2": True}
    assert m["label_correct"] is True             # t0 is ungraded
    assert m["gbps_hours_lost_total"] == 200.0 and m["regret_gbps_h"] == 50.0
    assert m["inert_commits"] == 0
    assert m["cites_flip_variable_frac"] == 1.0   # "cooling" and "margin" both cited
    assert decision_label(s, t) == "750,700,750"
    assert "mode_by_hour" in LABEL_RULES


def test_one_graded_mismatch_fails_the_episode(write_scenario, heat_example_yaml):
    s = _scenario(write_scenario, heat_example_yaml)
    m = heat_episode_metrics(s, _trace(["800", "800", "750"], [0, 800, 50]))
    assert m["label_correct"] is False and m["mode_match_by_hour"]["t1"] is False
