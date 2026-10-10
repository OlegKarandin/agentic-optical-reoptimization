# src/storm_reoptimizer/eval/heat_runner.py
"""One heat rollout (heat spec §4.2-4.3, plan deviations 4 and 5). Per hour:
read live state -> build the observation -> ONE `decide_mode` call -> apply
the mode change through the direct `set_modulation_format` tool (never a
plan op: validate_plan blocks own-link overload) -> apply this hour's
realized ΔNF to every heated amp -> simulate IP routing -> loss on the SUT's
link. No ledger, no risk groups, no spares."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from ..events.filters import get_filter
from ..geo_mapper import load_edges, map_geo_event_to_amplifiers
from .decisions import HOLD_CHOICE, ModeDecision, check_mode_on_ladder
from .heat_margin import CURVE_TOLERANCE_DB, curve_grid, measure_margin_curve
from .heat_observation import (all_forecast_mixtures, build_heat_observation,
                               mode_ladder, project_heat_observation,
                               realized_nf_delta_at)
from .runner import EpisodeTrace, _CountingClient


class HeatEpisodeError(RuntimeError):
    """The live network disagrees with what the heat episode assumes."""


def check_heated_amplifiers(mapped: Iterable[str], declared: Iterable[str]) -> None:
    if set(mapped) != set(declared):
        raise HeatEpisodeError(
            f"heat polygon maps to {sorted(mapped)} but thermal.amplifiers "
            f"declares {sorted(declared)} -- re-author the polygon or the list")


def target_mode(decision: ModeDecision, current_mode_id: str) -> str | None:
    mode = current_mode_id if decision.mode == HOLD_CHOICE else decision.mode
    return None if mode == current_mode_id else mode


def mode_change_rejected(result: dict) -> bool:
    return result.get("ok") is False


def link_row(routing: dict, ip_link_id: str) -> dict:
    return next(r for r in routing["utilizations"] if r["ip_link_id"] == ip_link_id)


def hour_loss(row: dict, offered_gbps: float) -> float:
    return offered_gbps if row["down"] else max(0.0, offered_gbps - row["capacity_gbps"])


def bystanders_of(lightpaths: Iterable[dict], amplifiers: Iterable[str],
                  oms: Iterable[dict], lightpath_id: str) -> list[str]:
    amps = set(amplifiers)
    heated = {o["id"] for o in oms if amps & set(o["elements"])}
    return sorted(lp["id"] for lp in lightpaths
                  if lp["id"] != lightpath_id and heated & set(lp["oms_sequence"]))


async def run_heat_episode(client, scenario, decider, *, topology_path,
                           run_index: int = 0, trace_path=None) -> EpisodeTrace:
    if not hasattr(decider, "decide_mode"):
        raise TypeError(f"{getattr(decider, 'name', decider)!r} has no decide_mode; "
                        f"it cannot run heat episode {scenario.id}")
    started = time.monotonic()
    counting = _CountingClient(client)
    call = counting.call
    lp_id = scenario.lightpath_under_test

    optical = await call("get_topology", {"layer": "optical"})
    fiber_len = {f["id"]: f["length_km"] for f in optical["fibers"]}
    edges = load_edges(topology_path)
    amps = scenario.thermal.amplifiers
    for issue, polygon in scenario.heat_polygon.items():
        check_heated_amplifiers(map_geo_event_to_amplifiers(
            polygon, optical["oms"], fiber_len, edges, get_filter("heat")), amps)

    modes = await call("get_transceiver_modes", expect_list=True)
    lightpaths = await call("get_lightpaths", expect_list=True)
    provisioned = next(lp for lp in lightpaths if lp["id"] == lp_id)["mode_id"]
    ladder = mode_ladder(modes, provisioned)
    threshold = {m["mode_id"]: m["threshold_db"] for m in ladder}
    bystander_ids = bystanders_of(lightpaths, amps, optical["oms"], lp_id)
    ip = await call("get_ip_topology")
    link = next(l for l in ip["ip_links"] if l["lightpath_id"] == lp_id)
    ip_link_id, offered = link["id"], float(link["load_gbps"])

    realized = [realized_nf_delta_at(scenario, h) for h in scenario.hours]
    curve = await measure_margin_curve(
        call, amplifiers=amps, lightpath_id=lp_id, bystander_ids=bystander_ids,
        grid=curve_grid(all_forecast_mixtures(scenario), realized))

    applied = 0.0
    hours: list[dict] = []
    actions_taken: list[dict] = []
    last_rejection: dict | None = None
    routing: dict = {}
    for hour_index, hour in enumerate(scenario.hours):
        record: dict = {"hour": hour}
        rows = {lp["id"]: lp for lp in await call("get_lightpaths", expect_list=True)}
        current = rows[lp_id]["mode_id"]
        margin = rows[lp_id]["qot"]["margin_db"]
        expected = (curve.margin_at(applied) + threshold[provisioned] - threshold[current])
        if abs(margin - expected) > CURVE_TOLERANCE_DB:
            raise HeatEpisodeError(
                f"{scenario.id} {hour}: live margin {margin:.6f} != curve "
                f"{expected:.6f} at ΔNF {applied} mode {current} -- headroom is "
                f"not additive or the curve went stale")
        pre = link_row(await call("simulate_ip_routing"), ip_link_id)
        obs = build_heat_observation(
            scenario, hour, current_mode_id=current, current_margin_db=margin,
            ladder=ladder,
            link={"ip_link_id": ip_link_id, "offered_gbps": offered,
                  "capacity_gbps": pre["capacity_gbps"]},
            bystanders=tuple({"lightpath_id": b, "margin_db": rows[b]["qot"]["margin_db"]}
                             for b in bystander_ids),
            actions_taken=tuple(actions_taken), last_rejection=last_rejection,
            curve=curve)
        record["observation"] = obs.to_dict()
        record["projected"] = project_heat_observation(obs)

        decision = await decider.decide_mode(obs)
        check_mode_on_ladder(decision, [m["mode_id"] for m in ladder])
        record["mode_decision"] = decision.to_dict()
        target = target_mode(decision, current)
        record["mode_change"] = None
        outcome = "noop"
        if target is not None:
            result = await call("set_modulation_format",
                                {"lightpath_id": lp_id, "mode_id": target})
            record["mode_change"] = result
            if mode_change_rejected(result):
                outcome, last_rejection = "rejected", result
            else:
                outcome, last_rejection, current = "committed", None, target
        record["mode_change_outcome"] = outcome
        record["mode_in_force"] = current
        actions_taken.append({"hour": hour, "mode": decision.mode, "outcome": outcome})

        delta = realized[hour_index]
        if delta != applied:                                   # Review Focus 2: may be negative
            for amp in amps:
                await call("inject_degradation", {"asset_id": amp, "nf_delta": delta - applied})
            applied = delta
        record["nf_delta_applied_db"] = applied
        record["snapshot_id"] = (await call("snapshot_create"))["id"]
        routing = await call("simulate_ip_routing")
        row = link_row(routing, ip_link_id)
        after = {lp["id"]: lp for lp in await call("get_lightpaths", expect_list=True)}
        record.update(link_down=row["down"], capacity_gbps=row["capacity_gbps"],
                      loss_gbps=hour_loss(row, offered),
                      margin_realized_db=after[lp_id]["qot"]["margin_db"])
        hours.append(record)

    trace = EpisodeTrace(
        scenario_id=scenario.id, decider_name=decider.name, run_index=run_index,
        hours=tuple(hours), actions=(), ledger_debits=(), spares_remaining=0,
        terminal_status="converged", final_routing=routing, affected_by_hour={},
        tool_calls=counting.calls, wall_clock_s=time.monotonic() - started)
    if trace_path is not None:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        Path(trace_path).write_text(json.dumps(trace.to_dict(), indent=2, default=str),
                                    encoding="utf-8")
    return trace
