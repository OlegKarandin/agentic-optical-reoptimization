# tools/build_viewer_data.py
"""Fold a run's traces, its scenarios and the topology into one self-contained
HTML reader (run-viewer design, "The fold script" and "The viewer").

Reading a rollout today means holding an episode's geometry, the forecast
revision, the exposure arithmetic and up to eleven LLM calls per acting hour in
your head at once, across seven episodes and three runs each. Every number in
docs/superpowers/2026-08-28-control-arm-findings.md was hand-derived; its
sharpest finding is a claim about a point moving between two hours of one run
of one episode, and it took a manual reconstruction to see. It should have
taken a scrubber.

Post-hoc and read-only. It never re-runs, re-scores or edits anything -- a UI
that can mutate a run is a second harness.

Runs in THIS repo's env: pure stdlib plus pyyaml, and it imports nothing from
multilayer_optical_network. It is NOT the offline-build exception.

Run: python tools/build_viewer_data.py --out eval/viewer/index.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
TRACES_DIR = REPO_ROOT / "eval" / "traces"
SCENARIOS_DIR = (REPO_ROOT / "src" / "storm_reoptimizer" / "eval"
                 / "scenarios")
TOPOLOGY_PATH = (REPO_ROOT / "src" / "storm_reoptimizer" / "data"
                 / "toy_india_topology.json")
DEFAULT_OUT = REPO_ROOT / "eval" / "viewer" / "index.html"


def load_topology(path: Path) -> dict:
    """143 nodes and 180 edges, for geographic orientation and for the aerial
    overlay. utf-8-sig: this file has a BOM."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        "nodes": {n["id"]: [n["lat"], n["lon"]]
                  for n in raw["graph"]["nodes"]},
        "edges": [{"src": e["src"], "dst": e["dst"],
                   "mount_type": e["mount_type"],
                   "length_km": e["length_km"]}
                  for e in raw["graph"]["edges"]],
    }


def load_episode(path: Path) -> dict:
    """One scenario YAML's geometry and gold block.

    The cone rings are GeoJSON, hence lon-first. Flip to [lat, lon] HERE, once,
    so nothing downstream has to remember which convention it is holding."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    forecast: dict = {}
    for issued_at, horizons in (raw.get("forecast") or {}).items():
        forecast[issued_at] = {
            horizon: {
                "polygon": [[lat, lon] for lon, lat
                            in cone["cone"]["coordinates"][0]],
                "center": cone["center"],
                "width_km": cone["width_km"],
            }
            for horizon, cone in horizons.items()}
    gold = dict(raw.get("gold") or {})
    gold["gold_spare_action"] = (
        raw.get("metadata") or {}).get("gold_spare_action")
    return {
        "id": raw["id"], "pair": raw.get("pair"),
        "hours": raw["hours"], "decision_hour": raw["decision_hour"],
        "actionable_service": raw["service_under_test"],
        "damage_radius_km": raw["damage_radius_km"],
        "lead_time_hours": raw["lead_time_hours"],
        "spares_on_hand": raw["spares_on_hand"],
        "gold": gold, "forecast": forecast, "runs": [],
    }


def _exposure_rows(hour: dict) -> list[dict]:
    """One row per service the hour's ground truth exposed, per horizon,
    marked shown or omitted by comparing `observation` against `projected`.

    The gap between what was true and what the agent was shown is the entire
    subject of the eval-fairness design; this is where it becomes visible."""
    observation = hour.get("observation") or {}
    shown = set((hour.get("projected") or {}).get("exposure") or {})
    rows = []
    for svc, per_horizon in (observation.get("exposure") or {}).items():
        for horizon, entry in per_horizon.items():
            rows.append({"service_id": svc, "horizon": horizon,
                         "shown": svc in shown, **entry})
    rows.sort(key=lambda r: (-r.get("expected_capacity_at_risk_gbps", 0.0),
                            r["service_id"], r["horizon"]))
    return rows


def _mark_committed(hour: dict) -> None:
    """Flag the candidate that actually committed, so each hour opens on what
    happened and divergence is a deliberate click."""
    for step in hour.get("iterations") or ():
        chosen = (step.get("objective") or {}).get("choice")
        for candidate in ((step.get("menu") or {}).get("candidates") or ()):
            candidate["committed"] = (
                step.get("outcome") == "committed"
                and candidate.get("candidate_label") == chosen)


def load_run(path: Path) -> dict:
    """One trace, enriched. Tolerates a pre-Phase-A trace: every key added by
    the recording change is optional and defaults to empty, so the 21 archived
    control rollouts load beside a fresh run (run-viewer design, §6.2)."""
    trace = json.loads(path.read_text(encoding="utf-8"))
    hours = []
    for hour in trace.get("hours") or ():
        enriched = dict(hour)
        observation = enriched.get("observation") or {}
        enriched["actionable_service"] = (
            observation.get("actionable_service")
            or observation.get("service_under_test"))
        enriched["exposure_rows"] = _exposure_rows(enriched)
        enriched.setdefault("service_points", {})
        enriched.setdefault("service_paths", {})
        enriched.setdefault("unmapped_nodes", {})
        _mark_committed(enriched)
        hours.append(enriched)
    return {"decider_name": trace["decider_name"],
            "run_index": trace["run_index"],
            "terminal_status": trace["terminal_status"],
            "spares_remaining": trace["spares_remaining"],
            "ledger_debits": trace.get("ledger_debits") or [],
            "actions": trace.get("actions") or [],
            "oms_nodes": trace.get("oms_nodes") or {},
            "tool_calls": trace.get("tool_calls"),
            "hours": hours}


def fold(traces_dir: Path, scenarios_dir: Path, topology_path: Path) -> dict:
    episodes = {}
    for path in sorted(scenarios_dir.glob("*.yaml")):
        episode = load_episode(path)
        episodes[episode["id"]] = episode
    for path in sorted(traces_dir.glob("*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        episode = episodes.get(trace.get("scenario_id"))
        if episode is None:
            print(f"build_viewer_data: WARNING skipping {path.name}: no "
                  f"scenario {trace.get('scenario_id')!r}", flush=True)
            continue
        episode["runs"].append(load_run(path))
    for episode in episodes.values():
        episode["runs"].sort(
            key=lambda r: (r["decider_name"], r["run_index"]))
    return {"topology": load_topology(topology_path), "episodes": episodes}


def render_html(payload: dict) -> str:
    return json.dumps(payload)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="build_viewer_data",
        description="Fold eval traces into a single self-contained HTML "
                    "reader.")
    p.add_argument("--traces", default=str(TRACES_DIR),
                   help=f"Directory of trace JSON (default: {TRACES_DIR}).")
    p.add_argument("--scenarios", default=str(SCENARIOS_DIR),
                   help="Directory of scenario YAML.")
    p.add_argument("--topology", default=str(TOPOLOGY_PATH),
                   help="The topology JSON the runs were driven against.")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"Output HTML file (default: {DEFAULT_OUT}).")
    args = p.parse_args()
    payload = fold(Path(args.traces), Path(args.scenarios),
                   Path(args.topology))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    runs = sum(len(e["runs"]) for e in payload["episodes"].values())
    print(f"wrote {out}: {len(payload['episodes'])} episodes, {runs} runs, "
          f"{out.stat().st_size // 1024} KB", flush=True)


if __name__ == "__main__":
    main()
