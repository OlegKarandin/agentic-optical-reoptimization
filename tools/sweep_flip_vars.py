"""Print the whole-suite `FLIP_VARS` sweep `global_policy_report` computes,
against a REAL run of the shipped suite -- so README/CLAUDE.md/the rehearsal
docs can be regenerated from actual tool output instead of hand-transcribed
numbers (T2/T3 probe redesign plan, Task 13).

Needs STORM_REOPTIMIZER_MCP_SERVER_CMD (a JSON list, read by
mcp_client._default_server_command) and PYTHONPATH=src; see
tools/freeze_derived_scalars.py, which this tool's connection wiring copies
verbatim (scenarios can span several state files since the T2/T3 probe
redesign, spec §7, so no single `--state` flag suffices any more).

Usage: python tools/sweep_flip_vars.py
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from storm_reoptimizer.eval.assertions import global_policy_report
from storm_reoptimizer.eval.derived import FLIP_VARS, flip_scalars_for_suite
from storm_reoptimizer.eval.scenario_file import load_all_scenarios
from storm_reoptimizer.mcp_client import connect_server

REPO_ROOT = Path(__file__).parent.parent
TOPOLOGY = (REPO_ROOT / "src" / "storm_reoptimizer" / "data"
            / "toy_india_topology.json")


def _connect_for(state_file: str):
    # state_file is repo-relative (scenario_file.py's module docstring);
    # resolve against REPO_ROOT so this tool runs from any cwd.
    state = REPO_ROOT / state_file

    @asynccontextmanager
    async def _connect():
        async with connect_server(
            TOPOLOGY, env=dict(os.environ),
            extra_args=["--state", str(state)],
        ) as client:
            yield client
    return _connect


async def _sweep() -> tuple[dict, list, dict]:
    episodes = load_all_scenarios()
    scenarios = list(episodes.values())
    flip = await flip_scalars_for_suite(
        _connect_for, scenarios, topology_path=TOPOLOGY)
    flip_values = {sid: f.values() for sid, f in flip.items()}
    halves = sorted(sid for sid, e in episodes.items() if e.pair)
    actions = {sid: e.metadata.get("gold_spare_action")
              for sid, e in episodes.items() if e.pair}
    report = global_policy_report(scenarios, flip_values)
    return report, halves, actions


def render(report: dict, halves: list[str], actions: dict[str, str]) -> str:
    n = len(halves)
    lines = ["| Variable | Blocked by | Best achievable (of "
             f"{n} halves) |",
             "|---|---|---|"]
    for var in FLIP_VARS:
        entry = report.get(var)
        if entry is None:
            lines.append(f"| `{var}` | (not swept) | -- |")
            continue
        lines.append(
            f"| `{var}` | {entry['blocked_by'].upper()} | "
            f"{entry['best']}/{entry['n']} |")
    lines.append("")
    for var in FLIP_VARS:
        entry = report.get(var)
        if entry is None:
            continue
        lines.append(f"### `{var}` ({entry['blocked_by']})")
        lines.append("")
        for sid in halves:
            value = entry["values"].get(sid)
            lines.append(
                f"  {sid:<5} value={value!r:<22}  "
                f"gold_spare_action={actions.get(sid)}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    report, halves, actions = asyncio.run(_sweep())
    print(render(report, halves, actions))


if __name__ == "__main__":
    main()
