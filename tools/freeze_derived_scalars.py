"""Dump every scored episode's probability-model scalars to JSON.

Run this BEFORE a change that is claimed not to touch cone.py's model, and
the test in tests/eval/test_episodes.py turns that claim into an assertion.
Running it AFTER such a change re-freezes whatever the change did and proves
nothing -- which is the only way to use this tool wrongly.

Needs STORM_REOPTIMIZER_MCP_SERVER_CMD (a JSON list, read by
mcp_client._default_server_command) and PYTHONPATH=src; see the plan's
Global Constraints.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from storm_reoptimizer.eval.derived import (
    derived_scalars_for, flip_scalars_for,
)
from storm_reoptimizer.eval.scenario_file import load_all_scenarios
from storm_reoptimizer.mcp_client import connect_server

REPO_ROOT = Path(__file__).parent.parent
TOPOLOGY = (REPO_ROOT / "src" / "storm_reoptimizer" / "data"
            / "toy_india_topology.json")
DEFAULT_OUT = REPO_ROOT / "tests" / "eval" / "fixtures" / "frozen_derived_scalars.json"


async def _dump(state_path: Path) -> dict:
    @asynccontextmanager
    async def _connect():
        async with connect_server(
            TOPOLOGY, env=dict(os.environ),
            extra_args=["--state", str(state_path)],
        ) as client:
            yield client

    episodes = load_all_scenarios()
    scenarios = list(episodes.values())
    # Both reads are read-only (service_geometry + get_services), so one
    # connection serves both -- suite.main() uses two only because it makes
    # two separate asyncio.run() calls.
    async with _connect() as client:
        derived = await derived_scalars_for(client, scenarios,
                                            topology_path=TOPOLOGY)
        flip = await flip_scalars_for(client, scenarios,
                                      topology_path=TOPOLOGY)
    return {"derived": derived,
            "flip": {sid: f.values() for sid, f in flip.items()}}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="freeze_derived_scalars")
    p.add_argument("--state", required=True,
                   help="the loaded eval state file, e.g. eval/states/loaded-s17.json")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args(argv)
    payload = asyncio.run(_dump(Path(args.state)))
    Path(args.out).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
