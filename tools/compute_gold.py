# tools/compute_gold.py
"""Task 10 (T1 spend-or-hold redesign, plan; spec 2026-09-05, section 4.6):
the enumerator CLI. Runs a `spare_action_by_deadline` scenario's two
candidate decisions (spend the depot's spare on the service under test at
the decision hour; hold it) for real, against a live `multilayer-optical-mcp`
server, and reports the actual, simulated Gbps-hours lost under each --
`gold.gold_from_outcomes` turns that into an actual `Gold`.

Runs in THIS repo's env (storm-reoptimizer) against a LIVE server, in the
same spirit as `tools/derive_episodes.py` and `tools/probe_claimants.py`
(design doc's own precedent for exploratory/authoring tooling under
`tools/`) -- NOT the offline-build exception, and imports nothing from
`multilayer_optical_network`/`multilayer_optical_mcp` directly; every call
goes over MCP via `storm_reoptimizer.mcp_client`.

Prints the enumerator's own table (`gold.rationale`) either way. With
`--write`, builds the updated scenario via
`scenario_file.dump_scenario(dataclasses.replace(scenario, gold=...))` --
the reverse of `load_scenario`, so no human hand-edits the YAML -- and by
DEFAULT writes it to a NEW sibling file, `<name>.new.yaml`, next to
`--scenario`, never touching the original. `dump_scenario` round-trips the
file's DATA, not its YAML COMMENTS (`yaml.safe_load`/`safe_dump` have no
comment-preservation mechanism, confirmed directly), and every shipped
scenario carries load-bearing derivation-provenance comments (e.g. T1a.yaml's
"GEOMETRY FROZEN ... re-run the tool to reproduce every number below digit
for digit") that overwriting the original in place would silently destroy.
Pass `--in-place` to overwrite `--scenario` itself anyway once you've
manually carried forward (or deliberately dropped) whatever comments it had.

Run (from repo root, this repo's own env):

    C:/Users/olegk/miniconda3/envs/storm-reoptimizer/python.exe tools/compute_gold.py \\
        --scenario src/storm_reoptimizer/eval/scenarios/T1a.yaml --write \\
        --server-command '["C:/Users/olegk/miniconda3/envs/multilayer-optical-mcp/python.exe", "-c", "import sys; sys.argv = [\\'multilayer-optical-mcp\\', *sys.argv[1:]]; from multilayer_optical_mcp.server import main; main()"]'
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from storm_reoptimizer.eval.gold import enumerate_outcomes, gold_from_outcomes  # noqa: E402
from storm_reoptimizer.eval.scenario_file import (  # noqa: E402
    dump_scenario, load_scenario,
)
from storm_reoptimizer.mcp_client import connect_server  # noqa: E402

DEFAULT_TOPOLOGY = "src/storm_reoptimizer/data/toy_india_topology.json"
# This workspace's own workaround (tests/conftest.py's local_server_command):
# the sibling server repo's console script never gets a .exe here (a
# Cyrillic path trips the editable-install bug), so the server is launched
# by invoking its main() directly via `python -c`, in the SIBLING conda env.
DEFAULT_SERVER_PYTHON = (
    r"C:\Users\olegk\miniconda3\envs\multilayer-optical-mcp\python.exe")
DEFAULT_SERVER_COMMAND = [
    DEFAULT_SERVER_PYTHON, "-c",
    "import sys; sys.argv = ['multilayer-optical-mcp', *sys.argv[1:]]; "
    "from multilayer_optical_mcp.server import main; main()",
]


async def run(args: argparse.Namespace) -> None:
    scenario = load_scenario(args.scenario)
    server_command = (json.loads(args.server_command) if args.server_command
                      else DEFAULT_SERVER_COMMAND)
    env = dict(os.environ)

    @asynccontextmanager
    async def _connect():
        # A FRESH connection per call -- enumerate_outcomes calls this once
        # per choice, since run_episode mutates server state and the two
        # rollouts must each start from the scenario's own state file
        # untouched by the other.
        async with connect_server(
            args.topology, server_command=server_command, env=env,
            extra_args=["--state", scenario.state_file],
        ) as client:
            yield client

    outcomes = await enumerate_outcomes(
        _connect, scenario, topology_path=args.topology)
    gold = gold_from_outcomes(
        scenario, outcomes, min_margin_fraction=args.min_margin_fraction)

    print(f"{scenario.id}: outcomes for spend vs hold\n")
    print(gold.rationale)

    if args.write:
        updated = dataclasses.replace(scenario, gold=gold)
        dumped = dump_scenario(updated)
        scenario_path = Path(args.scenario)
        if args.in_place:
            print(f"WARNING: overwriting {scenario_path} IN PLACE -- this "
                  f"permanently drops any YAML comments it had "
                  f"(dump_scenario round-trips data only, never comments).")
            out_path = scenario_path
        else:
            out_path = scenario_path.with_name(
                f"{scenario_path.stem}.new{scenario_path.suffix}")
        out_path.write_text(dumped, encoding="utf-8")
        print(f"wrote the updated scenario (gold: block replaced) to "
             f"{out_path}"
             + ("" if args.in_place else
                f" -- {scenario_path} itself was left untouched; pass "
                f"--in-place to overwrite it directly"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="compute_gold")
    parser.add_argument("--scenario", required=True,
                        help="path to the scenario YAML to enumerate "
                        "outcomes for (must use the spare_action_by_"
                        "deadline label rule)")
    parser.add_argument("--write", action="store_true",
                        help="write the scenario with its gold: block "
                        "replaced by the freshly computed one. By DEFAULT "
                        "this writes a NEW sibling file, <name>.new.yaml, "
                        "next to --scenario, rather than the original -- "
                        "dump_scenario round-trips the file's data but "
                        "DROPS every YAML comment, and the shipped scenario "
                        "files carry load-bearing derivation-provenance "
                        "comments. Pass --in-place to overwrite --scenario "
                        "itself instead.")
    parser.add_argument("--in-place", action="store_true",
                        help="with --write, overwrite --scenario itself "
                        "instead of writing a <name>.new.yaml sibling. "
                        "PERMANENTLY DROPS any YAML comments the original "
                        "had -- make sure it's committed to git (or "
                        "otherwise safe to lose comments from) first.")
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    parser.add_argument("--server-command", default=None,
                        help="JSON list; defaults to this workspace's "
                        "multilayer-optical-mcp conda-env workaround")
    parser.add_argument("--min-margin-fraction", type=float, default=0.25,
                        dest="min_margin_fraction",
                        help="gold.min_margin_gbps_h = this fraction of the "
                        "smaller of the two SCOPE-ONLY totals (the service "
                        "under test plus its declared claimants, not the "
                        "network-wide total), floored at 1.0")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
