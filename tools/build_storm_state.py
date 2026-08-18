# tools/build_storm_state.py
"""Offline builder: seeds this app's demo services into a state file for the
real multilayer-optical-mcp server's --state flag. NOT part of the
storm_reoptimizer package (lives outside src/ and tests/) and imports
multilayer_optical_network directly -- the one narrow, documented exception
to CLAUDE.md's "no server-internals imports" rule (see CLAUDE.md's "sibling
side split" section). Run under the multilayer-optical-mcp conda env's
python (the only env with multilayer_optical_network installed); never
imported by this app's own runtime process.

Deterministic, not the simulator's generic gravity-load builder
(multilayer-optical-network-build): this app needs specific demands routed
onto specific, verified topology structure, not a random load. See
docs/superpowers/specs/2026-08-07-storm-scenario-design.md's 2026-08-18
addendum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multilayer_optical_network.model.allocation import (
    make_adapter_evaluator, solve_allocation_model,
)
from multilayer_optical_network.model.modes import default_modes
from multilayer_optical_network.model.qot_results import QoTCache, QoTResultStore
from multilayer_optical_network.model.solvers import SolverStatus
from multilayer_optical_network.state_file import dump_state, topology_fingerprint
from multilayer_optical_network.topology_loader import load_model_from_topology_file

# storm-svc-1: the forced-exposure demand (see the design spec's "Forced-
# exposure demand" section -- satna/jhansi/allahabad/rewa form a real ring,
# both halves aerial). unexposed-svc: a real negative control, far from the
# Hudhud corridor, for Task 7's e2e test.
DEMANDS = [
    {"id": "storm-svc-1", "src": "satna", "dst": "allahabad",
     "demand_gbps": 300.0, "protected": True},
    {"id": "unexposed-svc", "src": "mumbai", "dst": "pune",
     "demand_gbps": 300.0, "protected": True},
]
SPARE_INVENTORY = {"satna": 2, "allahabad": 2, "mumbai": 2, "pune": 2}


def build(topology_path: str, out_path: str) -> None:
    modes = default_modes()
    model = load_model_from_topology_file(topology_path, modes=modes)
    store = QoTResultStore()
    qot = make_adapter_evaluator(model, store, cache=QoTCache())

    result, work = solve_allocation_model(model, qot, DEMANDS, SPARE_INVENTORY)
    if result.status is not SolverStatus.SOLUTION or result.unplaced:
        raise SystemExit(
            f"build_storm_state: allocation did not fully place demands "
            f"(status={result.status}, unplaced={result.unplaced})")

    raw = json.loads(Path(topology_path).read_text(encoding="utf-8-sig"))
    fingerprint = topology_fingerprint(raw)
    doc = dump_state(work, fingerprint=fingerprint,
                      meta={"note": "storm-reoptimizer demo services",
                            "demand_ids": [d["id"] for d in DEMANDS]})
    Path(out_path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {out_path}: {len(doc['services'])} services", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="build_storm_state")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build(args.topology, args.out)


if __name__ == "__main__":
    main()
