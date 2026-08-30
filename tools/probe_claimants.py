# tools/probe_claimants.py -- offline, runs under the multilayer-optical-mcp
# conda env's python (it imports multilayer_optical_network directly, the one
# documented exception to CLAUDE.md's no-server-internals rule). Never
# imported by this app's runtime process.
import argparse, json
from pathlib import Path

import networkx as nx
from multilayer_optical_network.model.allocation import (
    make_adapter_evaluator, solve_allocation_model)
from multilayer_optical_network.model.modes import default_modes
from multilayer_optical_network.model.qot_results import QoTCache, QoTResultStore
from multilayer_optical_network.state_file import load_model_from_state_file

DEPOT = "satna"
# storm-svc-1's own aerial spans: working satna<->rewa, protection
# satna<->jhansi. Property 3 of §3.3 is "disjoint from these".
SUT_AERIAL = {("rewa", "satna"), ("jhansi", "satna")}


def aerial_spans(graph, path):
    return {tuple(sorted((a, b))) for a, b in zip(path, path[1:])
            if graph[a][b]["mount_type"] == "aerial"}


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe_claimants")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--demands", type=float, nargs="+",
                        default=[100.0, 200.0, 400.0])
    args = parser.parse_args()

    raw = json.loads(Path(args.topology).read_text(encoding="utf-8-sig"))
    graph = nx.Graph()
    for edge in raw["graph"]["edges"]:
        graph.add_edge(edge["src"], edge["dst"],
                       weight=edge["length_km"], mount_type=edge["mount_type"])

    # NOTE: default_modes() returns a ModeRegistry (has .get()/.list()), not
    # an iterable of modes -- iterating it directly raises TypeError. The
    # mode field is also `bitrate_gbps`, not `capacity_gbps` (brief drift
    # against the sibling library's current API, confirmed against
    # multilayer_optical_network.model.modes.TransceiverMode).
    print(f"max lightpath capacity: "
          f"{max(m.bitrate_gbps for m in default_modes().list())} Gbps")

    rows = []
    for dst in sorted(graph.nodes):
        if dst == DEPOT:
            continue
        path = nx.shortest_path(graph, DEPOT, dst, weight="weight")
        spans = aerial_spans(graph, path)
        if not spans or spans & SUT_AERIAL:
            continue        # property 2 or property 3 fails
        rows.append((dst, path, spans))

    for dst, path, spans in rows:
        others = [o for o, _, s in rows if o != dst and not (s & spans)]
        print(f"{dst:14s} aerial={sorted(spans)} "
              f"independent_of={sorted(others)} path={'->'.join(path)}")

    # Property: does it actually place? Pin each candidate onto the loaded
    # state and report the solver's own verdict rather than assuming. Load
    # against eval/states/loaded-s17.json itself (not a bare topology model)
    # so the check reflects the spectrum the eval's background traffic and
    # storm-svc-1 already consume -- the brief's snippet called
    # load_model_from_topology_file(args.topology, ...) here, which builds an
    # empty-spectrum model and never touches --state at all, silently at odds
    # with the interface contract ("against the existing state"). Fixed to
    # load_model_from_state_file, confirmed against
    # multilayer_optical_network.state_file's actual signature.
    modes = default_modes()
    model = load_model_from_state_file(args.topology, args.state, modes=modes)
    store = QoTResultStore()
    for dst, _, _ in rows:
        for demand in args.demands:
            qot = make_adapter_evaluator(model, store, cache=QoTCache())
            result, _ = solve_allocation_model(
                model, qot,
                [{"id": f"probe-{dst}", "src": DEPOT, "dst": dst,
                  "demand_gbps": demand, "protected": False}],
                {DEPOT: 4, dst: 4})
            print(f"  pin {dst:14s} {demand:6.0f}G -> {result.status.name} "
                  f"unplaced={result.unplaced}")


if __name__ == "__main__":
    main()
