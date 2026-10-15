# tools/find_heat_sut.py
"""Heat Task 5: rank (src, dst) corridors whose SHORTEST route over the toy
topology uses only edges carrying NO lightpath in the base state and
includes an aerial edge >= 100 km (so it has a mid-span amplifier hut, and
the heat filter admits it). A SUT pinned there has zero bystanders on the
heated hut -- spec §8 holds by construction, which loaded-s17 cannot offer
(plan deviation 3). Static; no server. The solver may still route a pin
differently: the live build is the truth.

    python tools/find_heat_sut.py [--base eval/states/loaded-s17.json] [--top 25]"""
from __future__ import annotations

import argparse
import collections
import heapq
import json
from pathlib import Path

TOPOLOGY = Path("src/storm_reoptimizer/data/toy_india_topology.json")


def _oms_ends(oms_id: str, nodes: set[str]) -> tuple[str, str]:
    rest = oms_id[len("oms_"):]
    for i, ch in enumerate(rest):
        if ch == "_" and rest[:i] in nodes and rest[i + 1:] in nodes:
            return rest[:i], rest[i + 1:]
    raise ValueError(oms_id)


def rank(base: Path, topology: Path = TOPOLOGY) -> list[tuple]:
    state = json.loads(base.read_text(encoding="utf-8"))
    graph = json.loads(topology.read_text(encoding="utf-8-sig"))["graph"]
    nodes = {n["id"] for n in graph["nodes"]}
    used = {frozenset(_oms_ends(o, nodes))
            for lp in state["lightpaths"] for o in lp["oms_sequence"]}
    adj, info = collections.defaultdict(list), {}
    for e in graph["edges"]:
        adj[e["src"]].append((e["dst"], e["length_km"]))
        adj[e["dst"]].append((e["src"], e["length_km"]))
        info[frozenset((e["src"], e["dst"]))] = (e["mount_type"], e["length_km"])
    out = []
    for a in sorted(nodes):
        dist, prev, pq = {a: 0.0}, {}, [(0.0, a)]
        while pq:
            c, u = heapq.heappop(pq)
            if c > dist[u]:
                continue
            for v, w in adj[u]:
                if c + w < dist.get(v, float("inf")):
                    dist[v], prev[v] = c + w, u
                    heapq.heappush(pq, (c + w, v))
        for b in dist:
            if b <= a:
                continue
            path = [b]
            while path[-1] != a:
                path.append(prev[path[-1]])
            path.reverse()
            edges = [frozenset(p) for p in zip(path, path[1:])]
            if any(e in used for e in edges):
                continue
            aerial = [e for e in edges if info[e][0] == "aerial" and info[e][1] >= 100]
            if aerial:
                out.append((round(dist[b]), len(edges), "-".join(path),
                            ["-".join(sorted(e)) for e in aerial]))
    return sorted(out, reverse=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="eval/states/loaded-s17.json")
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args()
    for row in rank(Path(args.base))[: args.top]:
        print(*row)
