"""One-time conversion: Internet Topology Zoo's TataNld.gml (Tata Communications'
real Indian national long-distance backbone, captured 2010:
https://github.com/mroughan/InternetTopologyZoo/blob/main/gml/TataNld.gml) into
the {graph, srlgs} topology JSON multilayer-optical-mcp's topology_loader
expects. Not part of the installed package -- run manually to regenerate
src/storm_reoptimizer/data/toy_india_topology.json when the source data or the
conversion rules below change.

Two synthetic modeling assumptions, made explicit because they are NOT present
in the source data:

- length_km per edge: the GML carries no link length, only endpoint lat/lon.
  Approximated as great-circle (haversine) distance -- a real fiber route is
  longer than the geodesic, but this is a toy/demo topology, not an as-built
  record.
- mount_type per edge (aerial/buried): Topology Zoo does not capture physical
  install method at all. Assigned by a documented heuristic, not measured: an
  edge is "buried" if EITHER endpoint is a "Million-Plus Urban Agglomeration"
  per India's 2011 Census (an externally checkable population threshold, see
  MILLION_PLUS_UAS below), else "aerial". Requiring BOTH endpoints to be
  million-plus was tried first and rejected -- this backbone is mostly
  small-town-to-small-town mesh, so that gave a near-degenerate 165/16
  aerial/buried split. OR gives a workable 73/108 split and is still a
  defensible pattern (a link touching a major metro is plausibly routed
  through that metro's engineered core segment), not a claim about Tata's
  actual as-built infrastructure.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

GML_PATH = Path(__file__).parent.parent / "src" / "storm_reoptimizer" / "data" / "raw" / "TataNld.gml"
OUT_PATH = Path(__file__).parent.parent / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"

# India's 2011 Census "Million-Plus Urban Agglomerations" -- 52 UAs with
# population >= 1,000,000 (https://en.wikipedia.org/wiki/List_of_million-plus_urban_agglomerations_in_India).
# An externally checkable population threshold, not a hand-picked rank cutoff.
# Matched against this graph's (often older/anglicized) node labels via
# MILLION_PLUS_LABEL_ALIASES below where the census name differs from the
# graph's label.
MILLION_PLUS_UAS = {
    "Mumbai", "Delhi", "Kolkata", "Chennai", "Bengaluru", "Hyderabad", "Ahmedabad",
    "Pune", "Surat", "Jaipur", "Kanpur", "Lucknow", "Nagpur", "Ghaziabad", "Indore",
    "Coimbatore", "Kochi", "Patna", "Kozhikode", "Bhopal", "Thrissur", "Vadodara",
    "Agra", "Visakhapatnam", "Malappuram", "Thiruvananthapuram", "Kannur",
    "Ludhiana", "Nashik", "Vijayawada", "Madurai", "Varanasi", "Meerut",
    "Faridabad", "Rajkot", "Jamshedpur", "Jabalpur", "Srinagar", "Asansol",
    "Vasai-Virar", "Allahabad", "Dhanbad", "Aurangabad", "Amritsar", "Jodhpur",
    "Ranchi", "Raipur", "Kollam", "Gwalior", "Durg-Bhilainagar", "Tiruchirappalli",
    "Kota",
}
# This graph's node labels use older/anglicized city names in several cases.
MILLION_PLUS_LABEL_ALIASES = {
    "Bangalore": "Bengaluru",
    "Trivandrum": "Thiruvananthapuram",
    "Cannonore": "Kannur",
    "Baroda": "Vadodara",
    "Nasik": "Nashik",
    "Trichy": "Tiruchirappalli",
    "Thirussur": "Thrissur",
    "Madural": "Madurai",
    "Vijayavada": "Vijayawada",
}


def _is_million_plus(label: str) -> bool:
    return MILLION_PLUS_LABEL_ALIASES.get(label, label) in MILLION_PLUS_UAS


def _parse_gml_blocks(text: str, block_name: str) -> list[dict[str, str]]:
    """Extract every `block_name [ key value ... ]` block's flat key/value
    pairs. Topology Zoo GML files nest only one level (graph > node/edge),
    with no lists or quoted-string escaping inside node/edge blocks, so a
    line-oriented scan is sufficient -- no general GML grammar needed."""
    blocks: list[dict[str, str]] = []
    pattern = re.compile(rf"^\s*{block_name}\s*\[", re.MULTILINE)
    for m in pattern.finditer(text):
        start = m.end()
        depth = 1
        i = start
        while depth > 0:
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
            i += 1
        body = text[start:i - 1]
        fields: dict[str, str] = {}
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            key, _, value = line.partition(" ")
            fields[key] = value.strip().strip('"')
        blocks.append(fields)
    return blocks


def _slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return round(r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)


def convert() -> dict:
    text = GML_PATH.read_text(encoding="utf-8")
    raw_nodes = _parse_gml_blocks(text, "node")
    raw_edges = _parse_gml_blocks(text, "edge")

    node_id_to_slug: dict[str, str] = {}
    seen_slugs: dict[str, str] = {}
    nodes: list[dict] = []
    dropped_node_ids: set[str] = set()
    for n in raw_nodes:
        gml_id = n["id"]
        label = n.get("label", "")
        if "Latitude" not in n or "Longitude" not in n:
            # Topology Zoo "hyperedge" junction markers (label "None"): a
            # topological construct joining 3+ real edges, not a real place.
            # Dropped rather than given fabricated coordinates -- every node
            # kept here is a real named city with real geometry.
            dropped_node_ids.add(gml_id)
            continue
        slug = _slugify(label) if label and label != "None" else f"node_{gml_id}"
        if slug in seen_slugs:
            raise ValueError(
                f"slug collision: {label!r} (id {gml_id}) and "
                f"{seen_slugs[slug]!r} both slugify to {slug!r}"
            )
        seen_slugs[slug] = label or f"<unlabeled id {gml_id}>"
        node_id_to_slug[gml_id] = slug
        nodes.append({
            "id": slug,
            "lat": float(n["Latitude"]),
            "lon": float(n["Longitude"]),
        })
    by_slug = {n["id"]: n for n in nodes}
    if dropped_node_ids:
        print(f"dropped {len(dropped_node_ids)} non-place hyperedge nodes: {sorted(dropped_node_ids)}")

    seen_pairs: set[tuple[str, str]] = set()
    edges: list[dict] = []
    dropped_parallel = 0
    dropped_hyperedge_touching = 0
    dropped_zero_length = 0
    for e in raw_edges:
        if e["source"] in dropped_node_ids or e["target"] in dropped_node_ids:
            dropped_hyperedge_touching += 1
            continue
        src = node_id_to_slug[e["source"]]
        dst = node_id_to_slug[e["target"]]
        pair = (src, dst) if src <= dst else (dst, src)
        if pair in seen_pairs:
            dropped_parallel += 1
            continue
        a, b = by_slug[src], by_slug[dst]
        length_km = _haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        if length_km == 0.0:
            # Source data artifact: two distinct node labels (e.g. "Goa" and
            # its capital "Panjim") sharing identical lat/lon in TataNld.gml.
            # A 0km span isn't a real fiber route -- drop it rather than model
            # a physically meaningless zero-loss link. Neither endpoint is
            # orphaned by this (both have other real edges in this topology).
            dropped_zero_length += 1
            continue
        seen_pairs.add(pair)
        src_label = seen_slugs[src]
        dst_label = seen_slugs[dst]
        # OR, not AND: buried if EITHER endpoint is a million-plus city. AND
        # (both endpoints must be million-plus) was tried first and produced
        # a near-degenerate 165/16 aerial/buried split -- this backbone is
        # mostly small-town-to-small-town mesh, so "both ends are a metro"
        # almost never fires. OR gives a workable 73/108 split and is still
        # a defensible pattern: a link touching even one major metro is
        # plausibly routed through that metro's engineered core/access
        # segment, not just cross-country lashed cable.
        mount_type = "buried" if _is_million_plus(src_label) or _is_million_plus(dst_label) else "aerial"
        edges.append({"src": src, "dst": dst, "length_km": length_km, "mount_type": mount_type})

    print(
        f"nodes: {len(nodes)}  edges: {len(edges)}  "
        f"(dropped {dropped_parallel} parallel duplicates, "
        f"{dropped_hyperedge_touching} edges touching hyperedge nodes, "
        f"{dropped_zero_length} zero-length duplicate-coordinate edges)"
    )
    hub_labels = sorted(seen_slugs[n["id"]] for n in nodes if _is_million_plus(seen_slugs[n["id"]]))
    print(f"million-plus nodes matched ({len(hub_labels)}/{len(nodes)}): {hub_labels}")
    aerial = sum(1 for e in edges if e["mount_type"] == "aerial")
    print(f"mount_type: {aerial} aerial / {len(edges) - aerial} buried")

    return {"graph": {"nodes": nodes, "edges": edges}, "srlgs": []}


if __name__ == "__main__":
    data = convert()
    OUT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
