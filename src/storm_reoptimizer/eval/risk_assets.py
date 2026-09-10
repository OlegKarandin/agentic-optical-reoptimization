# src/storm_reoptimizer/eval/risk_assets.py
"""A horizon's risk group, made nameable (decider allocation redesign spec,
2026-09-10, section 5.1).

`risk_group_ids` tells the agent a group EXISTS; it never told it what was
in one. D1's consequence, in the 2026-09-09 trace: the agent probed, learned
the full group has no solution, and avoided the full group anyway -- because
the only avoid it could express was the whole thing. Its gold avoid is eight
named fibres, and the named group disconnects satna outright, so "avoid the
group" and "reach the depot" were mutually exclusive and nothing in the
payload could say so.

Three facts per asset, no more. The id, so it can be named in
`avoid.assets`. That single span's own `p_cut` at this horizon's cone
(`cone.p_cut_region` over one span -- the same integral the service rows
use, restricted to one segment), so "which of these actually matters" is
arithmetic the agent does not have to guess at. And `on`, whether the span
lies on the actionable service's own working or protection path, so
"avoiding my own corridor" is visible before it removes every stay-put
candidate from the menu.

Pure: no server, no event vocabulary, no scenario object. The runner supplies
the group's asset ids (it already computes them to DEFINE the group) and the
topology's coordinates."""
from __future__ import annotations

from .cone import Segment, p_cut_region

_FIBER_PREFIX = "fiber_"


def fiber_span_index(oms: list[dict], coords: dict[str, tuple[float, float]]
                     ) -> dict[str, tuple[tuple[str, str], Segment]]:
    """`fiber_id -> ((src_node, dst_node), span)` for every `fiber_*` element
    of every OMS whose two endpoints the LOCAL topology can place.

    Every fibre of one OMS shares that OMS's span: they are strands of the
    same cable on the same route, and the storm cuts the cable. A node the
    local topology lacks yields no entry at all rather than a span computed
    from the half it does know -- the same policy
    `ServiceGeometry.unmapped_nodes` states."""
    index: dict[str, tuple[tuple[str, str], Segment]] = {}
    for record in oms:
        a, b = record["src_node_id"], record["dst_node_id"]
        if a not in coords or b not in coords:
            continue
        span = (coords[a], coords[b])
        for element in record.get("elements") or ():
            if element.startswith(_FIBER_PREFIX):
                index[element] = ((a, b), span)
    return index


def path_edges(nodes) -> set[frozenset[str]]:
    """The unordered node pairs a path's node walk traverses.

    UNORDERED because the local topology's edge direction and the server's
    OMS direction are independent facts that agree only by accident --
    `runner.horizon_risk_group_asset_ids` matches both orders for exactly
    this reason."""
    nodes = list(nodes or ())
    return {frozenset({a, b}) for a, b in zip(nodes, nodes[1:]) if a != b}


def risk_group_rows(asset_ids, index, *, center_lat: float, center_lon: float,
                    width_km: float, damage_radius_km: float,
                    working_edges: set, protection_edges: set) -> list[dict]:
    """One row per asset in the group, most likely to be cut first.

    `p_cut` is `cone.p_cut_region` over that ONE span -- deliberately not a
    joint or a service-level probability: this is a statement about a fibre,
    and the rows are what an `avoid.assets` list is drawn from. An asset the
    index cannot place is skipped rather than reported at zero, so a gap
    reads as a gap."""
    rows = []
    for asset_id in asset_ids:
        placed = index.get(asset_id)
        if placed is None:
            continue
        (a, b), span = placed
        edge = frozenset({a, b})
        rows.append({
            "asset_id": asset_id,
            "p_cut": round(p_cut_region((span,), center_lat, center_lon,
                                        width_km, damage_radius_km), 3),
            "on": ("working" if edge in working_edges
                   else "protection" if edge in protection_edges else "none"),
        })
    rows.sort(key=lambda r: (-r["p_cut"], r["asset_id"]))
    return rows
