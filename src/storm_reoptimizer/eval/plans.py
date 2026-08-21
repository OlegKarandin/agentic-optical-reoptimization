# src/storm_reoptimizer/eval/plans.py
"""Translate a route_service candidate into a committable plan (eval design
spec, build order item 3). Deterministic harness code, not a decision: the
agent's surface ends at CHOOSING a candidate.

This is what makes harness-side state advancement real. Without a commit,
spares are never consumed, groomed capacity never shrinks, and hour t+1's
menu does not reflect hour t's action -- so the spare contention T3 and
Decision 1 reason about would never exist in the model, and every T3
rationale would be scored against a state that never moved.

Note what the MCP serialization does and does not carry. A new run comes back
as {oms_sequence, lam, mode_id, gsnr_db, bitrate_gbps} -- no src_node/dst_node
(the in-process apply_candidate path reads those off the typed Placement,
which we do not have over stdio). The run's endpoints are therefore recovered
by walking the OMS chain: an OMS can be traversed in either direction, so a
run's terminal nodes are the ones NOT shared with the neighbouring OMS.

Every server interaction goes through mcp_client.call_tool_json -- no
server-internals imports, per CLAUDE.md's hard seam."""
from __future__ import annotations

from dataclasses import dataclass

from mcp.client import Client

from ..mcp_client import call_tool_json

# The server's SpectrumGrid.default(): 191.4 THz anchor, 100 GHz spacing.
GRID_ANCHOR_HZ = 191.4e12
GRID_SPACING_HZ = 100e9


class PlanTranslationError(ValueError):
    """A candidate that cannot be expressed as a plan against this topology."""


def lam_to_center_freq_hz(lam: int) -> float:
    """The same arithmetic SpectrumGrid.freq(slot) does server-side."""
    return GRID_ANCHOR_HZ + lam * GRID_SPACING_HZ


@dataclass(frozen=True)
class TopologyIndex:
    """The three lookups plan translation needs, read once per connection."""
    oms_by_id: dict[str, dict]
    router_by_site: dict[str, str]
    ip_link_by_lightpath: dict[str, dict]
    service_by_id: dict[str, dict]


async def build_topology_index(client: Client) -> TopologyIndex:
    optical = await call_tool_json(client, "get_topology", {"layer": "optical"})
    ip = await call_tool_json(client, "get_topology", {"layer": "ip"})
    services = await call_tool_json(client, "get_services")
    return TopologyIndex(
        oms_by_id={o["id"]: o for o in optical["oms"]},
        router_by_site={r["site"]: r["id"] for r in ip["routers"]},
        ip_link_by_lightpath={lnk["lightpath_id"]: lnk
                              for lnk in ip["ip_links"]
                              if lnk.get("lightpath_id")},
        service_by_id={s["id"]: s for s in services["services"]},
    )


def _endpoints(oms_by_id: dict[str, dict], oms_sequence: list[str]
               ) -> tuple[str, str]:
    """The two terminal node ids of a run, by chaining the OMS sequence. For
    a single-OMS run the endpoints are that OMS's own; otherwise each end's
    terminal node is the one it does not share with its neighbour."""
    try:
        first = oms_by_id[oms_sequence[0]]
        last = oms_by_id[oms_sequence[-1]]
    except KeyError as exc:
        raise PlanTranslationError(f"unknown OMS {exc} in run") from None
    if len(oms_sequence) == 1:
        return first["src_node_id"], first["dst_node_id"]

    def _free_end(oms: dict, neighbour: dict) -> str:
        ends = {oms["src_node_id"], oms["dst_node_id"]}
        shared = ends & {neighbour["src_node_id"], neighbour["dst_node_id"]}
        free = ends - shared
        if len(free) != 1:
            raise PlanTranslationError(
                f"OMS {oms['id']} and {neighbour['id']} do not chain "
                f"(ends {sorted(ends)}, shared {sorted(shared)})")
        return free.pop()

    return (_free_end(first, oms_by_id[oms_sequence[1]]),
            _free_end(last, oms_by_id[oms_sequence[-2]]))


def _stitch_ip_path(segments: list[tuple[str, str, str]],
                    src_router: str, dst_router: str) -> list[str]:
    """Order (a_router, z_router, ip_link_id) segments into a contiguous walk
    src -> dst. Each segment is usable in either orientation. Mirrors the
    server's own _stitch_ip_path, but raises on a broken walk instead of
    returning a truncated path: a truncated ip_path would fail deep inside
    commit_plan with an opaque message about contiguity."""
    remaining = list(segments)
    path: list[str] = []
    node = src_router
    while node != dst_router:
        for k, (a, z, ip_id) in enumerate(remaining):
            if a == node:
                path.append(ip_id)
                node = z
                remaining.pop(k)
                break
            if z == node:
                path.append(ip_id)
                node = a
                remaining.pop(k)
                break
        else:
            raise PlanTranslationError(
                f"candidate's segments do not form a walk {src_router} -> "
                f"{dst_router}: stalled at {node} with {remaining!r} unused")
    return path


def plan_from_candidate(index: TopologyIndex, candidate: dict,
                        service_id: str, *, prefix: str) -> dict:
    """A committable plan dict for `candidate`. `prefix` must be unique per
    commit within an episode (the runner passes f"eval-{hour}"), because the
    lightpath and IP-link ids are minted from it and the server rejects a
    duplicate id as invalid_plan."""
    service = index.service_by_id.get(service_id)
    if service is None:
        raise PlanTranslationError(f"unknown service {service_id!r}")

    ops: list[dict] = []
    segments: list[tuple[str, str, str]] = []

    for lp_id in candidate["reused_lightpaths"]:
        link = index.ip_link_by_lightpath.get(lp_id)
        if link is None:
            raise PlanTranslationError(
                f"reused lightpath {lp_id!r} has no bound IP link")
        segments.append((link["a_router"], link["z_router"], link["id"]))

    for i, run in enumerate(candidate["new_lightpaths"]):
        lp_id = f"lp-{prefix}-{service_id}-{i}"
        ipl_id = f"ipl-{prefix}-{service_id}-{i}"
        src_node, dst_node = _endpoints(index.oms_by_id, run["oms_sequence"])
        try:
            a = index.router_by_site[src_node]
            z = index.router_by_site[dst_node]
        except KeyError as exc:
            raise PlanTranslationError(
                f"no router at site {exc} for run {run['oms_sequence']}") from None
        ops.append({
            "op": "provision_lightpath",
            "lightpath": {
                "id": lp_id,
                "oms_sequence": list(run["oms_sequence"]),
                "mode_id": run["mode_id"],
                "center_freq_hz": lam_to_center_freq_hz(run["lam"]),
            },
            "ip_link": {"id": ipl_id, "a_router": a, "z_router": z},
        })
        segments.append((a, z, ipl_id))

    ip_path = _stitch_ip_path(segments, service["src_router"],
                              service["dst_router"])
    ops.append({"op": "reroute_service", "service_id": service_id,
                "ip_path": ip_path, "which": "working"})
    return {"ops": ops}


async def validate_candidate(client: Client, plan: dict, *,
                             basis: str, level: str) -> dict:
    """The typed violation list the AGENT retries against. commit_plan
    re-validates regardless, so this call exists to give the decider
    something to react to -- not to protect the model."""
    return await call_tool_json(client, "validate_plan", {
        "plan": plan, "basis": basis, "level": level})


async def commit_candidate(client: Client, plan: dict, *,
                           basis: str, level: str) -> dict:
    """The harness IS the approval gate (CLAUDE.md's "approval-gated"
    framing), so confirm=True is set here and never by a decider. The
    returned intended_snapshot_id is already a scoreable post-commit state --
    snapshot_create() is only needed after inject_failure."""
    return await call_tool_json(client, "commit_plan", {
        "plan": plan, "dry_run": False, "confirm": True,
        "basis": basis, "level": level})
