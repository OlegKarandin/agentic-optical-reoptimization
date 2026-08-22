# tools/probe_episode.py
"""Episode-authoring probe (eval design spec, "Rehearsal: author every
episode by walking it").

Episode YAML cannot be written blind: "storm-svc-1 at the cone's edge (~25%
containment) while svc-b is dead-centre (~90%)" is a claim about real
coordinates in the real loaded state, and "D1's menu contains no ip_reroute"
is a claim about a real menu. This prints those numbers so the author writes
down what is true rather than what sounds right.

Runs in THIS repo's env over MCP -- it is NOT the offline-build exception and
imports nothing from multilayer_optical_network."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from storm_reoptimizer.eval.cone import cone_polygon, cut_probability, radial_offset_km
from storm_reoptimizer.eval.ledger import pairs_needed
from storm_reoptimizer.eval.runner import service_points
from storm_reoptimizer.events.filters import get_filter
from storm_reoptimizer.geo_mapper import load_edges, map_geo_event_to_assets
from storm_reoptimizer.mcp_client import call_tool_json, connect_server


async def probe(topology: Path, state: Path, service: str, lat: float,
                lon: float, width_km: float, damage_radius_km: float,
                avoid: dict, server_command: list[str] | None,
                env: dict | None) -> None:
    async with connect_server(topology, server_command=server_command,
                              env=env,
                              extra_args=["--state", str(state)]) as client:
        points = await service_points(client, topology)
        services = (await call_tool_json(client, "get_services"))["services"]

        print(f"\n=== exposure at cone ({lat}, {lon}) width {width_km} km ===")
        print(f"{'service':<24}{'demand':>8}{'offset_km':>11}{'p_cut':>8}")
        for svc in sorted(services, key=lambda s: s["id"]):
            point = points.get(svc["id"])
            if point is None:
                continue
            offset = radial_offset_km(lat, lon, *point)
            p_cut = cut_probability(offset, width_km, damage_radius_km)
            print(f"{svc['id']:<24}{svc['demand_gbps']:>8.0f}"
                  f"{offset:>11.1f}{p_cut:>8.2f}")

        edges = load_edges(topology)
        exposed = map_geo_event_to_assets(
            cone_polygon(lat, lon, width_km), edges, get_filter("storm"))
        print(f"\n=== exposed aerial edges ({len(exposed)}) ===")
        for edge in exposed:
            print(f"  {edge.src} <-> {edge.dst}")

        menu = await call_tool_json(client, "route_service", {
            "service_id": service, "protected": False, "basis": "physical",
            "level": "link", "best_effort": False, "avoid": avoid})
        print(f"\n=== route_service({service}, avoid={avoid}) -> "
              f"{menu['status']} ===")
        for i, cand in enumerate(menu["candidates"]):
            print(f"  candidate_{i}: lever={cand['lever']:<16} "
                  f"restored={cand['restored_gbps']:>6.0f}G "
                  f"shortfall={cand['shortfall_gbps']:>6.0f}G "
                  f"pairs={pairs_needed(cand)}")
            print(f"      {json.dumps(cand['cost_vector'])}")
        if not menu["candidates"]:
            print("  (empty menu)")


def main() -> None:
    p = argparse.ArgumentParser(prog="probe_episode")
    p.add_argument("--topology", required=True, type=Path)
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--service", default="storm-svc-1")
    p.add_argument("--lat", required=True, type=float)
    p.add_argument("--lon", required=True, type=float)
    p.add_argument("--width-km", required=True, type=float)
    p.add_argument("--damage-radius-km", default=74.0, type=float)
    p.add_argument("--avoid", default="{}",
                   help='JSON, e.g. \'{"risk_groups": ["rg_t3"]}\'')
    p.add_argument("--server-command", default=None,
                   help="JSON list; see mcp_client.DEFAULT_SERVER_COMMAND")
    args = p.parse_args()
    asyncio.run(probe(
        args.topology, args.state, args.service, args.lat, args.lon,
        args.width_km, args.damage_radius_km, json.loads(args.avoid),
        json.loads(args.server_command) if args.server_command else None,
        None))


if __name__ == "__main__":
    main()
