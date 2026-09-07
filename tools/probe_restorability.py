# tools/probe_restorability.py
"""Ask the harness's own probe (eval/probe.py) one question against a state
file: what would route_service offer `--service` if the OMS between the
`--corridor A:B` endpoints (all of them) were in an avoided risk group.

This is spec §8 step 2's live feasibility check for T3's zero-spare groom,
kept as a tool so the answer can be re-asked after any state rebuild. Runs
in this repo's env against a live server, same seam as every tool here."""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import derive_episodes as de  # noqa: E402

from storm_reoptimizer.eval.probe import answer_probe  # noqa: E402
from storm_reoptimizer.eval.runner import service_geometry  # noqa: E402
from storm_reoptimizer.eval.scenario_file import ConeAtHorizon, Issuance  # noqa: E402
from storm_reoptimizer.geo_mapper import load_edges  # noqa: E402
from storm_reoptimizer.mcp_client import call_tool_json, connect_server  # noqa: E402


def _corridor_fibers(oms_by_id: dict, a: str, b: str) -> list[str]:
    ids = [o["id"] for o in oms_by_id.values()
           if {o["src_node_id"], o["dst_node_id"]} == {a, b}]
    if not ids:
        raise SystemExit(f"no OMS between {a!r} and {b!r}")
    return sorted(e for oms_id in ids for e in oms_by_id[oms_id]["elements"]
                  if e.startswith("fiber_"))


async def probe(args) -> dict:
    server_command = (json.loads(args.server_command) if args.server_command
                      else de.DEFAULT_SERVER_COMMAND)
    edges = load_edges(args.topology)
    async with connect_server(args.topology, server_command=server_command,
                              env=dict(os.environ),
                              extra_args=["--state", args.state]) as client:
        geometry = await service_geometry(client, args.topology, edges=edges)
        optical = await call_tool_json(client, "get_topology", {"layer": "optical"})
        oms_by_id = {o["id"]: o for o in optical["oms"]}
        assets = sorted({f for c in args.corridor
                         for f in _corridor_fibers(oms_by_id, *c.split(":"))}
                        | set(args.avoid_extra or []))
        rg_id = f"rg_probe_{args.service}"
        await call_tool_json(client, "define_risk_group", {
            "rg_id": rg_id, "asset_ids": assets,
            "metadata": {"tool": "probe_restorability"}})
        services = (await call_tool_json(client, "get_services"))["services"]
        demands = {s["id"]: float(s["demand_gbps"]) for s in services}
        # The probe needs an issuance only for residual_exposure; a dummy
        # cone far from everything makes that all-zero and irrelevant here.
        issuance = Issuance(issued_at="probe", horizons={"probe": ConeAtHorizon(
            cone={"type": "Polygon", "coordinates": []}, width_km=1.0,
            center={"lat": 0.0, "lon": 0.0})})
        answer = await answer_probe(
            functools.partial(call_tool_json, client), service_id=args.service,
            risk_group_id=rg_id, geometry=geometry, issuance=issuance,
            damage_radius_km=1.0, demands=demands)
        return {"service": args.service, "avoided_assets": assets,
                "working_oms": list(geometry.path_oms.get(args.service, {}).get("working", ())),
                **answer.to_dict()}


def main() -> None:
    p = argparse.ArgumentParser(prog="probe_restorability")
    p.add_argument("--topology", default=de.DEFAULT_TOPOLOGY)
    p.add_argument("--state", required=True)
    p.add_argument("--service", required=True)
    p.add_argument("--corridor", action="append", required=True, metavar="A:B",
                   help="put every fiber of the OMS between A and B in the avoided group; repeatable")
    p.add_argument("--avoid-extra", action="append", default=None, metavar="ASSET_ID")
    p.add_argument("--server-command", default=None, help="JSON list")
    args = p.parse_args()
    print(json.dumps(asyncio.run(probe(args)), indent=2))


if __name__ == "__main__":
    main()
