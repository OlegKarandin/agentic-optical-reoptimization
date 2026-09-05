# tools/find_sut.py
"""Task 13 (T1 spend-or-hold redesign plan): the diagnostic tool used to
CHOOSE the new T1 SUT (service under test) and its claimant corridor.

Two independent modes, on purpose -- one cheap and offline, one live and
expensive:

  --prefilter is pure GIS + a graph walk over the LOCAL topology JSON
    (Shapely LineStrings via geo_mapper.load_edges, networkx over the
    resulting graph). No server, no solver, runs in milliseconds. It answers
    "which sites even have the raw shape (>=2 aerial directions plus a
    genuine buried one) the canonical case needs", ranked by how close each
    sits to the scripted Hudhud track -- so the shortlist favours sites the
    demo's own storm would actually threaten.

  --evaluate is live: it shells out to tools/build_eval_state.py (this
    repo's own env invoking the SIBLING multilayer-optical-mcp conda env's
    python, same idiom tests/conftest.py's local_server_command/
    loaded_state_path fixtures use at :107-112) to pin a candidate SUT --
    and, once the SUT's own real legs are known, a claimant pair per unused
    aerial neighbour -- onto an existing base state, then starts the real
    MCP server against the result and reports the real numbers: per-leg and
    joint p_cut, the route_service menu under a risk-group avoid, and each
    claimant's own exposure. This is what Task 15 actually runs to pick the
    final site.

Runs under THIS repo's env (storm-reoptimizer): imports nothing from
multilayer_optical_network directly and never talks to the live model
except through mcp_client's stdio Client, per CLAUDE.md's hard seam.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from storm_reoptimizer.eval.cone import (                # noqa: E402
    cone_polygon, p_cut_region, p_cut_service, radial_offset_km,
)
from storm_reoptimizer.eval.decisions import ConstraintDecision  # noqa: E402
from storm_reoptimizer.eval.ledger import spares_needed          # noqa: E402
from storm_reoptimizer.eval.runner import (                      # noqa: E402
    horizon_risk_group_asset_ids, menu_with_path_facts, service_geometry,
)
from storm_reoptimizer.eval.scenario_file import (        # noqa: E402
    ConeAtHorizon, Issuance,
)
from storm_reoptimizer.events import amphan_track, hudhud_track  # noqa: E402
from storm_reoptimizer.events.filters import get_filter   # noqa: E402
from storm_reoptimizer.geo_mapper import load_edges       # noqa: E402
from storm_reoptimizer.mcp_client import call_tool_json, connect_server  # noqa: E402

# Same workspace workaround derive_episodes.py and tests/conftest.py already
# document: the sibling server repo's console script never gets a .exe here
# (Cyrillic path trips the editable-install bug), so it's launched by
# invoking its main() directly via `python -c`, in the SIBLING conda env.
# Duplicated here (not imported from derive_episodes.py) because tools/ is
# not a package -- every tool in this directory defines this pair itself.
DEFAULT_SERVER_PYTHON = (
    r"C:\Users\olegk\miniconda3\envs\multilayer-optical-mcp\python.exe")


def _server_command(server_python: str) -> list[str]:
    return [
        server_python, "-c",
        ("import sys; sys.argv = ['multilayer-optical-mcp', *sys.argv[1:]]; "
         "from multilayer_optical_mcp.server import main; main()"),
    ]


BUILD_EVAL_STATE_SCRIPT = Path(__file__).resolve().parent / "build_eval_state.py"

# --near-track's registry. Both are already-shipped scripted tracks (Hudhud
# replaced Amphan as the demo track -- hudhud_track.py's own module
# docstring -- but Amphan's ANCHORS are still real and importable).
TRACK_ANCHORS = {
    "hudhud": hudhud_track.ANCHORS,
    "amphan": amphan_track.ANCHORS,
}


def _node_coords(topology_path: str | Path) -> dict[str, tuple[float, float]]:
    """node_id -> (lat, lon), parsed straight from the topology JSON -- the
    same two-line convention runner.service_geometry's own `coords` local
    uses, kept local here rather than promoted to a shared helper since nowhere
    else needs it as a standalone function."""
    data = json.loads(Path(topology_path).read_text(encoding="utf-8-sig"))
    return {n["id"]: (n["lat"], n["lon"]) for n in data["graph"]["nodes"]}


# ---------------------------------------------------------------------------
# --prefilter: pure GIS + graph walk, no server
# ---------------------------------------------------------------------------

def _escapes_aerial_pair(graph: nx.Graph, site: str, buried: str,
                         a1: str, a2: str) -> bool | None:
    """Whether `buried` is a GENUINE alternate corridor out of `site`, not
    merely a stub that would have to double back through the two aerial
    neighbours (a1, a2) -- standing in for a hypothetical protected
    service's own working+protection legs -- to reconnect to anything.

    Removes the direct site<->buried edge (the trivial one-hop return) and
    asks networkx for the network's own NEXT-BEST way to reconnect `buried`
    to `site`. True iff that shortest path touches neither a1 nor a2 as an
    intermediate node -- i.e. the buried leg's onward connectivity does not
    itself depend on the very corridors a storm would be threatening. None
    when no such alternate path exists at all (an isolated buried stub, which
    is itself a useful thing for a human reader to see -- not conflated with
    a hard False).

    This is new graph-walk logic (the brief calls it out as the one thing in
    this file that isn't just composition of existing pieces) -- the exact
    semantics of "the shortest path from b onward avoids a1/a2" were
    underspecified in the brief and this is the interpretation chosen; see
    the task report for the reasoning. Not exercised by the one given test
    (which checks row shape/counts, not this field's values)."""
    trimmed = graph.copy()
    if trimmed.has_edge(site, buried):
        trimmed.remove_edge(site, buried)
    try:
        path = nx.shortest_path(trimmed, buried, site)
    except nx.NetworkXNoPath:
        return None
    return a1 not in path and a2 not in path


def prefilter(topology_path: str | Path, *, min_aerial: int = 2,
             near_track: str = "hudhud") -> list[dict]:
    """Every site with >= `min_aerial` aerial edges and >= 1 buried edge --
    a candidate to host the canonical case (protected service, both legs
    aerial inside a storm, plus a genuine buried escape route) -- sorted by
    distance to the nearest point of `near_track`'s scripted track.

    Each row: site, its aerial/buried neighbours (and counts), the track
    distance, and for every (buried neighbour, aerial-neighbour pair)
    combination, whether that buried leg's onward connectivity survives
    without the two aerial legs (see `_escapes_aerial_pair`)."""
    if near_track not in TRACK_ANCHORS:
        raise ValueError(
            f"unknown near_track {near_track!r}; known: "
            f"{sorted(TRACK_ANCHORS)}")
    edges = load_edges(topology_path)
    node_coords = _node_coords(topology_path)

    graph = nx.Graph()
    aerial: dict[str, set[str]] = {}
    buried: dict[str, set[str]] = {}
    for edge in edges:
        graph.add_edge(edge.src, edge.dst, mount_type=edge.mount_type)
        bucket = (aerial if edge.mount_type == "aerial"
                  else buried if edge.mount_type == "buried" else None)
        if bucket is None:
            continue
        bucket.setdefault(edge.src, set()).add(edge.dst)
        bucket.setdefault(edge.dst, set()).add(edge.src)

    anchors = TRACK_ANCHORS[near_track]
    rows: list[dict] = []
    for site in sorted(set(aerial) | set(buried)):
        aerial_n = sorted(aerial.get(site, ()))
        buried_n = sorted(buried.get(site, ()))
        if len(aerial_n) < min_aerial or not buried_n:
            continue
        lat, lon = node_coords[site]
        track_distance_km = min(
            radial_offset_km(a["lat"], a["lon"], lat, lon) for a in anchors)
        escape_checks = [
            {"buried": b, "aerial_pair": [a1, a2],
             "escapes": _escapes_aerial_pair(graph, site, b, a1, a2)}
            for b in buried_n for a1, a2 in combinations(aerial_n, 2)
        ]
        rows.append({
            "site": site,
            "aerial_neighbours": aerial_n,
            "buried_neighbours": buried_n,
            "n_aerial": len(aerial_n),
            "n_buried": len(buried_n),
            "track_distance_km": round(track_distance_km, 3),
            "escape_checks": escape_checks,
        })
    rows.sort(key=lambda r: r["track_distance_km"])
    return rows


def _print_prefilter_table(rows: list[dict]) -> None:
    header = (f"{'site':<16}{'aerial':>7}{'buried':>7}{'track_km':>11}  "
              f"{'aerial_neighbours':<40}{'buried_neighbours':<30}"
              f"{'escapes_ok/total'}")
    print(header)
    print("-" * len(header))
    for row in rows:
        checks = row["escape_checks"]
        ok = sum(1 for c in checks if c["escapes"] is True)
        print(f"{row['site']:<16}{row['n_aerial']:>7}{row['n_buried']:>7}"
              f"{row['track_distance_km']:>11.1f}  "
              f"{', '.join(row['aerial_neighbours']):<40}"
              f"{', '.join(row['buried_neighbours']):<30}"
              f"{ok}/{len(checks)}")


# ---------------------------------------------------------------------------
# --evaluate: live, shells out to build_eval_state.py, then a real server
# ---------------------------------------------------------------------------

def _run_build_eval_state(topology_path: str | Path, base_state: str | Path,
                          out_path: str | Path, pins: list[dict], *,
                          server_python: str) -> None:
    """Shell out to tools/build_eval_state.py's --base-state mode (same
    idiom tests/conftest.py:107-112 uses for build_storm_state.py/
    build_eval_state.py: invoke the SIBLING conda env's python directly on
    the script path, since the console script itself doesn't install here --
    see mcp_client.DEFAULT_SERVER_COMMAND's docstring). Raises with the
    subprocess's own stdout/stderr on failure."""
    args = [server_python, str(BUILD_EVAL_STATE_SCRIPT),
            "--topology", str(topology_path), "--out", str(out_path),
            "--seed", "0", "--base-state", str(base_state)]
    for pin in pins:
        args += ["--pin", json.dumps(pin)]
    proc = subprocess.run(args, env=dict(os.environ), capture_output=True,
                          text=True, timeout=1800, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"build_eval_state failed (rc={proc.returncode}) building "
            f"{out_path} from base={base_state} "
            f"pins={[p['id'] for p in pins]}:\n"
            f"stdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-2000:]}")


async def evaluate(topology_path: str | Path, base_state_path: str | Path,
                   site: str, dst: str, *, demand_gbps: float = 300.0,
                   width_km: float = 90.0, damage_radius_km: float = 74.0,
                   server_python: str | None = None,
                   workdir: str | None = None) -> dict:
    """Build a probe network state on top of `base_state_path` (a real,
    already-built state, e.g. eval/states/loaded-s17.json) with a candidate
    SUT `site`<->`dst` (protected, `demand_gbps`) pinned, discover its REAL
    working/protection legs, pin a claimant pair `site`<->A for every
    aerial neighbour of `site` NOT used by those legs, start the real
    server against the result, and report the real geometric/routing
    numbers Task 15 needs to pick the final site.

    Two build_eval_state.py invocations, not one: which of `site`'s aerial
    neighbours the SUT's own working/protection legs land on is a live
    solver decision (spectrum/disjoint-pair search), not something this
    tool can predict -- so stage 1 pins ONLY the SUT and stage 2 pins the
    claimants on top of stage 1's own output, once the legs are known
    (mirrors tools/derive_episodes.py's build_t1b_spec, which solves its own
    free parameter against a live read for the same reason)."""
    server_python = server_python or DEFAULT_SERVER_PYTHON
    server_command = _server_command(server_python)
    env = dict(os.environ)
    edges = load_edges(topology_path)
    node_coords = _node_coords(topology_path)
    site_lat, site_lon = node_coords[site]

    with tempfile.TemporaryDirectory(dir=workdir) as tmp_str:
        tmp = Path(tmp_str)
        sut_id = f"find-sut-{site}-{dst}"
        sut_pin = {"id": sut_id, "src": site, "dst": dst,
                  "demand_gbps": demand_gbps, "protected": True}
        stage1 = tmp / "stage1.json"
        _run_build_eval_state(topology_path, base_state_path, stage1,
                              [sut_pin], server_python=server_python)

        aerial_neighbours = sorted(
            (edge.dst if edge.src == site else edge.src)
            for edge in edges
            if edge.mount_type == "aerial" and site in (edge.src, edge.dst))

        async with connect_server(
                topology_path, server_command=server_command, env=env,
                extra_args=["--state", str(stage1)]) as client:
            geometry = await service_geometry(client, topology_path,
                                              edges=edges)
            working_oms = geometry.path_oms[sut_id]["working"]
            protection_oms = geometry.path_oms[sut_id]["protection"]
            used_edges = {
                tuple(sorted(geometry.oms_nodes[oms_id]))
                for oms_id in (*working_oms, *protection_oms)
                if oms_id in geometry.oms_nodes}

        claimant_neighbours = [
            a for a in aerial_neighbours
            if tuple(sorted((site, a))) not in used_edges]
        claimant_pins = [
            {"id": f"find-sut-claimant-{site}-{a}", "src": site, "dst": a,
             "demand_gbps": 100.0, "protected": False}
            for a in claimant_neighbours]

        if claimant_pins:
            stage2 = tmp / "stage2.json"
            _run_build_eval_state(topology_path, stage1, stage2,
                                  claimant_pins, server_python=server_python)
        else:
            stage2 = stage1

        async with connect_server(
                topology_path, server_command=server_command, env=env,
                extra_args=["--state", str(stage2)]) as client:
            geometry = await service_geometry(client, topology_path,
                                              edges=edges)
            working_spans = geometry.cuttable_spans.get(sut_id, ())
            protection_spans = geometry.protection_cuttable_spans.get(
                sut_id, ())

            p_cut_working = round(p_cut_region(
                working_spans, site_lat, site_lon, width_km,
                damage_radius_km), 6)
            p_cut_protection = round(p_cut_region(
                protection_spans, site_lat, site_lon, width_km,
                damage_radius_km), 6)
            p_cut_joint = round(p_cut_service(
                working_spans, protection_spans, site_lat, site_lon,
                width_km, damage_radius_km), 6)

            cone = ConeAtHorizon(
                cone=cone_polygon(site_lat, site_lon, width_km),
                width_km=width_km, center={"lat": site_lat, "lon": site_lon})
            optical = await call_tool_json(client, "get_topology",
                                           {"layer": "optical"})
            filter_fn = get_filter("storm")
            fiber_ids = horizon_risk_group_asset_ids(
                cone, damage_radius_km, edges=edges, oms=optical["oms"],
                filter_fn=filter_fn)
            rg_id = f"rg_find_sut_{site}_{dst}"
            await call_tool_json(client, "define_risk_group", {
                "rg_id": rg_id, "asset_ids": fiber_ids,
                "metadata": {"tool": "find_sut", "site": site, "dst": dst}})

            constraints = ConstraintDecision(
                avoid={"risk_groups": [rg_id]}, reasoning="find_sut probe")
            menu = await call_tool_json(
                client, "route_service",
                constraints.route_service_args(sut_id))
            issuance = Issuance(issued_at="probe", horizons={"probe": cone})
            annotated = menu_with_path_facts(
                menu, geometry, sut_id, issuance=issuance,
                damage_radius_km=damage_radius_km, demand_gbps=demand_gbps)

            candidates = [{
                "candidate_label": f"candidate_{i}",
                "lever": cand["lever"],
                "changes_working_path":
                    cand["path_delta"]["changes_working_path"],
                "collides_with_protection":
                    cand["collides_with_protection"]["collides"],
                "spares_needed": spares_needed(cand, geometry.oms_nodes),
                "residual_p_cut": cand["residual_exposure"]["probe"]["p_cut"],
            } for i, cand in enumerate(annotated.get("candidates") or [])]

            claimants = []
            for pin in claimant_pins:
                claimant_id = pin["id"]
                claimant_spans = geometry.cuttable_spans.get(claimant_id, ())
                a_lat, a_lon = node_coords[pin["dst"]]
                # "displaced toward the claimant corridor": the midpoint
                # between the site and the claimant's own far endpoint --
                # the same arithmetic-mean convention runner.service_geometry
                # already uses for a service's own representative point,
                # rather than a bearing/distance placement (cone.py exposes
                # no forward bearing-to-point helper, only
                # radial_offset_km's distance; introducing one just to
                # re-derive what averaging already gives here would be
                # unjustified complexity for a diagnostic display value).
                displaced_lat = (site_lat + a_lat) / 2.0
                displaced_lon = (site_lon + a_lon) / 2.0
                claimants.append({
                    "id": claimant_id, "dst": pin["dst"],
                    "p_cut_at_site_cone": round(p_cut_region(
                        claimant_spans, site_lat, site_lon, width_km,
                        damage_radius_km), 6),
                    "p_cut_at_displaced_cone": round(p_cut_region(
                        claimant_spans, displaced_lat, displaced_lon,
                        width_km, damage_radius_km), 6),
                })

            return {
                "site": site, "dst": dst, "service_id": sut_id,
                "demand_gbps": demand_gbps, "width_km": width_km,
                "damage_radius_km": damage_radius_km,
                "working_oms": list(working_oms),
                "protection_oms": list(protection_oms),
                "working_cuttable_spans": [
                    [list(pt) for pt in span] for span in working_spans],
                "protection_cuttable_spans": [
                    [list(pt) for pt in span] for span in protection_spans],
                "p_cut_working": p_cut_working,
                "p_cut_protection": p_cut_protection,
                "p_cut_joint": p_cut_joint,
                "risk_group_id": rg_id,
                "risk_group_asset_count": len(fiber_ids),
                "menu_status": annotated.get("status"),
                "candidates": candidates,
                "claimants": claimants,
                "aerial_neighbours": aerial_neighbours,
                "aerial_neighbours_used_by_sut": sorted(
                    a for a in aerial_neighbours
                    if a not in claimant_neighbours),
            }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="find_sut")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prefilter", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--min-aerial", type=int, default=2)
    parser.add_argument("--near-track", default="hudhud",
                        choices=sorted(TRACK_ANCHORS))
    parser.add_argument("--base-state",
                        help="--evaluate only: an existing state file to "
                             "pin the candidate SUT/claimants onto")
    parser.add_argument("--site", help="--evaluate only")
    parser.add_argument("--dst", help="--evaluate only")
    parser.add_argument("--demand", type=float, default=300.0)
    parser.add_argument("--width-km", type=float, default=90.0)
    parser.add_argument("--damage-radius-km", type=float, default=74.0)
    parser.add_argument("--server-python", default=None)
    args = parser.parse_args()

    if args.prefilter:
        rows = prefilter(args.topology, min_aerial=args.min_aerial,
                         near_track=args.near_track)
        _print_prefilter_table(rows)
        return

    if not (args.base_state and args.site and args.dst):
        parser.error("--evaluate requires --base-state, --site and --dst")
    result = asyncio.run(evaluate(
        args.topology, args.base_state, args.site, args.dst,
        demand_gbps=args.demand, width_km=args.width_km,
        damage_radius_km=args.damage_radius_km,
        server_python=args.server_python))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
