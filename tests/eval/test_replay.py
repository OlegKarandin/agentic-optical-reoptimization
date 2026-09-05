"""Post-cut restoration replay, pure (eval design spec, T1 spend-or-hold
redesign, section 5.1). No server: `counting` is a hand-rolled fake and
`index_factory` builds a TopologyIndex straight from it, exactly as
`build_topology_index` would from a real client -- the point is to exercise
`restore_after_cuts`'s own orchestration logic (scope, ordering, gating,
candidate selection, affordability, commit bookkeeping), not the MCP
plumbing that `tests/eval/test_runner.py`'s live tests already cover."""
import asyncio
import textwrap

from storm_reoptimizer.eval.ledger import SpareLedger
from storm_reoptimizer.eval.plans import TopologyIndex
from storm_reoptimizer.eval.replay import restore_after_cuts
from storm_reoptimizer.eval.runner import ServiceGeometry
from storm_reoptimizer.eval.scenario_file import ConeAtHorizon, Issuance, load_scenario

REPLAY_SCENARIO_YAML = textwrap.dedent("""
    id: REPLAY_TEST
    seed: 1
    state_file: eval/states/loaded-s17.json
    service_under_test: s
    track: hudhud
    hours: [t0, t1, t2, t3, t4]
    decision_hour: t1
    lead_time_hours: 2
    spares_on_hand: 1
    depot_site: tirupati
    spare_inventory: {tirupati: 1}
    damage_radius_km: 20
    reference_avoid: {}
    forecast:
      t0:
        t3: {cone: {type: Polygon, coordinates: [[[81.0, 25.0], [81.1, 25.0], [81.1, 25.1], [81.0, 25.0]]]}, width_km: 90, center: {lat: 25.0, lon: 81.0}}
    realized: {}
    gold:
      survived: []
      max_spares_wasted: 1
      decision_at_t0: wait
      label: wait
      rationale: replay test fixture; not scored
    flip_variable: [test]
    metadata:
      cone_width_km: 90
      cone_motion_kmh: 20
      n_future_claimants: 1
      exposure_horizon_hours: 2
      spares_on_hand: 1
      claimant_services: [c-fwd, c-rev]
      claimed_competing_ecar_gbps: 50.0
      claimed_competing_ecar_at: t3
""")


# One OMS, "oms_new" (tirupati<->nellore), is the only route every service's
# menu below can take -- a real new lightpath there is the ONLY workable
# candidate, and it charges one spare transponder at EACH site.
MENU_CANDIDATES = [
    # The inert reuse: no reused/new lightpaths at all, so its resolved OMS
    # set is empty -- identical to the (also empty) working-path OMS set
    # every service in this fixture's `geometry` is given below, so
    # `path_delta.changes_working_path` is False and the replay must never
    # pick it.
    {"lever": "ip_reroute", "reused_lightpaths": [], "new_lightpaths": [],
     "shortfall_gbps": 0.0},
    # The real new lightpath: moves the service onto oms_new and needs one
    # spare transponder at each of tirupati and nellore.
    {"lever": "optical_reroute", "reused_lightpaths": [],
     "new_lightpaths": [{"oms_sequence": ["oms_new"], "lam": 5,
                         "mode_id": "M", "gsnr_db": 10.0,
                         "bitrate_gbps": 300.0}],
     "shortfall_gbps": 0.0},
]


def _scenario(tmp_path):
    path = tmp_path / "REPLAY_TEST.yaml"
    path.write_text(REPLAY_SCENARIO_YAML, encoding="utf-8")
    return load_scenario(path)


def _geometry():
    empty_path = {"working": (), "protection": ()}
    return ServiceGeometry(
        points={}, paths={}, oms_nodes={"oms_new": ["tirupati", "nellore"]},
        unmapped_nodes={}, cuttable_spans={},
        endpoint_sites={"s": ("tirupati", "nellore"),
                       "c-fwd": ("tirupati", "nellore"),
                       "c-rev": ("nellore", "tirupati")},
        oms_sequences={}, cuttable_span_by_oms={},
        path_oms={"s": empty_path, "c-fwd": empty_path, "c-rev": empty_path},
        protection_cuttable_spans={})


def _issuance():
    return Issuance(issued_at="t1", horizons={"t3": ConeAtHorizon(
        cone={"type": "Polygon",
              "coordinates": [[[81.0, 25.0], [81.1, 25.0], [81.1, 25.1],
                               [81.0, 25.0]]]},
        width_km=90.0, center={"lat": 25.0, "lon": 81.0})})


class _FakeCounting:
    """Answers exactly the tool calls `restore_after_cuts` and its own
    `index_factory` (built from the same fake, below) can make -- no server,
    no `mcp_client`."""

    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []

    async def call(self, name, arguments=None, **kw):
        self.calls.append((name, arguments))
        if name == "route_service":
            return {"status": "solution",
                    "candidates": [dict(c) for c in MENU_CANDIDATES]}
        if name == "validate_plan":
            return {"ok": True}
        if name == "commit_plan":
            return {"status": "committed", "intended_snapshot_id": "snap"}
        if name == "get_topology":
            layer = (arguments or {}).get("layer")
            if layer == "optical":
                return {"oms": [{"id": "oms_new", "src_node_id": "tirupati",
                                 "dst_node_id": "nellore",
                                 "elements": ["fiber_x"]}]}
            if layer == "ip":
                return {"routers": [{"site": "tirupati", "id": "router_tirupati"},
                                    {"site": "nellore", "id": "router_nellore"}],
                        "ip_links": []}
            raise AssertionError(f"unexpected layer {layer!r}")
        if name == "get_lightpaths":
            return []
        if name == "get_services":
            return {"services": [
                {"id": "s", "demand_gbps": 300.0,
                 "src_router": "router_tirupati", "dst_router": "router_nellore"},
                {"id": "c-fwd", "demand_gbps": 100.0,
                 "src_router": "router_tirupati", "dst_router": "router_nellore"},
                {"id": "c-rev", "demand_gbps": 100.0,
                 "src_router": "router_nellore", "dst_router": "router_tirupati"},
            ]}
        raise AssertionError(f"unexpected call {name!r}")


def _index_factory(counting):
    async def factory():
        optical = await counting.call("get_topology", {"layer": "optical"})
        ip = await counting.call("get_topology", {"layer": "ip"})
        services = await counting.call("get_services")
        return TopologyIndex(
            oms_by_id={o["id"]: o for o in optical["oms"]},
            router_by_site={r["site"]: r["id"] for r in ip["routers"]},
            ip_link_by_lightpath={lnk["lightpath_id"]: lnk
                                  for lnk in ip["ip_links"]
                                  if lnk.get("lightpath_id")},
            service_by_id={s["id"]: s for s in services["services"]})
    return factory


def _ledger():
    return SpareLedger(inventory={"tirupati": 1}, depot_site="tirupati",
                       oms_nodes={"oms_new": ["tirupati", "nellore"]})


def _run(tmp_path, *, affected, priority):
    scenario = _scenario(tmp_path)
    counting = _FakeCounting()
    ledger = _ledger()
    return asyncio.run(restore_after_cuts(
        counting, scenario=scenario, hour="t1", hour_index=1,
        affected=affected, priority=priority, ledger=ledger,
        geometry=_geometry(), issuance=_issuance(), rg_for_cut_hour=None,
        index_factory=_index_factory(counting)))


def test_the_stated_priority_wins_the_last_spare(tmp_path):
    actions, records = _run(tmp_path, affected=["c-fwd", "s"],
                            priority=("c-fwd",))
    by_service = {r["service_id"]: r for r in records}
    assert by_service.keys() == {"c-fwd", "s"}
    assert by_service["c-fwd"]["outcome"] == "restored"
    assert by_service["c-fwd"]["effective_at_hour"] == "t3"    # t1 index 1 + lead 2
    assert by_service["c-fwd"]["lever"] == "optical_reroute"
    assert by_service["c-fwd"]["spares"] == {"tirupati": 1, "nellore": 1}
    assert by_service["s"]["outcome"] == "unaffordable"
    [action] = actions
    assert action.origin == "harness" and action.service_id == "c-fwd"
    assert action.effective_at_index == 3


def test_reversing_priority_swaps_who_gets_restored(tmp_path):
    _actions, records = _run(tmp_path, affected=["c-fwd", "s"],
                             priority=("s",))
    by_service = {r["service_id"]: r for r in records}
    assert by_service["s"]["outcome"] == "restored"
    assert by_service["c-fwd"]["outcome"] == "unaffordable"


def test_a_service_not_actually_dropped_gets_no_record(tmp_path):
    _actions, records = _run(tmp_path, affected=["s"], priority=())
    assert {r["service_id"] for r in records} == {"s"}
    assert records[0]["outcome"] == "restored"
