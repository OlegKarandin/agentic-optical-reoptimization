"""Post-cut restoration replay, pure (eval design spec, T1 spend-or-hold
redesign, section 5.1). No server: `counting` is a hand-rolled fake and
`index_factory` builds a TopologyIndex straight from it, exactly as
`build_topology_index` would from a real client -- the point is to exercise
`restore_after_cuts`'s own orchestration logic (scope, ordering, gating,
candidate selection, affordability, commit bookkeeping), not the MCP
plumbing that `tests/eval/test_runner.py`'s live tests already cover."""
import asyncio
import dataclasses
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
    track_revision_km_per_hour_ahead: 30
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


def _run(write_scenario, *, affected, priority):
    scenario = load_scenario(
        write_scenario(REPLAY_SCENARIO_YAML, "REPLAY_TEST.yaml"))
    counting = _FakeCounting()
    ledger = _ledger()
    return asyncio.run(restore_after_cuts(
        counting, scenario=scenario, hour="t1", hour_index=1,
        affected=affected, priority=priority, ledger=ledger,
        geometry=_geometry(), issuance=_issuance(), rg_for_cut_hour=None,
        index_factory=_index_factory(counting)))


def test_the_stated_priority_wins_the_last_spare(write_scenario):
    actions, records = _run(write_scenario, affected=["c-fwd", "s"],
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


def test_reversing_priority_swaps_who_gets_restored(write_scenario):
    _actions, records = _run(write_scenario, affected=["c-fwd", "s"],
                             priority=("s",))
    by_service = {r["service_id"]: r for r in records}
    assert by_service["s"]["outcome"] == "restored"
    assert by_service["c-fwd"]["outcome"] == "unaffordable"


def test_a_service_not_actually_dropped_gets_no_record(write_scenario):
    _actions, records = _run(write_scenario, affected=["s"], priority=())
    assert {r["service_id"] for r in records} == {"s"}
    assert records[0]["outcome"] == "restored"


class _TransientOutageCounting(_FakeCounting):
    """A server that refuses a WHOLE two-op restoration plan for the outage
    the plan is repairing, and accepts either half of it on its own -- the
    real `multilayer-optical-mcp` behaviour, confirmed live 2026-09-05
    (Task 15) and the reason `runner.try_commit` grew
    `split_transient_outage`. See that function's docstring: `baseline=
    "standing"` moves only NON-transient standing findings into
    `pre_existing`, and a plan that restores an already-dropped service is
    always at least two ops, so its own repaired outage comes back tagged
    transient and `ValidationReport.ok` is false."""

    def __init__(self):
        super().__init__()
        self.committed_op_lists: list[list[str]] = []

    async def call(self, name, arguments=None, **kw):
        if name == "validate_plan":
            self.calls.append((name, arguments))
            ops = arguments["plan"]["ops"]
            if len(ops) > 1:
                return {"ok": False, "violations": [
                    {"type": "dropped_traffic", "state_index": 0,
                     "asset_id": "c-fwd", "transient": True,
                     "reason": "link_down"}]}
            return {"ok": True}
        if name == "commit_plan":
            self.calls.append((name, arguments))
            self.committed_op_lists.append(
                [o["op"] for o in arguments["plan"]["ops"]])
            return {"status": "committed", "intended_snapshot_id": "snap"}
        return await super().call(name, arguments, **kw)


def test_a_transiently_dropped_service_is_restored_by_splitting_the_plan(
        write_scenario):
    scenario = load_scenario(
        write_scenario(REPLAY_SCENARIO_YAML, "REPLAY_TEST.yaml"))
    counting = _TransientOutageCounting()
    actions, records = asyncio.run(restore_after_cuts(
        counting, scenario=scenario, hour="t1", hour_index=1,
        affected=["c-fwd"], priority=("c-fwd",), ledger=_ledger(),
        geometry=_geometry(), issuance=_issuance(), rg_for_cut_hour=None,
        index_factory=_index_factory(counting)))

    assert [r["outcome"] for r in records] == ["restored"]
    assert records[0]["spares"] == {"tirupati": 1, "nellore": 1}
    assert len(actions) == 1 and actions[0].origin == "harness"
    # The whole plan was offered first and refused, then the SAME ops were
    # committed as provisioning-then-reroute.
    assert counting.committed_op_lists == [["provision_lightpath"],
                                           ["reroute_service"]]


def test_a_non_transient_violation_is_still_a_rejection(write_scenario):
    """The split is not a blanket "commit anyway": a violation that is NOT
    transient, or that names some OTHER asset, still rejects the candidate."""
    class _RealViolation(_TransientOutageCounting):
        async def call(self, name, arguments=None, **kw):
            if name == "validate_plan" and len(arguments["plan"]["ops"]) > 1:
                return {"ok": False, "violations": [
                    {"type": "disjointness_collapse", "state_index": 1,
                     "asset_id": "some-other-service", "transient": False}]}
            return await super().call(name, arguments, **kw)

    scenario = load_scenario(
        write_scenario(REPLAY_SCENARIO_YAML, "REPLAY_TEST.yaml"))
    counting = _RealViolation()
    actions, records = asyncio.run(restore_after_cuts(
        counting, scenario=scenario, hour="t1", hour_index=1,
        affected=["c-fwd"], priority=("c-fwd",), ledger=_ledger(),
        geometry=_geometry(), issuance=_issuance(), rg_for_cut_hour=None,
        index_factory=_index_factory(counting)))

    assert [r["outcome"] for r in records] == ["rejected"]
    assert actions == []
    assert counting.committed_op_lists == []


def test_a_duplicated_priority_id_is_attempted_only_once(write_scenario):
    # decisions._claim_priority only validates SHAPE, never uniqueness -- a
    # real decider can legally repeat an id. Without deduping in
    # `_ordered_scope`, "c-fwd" would be attempted twice: once (correctly)
    # restored, and a second time re-running route_service/try_commit
    # against a service that is no longer down, producing two records for
    # the same service_id (violating the documented "one record per
    # attempted service" contract) and wasting a redundant reroute attempt.
    actions, records = _run(write_scenario, affected=["c-fwd", "s"],
                            priority=("c-fwd", "c-fwd"))
    assert [r["service_id"] for r in records] == ["c-fwd", "s"]
    assert len(actions) == 1


GROOM_CANDIDATE = {
    # A zero-spare ip_reroute that genuinely MOVES the service: its reused
    # lightpath resolves to oms_alt, which is not on the (empty) working
    # path, so path_delta.changes_working_path is True.
    "lever": "ip_reroute", "reused_lightpaths": ["lp-groom"],
    "new_lightpaths": [], "shortfall_gbps": 0.0}


class _GroomLastCounting(_FakeCounting):
    """The lightpath first, the free groom LAST in menu order -- the order a
    first-workable pick would get wrong."""

    async def call(self, name, arguments=None, **kw):
        if name == "route_service":
            self.calls.append((name, arguments))
            return {"status": "solution",
                    "candidates": [dict(MENU_CANDIDATES[1]),
                                   dict(GROOM_CANDIDATE)]}
        if name == "get_topology":
            layer = (arguments or {}).get("layer")
            if layer == "optical":
                # Add oms_alt to the topology for the groom lightpath.
                return {"oms": [{"id": "oms_new", "src_node_id": "tirupati",
                                 "dst_node_id": "nellore",
                                 "elements": ["fiber_x"]},
                               {"id": "oms_alt", "src_node_id": "tirupati",
                                "dst_node_id": "nellore",
                                "elements": ["fiber_alt"]}]}
            if layer == "ip":
                # Add the groom lightpath that connects c-fwd endpoints.
                return {"routers": [{"site": "tirupati", "id": "router_tirupati"},
                                    {"site": "nellore", "id": "router_nellore"}],
                        "ip_links": [{"lightpath_id": "lp-groom",
                                      "id": "lp-groom",
                                      "a_router": "router_tirupati",
                                      "z_router": "router_nellore"}]}
            raise AssertionError(f"unexpected layer {layer!r}")
        if name == "get_lightpaths":
            # Return the groom lightpath so the commit can find it.
            return [{"id": "lp-groom", "oms_id": "oms_alt",
                     "src_site": "tirupati", "dst_site": "nellore"}]
        return await super().call(name, arguments, **kw)


def _groom_geometry():
    geometry = _geometry()
    return dataclasses.replace(
        geometry,
        oms_nodes={**geometry.oms_nodes, "oms_alt": ["tirupati", "nellore"]},
        oms_sequences={"lp-groom": ("oms_alt",)})


def test_the_replay_prefers_the_cheapest_workable_candidate(write_scenario):
    """Spec 5.4: a zero-spare groom must be taken over a spare-charging
    lightpath even when it is listed after it, else T3's 'free' restoration
    silently spends the spare."""
    scenario = load_scenario(
        write_scenario(REPLAY_SCENARIO_YAML, "REPLAY_TEST.yaml"))
    counting = _GroomLastCounting()
    # Ledger must include oms_alt for the groom lightpath.
    ledger = SpareLedger(
        inventory={"tirupati": 1}, depot_site="tirupati",
        oms_nodes={"oms_new": ["tirupati", "nellore"],
                   "oms_alt": ["tirupati", "nellore"]})
    actions, records = asyncio.run(restore_after_cuts(
        counting, scenario=scenario, hour="t1", hour_index=1,
        affected=["c-fwd"], priority=(), ledger=ledger,
        geometry=_groom_geometry(), issuance=_issuance(), rg_for_cut_hour=None,
        index_factory=_index_factory(counting)))
    [record] = records
    assert record["outcome"] == "restored"
    assert record["lever"] == "ip_reroute"
    assert record["spares"] == {}
    assert ledger.on_hand == 1, "the groom must not have spent the spare"
    assert actions[0].spares == {}


def test_ties_in_spare_cost_keep_menu_order(write_scenario):
    # Two lightpaths at equal cost: the first in menu order wins, as today.
    _actions, records = _run(write_scenario, affected=["c-fwd"], priority=())
    assert records[0]["lever"] == "optical_reroute"
