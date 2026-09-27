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


# The pre-2026-09-27 hard-coded roster, `{SUT} | claimant_services`, as the
# default: every test written before the replay's scope became "services
# shown" keeps exercising exactly the scope it was written against.
DEFAULT_SCOPE = frozenset({"s", "c-fwd", "c-rev"})


def _run(write_scenario, *, affected, priority, scope=DEFAULT_SCOPE):
    scenario = load_scenario(
        write_scenario(REPLAY_SCENARIO_YAML, "REPLAY_TEST.yaml"))
    counting = _FakeCounting()
    ledger = _ledger()
    return asyncio.run(restore_after_cuts(
        counting, scenario=scenario, hour="t1", hour_index=1,
        affected=affected, priority=priority, scope=scope, ledger=ledger,
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


# -- scope = services shown (2026-09-27 spec, 4.5) --------------------------

def test_scope_not_the_claimant_roster_bounds_restoration(write_scenario):
    # c-fwd IS a roster claimant and IS dropped, but is not in `scope` (never
    # shown this episode), so the replay must not touch it.
    _a, records = _run(write_scenario, affected=["c-fwd", "s"], priority=(),
                       scope={"s"})
    assert {r["service_id"] for r in records} == {"s"}


def test_a_shown_non_roster_service_is_restored(write_scenario):
    # REPLAY yaml's claimant_services is [c-fwd, c-rev]; drop them from scope
    # and add s only via scope -- s is restored from scope alone.
    _a, records = _run(write_scenario, affected=["s"], priority=(), scope={"s"})
    assert records[0]["outcome"] == "restored"


# A depot service that is neither the SUT nor any claimant_services entry --
# one the decider was merely SHOWN. 300 G, the SUT's own demand, and an id
# that sorts BEFORE "s": exactly T2's indore-vs-SUT shape (plan Task 5, "Why
# the priority is no longer stripped").
SHOWN_ONLY = "a-shown"


class _ShownOnlyCounting(_FakeCounting):
    async def call(self, name, arguments=None, **kw):
        result = await super().call(name, arguments, **kw)
        if name == "get_services":
            result = {"services": [*result["services"], {
                "id": SHOWN_ONLY, "demand_gbps": 300.0,
                "src_router": "router_tirupati",
                "dst_router": "router_nellore"}]}
        return result


def _run_with_shown_only(write_scenario, *, affected, priority, scope):
    scenario = load_scenario(
        write_scenario(REPLAY_SCENARIO_YAML, "REPLAY_TEST.yaml"))
    counting = _ShownOnlyCounting()
    geometry = _geometry()
    geometry = dataclasses.replace(
        geometry,
        endpoint_sites={**geometry.endpoint_sites,
                        SHOWN_ONLY: ("tirupati", "nellore")},
        path_oms={**geometry.path_oms,
                  SHOWN_ONLY: {"working": (), "protection": ()}})
    return asyncio.run(restore_after_cuts(
        counting, scenario=scenario, hour="t1", hour_index=1,
        affected=affected, priority=priority, scope=scope, ledger=_ledger(),
        geometry=geometry, issuance=_issuance(), rg_for_cut_hour=None,
        index_factory=_index_factory(counting)))


def test_a_shown_dropped_non_roster_service_is_restored(write_scenario):
    """Spec 8: 'a shown, dropped, restorable non-roster service is
    restored' -- the old `{SUT} | claimant_services` roster would never
    have attempted it at all."""
    _a, records = _run_with_shown_only(
        write_scenario, affected=[SHOWN_ONLY], priority=(),
        scope={"s", "c-fwd", "c-rev", SHOWN_ONLY})
    assert [(r["service_id"], r["outcome"]) for r in records] == [
        (SHOWN_ONLY, "restored")]


def test_a_never_shown_dropped_service_is_ignored(write_scenario):
    """Spec 8: 'a never-shown dropped depot service is ignored'."""
    _a, records = _run_with_shown_only(
        write_scenario, affected=[SHOWN_ONLY, "s"], priority=(), scope={"s"})
    assert [r["service_id"] for r in records] == ["s"]


def test_a_ranked_sut_keeps_its_place_over_an_equal_demand_shown_service(
        write_scenario):
    """The ranking reaches the replay UNSTRIPPED (spec 4.5: 'standing
    ranking first'): a SUT the decider ranked keeps first call on the spare
    even though an unranked shown service of equal demand, whose id sorts
    first, is dropped in the same cut."""
    _a, records = _run_with_shown_only(
        write_scenario, affected=[SHOWN_ONLY, "s"], priority=("s",),
        scope={"s", SHOWN_ONLY})
    assert [(r["service_id"], r["outcome"]) for r in records] == [
        ("s", "restored"), (SHOWN_ONLY, "unaffordable")]
    # And unranked, equal demand falls back to id order -- the case the
    # stripped priority used to hit.
    _a, records = _run_with_shown_only(
        write_scenario, affected=[SHOWN_ONLY, "s"], priority=(),
        scope={"s", SHOWN_ONLY})
    assert [r["service_id"] for r in records] == [SHOWN_ONLY, "s"]


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
        affected=["c-fwd"], priority=("c-fwd",), scope=DEFAULT_SCOPE,
        ledger=_ledger(),
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
        affected=["c-fwd"], priority=("c-fwd",), scope=DEFAULT_SCOPE,
        ledger=_ledger(),
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
        affected=["c-fwd"], priority=(), scope=DEFAULT_SCOPE, ledger=ledger,
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


FWD_LIGHTPATH = {"lever": "optical_reroute", "reused_lightpaths": [],
                 "new_lightpaths": [{"oms_sequence": ["oms_fwd"], "lam": 5,
                                     "mode_id": "M", "gsnr_db": 10.0,
                                     "bitrate_gbps": 300.0}],
                 "shortfall_gbps": 0.0}
REV_LIGHTPATH = {"lever": "optical_reroute", "reused_lightpaths": [],
                 "new_lightpaths": [{"oms_sequence": ["oms_rev"], "lam": 6,
                                     "mode_id": "M", "gsnr_db": 10.0,
                                     "bitrate_gbps": 300.0}],
                 "shortfall_gbps": 0.0}


class _MatePairCounting(_FakeCounting):
    """The real T1a shape: c-fwd's menu offers a new lightpath on oms_fwd
    (tirupati->nellore); c-rev's offers oms_rev (nellore->tirupati) -- each
    direction served by a physically DIFFERENT OMS/lightpath id, exactly
    like T1a's own t1-claimant-jalgaon-dhulia-fwd/-rev (CLAUDE.md,
    2026-09-21)."""

    async def call(self, name, arguments=None, **kw):
        if name == "route_service":
            self.calls.append((name, arguments))
            lp = (FWD_LIGHTPATH if arguments["service_id"] == "c-fwd"
                 else REV_LIGHTPATH)
            return {"status": "solution",
                    "candidates": [dict(MENU_CANDIDATES[0]), dict(lp)]}
        if name == "get_topology":
            layer = (arguments or {}).get("layer")
            if layer == "optical":
                return {"oms": [
                    {"id": "oms_fwd", "src_node_id": "tirupati",
                     "dst_node_id": "nellore", "elements": ["fiber_fwd"]},
                    {"id": "oms_rev", "src_node_id": "nellore",
                     "dst_node_id": "tirupati", "elements": ["fiber_rev"]}]}
            if layer == "ip":
                return {"routers": [{"site": "tirupati", "id": "router_tirupati"},
                                    {"site": "nellore", "id": "router_nellore"}],
                        "ip_links": []}
            raise AssertionError(f"unexpected layer {layer!r}")
        return await super().call(name, arguments, **kw)


def _mate_pair_geometry():
    geometry = _geometry()
    return dataclasses.replace(
        geometry, oms_nodes={"oms_fwd": ["tirupati", "nellore"],
                             "oms_rev": ["nellore", "tirupati"]})


def _mate_pair_ledger():
    return SpareLedger(inventory={"tirupati": 1}, depot_site="tirupati",
                       oms_nodes={"oms_fwd": ["tirupati", "nellore"],
                                 "oms_rev": ["nellore", "tirupati"]})


def test_the_t1a_shape_end_to_end_one_spare_a_mate_pair_cut_both_restored(
        write_scenario):
    """T1a's own shape: one spare on hand, a bidirectional claim cut as two
    unidirectional pins riding physically separate lightpaths. Before Task
    1, c-rev was unaffordable once c-fwd spent the depot's only pair; after
    it, c-rev mates against c-fwd's already-lit run and restores for free."""
    scenario = load_scenario(
        write_scenario(REPLAY_SCENARIO_YAML, "REPLAY_TEST.yaml"))
    counting = _MatePairCounting()
    ledger = _mate_pair_ledger()
    actions, records = asyncio.run(restore_after_cuts(
        counting, scenario=scenario, hour="t1", hour_index=1,
        affected=["c-fwd", "c-rev"], priority=("c-fwd", "c-rev"),
        scope=DEFAULT_SCOPE,
        ledger=ledger, geometry=_mate_pair_geometry(), issuance=_issuance(),
        rg_for_cut_hour=None, index_factory=_index_factory(counting)))
    by_service = {r["service_id"]: r for r in records}
    assert by_service["c-fwd"]["outcome"] == "restored"
    assert by_service["c-fwd"]["spares"] == {"tirupati": 1, "nellore": 1}
    assert by_service["c-rev"]["outcome"] == "restored"
    assert by_service["c-rev"]["spares"] == {}
    assert ledger.on_hand == 0
    assert len(actions) == 2


ALT_REV_LIGHTPATH = {"lever": "optical_reroute", "reused_lightpaths": [],
                     "new_lightpaths": [{"oms_sequence": ["oms_rev_alt"], "lam": 7,
                                         "mode_id": "M", "gsnr_db": 10.0,
                                         "bitrate_gbps": 300.0}],
                     "shortfall_gbps": 0.0}


class _MatePairChoiceCounting(_MatePairCounting):
    """c-rev's menu offers TWO workable candidates: a fresh, unrelated
    lightpath (oms_rev_alt, full price, nellore<->vijayawada) listed
    FIRST, and the true reverse mate (oms_rev, free once c-fwd is lit)
    listed SECOND -- proving the cheapest-first sort (replay.py's
    lit_runs-aware spares_needed, not menu order) is what picks the mate
    (transponder-pairing spec, 2026-09-21, replay.py:178)."""

    async def call(self, name, arguments=None, **kw):
        if name == "route_service" and arguments["service_id"] == "c-rev":
            self.calls.append((name, arguments))
            return {"status": "solution",
                    "candidates": [dict(MENU_CANDIDATES[0]),
                                   dict(ALT_REV_LIGHTPATH), dict(REV_LIGHTPATH)]}
        if name == "get_topology":
            layer = (arguments or {}).get("layer")
            if layer == "optical":
                return {"oms": [
                    {"id": "oms_fwd", "src_node_id": "tirupati",
                     "dst_node_id": "nellore", "elements": ["fiber_fwd"]},
                    {"id": "oms_rev", "src_node_id": "nellore",
                     "dst_node_id": "tirupati", "elements": ["fiber_rev"]},
                    {"id": "oms_rev_alt", "src_node_id": "nellore",
                     "dst_node_id": "vijayawada", "elements": ["fiber_alt"]}]}
            if layer == "ip":
                return {"routers": [
                    {"site": "tirupati", "id": "router_tirupati"},
                    {"site": "nellore", "id": "router_nellore"},
                    {"site": "vijayawada", "id": "router_vijayawada"}],
                        "ip_links": []}
            raise AssertionError(f"unexpected layer {layer!r}")
        return await super().call(name, arguments, **kw)


def _mate_pair_choice_geometry():
    geometry = _mate_pair_geometry()
    return dataclasses.replace(
        geometry, oms_nodes={**geometry.oms_nodes,
                             "oms_rev_alt": ["nellore", "vijayawada"]})


def _mate_pair_choice_ledger():
    return SpareLedger(inventory={"tirupati": 1}, depot_site="tirupati",
                       oms_nodes={"oms_fwd": ["tirupati", "nellore"],
                                 "oms_rev": ["nellore", "tirupati"],
                                 "oms_rev_alt": ["nellore", "vijayawada"]})


def test_the_cheapest_first_sort_picks_the_mate_over_menu_order(write_scenario):
    """replay.py:178 -- the sort must rank by REAL (lit_runs-aware) cost,
    not by menu order. c-rev's menu lists the expensive, unrelated
    candidate FIRST and the free reverse mate SECOND; the replay must
    still pick the free one."""
    scenario = load_scenario(
        write_scenario(REPLAY_SCENARIO_YAML, "REPLAY_TEST.yaml"))
    counting = _MatePairChoiceCounting()
    ledger = _mate_pair_choice_ledger()
    actions, records = asyncio.run(restore_after_cuts(
        counting, scenario=scenario, hour="t1", hour_index=1,
        affected=["c-fwd", "c-rev"], priority=("c-fwd", "c-rev"),
        scope=DEFAULT_SCOPE,
        ledger=ledger, geometry=_mate_pair_choice_geometry(), issuance=_issuance(),
        rg_for_cut_hour=None, index_factory=_index_factory(counting)))
    by_service = {r["service_id"]: r for r in records}
    assert by_service["c-rev"]["outcome"] == "restored"
    assert by_service["c-rev"]["spares"] == {}       # picked the free mate, not the alt
    assert ledger.on_hand == 0
