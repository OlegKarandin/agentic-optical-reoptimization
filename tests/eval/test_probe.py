"""probe.py, pure: a fake `call` answering route_service, no server. The
menu is the same shape tests/eval/test_replay.py's MENU_CANDIDATES uses."""
import asyncio
import functools
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from storm_reoptimizer.eval import oracle
from storm_reoptimizer.eval.probe import (
    MAX_PROBES_PER_DECISION, ProbeBinding, ProbeError, answer_probe,
)
from storm_reoptimizer.eval.runner import (
    EVENT_TYPE, ServiceGeometry, horizon_risk_group_asset_ids, run_episode,
    service_geometry,
)
from storm_reoptimizer.eval.scenario_file import (
    ConeAtHorizon, Issuance, load_all_scenarios,
)
from storm_reoptimizer.events.filters import get_filter
from storm_reoptimizer.geo_mapper import load_edges
from storm_reoptimizer.mcp_client import call_tool_json, connect_server

TOPOLOGY_PATH = (Path(__file__).parent.parent.parent / "src"
                 / "storm_reoptimizer" / "data" / "toy_india_topology.json")

INERT = {"lever": "ip_reroute", "reused_lightpaths": [], "new_lightpaths": [],
         "shortfall_gbps": 0.0}
LIGHTPATH = {"lever": "optical_reroute", "reused_lightpaths": [],
             "new_lightpaths": [{"oms_sequence": ["oms_new"], "lam": 5,
                                 "mode_id": "M", "gsnr_db": 10.0,
                                 "bitrate_gbps": 300.0}],
             "shortfall_gbps": 0.0}
GROOM = {"lever": "ip_reroute", "reused_lightpaths": ["lp-groom"],
         "new_lightpaths": [], "shortfall_gbps": 0.0}
PARTIAL = {"lever": "optical_reroute", "reused_lightpaths": [],
           "new_lightpaths": [{"oms_sequence": ["oms_new"], "lam": 6,
                               "mode_id": "M", "gsnr_db": 10.0,
                               "bitrate_gbps": 100.0}],
           "shortfall_gbps": 100.0}


def _geometry():
    empty = {"working": (), "protection": ()}
    return ServiceGeometry(
        points={}, paths={},
        oms_nodes={"oms_new": ["jalgaon", "khandwa"],
                   "oms_alt": ["jalgaon", "aurangabad"]},
        unmapped_nodes={}, cuttable_spans={},
        endpoint_sites={"c": ("jalgaon", "khandwa")},
        oms_sequences={"lp-groom": ("oms_alt",)}, cuttable_span_by_oms={},
        path_oms={"c": empty}, protection_cuttable_spans={})


def _issuance():
    return Issuance(issued_at="t1", horizons={"t3": ConeAtHorizon(
        cone={"type": "Polygon", "coordinates": [[[75.0, 21.0], [75.1, 21.0],
                                                   [75.1, 21.1], [75.0, 21.0]]]},
        width_km=60.0, center={"lat": 21.0, "lon": 75.0})})


def _call_returning(status, candidates):
    calls = []

    async def call(name, arguments=None, **kw):
        calls.append((name, arguments))
        assert name == "route_service"
        return {"status": status, "candidates": [dict(c) for c in candidates]}
    call.calls = calls
    return call


def _answer(call, service_id="c", risk_group_id="rg_x"):
    return asyncio.run(answer_probe(
        call, service_id=service_id, risk_group_id=risk_group_id,
        geometry=_geometry(), issuance=_issuance(), damage_radius_km=50.0,
        demands={"c": 200.0}))


def test_the_probe_uses_the_replays_exact_route_service_posture():
    call = _call_returning("solution", [LIGHTPATH])
    _answer(call)
    assert call.calls == [("route_service", {
        "service_id": "c", "protected": False, "basis": "physical",
        "level": "link", "best_effort": False,
        "avoid": {"risk_groups": ["rg_x"]}})]


def test_inert_and_partial_candidates_are_not_full_restores():
    answer = _answer(_call_returning("solution", [INERT, PARTIAL, LIGHTPATH]))
    assert answer.status == "solution"
    assert answer.full_restore_candidates == 1
    assert answer.min_spares_needed_by_site == {"jalgaon": 1, "khandwa": 1}
    assert answer.levers == ("optical_reroute",)


def test_a_zero_spare_groom_is_the_cheapest_full_restore():
    answer = _answer(_call_returning("solution", [LIGHTPATH, GROOM]))
    assert answer.full_restore_candidates == 2
    assert answer.min_spares_needed_by_site == {}
    assert answer.levers == ("ip_reroute", "optical_reroute")


def test_no_solution_reports_no_candidates_and_null_spares():
    answer = _answer(_call_returning("no_solution", []))
    assert answer.to_dict() == {
        "status": "no_solution", "full_restore_candidates": 0,
        "min_spares_needed_by_site": None, "levers": []}


def test_an_unknown_service_demand_is_a_probe_error():
    with pytest.raises(ProbeError, match="demand"):
        _answer(_call_returning("solution", [LIGHTPATH]), service_id="ghost")


def _binding(**kw):
    # NOTE: `answer` must call `answer_probe` directly (awaited) rather than
    # through the sync `_answer` helper above -- `_answer` wraps its call in
    # `asyncio.run(...)`, and this function is itself driven by
    # `asyncio.run(binding(...))` in the tests below, so going through
    # `_answer` here raises "asyncio.run() cannot be called from a running
    # event loop". Same fixed inputs `_answer("c", "rg_x")` would use.
    async def answer(service_id, risk_group_id):
        return await answer_probe(
            _call_returning("solution", [LIGHTPATH]), service_id=service_id,
            risk_group_id=risk_group_id, geometry=_geometry(),
            issuance=_issuance(), damage_radius_km=50.0,
            demands={"c": 200.0})
    return ProbeBinding(answer=answer, service_ids={"c", "s"},
                        risk_group_ids={"rg_x"}, **kw)


def test_a_bound_probe_records_the_decision_arguments_and_answer():
    binding = _binding()
    binding.begin("timing")
    result = asyncio.run(binding("c", "rg_x"))
    assert result["full_restore_candidates"] == 1
    assert binding.records == [{
        "decision": "timing", "service_id": "c", "risk_group_id": "rg_x",
        "answer": result, "error": None}]


@pytest.mark.parametrize("service_id, risk_group_id, needle", [
    ("ghost", "rg_x", "exposure"),
    ("c", "rg_nope", "risk_group_ids"),
])
def test_guards_reject_and_record_unshown_ids(service_id, risk_group_id, needle):
    binding = _binding()
    binding.begin("timing")
    with pytest.raises(ProbeError, match=needle):
        asyncio.run(binding(service_id, risk_group_id))
    assert binding.records[0]["answer"] is None
    assert needle in binding.records[0]["error"]


def test_the_cap_is_per_decision_and_begin_resets_it():
    binding = _binding()
    binding.begin("timing")
    for _ in range(MAX_PROBES_PER_DECISION):
        asyncio.run(binding("c", "rg_x"))
    with pytest.raises(ProbeError, match="cap"):
        asyncio.run(binding("c", "rg_x"))
    binding.begin("constraints")
    asyncio.run(binding("c", "rg_x"))   # a fresh decision, a fresh cap
    assert len(binding.records) == MAX_PROBES_PER_DECISION + 2


def test_a_probe_outside_any_decision_is_rejected():
    with pytest.raises(ProbeError, match="decision"):
        asyncio.run(_binding()("c", "rg_x"))


def test_the_probe_answers_what_the_replay_then_does_on_t1a(
    loaded_state_path, local_server_command, local_server_env,
):
    """Spec 8.1: for the same service and group, the probe's cheapest full
    restore is exactly what `restore_after_cuts` spends on it after the cut.
    T1a: the hold rollout restores the dhulia claimant at t3 with one pair
    at jalgaon and one at dhulia."""
    scenario = load_all_scenarios()["T1a"]
    claimant = scenario.metadata["claimant_services"][0]
    horizon, rg_id = oracle.spend_risk_group(scenario)

    @asynccontextmanager
    async def _connect():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            yield client

    async def _probe():
        edges = load_edges(TOPOLOGY_PATH)
        async with _connect() as client:
            geometry = await service_geometry(client, TOPOLOGY_PATH, edges=edges)
            topo = await call_tool_json(client, "get_topology", {"layer": "optical"})
            issuance = scenario.forecast[scenario.decision_hour]
            await call_tool_json(client, "define_risk_group", {
                "rg_id": rg_id,
                "asset_ids": horizon_risk_group_asset_ids(
                    issuance.horizons[horizon], scenario.damage_radius_km,
                    edges=edges, oms=topo["oms"], filter_fn=get_filter(EVENT_TYPE)),
                "metadata": {"test": "probe-equals-replay"}})
            services = (await call_tool_json(client, "get_services"))["services"]
            return await answer_probe(
                functools.partial(call_tool_json, client),
                service_id=claimant, risk_group_id=rg_id, geometry=geometry,
                issuance=issuance, damage_radius_km=scenario.damage_radius_km,
                demands={s["id"]: float(s["demand_gbps"]) for s in services})

    async def _replay():
        async with _connect() as client:
            return await run_episode(client, scenario, oracle.hold_decider(scenario),
                                     topology_path=TOPOLOGY_PATH)

    answer = asyncio.run(_probe())
    trace = asyncio.run(_replay())
    restored = next(r for r in trace.restorations
                    if r["service_id"] == claimant and r["outcome"] == "restored")
    assert answer.full_restore_candidates > 0
    assert answer.min_spares_needed_by_site == restored["spares"]
    assert restored["lever"] in answer.levers
