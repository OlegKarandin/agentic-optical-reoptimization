"""probe.py, pure: a fake `call` answering route_service, no server. The
menu is the same shape tests/eval/test_replay.py's MENU_CANDIDATES uses."""
import asyncio

import pytest

from storm_reoptimizer.eval.probe import (
    MAX_PROBES_PER_DECISION, ProbeBinding, ProbeError, answer_probe,
)
from storm_reoptimizer.eval.runner import ServiceGeometry
from storm_reoptimizer.eval.scenario_file import ConeAtHorizon, Issuance

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
