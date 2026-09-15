"""The LLM decider (agent decider design spec, 2026-08-24). Unit tests only:
a fake Anthropic client, no network, no cost, and no `anthropic` install."""
import ast
import asyncio
import dataclasses
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from storm_reoptimizer.eval import agent as agent_module
from storm_reoptimizer.eval.agent import (
    CONSTRAINT_TOOL, DEFAULT_MODEL, MAX_ATTEMPTS, OBJECTIVE_INSTRUCTION,
    OBJECTIVE_TOOL, P_CUT_ENUMERATION_THRESHOLD, SYSTEM_PROMPT, TIMING_TOOL,
    ClaudeDecider, _project_exposure_entry, project_observation,
    strict_tool_schema,
)
from storm_reoptimizer.eval.baseline import ForecastBlindBaseline
from storm_reoptimizer.eval.decisions import (
    CONSTRAINT_JSON_SCHEMA, ConstraintDecision, DecisionError,
    OBJECTIVE_JSON_SCHEMA, ObjectiveDecision, TIMING_JSON_SCHEMA,
    TimingDecision,
)
from storm_reoptimizer.eval.observation import Observation
from storm_reoptimizer.eval.probe import (
    MAX_PROBES_PER_DECISION, PROBE_TOOL, ProbeBinding, ProbeError,
)
from storm_reoptimizer.eval.scenario_file import ConeAtHorizon, Issuance
from storm_reoptimizer.eval.suite import build_arg_parser, build_deciders

HORIZON = "t3"
# T3a's own depot_site (scenarios/T3a.yaml). storm-svc-1 terminates
# satna<->allahabad -- the only service in the shipped state that touches it.
DEPOT_SITE = "satna"


def _run(coro):
    return asyncio.run(coro)


def _obs(*, others=(), sut_p_cut=0.3410, iteration=0, last_rejection=None,
        standing_claim_priority=("storm-svc-1",)):
    """T3a's shape at t1: the service under test centred in the far cone,
    plus whichever other claimants the test wants to place. `others` is a
    sequence of (service_id, p_cut, demand_gbps).

    Every claimant placed this way terminates at `DEPOT_SITE` (paired with a
    site unique to it, so co-terminating grouping never merges two of them),
    which makes it depot-eligible by construction -- these are the pre-Task-11
    enumeration-threshold tests, not the eligibility-filter ones
    (`_wide_observation` below is that one), so the claimants here should
    survive projection exactly as they did before the eligibility split.

    `standing_claim_priority` defaults to a non-empty ranking -- this
    helper builds the shape of an hour ALREADY mid-episode, which is what
    every pre-existing mechanics test here means to exercise (tool schema,
    retries, the audit sidecar, probe wiring), none of which are about the
    Task 4 first-call ranking rule. A test of that rule (test_agent.py's
    `test_the_first_timing_call_of_an_episode_must_state_a_ranking` and its
    sibling) builds its payload directly and overrides the field itself, so
    this default never masks either branch."""
    exposure = {"storm-svc-1": {HORIZON: {
        "hours_ahead": 2, "offset_km": 0.0, "width_km": 320.0,
        "p_cut": sut_p_cut, "demand_gbps": 300.0}}}
    services = [{"id": "storm-svc-1", "demand_gbps": 300.0,
                 "src_router": "router_satna",
                 "dst_router": "router_allahabad",
                 "working_path": ["ipl-cand-storm-svc-1-0"],
                 "protection_path": ["ipl-prot-storm-svc-1-0"]}]
    groups = []
    for svc_id, p_cut, demand in others:
        exposure[svc_id] = {HORIZON: {
            "hours_ahead": 2, "offset_km": 129.3, "width_km": 320.0,
            "p_cut": p_cut, "demand_gbps": demand}}
        services.append({"id": svc_id, "demand_gbps": demand,
                         "src_router": "router_a", "dst_router": "router_b",
                         "working_path": [], "protection_path": []})
        groups.append({"endpoints": (DEPOT_SITE, svc_id), "members": (svc_id,),
                       "ecar_gbps": round(p_cut * demand, 3)})
    return Observation(
        scenario_id="T3a", service_under_test="storm-svc-1", hour="t1",
        hour_index=1, hours_remaining=2,
        issuance=Issuance(issued_at="t1", horizons={
            HORIZON: ConeAtHorizon(
                cone={"type": "Polygon", "coordinates": []}, width_km=320.0,
                center={"lat": 24.855553, "lon": 81.327777})}),
        exposure=exposure, services=tuple(services), spares_on_hand=1,
        lead_time_hours=1, risk_group_ids={HORIZON: "rg_T3a_t1_t3"},
        iteration=iteration, last_rejection=last_rejection,
        restorable_groups={HORIZON: tuple(groups)} if groups else {},
        standing_claim_priority=standing_claim_priority)


def _wide_observation():
    """T3a's shape at t1, widened with a full eligible/ineligible/quiet
    claimant roster (design spec §4.2's worked example): svc-b and svc-c
    terminate at `DEPOT_SITE` and clear the enumeration threshold; svc-d is
    heavily exposed but terminates kolkata<->mumbai, nowhere near the depot;
    svc-e terminates at `DEPOT_SITE` too but clears no threshold at all."""
    exposure = {"storm-svc-1": {HORIZON: {
        "hours_ahead": 2, "offset_km": 0.0, "width_km": 320.0,
        "p_cut": 0.3410, "demand_gbps": 300.0}}}
    services = [{"id": "storm-svc-1", "demand_gbps": 300.0,
                 "src_router": "router_satna",
                 "dst_router": "router_allahabad",
                 "working_path": ["ipl-cand-storm-svc-1-0"],
                 "protection_path": ["ipl-prot-storm-svc-1-0"]}]
    # (service_id, p_cut, demand_gbps, depot-eligible?)
    claimants = [("svc-b", 0.30, 100.0, True),
                 ("svc-c", 0.20, 100.0, True),
                 ("svc-d", 0.90, 100.0, False),
                 ("svc-e", 0.001, 100.0, True)]
    groups = []
    for svc_id, p_cut, demand, eligible in claimants:
        exposure[svc_id] = {HORIZON: {
            "hours_ahead": 2, "offset_km": 129.3, "width_km": 320.0,
            "p_cut": p_cut, "demand_gbps": demand}}
        if eligible:
            src, dst = f"router_{DEPOT_SITE}", f"router_{svc_id}"
            groups.append({"endpoints": (DEPOT_SITE, svc_id),
                           "members": (svc_id,),
                           "ecar_gbps": round(p_cut * demand, 3)})
        else:
            src, dst = "router_kolkata", "router_mumbai"
        services.append({"id": svc_id, "demand_gbps": demand,
                         "src_router": src, "dst_router": dst,
                         "working_path": [], "protection_path": []})
    return Observation(
        scenario_id="T3a", service_under_test="storm-svc-1", hour="t1",
        hour_index=1, hours_remaining=2,
        issuance=Issuance(issued_at="t1", horizons={
            HORIZON: ConeAtHorizon(
                cone={"type": "Polygon", "coordinates": []}, width_km=320.0,
                center={"lat": 24.855553, "lon": 81.327777})}),
        exposure=exposure, services=tuple(services), spares_on_hand=1,
        lead_time_hours=1, risk_group_ids={HORIZON: "rg_T3a_t1_t3"},
        restorable_groups={HORIZON: tuple(groups)},
        # sut_ecar_gbps = 0.3410*300; largest_restorable_group_ecar_gbps is
        # the max over svc-b/svc-c/svc-e's own groups (svc-b's 30.0);
        # non_sut_ineligible_ecar_gbps is svc-d's, the only ineligible one.
        horizon_totals={HORIZON: {
            "sut_ecar_gbps": 102.3,
            "largest_restorable_group_ecar_gbps": 30.0,
            "non_sut_ineligible_ecar_gbps": 90.0}})


def _settled_observation():
    """The far cone has passed and the service under test reads zero
    exposure everywhere -- it must still be the one thing the projection
    unconditionally keeps, because it is the only service these tools can
    act on."""
    return Observation(
        scenario_id="T3a", service_under_test="storm-svc-1", hour="t2",
        hour_index=2, hours_remaining=1,
        issuance=Issuance(issued_at="t1", horizons={
            HORIZON: ConeAtHorizon(
                cone={"type": "Polygon", "coordinates": []}, width_km=320.0,
                center={"lat": 24.855553, "lon": 81.327777})}),
        exposure={"storm-svc-1": {HORIZON: {
            "hours_ahead": 1, "offset_km": 400.0, "width_km": 320.0,
            "p_cut": 0.0, "demand_gbps": 300.0}}},
        services=({"id": "storm-svc-1", "demand_gbps": 300.0,
                   "src_router": "router_satna",
                   "dst_router": "router_allahabad",
                   "working_path": ["ipl-cand-storm-svc-1-0"],
                   "protection_path": ["ipl-prot-storm-svc-1-0"]},),
        spares_on_hand=1, lead_time_hours=1,
        risk_group_ids={HORIZON: "rg_T3a_t2_t3"})


# The three claimants T3a's authoring note enumerates, plus two the same note
# excludes ("the only non-SUT services above p_cut 0.005 ... in EITHER half").
CLAIMANTS = [("d0363", 0.2639, 100.0), ("d0462", 0.1207, 100.0),
             ("d0212", 0.1207, 100.0)]
BELOW_THRESHOLD = [("d0001", 0.004, 100.0), ("d0002", 0.002, 300.0)]


def test_the_threshold_is_the_one_the_episode_authors_enumerated_against():
    # scenarios/T3a.yaml names 0.005 as the cut-off it used to decide which
    # non-SUT services were worth naming at all. Reusing it keeps the
    # projection's notion of "relevant" the same as the gold rationale's.
    assert P_CUT_ENUMERATION_THRESHOLD == 0.005


def test_a_claimant_above_the_threshold_survives_projection():
    payload = project_observation(_obs(others=CLAIMANTS + BELOW_THRESHOLD))
    assert set(payload["exposure"]) == {"storm-svc-1", "d0363", "d0462",
                                        "d0212"}
    assert "services" not in payload


def test_the_service_under_test_survives_even_when_its_own_p_cut_is_zero():
    payload = project_observation(_obs(sut_p_cut=0.0, others=CLAIMANTS))
    assert "storm-svc-1" in payload["exposure"]


def test_projection_keys_on_cut_probability_not_cone_containment():
    # D1's lesson: offset 58.4 km against a 15 km-wide cone (half-width
    # 7.5 km) is OUTSIDE the polygon, yet p_cut is 0.976. A containment
    # filter would drop the most exposed service in that episode. The wire
    # no longer carries `offset_km` at all (it is trimmed from the projected
    # exposure entry), so this fixture's containment-vs-p_cut contrast lives
    # only in the raw observation this test builds, never in what the
    # decider is shown.
    obs = _obs(sut_p_cut=0.0)
    obs.exposure["d0001"] = {HORIZON: {
        "hours_ahead": 2, "offset_km": 58.4, "width_km": 15.0,
        "p_cut": 0.976, "demand_gbps": 100.0}}
    obs.restorable_groups[HORIZON] = (
        {"endpoints": (DEPOT_SITE, "d0001"), "members": ("d0001",),
         "ecar_gbps": 97.6},)
    payload = project_observation(obs)
    assert "d0001" in payload["exposure"]


def test_the_omission_summary_accounts_for_every_service_it_dropped():
    payload = project_observation(_obs(others=CLAIMANTS + BELOW_THRESHOLD))
    omitted = payload["omitted_services"]
    assert omitted["p_cut_threshold"] == 0.005
    # CLAIMANTS and BELOW_THRESHOLD are all depot-eligible by _obs's own
    # construction, so the below-threshold two land in that bucket and
    # nothing lands in ineligible_for_depot.
    assert omitted["below_threshold"]["count"] == 2
    assert omitted["below_threshold"]["max_p_cut"] == 0.004
    # 0.004*100 + 0.002*300 = 0.4 + 0.6
    assert omitted["below_threshold"][
        "summed_expected_capacity_at_risk_gbps"] == pytest.approx(1.0)
    assert omitted["ineligible_for_depot"] == {"count": 0}
    assert payload["n_services_total"] == 6


def test_nothing_is_omitted_when_every_service_clears_the_threshold():
    payload = project_observation(_obs(others=CLAIMANTS))
    assert payload["omitted_services"] == {
        "p_cut_threshold": 0.005,
        "below_threshold": {"count": 0, "max_p_cut": 0.0,
                            "summed_expected_capacity_at_risk_gbps": 0.0},
        "ineligible_for_depot": {"count": 0},
    }


def test_projection_preserves_every_field_the_decision_points_read():
    obs = _obs(others=CLAIMANTS, iteration=2,
               last_rejection={"type": "insufficient_spares", "needed": 2,
                               "on_hand": 1})
    payload = project_observation(obs)
    raw = obs.to_dict()
    for key in ("scenario_id", "actionable_service", "hour",
                "hours_remaining", "issued_at", "spares_on_hand",
                "lead_time_hours", "risk_group_ids", "iteration",
                "last_rejection", "next_issuance"):
        assert payload[key] == raw[key]
    assert payload["horizons"] == [HORIZON]
    assert "cones" not in payload
    assert "damage_radius_km" not in payload


def test_projection_strips_geometry_from_exposure_entries():
    # `offset_km`/`width_km` are the cone geometry `p_cut` already
    # integrates -- distractors, not decision-relevant content (D1's own
    # lesson: p_cut, not containment/offset, is what tracks the real risk).
    # Kept as a fresh dict per agent.py's `_project_exposure_entry`, never a
    # `del` on the observation's own inner dicts.
    obs = _obs(others=CLAIMANTS)
    payload = project_observation(obs)
    for per_horizon in payload["exposure"].values():
        for entry in per_horizon.values():
            assert set(entry) == {"hours_ahead", "p_cut", "demand_gbps",
                                  "expected_capacity_at_risk_gbps"}
    # The recompute (agent.py's `_project_exposure_entry`) must land on the
    # same value `observation.py` would have stored: p_cut(0.3410) x
    # demand_gbps(300.0), rounded to 3 decimals -- not just the right keys.
    sut_entry = payload["exposure"]["storm-svc-1"][HORIZON]
    assert sut_entry["expected_capacity_at_risk_gbps"] == pytest.approx(102.3)
    # The raw observation is untouched -- projection must not mutate it.
    assert set(obs.exposure["storm-svc-1"][HORIZON]) == {
        "hours_ahead", "offset_km", "width_km", "p_cut", "demand_gbps"}


def test_projection_bounds_the_prompt_against_a_full_573_service_roster():
    # eval/states/loaded-s17.json carries 573 services (97,654 JSON chars for
    # the roster alone, plus one exposure entry per service per horizon).
    # Sending that raw would be ~50-70K tokens on every one of the up-to-11
    # calls an acting hour makes. This test is the guard on that.
    obs = _obs(others=[(f"d{i:04d}", 0.0, 100.0) for i in range(571)])
    assert len(json.dumps(obs.to_dict())) > 100_000
    payload = project_observation(obs)
    assert payload["n_services_total"] == 572
    assert payload["omitted_services"]["below_threshold"]["count"] == 571
    assert payload["omitted_services"]["ineligible_for_depot"]["count"] == 0
    # Was < 5_000 before `services`/`horizon_totals` left the wire; the
    # roster was the payload's largest block, so the bound shrinks with it
    # (measured 798 chars).
    assert len(json.dumps(payload)) < 1_500


def test_the_projection_shows_the_restorable_groups_not_the_totals():
    # `horizon_totals`' three figures are all re-derivable from rows the
    # payload still shows (spec 5.3); `restorable_groups` is the one that
    # survives onto the wire.
    obs = _obs(others=CLAIMANTS)
    payload = project_observation(obs)
    assert payload["restorable_groups"] == obs.restorable_groups
    assert "horizon_totals" not in payload


def test_the_projection_does_not_mutate_the_observation():
    # payload["exposure"] aliases the Observation's own inner dicts
    # (agent.py:87-89); a `del` on the wrong object would corrupt the caller.
    obs = _obs(others=CLAIMANTS)
    before = dict(obs.horizon_totals)
    project_observation(obs)
    assert obs.horizon_totals == before


def test_the_omission_account_separates_quiet_from_ineligible():
    # Trimming silently would be worse than not trimming (agent.py's own
    # docstring). But a single bucket hands the agent a large `max_p_cut` it
    # cannot interpret: a badly-exposed service that cannot draw on THIS
    # depot is not a competing claim, and folding it in with the quiet ones
    # invites exactly the overstatement D2 found in the gold rationales.
    payload = project_observation(_wide_observation())
    omitted = payload["omitted_services"]
    assert set(omitted["below_threshold"]) == {
        "count", "max_p_cut", "summed_expected_capacity_at_risk_gbps"}
    # ineligible_for_depot is reduced to a bare count (spec 5.3) --
    # test_the_ineligible_bucket_is_reduced_to_a_count below covers why.
    assert set(omitted["ineligible_for_depot"]) == {"count"}
    # svc-d is heavily exposed but terminates kolkata <-> mumbai.
    assert omitted["ineligible_for_depot"]["count"] == 1
    assert omitted["below_threshold"]["max_p_cut"] < 0.005


def test_the_projection_drops_the_service_roster():
    """`exposure` already carries every service's demand and
    `restorable_groups` carries the endpoints; the roster was redundant and
    was the single largest block in the payload (spec 5.3)."""
    payload = project_observation(_wide_observation())
    assert "services" not in payload
    # _wide_observation's own roster: storm-svc-1 + svc-b/c/d/e.
    assert payload["n_services_total"] == 5


def test_the_projection_drops_the_horizon_totals():
    """Every figure in it is re-derivable from a row the payload still shows,
    and `non_sut_ineligible_ecar_gbps` was described by the prompt itself as
    something to ignore (spec 5.3)."""
    obs = _wide_observation()
    assert obs.horizon_totals, "the Observation must still carry them"
    assert "horizon_totals" not in project_observation(obs)


def test_the_ineligible_bucket_is_reduced_to_a_count():
    payload = project_observation(_wide_observation())
    assert set(payload["omitted_services"]["ineligible_for_depot"]) == {"count"}
    # The quiet bucket keeps its full account: a large number there IS
    # something the decider can act on, unlike an ineligible service's.
    assert set(payload["omitted_services"]["below_threshold"]) == {
        "count", "max_p_cut", "summed_expected_capacity_at_risk_gbps"}


def test_the_projection_carries_next_issuance_through():
    obs = dataclasses.replace(_wide_observation(),
                              next_issuance={"hour": "t1"})
    assert project_observation(obs)["next_issuance"] == {"hour": "t1"}


def test_the_projection_keeps_only_depot_eligible_claimants():
    payload = project_observation(_wide_observation())
    assert set(payload["exposure"]) == {"storm-svc-1", "svc-b", "svc-c"}


def test_a_mixed_group_survives_whole_not_partially_trimmed():
    # svc-f and svc-g co-terminate (one restoring lightpath serves both), but
    # only svc-f individually clears the enumeration threshold. Trimming the
    # group to ONLY kept members (or dropping the group outright because not
    # every member is kept) would either understate `ecar_gbps` below what
    # `horizon_totals.largest_restorable_group_ecar_gbps` maxes over, or hide
    # the group entirely even though svc-f -- a service the agent IS shown --
    # is a member of it. The group must appear whole: both members, and the
    # same `ecar_gbps` the group total is built from.
    obs = _obs(others=[("svc-f", 0.30, 100.0)])
    obs.exposure["svc-g"] = {HORIZON: {
        "hours_ahead": 2, "offset_km": 129.3, "width_km": 320.0,
        "p_cut": 0.001, "demand_gbps": 50.0}}
    obs.restorable_groups[HORIZON] = (
        {"endpoints": (DEPOT_SITE, "raipur"), "members": ("svc-f", "svc-g"),
         "ecar_gbps": 30.05},)
    payload = project_observation(obs)
    assert "svc-f" in payload["exposure"]
    assert "svc-g" not in payload["exposure"]
    groups = payload["restorable_groups"][HORIZON]
    assert len(groups) == 1
    assert groups[0]["members"] == ("svc-f", "svc-g")
    assert groups[0]["ecar_gbps"] == pytest.approx(30.05)


def test_the_actionable_service_survives_even_at_zero_exposure():
    payload = project_observation(_settled_observation())
    assert "storm-svc-1" in payload["exposure"]


def test_the_prompt_no_longer_claims_a_shared_global_depot():
    assert ("Other services on this network draw on the same inventory"
           not in SYSTEM_PROMPT)
    assert "spare_inventory" in SYSTEM_PROMPT
    assert "each of its two endpoint sites" in SYSTEM_PROMPT
    assert "ineligible_for_depot" in SYSTEM_PROMPT
    # The transponders cost-term warning survives, re-pointed.
    assert "reason about spares from `spares_needed`" in SYSTEM_PROMPT.lower() \
        or "spares_needed" in SYSTEM_PROMPT
    assert "pairs_needed" not in SYSTEM_PROMPT


def test_a_contested_claim_must_still_name_a_service_in_exposure():
    # Free consequence of the eligibility filter: agent._check_named_services's
    # existing "must be in `exposure`" rule becomes exactly the right
    # depot-eligibility check, with no new validation logic. svc-d is real
    # (it is in the full Observation's `exposure`) but was filtered out of
    # the payload for being depot-ineligible, so naming it must still fail.
    payload = project_observation(_wide_observation())
    decision = SimpleNamespace(contested_claim={
        "service_id": "svc-d", "expected_capacity_at_risk_gbps": 90.0})
    with pytest.raises(DecisionError, match="svc-d"):
        ClaudeDecider._check_named_services(decision, payload, TIMING_TOOL)


def test_claim_priority_must_name_shown_services():
    payload = {"exposure": {"s": {}, "c": {}}}
    ok = TimingDecision("wait", "x", None, ("c",))
    ClaudeDecider._check_named_services(ok, payload, TIMING_TOOL)
    with pytest.raises(DecisionError):
        ClaudeDecider._check_named_services(
            TimingDecision("wait", "x", None, ("ghost",)),
            payload, TIMING_TOOL)


MENU = {
    "status": "solution",
    "candidates": [
        {"lever": "optical_reroute", "reused_lightpaths": [],
         "new_lightpaths": [{"oms_sequence": ["oms_1"], "lam": 0,
                             "mode_id": "300G@4.8dB", "gsnr_db": 12.0,
                             "bitrate_gbps": 300.0}],
         "restored_gbps": 300.0, "shortfall_gbps": 0.0,
         "cost_vector": {"dropped_traffic": 0.0, "transponders": 40.0,
                         "services_at_risk": 0, "total_margin": 5.0,
                         "added_latency": 9.0, "spectrum_used": 20,
                         "max_util": 0.7}},
        {"lever": "ip_reroute", "reused_lightpaths": ["lp-x"],
         "new_lightpaths": [], "restored_gbps": 250.0, "shortfall_gbps": 50.0,
         "cost_vector": {"dropped_traffic": 50.0, "transponders": 38.0,
                         "services_at_risk": 0, "total_margin": 5.0,
                         "added_latency": 4.0, "spectrum_used": 18,
                         "max_util": 0.8}},
    ],
    "pairs": [],
}

# oms_1's two endpoint SITES, for spares_needed to resolve MENU's/PROBE's one
# real new lightpath against.
OMS_NODES = {"oms_1": ["site_a", "site_b"]}


class FakeToolUse:
    """Mimics exactly the four attributes the decider reads off a real
    anthropic tool_use block. The real SDK types are unavailable here --
    `anthropic` is deliberately not a hard dependency of this package."""
    type = "tool_use"

    def __init__(self, name, payload, block_id="toolu_test"):
        self.id = block_id
        self.name = name
        self.input = payload


class FakeText:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeThinking:
    type = "thinking"

    def __init__(self, thinking=""):
        self.thinking = thinking


class FakeResponse:
    def __init__(self, *content):
        self.content = list(content)


def _response(tool_name, payload, block_id="toolu_test"):
    """`FakeResponse(FakeToolUse(tool_name, payload))`, spelled once -- the
    shape every test above already builds by hand."""
    return FakeResponse(FakeToolUse(tool_name, payload, block_id=block_id))


class FakeMessages:
    def __init__(self, responses):
        self.queued = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.queued:
            raise AssertionError(
                "the decider made more API calls than the test queued "
                f"responses for ({len(self.calls)} calls)")
        return self.queued.pop(0)


class FakeAnthropic:
    def __init__(self, *responses):
        self.messages = FakeMessages(responses)


TIMING_OK = {"action": "wait", "reasoning": "the far cone is 2h out"}
CONSTRAINT_OK = {"avoid": {"risk_groups": ["rg_T3a_t1_t3"]},
                 "reasoning": "route around the t3 cone"}
OBJECTIVE_OK = {"choice": "candidate_1",
                "reasoning": "the ip_reroute costs no spare pairs"}


def _decider(*responses, model=DEFAULT_MODEL, oms_nodes=OMS_NODES, **kwargs):
    client = FakeAnthropic(*responses)
    return (ClaudeDecider(model=model, client=client, oms_nodes=oms_nodes,
                          **kwargs),
            client)


def test_the_decider_name_namespaces_the_model():
    decider, _ = _decider()
    assert decider.name == "agent:claude-sonnet-5"
    assert ClaudeDecider(model="claude-opus-5",
                         client=object()).name == "agent:claude-opus-5"


def test_the_default_client_is_built_lazily_so_construction_needs_no_sdk():
    # suite.py builds the decider before any rollout starts, and every unit
    # test here constructs one. `anthropic` is an optional extra, so nothing
    # may import it until a real API call is actually made.
    assert ClaudeDecider()._client is None


def test_anthropic_is_never_imported_at_module_scope():
    tree = ast.parse(Path(agent_module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert not any(n.split(".")[0] == "anthropic" for n in names)


def test_strict_tool_schemas_drop_the_keywords_the_api_rejects():
    # Strict tool use supports neither `minLength` nor a type-union list.
    # decisions.py's schemas use both, so sending them raw is a 400 on the
    # very first real call.
    assert strict_tool_schema(TIMING_JSON_SCHEMA) == {
        "type": "object", "additionalProperties": False,
        "required": ["reasoning", "contested_claim", "claim_priority",
                     "action"],
        "properties": {
            "reasoning": {"type": "string"},
            "contested_claim": {"anyOf": [
                {"type": "object", "additionalProperties": False,
                 "required": ["service_id",
                              "expected_capacity_at_risk_gbps"],
                 "properties": {
                     "service_id": {"type": "string"},
                     "expected_capacity_at_risk_gbps": {"type": "number"},
                 }},
                {"type": "null"},
            ]},
            "claim_priority": {"type": "array",
                               "items": {"type": "string"}},
            "action": {"type": "string", "enum": ["act", "wait"]},
        },
    }
    assert strict_tool_schema(OBJECTIVE_JSON_SCHEMA) == {
        "type": "object", "additionalProperties": False,
        "required": ["reasoning", "choice"],
        "properties": {
            "reasoning": {"type": "string"},
            "choice": {"type": "string"},
        },
    }
    avoid = strict_tool_schema(CONSTRAINT_JSON_SCHEMA)["properties"]["avoid"]
    assert avoid["additionalProperties"] is False
    assert sorted(avoid["properties"]) == ["assets", "risk_groups"]


def test_the_canonical_schemas_are_not_mutated_by_adaptation():
    strict_tool_schema(TIMING_JSON_SCHEMA)
    assert TIMING_JSON_SCHEMA["properties"]["reasoning"]["minLength"] == 1


def test_timing_forces_its_own_tool_and_parses_the_action():
    decider, client = _decider(
        FakeResponse(FakeThinking(), FakeToolUse(TIMING_TOOL, TIMING_OK)))
    decision = _run(decider.timing(_obs(others=CLAIMANTS)))
    assert decision.action == "wait"
    assert decision.reasoning == "the far cone is 2h out"
    assert client.messages.calls[0]["tool_choice"] == {
        "type": "any", "disable_parallel_tool_use": True}


def test_constraints_forces_its_own_tool_and_parses_the_posture():
    decider, client = _decider(
        FakeResponse(FakeToolUse(CONSTRAINT_TOOL, CONSTRAINT_OK)))
    decision = _run(decider.constraints(_obs(others=CLAIMANTS)))
    assert decision.avoid == {"risk_groups": ["rg_T3a_t1_t3"]}
    # The posture is DERIVED from the avoid-only payload, not stated: a
    # named risk group derives risk_group/risk_group (spec 6.4).
    assert (decision.protected, decision.best_effort, decision.basis,
            decision.level) == (False, False, "risk_group", "risk_group")


def test_objective_forces_its_own_tool_and_parses_the_choice():
    decider, client = _decider(
        FakeResponse(FakeToolUse(OBJECTIVE_TOOL, OBJECTIVE_OK)))
    decision = _run(decider.objective(_obs(others=CLAIMANTS), MENU))
    assert decision.choice == "candidate_1"
    # `priority` is no longer read from a payload (spec 6.5).
    assert decision.priority is None


def test_every_request_declares_all_three_tools_so_one_cache_prefix_serves():
    # Cache invalidation hierarchy: changing `tool_choice` does NOT
    # invalidate the tools+system prefix, but changing TOOL DEFINITIONS does.
    # One tool per decision point would give three separate prefixes across
    # an hour-loop; declaring all three gives one.
    decider, client = _decider(
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)),
        FakeResponse(FakeToolUse(CONSTRAINT_TOOL, CONSTRAINT_OK)))
    obs = _obs(others=CLAIMANTS)
    _run(decider.timing(obs))
    _run(decider.constraints(obs))
    first, second = client.messages.calls
    assert [t["name"] for t in first["tools"]] == [
        TIMING_TOOL, CONSTRAINT_TOOL, OBJECTIVE_TOOL, PROBE_TOOL]
    assert first["tools"] == second["tools"]
    assert first["system"] == second["system"]


def test_the_system_block_is_cached_and_thinking_is_adaptive():
    decider, client = _decider(
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    _run(decider.timing(_obs(others=CLAIMANTS)))
    call = client.messages.calls[0]
    assert call["system"] == [{"type": "text", "text": SYSTEM_PROMPT,
                               "cache_control": {"type": "ephemeral"}}]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["model"] == "claude-sonnet-5"


def test_no_parameter_sonnet_5_rejects_is_ever_sent():
    # Sonnet 5 returns 400 for temperature/top_p/top_k and for
    # thinking.budget_tokens. A unit test is the only place this gets caught
    # before a real run burns a rollout.
    decider, client = _decider(
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    _run(decider.timing(_obs(others=CLAIMANTS)))
    call = client.messages.calls[0]
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in call
    assert "budget_tokens" not in call["thinking"]


def test_the_user_turn_carries_the_projection_not_the_raw_observation():
    decider, client = _decider(
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    _run(decider.timing(_obs(others=CLAIMANTS + BELOW_THRESHOLD)))
    content = client.messages.calls[0]["messages"][0]["content"]
    assert "d0363" in content            # a claimant above the threshold
    assert "d0001" not in content        # one below it
    assert "omitted_services" in content
    assert "n_services_total" in content


def test_the_objective_prompt_states_each_candidates_own_spare_cost():
    # cost_vector["transponders"] is 2.0 * the WHOLE network's lightpath
    # count on a clone (ledger.py's docstring); the candidate's own cost is
    # len(new_lightpaths). Handing the model the precomputed figure is the
    # only reliable way it reasons about spares correctly.
    decider, client = _decider(
        FakeResponse(FakeToolUse(OBJECTIVE_TOOL, OBJECTIVE_OK)))
    _run(decider.objective(_obs(others=CLAIMANTS), MENU))
    content = client.messages.calls[0]["messages"][0]["content"]
    menu = json.loads(content.split("\n\n")[0])["menu"]
    assert [c["spares_needed"] for c in menu["candidates"]] == [
        {"site_a": 1, "site_b": 1}, {}]
    assert [c["candidate_label"] for c in menu["candidates"]] == [
        "candidate_0", "candidate_1"]


def test_build_deciders_wiring_gap_is_closed(monkeypatch):
    """Whole-branch final review, finding 1: suite.build_deciders() builds a
    ClaudeDecider with no `oms_nodes` (suite.py never passes that kwarg), and
    -- before runner.py's fix -- nothing else ever set it either: only
    `ledger.oms_nodes` got assigned from `geometry.oms_nodes` each hour. The
    reviewer reproduced the resulting crash live, on the first real costed
    candidate: `ValueError: lightpath [...] does not resolve to exactly two
    endpoint sites (got [])`, raised out of ledger._lightpath_endpoints via
    _menu_for_prompt/spares_needed, before any API call.

    This constructs the decider exactly the way build_deciders() does (not
    the `_decider()` test helper above, which always passes oms_nodes) and
    proves both halves: unwired, a real costed candidate (MENU's
    optical_reroute, with a real new_lightpaths entry) still crashes exactly
    that way; wired the way runner.run_episode's per-hour loop now does it
    (`if hasattr(decider, "oms_nodes"): decider.oms_nodes =
    geometry.oms_nodes`, mirroring the ledger's own line beside it), the same
    menu renders onto the wire with no crash."""
    monkeypatch.setattr(
        "storm_reoptimizer.eval.suite.AGENT_AUDIT_PATH", None)
    args = SimpleNamespace(include_agent=True, agent_model=DEFAULT_MODEL)
    decider = build_deciders(args)[-1]
    assert isinstance(decider, ClaudeDecider)
    assert decider.oms_nodes == {}, (
        "build_deciders() is expected to leave oms_nodes unset -- runner.py, "
        "not suite.py, is where it gets wired")

    decider._client = FakeAnthropic(
        FakeResponse(FakeToolUse(OBJECTIVE_TOOL, OBJECTIVE_OK)))
    with pytest.raises(ValueError, match="does not resolve to exactly two"):
        _run(decider.objective(_obs(others=CLAIMANTS), MENU))

    # runner.run_episode's wiring, reproduced here rather than invoked
    # through a live rollout: `if hasattr(decider, "oms_nodes"):
    # decider.oms_nodes = geometry.oms_nodes`.
    if hasattr(decider, "oms_nodes"):
        decider.oms_nodes = OMS_NODES
    decider._client = FakeAnthropic(
        FakeResponse(FakeToolUse(OBJECTIVE_TOOL, OBJECTIVE_OK)))
    _run(decider.objective(_obs(others=CLAIMANTS), MENU))  # no crash


PROBE = {"status": "solution",
         "candidates": [{"candidate_label": "candidate_0",
                         "lever": "ip_reroute", "pairs_needed": 0},
                        {"candidate_label": "candidate_1",
                         "lever": "optical_reroute", "pairs_needed": 1}]}


def test_no_prompt_carries_the_unconstrained_menu():
    # Trace evidence showed the model treating unconstrained_menu as an
    # invitation to act on options that don't survive the real constraint
    # step -- a distractor it never cited correctly. runner.py still
    # computes, records and passes it (for the audit/viewer, and because the
    # Decider protocol requires the parameter), but no prompt -- for any of
    # the three decisions -- may put it in front of the model.
    decider, client = _decider(
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)),
        FakeResponse(FakeToolUse(CONSTRAINT_TOOL, CONSTRAINT_OK)),
        FakeResponse(FakeToolUse(OBJECTIVE_TOOL, OBJECTIVE_OK)))
    _run(decider.timing(_obs(others=CLAIMANTS)))
    _run(decider.constraints(_obs(others=CLAIMANTS), PROBE))
    _run(decider.objective(_obs(others=CLAIMANTS), MENU))
    for call in client.messages.calls:
        body = json.loads(call["messages"][0]["content"].split("\n\n")[0])
        assert "unconstrained_menu" not in body
    assert "unconstrained_menu" not in SYSTEM_PROMPT


def test_the_system_prompt_warns_that_transponders_is_a_network_wide_count():
    assert "transponders" in SYSTEM_PROMPT
    assert "spares_needed" in SYSTEM_PROMPT


def test_the_system_prompt_never_leaks_scoring_internals():
    # Coaching the model toward cites_flip_variable would pass a
    # NECESSARY-NOT-SUFFICIENT check while destroying its only purpose.
    # Merged with the mechanics-not-answers leaks (spec 3, Task 8 brief):
    # the prompt must never state the comparison or its answer.
    lowered = SYSTEM_PROMPT.lower()
    for leak in ("flip_variable", "flip variable", "pair_solved", "gold",
                 "cites_", "label_correct", "scoring", "spend",
                 "hold the spare for", "claimant"):
        assert leak not in lowered, leak
    # Episode names are checked case-sensitively: the prompt legitimately
    # uses lowercase hour labels like "t1"/"t2"/"t3" (e.g. the
    # `[t0, t1, t2, t6]` example in the action-history bullet), which a
    # lowered check would false-positive on.
    for leak in ("T1", "T2", "T3"):
        assert leak not in SYSTEM_PROMPT, leak


def test_the_system_prompt_describes_the_agents_own_action_history():
    # The payload carries these keys (W2.2); an undocumented JSON key is a
    # worse failure mode than a described one.
    assert "actions_taken" in SYSTEM_PROMPT
    assert "spares_spent" in SYSTEM_PROMPT
    # Final-review finding: a bare index into scenario.hours is misleading
    # for T2a-style non-positional hours ([t0, t1, t2, t6]) -- the prompt
    # must point the model at the resolved hour LABEL, not just the index.
    assert "effective_at_hour" in SYSTEM_PROMPT


def test_the_system_prompt_clears_sonnet_5s_minimum_cacheable_prefix():
    # Sonnet 5 will not create a cache entry below 1024 tokens -- silently,
    # with no error. ~4 chars/token is the working estimate.
    assert len(SYSTEM_PROMPT) > 4 * 1024


def test_the_system_prompt_names_the_precomputed_risk_field():
    # The payload carries this key (W3.1); an undocumented JSON key is a
    # worse failure mode than a described one.
    assert "expected_capacity_at_risk_gbps" in SYSTEM_PROMPT


def test_prompt_states_schedule_deadline_priority_and_inert_rule():
    p = SYSTEM_PROMPT
    assert "You never see a future issuance" not in p
    for needle in ("`issuance_schedule`", "`deadline_hour`", "`claim_priority`",
                   "recorded as a wait"):
        assert needle in p


def test_the_prompt_states_that_the_depot_is_shared():
    # Rewritten for the node-local depot (2026-08-30): shared with the
    # services that terminate at the SAME site, not the whole network.
    assert "That depot is SHARED" in SYSTEM_PROMPT
    assert "their claim on this site's inventory is real" in SYSTEM_PROMPT


def test_the_prompt_states_the_queue_discipline_both_ways():
    lowered = SYSTEM_PROMPT.lower()
    assert "earlier hour reaches the depot before one cut later" in lowered
    assert "larger demand has the stronger claim" in lowered


def test_the_prompt_never_states_the_comparison_or_its_answer():
    # The judgement under test. Facts about the world, never the answer key
    # (eval-fairness design, §3).
    lowered = SYSTEM_PROMPT.lower()
    for banned in ("conserve if", "if theirs is bigger", "sum the claimants",
                   "add up the", "threshold", "usually larger"):
        assert banned not in lowered


def test_the_prompt_names_the_new_path_fact_fields():
    # T1 inert-reroute finding (2026-08-31): the menu now carries these three
    # fields (runner.menu_with_path_facts); an undocumented JSON key is a
    # worse failure mode than a described one, same rule as the risk field.
    for field in ("path_delta", "residual_exposure", "collides_with_protection"):
        assert field in SYSTEM_PROMPT


def test_the_prompt_states_avoid_survival_is_not_exposure_reduction():
    # The finding's own root cause: a candidate can survive `avoid` (hard
    # polygon containment) while carrying the SAME continuous p_cut as the
    # status quo, because the two tests disagree at exactly the offset band
    # some episodes are built at. The prompt must say so in the terms the
    # payload actually uses, not leave it to be inferred from candidate
    # numbers alone.
    lowered = SYSTEM_PROMPT.lower()
    assert "changes_working_path" in lowered
    assert "not the same claim as" in lowered or \
        "is not the same as" in lowered


def test_the_prompt_never_recommends_a_candidate_over_another():
    # Facts about the world, never the answer key -- same discipline as the
    # depot-claim comparison above. The fix hands over deterministic facts;
    # which candidate to pick stays the model's judgement.
    lowered = SYSTEM_PROMPT.lower()
    for banned in ("prefer the candidate", "always choose", "never choose",
                   "candidate_0 is usually", "pick the one that"):
        assert banned not in lowered


def test_the_reasoning_instruction_fires_in_both_directions():
    assert "say what claim you preferred over the claims you did not " \
           "serve" in SYSTEM_PROMPT
    assert "If you held one, say what you held it for" in SYSTEM_PROMPT


def test_the_prompt_says_the_actionable_service_is_shown_mechanically():
    assert "not because it has the stronger claim" in SYSTEM_PROMPT
    assert "real competing claim on the same depot" in SYSTEM_PROMPT
    assert "service under test" not in SYSTEM_PROMPT


TIMING_BAD_ACTION = {"action": "hedge", "reasoning": "neither act nor wait"}
TIMING_NO_REASONING = {"action": "wait", "reasoning": "   "}


def test_three_attempts_is_the_contract():
    assert MAX_ATTEMPTS == 3


def test_an_invalid_payload_is_returned_as_an_error_tool_result_and_retried():
    decider, client = _decider(
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_BAD_ACTION,
                                 block_id="toolu_first")),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    assert _run(decider.timing(_obs(others=CLAIMANTS))).action == "wait"
    retry = client.messages.calls[1]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    result = retry[2]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_first"
    assert result["is_error"] is True
    assert "action" in result["content"]


def test_the_retry_echoes_the_assistant_content_so_thinking_blocks_survive():
    # Adaptive thinking means the assistant turn carries a thinking block
    # alongside the tool_use. Continuing the same turn requires echoing it
    # back unchanged, so the whole content list goes back, not just the text.
    thinking = FakeThinking("weighing the near cone against the spare")
    tool_use = FakeToolUse(TIMING_TOOL, TIMING_BAD_ACTION)
    decider, client = _decider(
        FakeResponse(thinking, tool_use),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    _run(decider.timing(_obs(others=CLAIMANTS)))
    echoed = client.messages.calls[1]["messages"][1]
    assert echoed["role"] == "assistant"
    assert echoed["content"] == [thinking, tool_use]


def test_a_response_with_no_tool_use_block_is_retried_as_a_correction():
    decider, client = _decider(
        FakeResponse(FakeText("I would rather discuss this in prose.")),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    assert _run(decider.timing(_obs(others=CLAIMANTS))).action == "wait"
    retry = client.messages.calls[1]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    # No tool_use block means no tool_use_id to answer, so the correction is
    # a plain user turn rather than a tool_result.
    assert isinstance(retry[2]["content"], str)
    assert TIMING_TOOL in retry[2]["content"]


def test_a_tool_use_block_naming_a_different_tool_is_not_accepted():
    decider, client = _decider(
        FakeResponse(FakeToolUse(OBJECTIVE_TOOL, OBJECTIVE_OK)),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    assert _run(decider.timing(_obs(others=CLAIMANTS))).action == "wait"
    assert len(client.messages.calls) == 2
    correction = client.messages.calls[1]["messages"][2]["content"][0]
    assert correction["type"] == "tool_result" and correction["is_error"] is True
    assert TIMING_TOOL in correction["content"]


def test_a_claim_naming_an_unshown_service_is_retried_not_accepted():
    decider, client = _decider(
        FakeResponse(FakeToolUse(TIMING_TOOL, {
            "action": "act", "reasoning": "d9999 outranks me",
            "contested_claim": {"service_id": "d9999",
                                "expected_capacity_at_risk_gbps": 900.0}},
                                 block_id="toolu_first")),
        FakeResponse(FakeToolUse(TIMING_TOOL, {
            "action": "act", "reasoning": "nothing else is exposed",
            "contested_claim": None})))
    decision = _run(decider.timing(_obs(others=CLAIMANTS)))
    assert decision.contested_claim is None
    assert len(client.messages.calls) == 2
    correction = client.messages.calls[1]["messages"][-1]["content"][0]
    assert correction["is_error"] is True
    assert "d9999" in correction["content"]


def test_a_claim_naming_a_shown_service_is_accepted_first_time():
    shown = CLAIMANTS[0][0]
    decider, client = _decider(FakeResponse(FakeToolUse(TIMING_TOOL, {
        "action": "wait", "reasoning": "they are ahead of me in the queue",
        "contested_claim": {"service_id": shown,
                            "expected_capacity_at_risk_gbps": 12.0}})))
    decision = _run(decider.timing(_obs(others=CLAIMANTS)))
    assert decision.contested_claim["service_id"] == shown
    assert len(client.messages.calls) == 1


def test_recovery_on_the_third_attempt_still_returns_a_decision():
    decider, client = _decider(
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_BAD_ACTION)),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_NO_REASONING)),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    assert _run(decider.timing(_obs(others=CLAIMANTS))).action == "wait"
    assert len(client.messages.calls) == 3


def test_exhausting_every_attempt_raises_rather_than_degrading():
    decider, client = _decider(
        *[FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_BAD_ACTION))
          for _ in range(MAX_ATTEMPTS)])
    with pytest.raises(DecisionError, match="action"):
        _run(decider.timing(_obs(others=CLAIMANTS)))
    assert len(client.messages.calls) == MAX_ATTEMPTS


def test_a_response_that_never_calls_the_tool_raises_a_decision_error():
    # Exactly one exception type ever escapes this decider, so runner.py's
    # "a decider exception kills the rollout" contract stays legible.
    decider, _ = _decider(
        *[FakeResponse(FakeText("no")) for _ in range(MAX_ATTEMPTS)])
    with pytest.raises(DecisionError, match="no tool_use block"):
        _run(decider.timing(_obs(others=CLAIMANTS)))


def test_each_decision_starts_a_fresh_conversation():
    # suite.py reuses ONE decider across all 7 episodes and every run.
    # Anything carried on self would leak one rollout's context into another.
    decider, client = _decider(
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_BAD_ACTION)),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    obs = _obs(others=CLAIMANTS)
    _run(decider.timing(obs))
    _run(decider.timing(obs))
    assert len(client.messages.calls[2]["messages"]) == 1
    assert client.messages.calls[2]["messages"][0]["role"] == "user"


def test_the_audit_sidecar_records_what_the_model_was_shown(tmp_path):
    audit = tmp_path / "agent-calls.jsonl"
    decider, _ = _decider(
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)), audit_path=audit)
    _run(decider.timing(_obs(others=CLAIMANTS + BELOW_THRESHOLD)))
    record = json.loads(audit.read_text(encoding="utf-8").strip())
    assert record["decider"] == "agent:claude-sonnet-5"
    assert record["scenario_id"] == "T3a"
    assert record["hour"] == "t1"
    assert record["decision"] == TIMING_TOOL
    assert record["attempts"] == 1
    assert record["shown_services"] == ["d0212", "d0363", "d0462",
                                        "storm-svc-1"]
    assert record["omitted_services"]["below_threshold"]["count"] == 2
    assert record["n_services_total"] == 6
    assert record["result"] == {**TIMING_OK, "contested_claim": None,
                                "claim_priority": []}


def test_the_audit_records_how_many_attempts_a_decision_took(tmp_path):
    audit = tmp_path / "agent-calls.jsonl"
    decider, client = _decider(
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_BAD_ACTION)),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)), audit_path=audit)
    _run(decider.timing(_obs(others=CLAIMANTS)))
    assert json.loads(audit.read_text(encoding="utf-8"))["attempts"] == 2


def test_no_audit_file_is_written_when_none_is_configured(tmp_path):
    decider, _ = _decider(FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    _run(decider.timing(_obs(others=CLAIMANTS)))
    assert list(tmp_path.iterdir()) == []


def test_the_audit_record_shows_the_risk_figure_for_every_shown_service(
        tmp_path):
    # W3.1's acceptance: "the audit sidecar shows the field for every shown
    # service". A bare id list cannot answer "was the claimant the episode
    # names actually shown, and at what magnitude?".
    path = tmp_path / "calls.jsonl"
    decider, _ = _decider(FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)),
                          audit_path=path)
    _run(decider.timing(_obs(others=CLAIMANTS)))
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    shown = record["shown_expected_capacity_at_risk_gbps"]
    assert sorted(shown) == record["shown_services"]
    assert shown["storm-svc-1"] == pytest.approx(0.3410 * 300.0, abs=1e-3)
    assert "rival_totals_shown" not in record


def test_the_arm_is_named_so_one_results_table_can_hold_both():
    # suite.run_suite keys results on decider.name AND derives the trace
    # filename from it (suite.py:94-95).
    decider, _ = _decider()
    assert decider.name == "agent:claude-sonnet-5"


def test_there_is_one_prompt_and_it_names_the_restorable_groups():
    # Rewritten for Task 8 (spec 4.9): `horizon_totals` -- and the
    # `sut_ecar_gbps`/`largest_restorable_group_ecar_gbps`/
    # `non_sut_ineligible_ecar_gbps` figures it carried -- left the wire in
    # an earlier task; the bullet describing them is now gone from the
    # prompt too. `restorable_groups` is the one that survives: it is the
    # only view of who shares a restoring lightpath.
    assert "restorable_groups" in SYSTEM_PROMPT
    assert not hasattr(agent_module, "SYSTEM_PROMPT_WITH_RIVAL_TOTALS")


PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


def _names(argv):
    return [d.name
            for d in build_deciders(build_arg_parser().parse_args(argv))]


def test_a_plain_suite_run_stays_free_and_keyless():
    # `python -m storm_reoptimizer.eval.suite` must not spend API credits or
    # need credentials unless it was asked to.
    assert _names([]) == ["baseline:immediate", "baseline:at_deadline"]


def test_include_agent_appends_exactly_one_claude_decider():
    assert _names(["--include-agent"]) == [
        "baseline:immediate", "baseline:at_deadline", "agent:claude-sonnet-5"]


def test_the_agent_model_is_overridable_from_the_command_line():
    assert _names(["--include-agent", "--agent-model", "claude-opus-5"])[-1] \
        == "agent:claude-opus-5"


def test_naming_a_model_without_the_flag_still_runs_baselines_only():
    assert _names(["--agent-model", "claude-opus-5"]) == [
        "baseline:immediate", "baseline:at_deadline"]


def test_only_and_runs_default_to_unset_and_the_module_constant():
    args = build_arg_parser().parse_args([])
    assert args.only is None
    from storm_reoptimizer.eval.suite import RUNS_PER_EPISODE
    assert args.runs == RUNS_PER_EPISODE


def test_only_and_runs_are_parsed_from_the_command_line():
    args = build_arg_parser().parse_args(
        ["--include-agent", "--only", "T1a,T1b", "--runs", "1"])
    assert args.only == "T1a,T1b"
    assert args.runs == 1


def test_the_agent_decider_is_not_collapsed_to_a_single_rollout():
    # collapse_deterministic special-cases ForecastBlindBaseline only, which
    # is what gives the agent its intended N=3 runs per episode for free.
    agent = build_deciders(
        build_arg_parser().parse_args(["--include-agent"]))[-1]
    assert not isinstance(agent, ForecastBlindBaseline)


def test_the_agent_name_survives_the_trace_filename_sanitizer():
    # run_suite writes traces via name.replace(":", "_"); a colon would be an
    # illegal Windows filename.
    agent = build_deciders(
        build_arg_parser().parse_args(["--include-agent"]))[-1]
    assert agent.name.replace(":", "_") == "agent_claude-sonnet-5"


def test_the_agent_arm_writes_its_audit_under_eval_traces():
    agent = build_deciders(
        build_arg_parser().parse_args(["--include-agent"]))[-1]
    assert agent._audit_path.name == "agent-calls.jsonl"
    assert agent._audit_path.parent.name == "traces"


def test_the_agent_arm_has_one_name_and_one_sidecar():
    assert _names(["--include-agent"]) == [
        "baseline:immediate", "baseline:at_deadline", "agent:claude-sonnet-5"]
    built = build_deciders(build_arg_parser().parse_args(["--include-agent"]))
    assert built[-1]._audit_path.name == "agent-calls.jsonl"


def test_the_anthropic_sdk_is_an_optional_extra_not_a_hard_dependency():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    assert project["optional-dependencies"]["agent"] == ["anthropic>=0.40"]
    assert not any(d.startswith("anthropic") for d in project["dependencies"])


def test_the_agent_and_the_trace_project_the_menu_through_one_function():
    # The recorded menu must BE what the model read, not a re-derivation that
    # can drift from it (run-viewer design, §5.1).
    from storm_reoptimizer.eval import runner as runner_mod
    assert agent_module._menu_for_prompt is runner_mod.menu_for_prompt


def test_the_decider_exposes_the_payload_it_last_put_on_the_wire():
    decider, _ = _decider()
    assert decider.last_projection is None
    obs = _obs(others=CLAIMANTS)
    payload = decider._project(obs)
    assert decider.last_projection == payload
    assert decider.last_projection is payload


def test_last_projection_is_telemetry_and_never_reaches_a_request():
    # ClaudeDecider is stateless BY CONSTRUCTION -- suite.py builds one and
    # reuses it across every episode, so per-instance memory would leak T1a's
    # context into T2b's prompt. last_projection is write-only.
    decider, fake = _decider(
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    _run(decider.timing(_obs(others=CLAIMANTS)))
    first = fake.messages.calls[-1]
    _run(decider.timing(_obs(others=CLAIMANTS)))
    assert fake.messages.calls[-1]["messages"] == first["messages"]


PROBE_OK = {"service_id": CLAIMANTS[0][0], "risk_group_id": "rg_T3a_t1_t3"}
PROBE_ANSWER = {"status": "solution", "full_restore_candidates": 1,
                "min_spares_needed_by_site": {"satna": 1, "x": 1},
                "levers": ["optical_reroute"]}


def _bound(decider, *, fail_with=None):
    """Bind a fake probe that answers PROBE_ANSWER (or raises)."""
    seen = []

    async def probe(service_id, risk_group_id):
        seen.append((service_id, risk_group_id))
        if fail_with:
            raise ProbeError(fail_with)
        return dict(PROBE_ANSWER)
    decider.bind_probe(probe)
    return seen


def test_every_request_declares_the_probe_tool_last():
    decider, client = _decider(FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    _run(decider.timing(_obs(others=CLAIMANTS)))
    call = client.messages.calls[0]
    assert [t["name"] for t in call["tools"]] == [
        TIMING_TOOL, CONSTRAINT_TOOL, OBJECTIVE_TOOL, PROBE_TOOL]
    assert call["tool_choice"] == {"type": "any",
                                   "disable_parallel_tool_use": True}


def test_a_probe_call_is_answered_and_the_decision_still_arrives():
    decider, client = _decider(
        FakeResponse(FakeToolUse(PROBE_TOOL, PROBE_OK, block_id="toolu_p")),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    seen = _bound(decider)
    assert _run(decider.timing(_obs(others=CLAIMANTS))).action == "wait"
    assert seen == [(CLAIMANTS[0][0], "rg_T3a_t1_t3")]
    second = client.messages.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    result = second[2]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_p"
    assert result["is_error"] is False
    assert json.loads(result["content"]) == PROBE_ANSWER


def test_accepted_probes_do_not_consume_attempts():
    decider, client = _decider(
        *[FakeResponse(FakeToolUse(PROBE_TOOL, PROBE_OK))
          for _ in range(MAX_ATTEMPTS + 1)],
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    _bound(decider)
    assert _run(decider.timing(_obs(others=CLAIMANTS))).action == "wait"
    assert len(client.messages.calls) == MAX_ATTEMPTS + 2


def test_a_rejected_probe_is_an_error_result_and_consumes_an_attempt():
    decider, client = _decider(
        *[FakeResponse(FakeToolUse(PROBE_TOOL, PROBE_OK))
          for _ in range(MAX_ATTEMPTS)])
    _bound(decider, fail_with="probe cap reached")
    with pytest.raises(DecisionError, match="probe"):
        _run(decider.timing(_obs(others=CLAIMANTS)))
    assert len(client.messages.calls) == MAX_ATTEMPTS
    result = client.messages.calls[1]["messages"][2]["content"][0]
    assert result["is_error"] is True and "cap" in result["content"]


def test_a_probe_with_nothing_bound_is_rejected_not_crashed():
    decider, client = _decider(
        FakeResponse(FakeToolUse(PROBE_TOOL, PROBE_OK)),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)))
    assert _run(decider.timing(_obs(others=CLAIMANTS))).action == "wait"
    result = client.messages.calls[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert "not available" in result["content"]


def test_the_audit_sidecar_records_probes(tmp_path):
    audit = tmp_path / "audit.jsonl"
    decider, _ = _decider(
        FakeResponse(FakeToolUse(PROBE_TOOL, PROBE_OK)),
        FakeResponse(FakeToolUse(TIMING_TOOL, TIMING_OK)),
        audit_path=audit)
    _bound(decider)
    _run(decider.timing(_obs(others=CLAIMANTS)))
    record = json.loads(audit.read_text().splitlines()[-1])
    assert record["probes"] == [
        {"arguments": PROBE_OK, "answer": PROBE_ANSWER, "error": None}]


def test_the_prompt_describes_the_probe_without_saying_when_to_use_it():
    p = SYSTEM_PROMPT
    assert f"`{PROBE_TOOL}`" in p
    for needle in ("full_restore_candidates", "min_spares_needed_by_site",
                   "changes nothing", str(MAX_PROBES_PER_DECISION)):
        assert needle in p
    lowered = p.lower()
    for banned in ("probe before", "you should probe", "always probe",
                   "probe when", "use the probe if", "probe the claimant",
                   "probe a competing", "probe other"):
        assert banned not in lowered


def test_the_prompt_states_that_a_restoration_may_cost_zero_spares():
    # The zero-spares-on-restore sentence lived in the claim_priority
    # bullet, which Task 8 (spec 4.3) replaced wholesale; the underlying
    # fact -- an `ip_reroute` restoration charges no transponder -- is still
    # stated, now under `spares_on_hand` where the lever costs are defined.
    assert "an `ip_reroute` consumes none" in SYSTEM_PROMPT


def test_the_prompt_keeps_the_joint_p_cut_sentence_and_drops_legs():
    assert "BOTH its working and protection paths are cut" in SYSTEM_PROMPT
    assert "`legs`" not in SYSTEM_PROMPT


def test_the_prompt_no_longer_teaches_geometry():
    p = SYSTEM_PROMPT
    for banned in ("`offset_km`", "`width_km`", "`damage_radius_km`",
                   "`unconstrained_menu`", "`cones`"):
        assert banned not in p
    assert "`horizons`" in p


def test_the_prompt_says_claim_priority_cannot_create_inventory():
    # Task 8 (spec 4.3) replaced the claim_priority bullet's own
    # "cannot create inventory" sentence with the rejection rule instead
    # (a contradictory spend is refused outright, rather than silently
    # restoring nothing); the same "you can't restore more than remains"
    # fact is still stated, in the restoration-order sentence itself.
    p = SYSTEM_PROMPT
    assert "whatever spares REMAIN" in p
    assert "one way you act on behalf" not in p


def test_the_prompt_describes_hold_and_infeasible_as_distinct():
    p = SYSTEM_PROMPT
    assert "`hold`" in p
    assert "as if you had waited" in p


def test_the_probe_section_precedes_the_three_decisions():
    p = SYSTEM_PROMPT
    assert p.index("## One question you may ask") < p.index("## The three decisions")
    assert "at any of the three decisions" in p


def test_the_first_timing_call_of_an_episode_must_state_a_ranking():
    """An empty ranking means 'keep the standing one'. On the first call
    there is none to keep, so an empty one is a refusal to answer."""
    obs = _obs()
    payload = project_observation(obs)
    payload["standing_claim_priority"] = []
    decision = TimingDecision.from_dict(
        {"reasoning": "hold", "action": "wait", "contested_claim": None,
         "claim_priority": []})
    with pytest.raises(DecisionError, match="claim_priority"):
        ClaudeDecider._check_named_services(decision, payload, TIMING_TOOL)


def test_an_empty_ranking_is_fine_once_one_is_standing():
    obs = _obs()
    payload = project_observation(obs)
    payload["standing_claim_priority"] = ["storm-svc-1"]
    decision = TimingDecision.from_dict(
        {"reasoning": "unchanged", "action": "wait", "contested_claim": None,
         "claim_priority": []})
    ClaudeDecider._check_named_services(decision, payload, TIMING_TOOL)


def test_the_projection_carries_the_revision_band_through():
    entry = {"hours_ahead": 2, "p_cut": 0.914, "demand_gbps": 300.0,
             "p_cut_if_track_revised": {"revision_radius_km": 90.0,
                                        "min": 0.09, "max": 0.59,
                                        "mean": 0.283}}
    projected = _project_exposure_entry(entry)
    assert projected["p_cut_if_track_revised"] == entry["p_cut_if_track_revised"]


def test_a_row_without_a_band_projects_without_the_key():
    entry = {"hours_ahead": 2, "p_cut": 0.914, "demand_gbps": 300.0}
    assert "p_cut_if_track_revised" not in _project_exposure_entry(entry)


def test_only_the_constraints_step_is_shown_the_groups_contents():
    """The asset rows are what make `avoid.assets` nameable, and they are
    large; the timing and objective steps have no use for them."""
    obs = dataclasses.replace(_wide_observation(), risk_group_assets=(
        {"horizon": "t3", "assets": [{"asset_id": "fiber_a_b_0",
                                      "p_cut": 0.97, "on": "working"}]},))
    assert "risk_groups" not in project_observation(obs)
    at_constraints = project_observation(obs, include_risk_group_assets=True)
    # The projected risk_groups include risk_group_id, not just the raw entry.
    assert len(at_constraints["risk_groups"]) == 1
    entry = at_constraints["risk_groups"][0]
    assert entry["horizon"] == "t3"
    assert entry["risk_group_id"] == obs.risk_group_ids.get("t3")
    assert entry["assets"] == [{"asset_id": "fiber_a_b_0",
                                "p_cut": 0.97, "on": "working"}]


def test_the_decider_shows_the_groups_contents_at_constraints_only():
    decider, _ = _decider(
        _response(TIMING_TOOL, {"reasoning": "r", "action": "act",
                                "contested_claim": None,
                                "claim_priority": ["storm-svc-1"]}),
        _response(CONSTRAINT_TOOL, {"reasoning": "r", "avoid": {"risk_groups": ["rg_T3a_t1_t3"]}}),
        _response(OBJECTIVE_TOOL, {"reasoning": "r", "choice": "hold"}))
    obs = dataclasses.replace(_wide_observation(), risk_group_assets=(
        {"horizon": "t3", "assets": []},))
    _run(decider.timing(obs))
    assert "risk_groups" not in decider.last_projection
    _run(decider.constraints(obs))
    assert "risk_groups" in decider.last_projection
    _run(decider.objective(obs, {"status": "ok", "candidates": []}))
    assert "risk_groups" not in decider.last_projection


# Task 8 (spec 4, decider-allocation-redesign): the prompt rewrite. Every
# test below names the real trace fault that made the rewrite necessary.


def test_the_timing_decision_is_described_as_the_spare_allocation():
    """Fault 1: the question was posed as information and graded as
    allocation, so at the last issuance 'wait' read as pointless."""
    assert "This decision is what allocates" in SYSTEM_PROMPT
    assert "that spare leaves\nthe depot now, before any cut, ahead of " \
           "every other service" in SYSTEM_PROMPT.replace("\\\n", "")
    assert "No other service can draw on the depot before it is cut" \
        in SYSTEM_PROMPT


def test_the_prompt_denies_the_mechanism_the_agent_invented():
    """T1a/T1b t0: 'delaying risks losing the spare to an earlier-resolving
    claim'. No such mechanism exists."""
    assert "waiting cannot lose the spare to a competitor" in SYSTEM_PROMPT


def test_the_deadline_is_described_as_a_date_not_a_cliff():
    """Fault 2: 'the t1 issuance arrives too late for the optical lever',
    where the optical deadline IS t1."""
    assert "Acting AT that hour is on time" in SYSTEM_PROMPT
    assert "Acting earlier buys nothing unless no issuance is scheduled in " \
           "between" in SYSTEM_PROMPT


def test_the_prompt_says_when_the_ranking_is_read_and_that_it_binds():
    """Fault 4: T3a ranked the SUT fourth and then spent the spare on it."""
    assert "INCLUDING the actionable service" in SYSTEM_PROMPT
    assert "the harness rejects that commit" in SYSTEM_PROMPT
    assert "standing_claim_priority" in SYSTEM_PROMPT
    assert "an empty `claim_priority` keeps it" in SYSTEM_PROMPT


def test_the_prompt_says_what_a_probe_answer_means_for_the_spare():
    """Fault 5: 8 of 8 probes named the SUT. The prompt described what the
    tool returns and never what the answer implies."""
    assert "cannot use the spare after its cut, however exposed it is" \
        in SYSTEM_PROMPT
    assert "does not need it either" in SYSTEM_PROMPT
    assert "as relevant to the\nservices you would keep the spare for" \
        in SYSTEM_PROMPT.replace("\\\n", "")


def test_the_prompt_explains_the_revision_band():
    assert "p_cut_if_track_revised" in SYSTEM_PROMPT
    assert "how much that number can change" in SYSTEM_PROMPT


def test_the_prompt_says_the_later_steps_execute_the_timing_decision():
    assert "decided_this_hour" in SYSTEM_PROMPT
    assert "EXECUTE that decision" in SYSTEM_PROMPT
    assert "answer `hold`" in SYSTEM_PROMPT
    assert "attempts_this_hour" in SYSTEM_PROMPT
    assert "Repeating an avoid set that produced no menu cannot produce one" \
        in SYSTEM_PROMPT


def test_the_constraints_block_says_a_group_can_disconnect_a_site():
    assert "a group naming all of a site's spans leaves that site unroutable" \
        in SYSTEM_PROMPT
    # The constraints paragraph now describes the real risk_groups shape: a LIST
    # with risk_group_id in each entry, not a mapping keyed by id.
    assert "`risk_group_id`" in SYSTEM_PROMPT
    assert "a LIST with one entry per horizon" in SYSTEM_PROMPT


def test_the_seven_cost_terms_are_gone():
    """Every menu in the 2026-09-09 run was eight near-identical candidates;
    the ordering never separated them (spec 2)."""
    for term in ("spectrum_used", "max_util", "added_latency", "total_margin",
                 "services_at_risk", "dropped_traffic"):
        assert term not in SYSTEM_PROMPT, term
    assert "incommensurate units" not in SYSTEM_PROMPT
    assert "spares_needed" in SYSTEM_PROMPT
    assert "cost_vector" in SYSTEM_PROMPT


def test_the_objective_instruction_offers_hold():
    """T2a t1's objective call, seeing eight spare-charging candidates and an
    instruction offering only a candidate or `infeasible`, wrote 'I'm
    choosing the candidate that gets the most value for that single spare'
    -- after its own timing step had planned a zero-spare reroute."""
    assert "`hold` to commit nothing this hour" in OBJECTIVE_INSTRUCTION


def test_the_removed_bullets_are_actually_removed():
    assert "horizon_totals" not in SYSTEM_PROMPT
    assert "non_sut_ineligible_ecar_gbps" not in SYSTEM_PROMPT
    assert "`services` -- the roster" not in SYSTEM_PROMPT
    assert "protection posture" not in SYSTEM_PROMPT
    assert "basis=" not in SYSTEM_PROMPT
    # restorable_groups SURVIVES -- it is the only view of who shares a
    # restoring lightpath.
    assert "restorable_groups" in SYSTEM_PROMPT


# test_the_prompt_still_never_states_the_comparison_or_its_answer (brief
# Step 1) is deliberately not shipped as a separate test: its forbidden-word
# list ("spend", "hold the spare for", "gold", "T1"/"T2"/"T3", "claimant")
# is merged into test_the_system_prompt_never_leaks_scoring_internals above,
# per the brief's own instruction not to ship two overlapping lists.


RISK_GROUP_ASSETS = (
    {"horizon": HORIZON,
     "assets": [{"asset_id": "fiber_satna_rewa_0", "p_cut": 0.91,
                 "on": "working"},
                {"asset_id": "fiber_satna_jhansi_0", "p_cut": 0.88,
                 "on": "protection"},
                {"asset_id": "fiber_satna_jabalpur_0", "p_cut": 0.77,
                 "on": "none"}]},
)


def _obs_with_groups(**kwargs):
    """`_obs()` plus the asset rows the constraints step is the only step to
    see. `_obs` already sets `risk_group_ids={HORIZON: "rg_T3a_t1_t3"}`, so
    the horizon here resolves to a real id."""
    return dataclasses.replace(_obs(**kwargs),
                               risk_group_assets=RISK_GROUP_ASSETS)


def test_each_projected_risk_group_entry_names_its_own_group_id():
    """The prompt tells the model to draw asset ids from a structure keyed by
    risk-group id. Before this change the wire carried a list keyed by
    horizon with no id in it at all, and the model had to cross-reference
    `risk_group_ids` to connect the two (harness explainer, §5.1's second
    consequence)."""
    payload = project_observation(_obs_with_groups(others=CLAIMANTS),
                                  include_risk_group_assets=True)
    assert [e["horizon"] for e in payload["risk_groups"]] == [HORIZON]
    entry = payload["risk_groups"][0]
    assert entry["risk_group_id"] == "rg_T3a_t1_t3"
    assert entry["risk_group_id"] == payload["risk_group_ids"][HORIZON]
    assert [a["asset_id"] for a in entry["assets"]] == [
        "fiber_satna_rewa_0", "fiber_satna_jhansi_0",
        "fiber_satna_jabalpur_0"]


def test_the_projected_entries_are_fresh_dicts_not_the_observations_own():
    """`_project_exposure_entry`'s discipline, applied here too: the payload
    is handed to json.dumps and to the trace, and nothing downstream should be
    able to reach back into a frozen Observation's rows."""
    obs = _obs_with_groups()
    payload = project_observation(obs, include_risk_group_assets=True)
    assert payload["risk_groups"][0] is not obs.risk_group_assets[0]


def test_the_risk_group_block_is_still_absent_from_timing_and_objective():
    obs = _obs_with_groups(others=CLAIMANTS)
    assert "risk_groups" not in project_observation(obs)
    assert "risk_group_assets" not in project_observation(obs)


def test_the_constraints_paragraph_describes_the_real_risk_group_shape():
    """The paragraph used to say `risk_groups[<id>].assets` -- a mapping that
    has never existed on the wire."""
    assert "risk_groups[<id>]" not in SYSTEM_PROMPT
    assert "`risk_group_id`" in SYSTEM_PROMPT


def _constraint_check(avoid, *, obs=None):
    """Run the guard the way `_decide` does: a real ConstraintDecision plus
    the constraints step's own projection."""
    obs = obs if obs is not None else _obs_with_groups(others=CLAIMANTS)
    payload = project_observation(obs, include_risk_group_assets=True)
    decision = ConstraintDecision(avoid=avoid, reasoning="r")
    ClaudeDecider._check_named_services(decision, payload, CONSTRAINT_TOOL)


@pytest.mark.parametrize("avoid", [
    {},
    {"risk_groups": []},
    {"assets": []},
    {"assets": [], "risk_groups": []},
])
def test_an_avoid_that_binds_nothing_is_rejected(avoid):
    """D1 run B seeds 1 and 2: the reasoning named the right eight fibres, the
    payload bound nothing, route_service returned the full menu including the
    current path as a zero-spare ip_reroute, and the commit was inert
    (2026-09-12 failure analysis, finding 2)."""
    with pytest.raises(DecisionError, match="binds nothing"):
        _constraint_check(avoid)


def test_an_empty_string_risk_group_id_is_rejected_as_an_unknown_id():
    """Seed 2's literal payload. It is the hallucinated-id case the guard's
    own docstring already names, walking through the one field it skipped."""
    with pytest.raises(DecisionError, match="risk_groups"):
        _constraint_check({"risk_groups": [""]})


def test_an_unknown_asset_id_is_rejected_even_beside_a_real_one():
    with pytest.raises(DecisionError, match="bogus"):
        _constraint_check({"assets": ["fiber_satna_rewa_0", "bogus"]})


def test_a_narrow_avoid_naming_real_assets_passes():
    _constraint_check({"assets": ["fiber_satna_rewa_0",
                                  "fiber_satna_jhansi_0"]})


def test_a_whole_group_avoid_naming_a_real_group_passes():
    _constraint_check({"risk_groups": ["rg_T3a_t1_t3"]})


def test_a_mixed_avoid_naming_a_real_asset_and_a_real_group_passes():
    """decisions.ConstraintDecision's own docstring documents the mixed avoid
    as legal -- the server unions both halves before build_layered_graph sees
    them -- so the guard must not reject it either."""
    _constraint_check({"assets": ["fiber_satna_rewa_0"],
                       "risk_groups": ["rg_T3a_t1_t3"]})


def test_the_guard_does_not_trap_a_model_with_nothing_to_name():
    """An observation carrying neither asset rows nor risk-group ids gives the
    model no legal avoid to state, and three rejected attempts would kill the
    rollout. Unreachable in the shipped suite (a decidable hour always has a
    group defined for its exposed horizon), but a trap is a trap."""
    bare = dataclasses.replace(_obs(), risk_group_ids={},
                               risk_group_assets=())
    _constraint_check({}, obs=bare)


def test_the_avoid_guard_does_not_fire_on_the_other_two_decisions():
    payload = project_observation(_obs_with_groups(), include_risk_group_assets=True)
    ClaudeDecider._check_named_services(
        TimingDecision.from_dict(
            {"reasoning": "r", "action": "wait",
             "claim_priority": ["storm-svc-1"]}),
        payload, TIMING_TOOL)
    ClaudeDecider._check_named_services(
        ObjectiveDecision.from_dict({"reasoning": "r", "choice": "hold"}),
        payload, OBJECTIVE_TOOL)
