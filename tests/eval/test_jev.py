"""JevDecider (spec 2026-09-28). Unit tests only: a fake TypeSafe client, no
network, no cost, no `typesafe-sdk` install."""
import asyncio
import json
import re
from types import SimpleNamespace

import pytest

import storm_reoptimizer.eval.jev as jev_module
from storm_reoptimizer.eval.agent import SYSTEM_PROMPT, ClaudeDecider
from storm_reoptimizer.eval.jev import (
    CLAIM_LEVELS,
    CLAIM_SCORE_INSTRUCTION,
    CONSTRAINT_INSTRUCTION,
    DEFAULT_JEV_MODEL,
    HOLD_CRITERION,
    INFEASIBLE_CRITERION,
    OBJECTIVE_INSTRUCTION,
    PROBE_GATE_INSTRUCTION,
    TIMING_ACTION_CRITERIA,
    TIMING_ACTION_INSTRUCTION,
    TOTALS_SENTENCE,
    JevDecider,
)
from storm_reoptimizer.eval.observation import Observation
from storm_reoptimizer.eval.scenario_file import ConeAtHorizon, Issuance

HORIZON = "t3"
DEPOT_SITE = "jalgaon"
SUT = "t1-svc-jalgaon-indore"
RG = "rg_T1a_t0_t3"
CLAIMANTS = [("claim-a", 0.30, 100.0), ("claim-b", 0.20, 100.0)]


def _obs(*, others=CLAIMANTS, risk_group_ids=None, risk_group_assets=(),
         iteration=0, last_rejection=None):
    """T1's shape at t0: the SUT plus depot-eligible claimants, each in its
    own co-terminating group, with horizon_totals populated so the `totals`
    arm has something to add."""
    exposure = {SUT: {HORIZON: {"hours_ahead": 3, "offset_km": 0.0,
                                "width_km": 200.0, "p_cut": 0.38,
                                "demand_gbps": 300.0}}}
    services = [{"id": SUT, "demand_gbps": 300.0,
                 "src_router": "router_jalgaon", "dst_router": "router_indore",
                 "working_path": ["w"], "protection_path": ["p"]}]
    groups = []
    for svc, p_cut, demand in others:
        exposure[svc] = {HORIZON: {"hours_ahead": 3, "offset_km": 50.0,
                                   "width_km": 200.0, "p_cut": p_cut,
                                   "demand_gbps": demand}}
        services.append({"id": svc, "demand_gbps": demand,
                         "src_router": "router_jalgaon",
                         "dst_router": f"router_{svc}",
                         "working_path": [], "protection_path": []})
        groups.append({"endpoints": (DEPOT_SITE, svc), "members": (svc,),
                       "ecar_gbps": round(p_cut * demand, 3)})
    return Observation(
        scenario_id="T1a", service_under_test=SUT, hour="t0", hour_index=0,
        hours_remaining=4,
        issuance=Issuance(issued_at="t0", horizons={
            HORIZON: ConeAtHorizon(cone={"type": "Polygon", "coordinates": []},
                                   width_km=200.0,
                                   center={"lat": 21.0, "lon": 75.5})}),
        exposure=exposure, services=tuple(services), spares_on_hand=1,
        lead_time_hours=1,
        risk_group_ids=(risk_group_ids if risk_group_ids is not None
                        else {HORIZON: RG}),
        risk_group_assets=tuple(risk_group_assets),
        iteration=iteration, last_rejection=last_rejection,
        restorable_groups={HORIZON: tuple(groups)} if groups else {},
        horizon_totals={HORIZON: {"sut_ecar_gbps": 114.0,
                                  "largest_restorable_group_ecar_gbps": 30.0,
                                  "non_sut_ineligible_ecar_gbps": 0.0}})


def test_arm_names_are_distinct_and_filename_safe():
    raw, totals = JevDecider(), JevDecider(include_totals=True)
    assert raw.name == f"jev:{DEFAULT_JEV_MODEL}"
    assert totals.name == f"jev:{DEFAULT_JEV_MODEL}+totals"
    assert DEFAULT_JEV_MODEL != "jev-latest"
    for name in (raw.name, totals.name):
        assert re.fullmatch(r"[A-Za-z0-9._+-]+", name.replace(":", "_"))


def test_construction_needs_neither_the_extra_nor_a_key():
    decider = JevDecider()          # must not import typesafe_sdk
    assert decider._client is None


def test_raw_projection_is_claudes_projection():
    obs = _obs()
    jev, claude = JevDecider(), ClaudeDecider(client=object())
    for kwargs in ({}, {"include_risk_group_assets": True}):
        assert jev._project(obs, **kwargs) == claude._project(obs, **kwargs)
        assert jev.last_projection == claude.last_projection
    assert "horizon_totals" not in jev.last_projection


def test_totals_projection_adds_exactly_horizon_totals():
    obs = _obs()
    raw = JevDecider()._project(obs)
    totals = JevDecider(include_totals=True)._project(obs)
    assert set(totals) - set(raw) == {"horizon_totals"}
    assert {k: v for k, v in totals.items() if k != "horizon_totals"} == raw
    assert totals["horizon_totals"] == obs.horizon_totals


def test_with_totals_appends_only_the_totals_sentence():
    assert JevDecider()._with_totals("Q?") == "Q?"
    assert JevDecider(include_totals=True)._with_totals("Q?") == \
        f"Q? {TOTALS_SENTENCE}"
    assert "horizon_totals" in TOTALS_SENTENCE


def _all_question_text():
    parts = [TIMING_ACTION_INSTRUCTION, CLAIM_SCORE_INSTRUCTION,
             CONSTRAINT_INSTRUCTION, OBJECTIVE_INSTRUCTION,
             PROBE_GATE_INSTRUCTION, *CLAIM_LEVELS,
             json.dumps(TIMING_ACTION_CRITERIA), json.dumps(HOLD_CRITERION),
             json.dumps(INFEASIBLE_CRITERION)]
    return "\n".join(parts)


def test_rule2_raw_text_never_names_horizon_totals():
    assert "horizon_totals" not in _all_question_text()


def test_rule2_observation_paths_name_only_fields_claude_is_told_about():
    claude_fields = set(re.findall(r"`([a-z_]+)`", SYSTEM_PROMPT))
    paths = re.findall(r"`observation\.([a-z_]+)", _all_question_text())
    assert paths, "the constants should point at evidence"
    assert set(paths) <= claude_fields, set(paths) - claude_fields


def test_rule4_confusable_options_carry_structured_criteria():
    for crit in (TIMING_ACTION_CRITERIA["act"], TIMING_ACTION_CRITERIA["wait"],
                 HOLD_CRITERION, INFEASIBLE_CRITERION):
        assert set(crit) == {"what", "not_for", "examples"}
        assert crit["examples"]
    assert set(TIMING_ACTION_CRITERIA) == {"act", "wait"}


def test_rule5_score_levels_carry_no_numbers():
    assert 2 <= len(CLAIM_LEVELS) <= 10
    assert not any(ch.isdigit() for level in CLAIM_LEVELS for ch in level)


# ---------------------------------------------------------------------------
# Test harness: fake TypeSafe client and timing tests
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async function synchronously in tests."""
    return asyncio.run(coro)


def _choice_answer(choice, probs=None, confidence=0.7):
    probs = probs or {choice: 1.0}
    return SimpleNamespace(type="choice", choice=choice, probabilities=probs,
                           confidence=confidence)


def _score_answer(score, confidence=0.6):
    return SimpleNamespace(type="score", score=score, confidence=confidence,
                           probabilities={round(score): 1.0})


def _noul_answer(p):
    return SimpleNamespace(type="noul", noul=p)


class FakeTypeSafe:
    """Mimics `AsyncTypeSafeClient.system_one` and the response attributes
    the decider reads. Each queued responder is a dict of answers or a
    callable(questions) -> dict of answers, keyed by question id."""

    def __init__(self, *responders):
        self.queued = list(responders)
        self.calls = []

    async def system_one(self, state, questions, *, model=None, **kwargs):
        self.calls.append({"state": state, "questions": questions,
                           "model": model})
        if not self.queued:
            raise AssertionError(
                f"decider made more Jev requests than queued ({len(self.calls)})")
        answers = self.queued.pop(0)
        if callable(answers):
            answers = answers(questions)
        def by(kind):
            return {k: v for k, v in answers.items() if v.type == kind}
        return SimpleNamespace(
            model="jev-1.13.0",
            usage=SimpleNamespace(input_tokens=1000, output_tokens=5),
            choices=by("choice"), scores=by("score"), nouls=by("noul"))


def _timing_answers(action="wait", scores=None):
    """Responder for a timing decision call: every claim_i Score gets
    `scores[service_id]` (resolved through the question text), default 2.0."""
    scores = scores or {}

    def respond(questions):
        out = {"action": _choice_answer(action, {action: 0.8, "wait" if action == "act" else "act": 0.2})}
        for qid, q in questions.items():
            if qid.startswith("claim_"):
                svc = re.search(r"`([^`]+)`", q["instructions"]).group(1)
                out[qid] = _score_answer(scores.get(svc, 2.0))
        return out
    return respond


def _decider(*responders, **kwargs):
    client = FakeTypeSafe(*responders)
    return JevDecider(client=client, **kwargs), client


def test_timing_sends_one_choice_and_one_score_per_shown_service():
    decider, fake = _decider(_timing_answers())
    _run(decider.timing(_obs()))
    (call,) = fake.calls
    qs = call["questions"]
    assert qs["action"]["type"] == "choice"
    assert set(qs["action"]["criteria"]) == {"act", "wait"}
    claims = {k: v for k, v in qs.items() if k.startswith("claim_")}
    named = {re.search(r"`([^`]+)`", q["instructions"]).group(1)
             for q in claims.values()}
    assert named == {SUT, "claim-a", "claim-b"}          # rule 1
    assert all(q["type"] == "score" and q["criteria"] == jev_module.CLAIM_LEVELS
               for q in claims.values())
    assert SUT in qs["action"]["instructions"]            # rule 1
    assert call["model"] == DEFAULT_JEV_MODEL


def test_timing_maps_action_and_sorts_claims_by_score_then_id():
    decider, _ = _decider(_timing_answers(
        "act", {SUT: 3.4, "claim-a": 1.2, "claim-b": 3.4}))
    decision = _run(decider.timing(_obs()))
    assert decision.action == "act"
    # ties at 3.4 broken by service id ascending: "claim-b" < "t1-svc-..."
    assert decision.claim_priority == ("claim-b", SUT, "claim-a")


def test_all_tied_claims_fall_back_to_service_id_order():
    decider, _ = _decider(_timing_answers(scores={}))  # every score 2.0
    decision = _run(decider.timing(_obs()))
    assert decision.claim_priority == tuple(sorted([SUT, "claim-a", "claim-b"]))


def test_raw_state_is_claudes_payload_plus_empty_probe_answers():
    obs = _obs()
    decider, fake = _decider(_timing_answers())
    _run(decider.timing(obs))
    claude_payload = ClaudeDecider(client=object())._project(obs)
    state = fake.calls[0]["state"]
    assert state == jev_module._wire({"observation": claude_payload,
                                      "probe_answers": []})
    assert state == json.loads(json.dumps(state))       # Review Focus 1


def test_totals_state_and_questions_differ_only_by_horizon_totals():
    obs = _obs()
    raw, raw_fake = _decider(_timing_answers())
    tot, tot_fake = _decider(_timing_answers(), include_totals=True)
    _run(raw.timing(obs)), _run(tot.timing(obs))
    rs, ts = raw_fake.calls[0]["state"], tot_fake.calls[0]["state"]
    assert set(ts["observation"]) - set(rs["observation"]) == {"horizon_totals"}
    ts_obs = dict(ts["observation"]); ts_obs.pop("horizon_totals")
    assert {**ts, "observation": ts_obs} == rs
    rq, tq = raw_fake.calls[0]["questions"], tot_fake.calls[0]["questions"]
    assert rq.keys() == tq.keys()
    for qid in rq:
        assert "horizon_totals" not in json.dumps(rq[qid])
        assert tq[qid]["instructions"] == f"{rq[qid]['instructions']} {TOTALS_SENTENCE}"
        assert tq[qid]["criteria"] == rq[qid]["criteria"]


def test_reused_decider_sends_identical_requests():       # Review Focus 3
    decider, fake = _decider(_timing_answers(), _timing_answers())
    _run(decider.timing(_obs()))
    _run(decider.timing(_obs()))
    assert fake.calls[0] == fake.calls[1]


def test_reasoning_is_machine_summary_that_passes_the_guard():
    decider, _ = _decider(_timing_answers("wait"))
    decision = _run(decider.timing(_obs()))
    assert decision.reasoning.startswith("jev-1.13.0 choice=wait p={")
    assert "conf=" in decision.reasoning
    assert "</" not in decision.reasoning and "<parameter" not in decision.reasoning


def test_audit_none_writes_nothing(tmp_path):              # Review Focus 5
    decider, _ = _decider(_timing_answers())
    _run(decider.timing(_obs()))
    assert list(tmp_path.iterdir()) == []


def test_audit_line_carries_base_fields_and_jev_block(tmp_path):
    path = tmp_path / "jev-calls.jsonl"
    decider, _ = _decider(_timing_answers(), audit_path=path)
    _run(decider.timing(_obs()))
    (line,) = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(line)
    assert rec["decider"] == decider.name and rec["attempts"] == 1
    for key in ("scenario_id", "hour", "iteration", "decision", "probes",
                "shown_services", "shown_expected_capacity_at_risk_gbps",
                "omitted_services", "n_services_total", "result"):
        assert key in rec
    (req,) = rec["jev"]
    assert req["purpose"] == "decision" and req["model"] == "jev-1.13.0"
    assert req["usage"] == {"input_tokens": 1000, "output_tokens": 5}
    assert req["answers"]["action"]["probabilities"]
    assert "confidence" in req["answers"]["action"]
    assert set(req["questions"]) == set(req["answers"])


def test_mapping_failure_raises_decision_error_with_raw_answers(monkeypatch):
    from storm_reoptimizer.eval.decisions import DecisionError
    decider, _ = _decider(_timing_answers())
    monkeypatch.setattr(jev_module.ClaudeDecider, "_check_named_services",
                        staticmethod(lambda *a: (_ for _ in ()).throw(
                            DecisionError("boom"))))
    with pytest.raises(DecisionError, match=r"boom.*jev answers"):
        _run(decider.timing(_obs()))
