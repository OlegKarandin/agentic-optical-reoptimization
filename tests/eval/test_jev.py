"""JevDecider (spec 2026-09-28). Unit tests only: a fake TypeSafe client, no
network, no cost, no `typesafe-sdk` install."""
import asyncio
import json
import re
from types import SimpleNamespace

import pytest

import storm_reoptimizer.eval.jev as jev_module
from storm_reoptimizer.eval.agent import SYSTEM_PROMPT, ClaudeDecider
from storm_reoptimizer.eval.decisions import DecisionError
from storm_reoptimizer.eval.jev import (
    CLAIM_LEVELS,
    CLAIM_SCORE_INSTRUCTION,
    CONSTRAINT_INSTRUCTION,
    DEFAULT_JEV_MODEL,
    HOLD_CRITERION,
    INFEASIBLE_CRITERION,
    OBJECTIVE_INSTRUCTION,
    PROBE_STEP_INSTRUCTION,
    PROBE_STOP_CRITERION,
    TIMING_ACTION_CRITERIA,
    TIMING_ACTION_INSTRUCTION,
    TOTALS_SENTENCE,
    JevDecider,
)
from storm_reoptimizer.eval.observation import Observation
from storm_reoptimizer.eval.probe import ProbeAnswer, ProbeBinding, ProbeError
from storm_reoptimizer.eval.scenario_file import ConeAtHorizon, Issuance

HORIZON = "t3"
DEPOT_SITE = "jalgaon"
SUT = "t1-svc-jalgaon-indore"
RG = "rg_T1a_t0_t3"
CLAIMANTS = [("claim-a", 0.30, 100.0), ("claim-b", 0.20, 100.0)]


def _obs(*, others=CLAIMANTS, risk_group_ids=None, risk_group_assets=(),
         iteration=0, last_rejection=None, decided_this_hour=None,
         probe_answers_this_episode=()):
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
        decided_this_hour=decided_this_hour,
        probe_answers_this_episode=tuple(probe_answers_this_episode),
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
             PROBE_STEP_INSTRUCTION, *CLAIM_LEVELS,
             json.dumps(TIMING_ACTION_CRITERIA), json.dumps(HOLD_CRITERION),
             json.dumps(INFEASIBLE_CRITERION), json.dumps(PROBE_STOP_CRITERION),
             json.dumps(jev_module._probe_option("svc", "rg"))]
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


def test_objective_question_points_at_the_previous_rejection():
    text = OBJECTIVE_INSTRUCTION.format(sut="svc-x")
    assert "`observation.last_rejection`" in text
    assert "`observation.iteration`" in text
    assert "svc-x" in text


def test_infeasible_criterion_does_not_claim_constraints_can_be_loosened():
    # Jev's constraints step is a Choice over whole risk groups (or a fixed
    # answer when one group is shown): it can never loosen anything.
    what = INFEASIBLE_CRITERION["what"]
    assert "loosen" not in what.lower()
    assert "asked for again" not in what
    assert set(INFEASIBLE_CRITERION) == {"what", "not_for", "examples"}


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
            choices=by("choice"), scores=by("score"))


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


def test_audit_none_writes_nothing(tmp_path, monkeypatch):  # Review Focus 5
    monkeypatch.chdir(tmp_path)
    decider, _ = _decider(_timing_answers(), audit_path=None)
    decision = _run(decider.timing(_obs()))
    assert decision.action in ("act", "wait")  # verify decision was returned
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
    decider, _ = _decider(_timing_answers())
    monkeypatch.setattr(jev_module.ClaudeDecider, "_check_named_services",
                        staticmethod(lambda *a: (_ for _ in ()).throw(
                            DecisionError("boom"))))
    with pytest.raises(DecisionError, match=r"boom.*jev answers"):
        _run(decider.timing(_obs()))


# ---------------------------------------------------------------------------
# Probe gate tests (Task 3)
# ---------------------------------------------------------------------------

PROBE_ANSWER = {"status": "no_solution", "full_restore_candidates": 0,
                "min_spares_needed_by_site": None, "levers": [], "scope": ""}


def _binding(service_ids, rgs=(RG,), max_per_decision=4):
    seen = []
    async def answer(service_id, risk_group_id):
        seen.append((service_id, risk_group_id))
        return ProbeAnswer(status="no_solution", full_restore_candidates=0,
                          min_spares_needed_by_site=None, levers=[])
    binding = ProbeBinding(answer=answer, service_ids=set(service_ids),
                           risk_group_ids=set(rgs),
                           max_per_decision=max_per_decision)
    binding.begin("timing")
    return binding, seen


def test_no_gate_request_when_nothing_is_bound():            # Review Focus 4
    decider, fake = _decider(_timing_answers())
    _run(decider.timing(_obs()))
    assert len(fake.calls) == 1
    assert fake.calls[0]["state"]["probe_answers"] == []


def test_no_gate_request_when_no_risk_group_is_shown():
    decider, fake = _decider(_timing_answers())
    decider.bind_probe(_binding([SUT])[0])
    _run(decider.timing(_obs(risk_group_ids={})))
    assert len(fake.calls) == 1
    assert fake.calls[0]["state"]["probe_answers"] == []


def _offered(questions):
    """(service_id, risk_group_id) per offered probe option, in key order."""
    return [tuple(re.findall(r"`([^`]+)`", c["what"]))
            for k, c in questions["probe"]["criteria"].items() if k != "stop"]


def _key_for(questions, service_id, risk_group_id=RG):
    for key, c in questions["probe"]["criteria"].items():
        if key != "stop" and tuple(re.findall(r"`([^`]+)`", c["what"])) == (
                service_id, risk_group_id):
            return key
    raise AssertionError(f"({service_id}, {risk_group_id}) not offered")


def _step(pick, risk_group_id=RG, p=0.7):
    """Probe-step responder: choose `pick`'s option (a service id) or "stop"."""
    def respond(questions):
        key = "stop" if pick == "stop" else _key_for(questions, pick, risk_group_id)
        probs = {k: 0.0 for k in questions["probe"]["criteria"]}
        probs[key] = p
        if key != "stop":
            probs["stop"] = round(1 - p, 3)
        return {"probe": _choice_answer(key, probs)}
    return respond


def _first_pair():
    """Probe-step responder that never stops: always the first offered pair."""
    def respond(questions):
        key = next(k for k in questions["probe"]["criteria"] if k != "stop")
        return {"probe": _choice_answer(key, {key: 0.9, "stop": 0.1})}
    return respond

THREE = [SUT, "claim-a", "claim-b"]


def test_probe_step_wording_is_the_specs():
    text = PROBE_STEP_INSTRUCTION.format(decision="timing", cap=4)
    assert text.startswith("Before the timing decision")
    assert "At most 4 questions" in text
    assert "`probe_budget.remaining`" in text
    assert text.endswith("as to the one that can act.")       # Claude's nudge
    assert set(PROBE_STOP_CRITERION) == {"what", "not_for"}
    opt = jev_module._probe_option("svc-x", "rg_y")
    assert set(opt) == {"what", "not_for"}
    assert "`svc-x`" in opt["what"] and "`rg_y`" in opt["what"]


def test_stop_at_step_zero_is_one_request_and_no_probe():
    decider, fake = _decider(_step("stop"), _timing_answers())
    binding, seen = _binding(THREE)
    decider.bind_probe(binding)
    _run(decider.timing(_obs()))
    assert seen == [] and binding.records == []
    assert len(fake.calls) == 2
    q = fake.calls[0]["questions"]
    assert set(q) == {"probe"} and q["probe"]["type"] == "choice"
    crit = q["probe"]["criteria"]
    # timing: decided_this_hour absent -> every pair offered, sorted
    assert _offered(q) == [("claim-a", RG), ("claim-b", RG), (SUT, RG)]
    assert list(crit) == ["probe_0", "probe_1", "probe_2", "stop"]
    assert crit["probe_0"] == jev_module._probe_option("claim-a", RG)
    assert crit["stop"] == PROBE_STOP_CRITERION
    assert q["probe"]["instructions"] == PROBE_STEP_INSTRUCTION.format(
        decision="timing", cap=4)
    assert fake.calls[1]["state"]["probe_answers"] == []


def test_second_step_sees_first_answer_and_a_smaller_budget():
    decider, fake = _decider(_step("claim-b"), _step("stop"), _timing_answers())
    binding, seen = _binding(THREE)
    decider.bind_probe(binding)
    _run(decider.timing(_obs()))
    assert seen == [("claim-b", RG)]
    s0, s1 = fake.calls[0]["state"], fake.calls[1]["state"]
    assert s0["probe_answers"] == []
    assert s0["probe_budget"] == {"max_per_decision": 4, "remaining": 4}
    answered = [{"service_id": "claim-b", "risk_group_id": RG,
                 "answer": PROBE_ANSWER}]
    assert s1["probe_answers"] == answered
    assert s1["probe_budget"] == {"max_per_decision": 4, "remaining": 3}
    assert ("claim-b", RG) not in _offered(fake.calls[1]["questions"])
    assert fake.calls[2]["state"]["probe_answers"] == answered


def test_the_budget_bounds_the_loop_with_no_rejected_record():
    decider, fake = _decider(_first_pair(), _first_pair(), _timing_answers())
    binding, seen = _binding(THREE, max_per_decision=2)
    decider.bind_probe(binding)
    decision = _run(decider.timing(_obs()))
    assert len(seen) == 2
    assert len(fake.calls) == 3          # 2 steps, no trailing stop, 1 decision
    assert len(binding.records) == 2
    assert all(r["error"] is None for r in binding.records)
    assert "rejected" not in decision.reasoning


def test_pairs_answered_earlier_this_hour_are_not_offered_again():
    flat = {k: v for k, v in PROBE_ANSWER.items()}
    obs = _obs(
        decided_this_hour={"timing": {"action": "act"}, "probe_answers": [
            {"service_id": "claim-a", "risk_group_id": RG, **flat}]},
        probe_answers_this_episode=(
            {"hour": "t0", "decision": "timing", "service_id": "claim-b",
             "risk_group_id": RG, **flat},))
    decider, fake = _decider(_step("stop"), {"choice": _choice_answer("hold")},
                             oms_nodes=OMS_NODES)
    binding, _ = _binding(THREE)
    binding.begin("objective")
    decider.bind_probe(binding)
    _run(decider.objective(obs, MENU))
    # claim-a answered earlier THIS hour: not offered; claim-b only in the
    # episode list (an earlier hour): still offered.
    assert _offered(fake.calls[0]["questions"]) == [("claim-b", RG), (SUT, RG)]
    assert fake.calls[0]["questions"]["probe"]["instructions"].startswith(
        "Before the objective decision")


def test_decision_state_is_claudes_payload_plus_probe_answers_only():
    obs = _obs()
    decider, fake = _decider(_step("claim-a"), _step("stop"), _timing_answers())
    decider.bind_probe(_binding(THREE)[0])
    _run(decider.timing(obs))
    claude = ClaudeDecider(client=object())._project(obs)
    assert fake.calls[-1]["state"] == jev_module._wire({
        "observation": claude,
        "probe_answers": [{"service_id": "claim-a", "risk_group_id": RG,
                           "answer": PROBE_ANSWER}]})
    assert all("probe_budget" in c["state"] for c in fake.calls[:-1])


def test_a_choice_naming_an_unoffered_key_is_a_mapping_bug():
    decider, _ = _decider(lambda qs: {"probe": _choice_answer("probe_99")})
    decider.bind_probe(_binding(THREE)[0])
    with pytest.raises(DecisionError, match="probe_99"):
        _run(decider.timing(_obs()))


def test_probe_steps_are_audited_with_decodable_options(tmp_path):
    path = tmp_path / "a.jsonl"
    decider, _ = _decider(_step("claim-b", p=0.7), _step("stop"),
                          _timing_answers(), audit_path=path)
    decider.bind_probe(_binding(THREE)[0])
    _run(decider.timing(_obs()))
    rec = json.loads(path.read_text(encoding="utf-8"))
    steps = [e for e in rec["jev"] if e["purpose"] == "probe_step"]
    assert [e["step"] for e in steps] == [0, 1]
    assert [e["remaining"] for e in steps] == [4, 3]
    chosen = steps[0]["answers"]["probe"]["choice"]
    assert steps[0]["options"][chosen] == {"service_id": "claim-b",
                                           "risk_group_id": RG}
    assert "stop" in steps[0]["answers"]["probe"]["probabilities"]
    (probe,) = rec["probes"]
    assert (probe["step"], probe["p_chosen"], probe["p_stop"]) == (0, 0.7, 0.3)
    assert "p" not in probe
    assert rec["jev"][-1]["purpose"] == "decision"


# Review Focus 1
def test_a_probe_error_ends_the_gate_but_not_the_decision():
    decider, fake = _decider(_step("claim-b"), _timing_answers())
    binding, seen = _binding([SUT, "claim-a"])        # claim-b not bindable
    decider.bind_probe(binding)
    decision = _run(decider.timing(_obs()))
    assert seen == [] and len(fake.calls) == 2         # no second step
    assert fake.calls[1]["state"]["probe_answers"] == []
    assert "claim-b@" in decision.reasoning and "rejected" in decision.reasoning


# Review Focus 2
def test_a_choice_without_a_stop_probability_records_p_stop_none(tmp_path):
    path = tmp_path / "a.jsonl"
    def respond(questions):
        key = _key_for(questions, "claim-a")
        return {"probe": _choice_answer(key, {key: 1.0})}
    decider, _ = _decider(respond, _step("stop"), _timing_answers(),
                          audit_path=path)
    decider.bind_probe(_binding(THREE)[0])
    _run(decider.timing(_obs()))
    (probe,) = json.loads(path.read_text(encoding="utf-8"))["probes"]
    assert probe["p_stop"] is None and probe["p_chosen"] == 1.0


# Review Focus 3
def test_open_pairs_over_the_choice_limit_are_truncated_in_order(
        tmp_path, monkeypatch):
    monkeypatch.setattr(jev_module, "MAX_CHOICE_OPTIONS", 3)
    path = tmp_path / "a.jsonl"
    decider, fake = _decider(_step("stop"), _timing_answers(), audit_path=path)
    decider.bind_probe(_binding(THREE)[0])
    _run(decider.timing(_obs()))
    assert _offered(fake.calls[0]["questions"]) == [("claim-a", RG),
                                                    ("claim-b", RG)]
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["jev"][0]["truncated_from"] == 3


# Review Focus 4
def test_reused_decider_sends_identical_gate_requests():
    decider, fake = _decider(_step("claim-a"), _step("stop"), _timing_answers(),
                             _step("claim-a"), _step("stop"), _timing_answers())
    decider.bind_probe(_binding(THREE)[0])
    _run(decider.timing(_obs()))
    decider.bind_probe(_binding(THREE)[0])
    _run(decider.timing(_obs()))
    assert fake.calls[:3] == fake.calls[3:]


def test_totals_probe_step_gains_only_the_totals_sentence():
    raw, rf = _decider(_step("stop"), _timing_answers())
    tot, tf = _decider(_step("stop"), _timing_answers(), include_totals=True)
    raw.bind_probe(_binding(THREE)[0]); tot.bind_probe(_binding(THREE)[0])
    _run(raw.timing(_obs())), _run(tot.timing(_obs()))
    rq, tq = rf.calls[0]["questions"]["probe"], tf.calls[0]["questions"]["probe"]
    assert tq["instructions"] == f"{rq['instructions']} {TOTALS_SENTENCE}"
    assert tq["criteria"] == rq["criteria"]


def test_nothing_left_to_ask_ends_without_a_stop_request():
    decider, fake = _decider(_first_pair(), _timing_answers())
    binding, seen = _binding([SUT])
    decider.bind_probe(binding)
    decision = _run(decider.timing(_obs(others=[])))   # one service, one group
    assert seen == [(SUT, RG)]
    assert len(fake.calls) == 2                         # no trailing stop
    assert decision.reasoning.endswith("; probe gate: nothing left to ask")


def test_budget_exhaustion_is_reported_in_the_reasoning():
    decider, _ = _decider(_first_pair(), _first_pair(), _timing_answers())
    decider.bind_probe(_binding(THREE, max_per_decision=2)[0])
    decision = _run(decider.timing(_obs()))
    assert decision.reasoning.endswith("; probe gate: budget exhausted")


# Review Focus 5
def test_a_budget_already_spent_sends_no_gate_request():
    decider, fake = _decider(_timing_answers())
    binding, seen = _binding(THREE, max_per_decision=0)
    decider.bind_probe(binding)
    decision = _run(decider.timing(_obs()))
    assert seen == [] and len(fake.calls) == 1
    assert "probe gate: budget exhausted" in decision.reasoning


def test_stop_adds_no_gate_note():
    decider, _ = _decider(_step("stop"), _timing_answers())
    decider.bind_probe(_binding(THREE)[0])
    decision = _run(decider.timing(_obs()))
    assert "probe gate:" not in decision.reasoning


def test_objective_reasoning_carries_the_gate_note():
    decider, _ = _objective_decider(_first_pair(), {"choice": _choice_answer("hold")})
    binding, _ = _binding([SUT])
    binding.begin("objective")
    decider.bind_probe(binding)
    decision = _run(decider.objective(_obs(others=[]), MENU))
    assert decision.reasoning.endswith("; probe gate: nothing left to ask")


# ---------------------------------------------------------------------------
# Constraints tests (Task 4)
# ---------------------------------------------------------------------------

def ASSETS(n):
    return [{"asset_id": f"fiber_{i}", "p_cut": 0.3, "on": "working"}
            for i in range(n)]


def test_single_group_constraints_sends_no_request():
    decider, fake = _decider()
    decision = _run(decider.constraints(_obs(), None))
    assert fake.calls == []
    assert decision.avoid == {"risk_groups": [RG]}
    assert "single option, no call" in decision.reasoning


def test_two_groups_is_a_choice_described_by_horizon_and_asset_count():
    obs = _obs(risk_group_ids={"t2": "rg_t2", "t3": "rg_t3"},
               risk_group_assets=({"horizon": "t2", "assets": ASSETS(2)},
                                  {"horizon": "t3", "assets": ASSETS(5)}))
    decider, fake = _decider({"group": _choice_answer("rg_t3")})
    decision = _run(decider.constraints(obs, None))
    q = fake.calls[0]["questions"]["group"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"rg_t2", "rg_t3"}
    assert "t3" in q["criteria"]["rg_t3"] and "5" in q["criteria"]["rg_t3"]
    assert SUT in q["instructions"]                          # rule 1
    assert "risk_groups" in fake.calls[0]["state"]["observation"]
    assert decision.avoid == {"risk_groups": ["rg_t3"]}


def test_no_group_shown_raises_rather_than_inventing_an_avoid():
    decider, _ = _decider()
    with pytest.raises(DecisionError, match="no risk group"):
        _run(decider.constraints(_obs(risk_group_ids={}), None))


def test_constraints_last_projection_is_claudes_constraints_projection():
    obs = _obs()
    decider, _ = _decider()
    _run(decider.constraints(obs, None))
    assert decider.last_projection == ClaudeDecider(client=object())._project(
        obs, include_risk_group_assets=True)


# ---------------------------------------------------------------------------
# Objective tests (Task 5)
# ---------------------------------------------------------------------------

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
                         "max_util": 0.7},
         "path_delta": {"changes_working_path": True, "oms_added": ["oms_1"],
                        "oms_removed": [], "oms_retained_cuttable": []},
         "collides_with_protection": {"collides": False,
                                      "oms_shared_with_protection": []},
         "residual_exposure": {HORIZON: {"p_cut": 0.0, "ecar_gbps": 0.0}}},
        {"lever": "ip_reroute", "reused_lightpaths": ["lp-x"],
         "new_lightpaths": [], "restored_gbps": 250.0, "shortfall_gbps": 50.0,
         "cost_vector": {"dropped_traffic": 50.0, "transponders": 38.0,
                         "services_at_risk": 0, "total_margin": 5.0,
                         "added_latency": 4.0, "spectrum_used": 18,
                         "max_util": 0.8},
         "path_delta": {"changes_working_path": False, "oms_added": [],
                        "oms_removed": [], "oms_retained_cuttable": ["oms_9"]},
         "collides_with_protection": {"collides": False,
                                      "oms_shared_with_protection": []},
         "residual_exposure": {HORIZON: {"p_cut": 0.38, "ecar_gbps": 114.0}}},
    ],
    "pairs": [],
}
OMS_NODES = {"oms_1": ["site_a", "site_b"]}


def _objective_decider(*responders, **kwargs):
    decider, fake = _decider(*responders, oms_nodes=OMS_NODES, **kwargs)
    return decider, fake


def test_objective_choice_offers_every_label_plus_hold_and_infeasible():
    decider, fake = _objective_decider({"choice": _choice_answer("candidate_0")})
    decision = _run(decider.objective(_obs(), MENU))
    q = fake.calls[0]["questions"]["choice"]
    assert set(q["criteria"]) == {"candidate_0", "candidate_1", "hold",
                                  "infeasible"}
    assert q["criteria"]["hold"] == jev_module.HOLD_CRITERION
    assert q["criteria"]["infeasible"] == jev_module.INFEASIBLE_CRITERION
    assert "optical_reroute" in q["criteria"]["candidate_0"]
    assert "spares_needed" in q["criteria"]["candidate_0"]
    assert "records as a wait" in q["criteria"]["candidate_1"]   # inert marker
    assert SUT in q["instructions"]
    assert decision.choice == "candidate_0"


def test_objective_menu_state_is_claudes_rendered_menu():
    from storm_reoptimizer.eval.runner import menu_for_prompt
    obs = _obs()
    decider, fake = _objective_decider({"choice": _choice_answer("hold")})
    decider.lit_runs = [("site_a", "site_b")]
    _run(decider.objective(obs, MENU))
    state = fake.calls[0]["state"]
    claude = ClaudeDecider(client=object())
    assert state == jev_module._wire({
        "observation": claude._project(obs),
        "menu": menu_for_prompt(MENU, OMS_NODES, lit_runs=[("site_a", "site_b")]),
        "probe_answers": []})


def test_objective_empty_menu_still_offers_hold_and_infeasible():  # Review Focus 2
    decider, fake = _objective_decider({"choice": _choice_answer("infeasible")})
    decision = _run(decider.objective(
        _obs(), {"status": "no_solution", "candidates": [], "pairs": []}))
    assert set(fake.calls[0]["questions"]["choice"]["criteria"]) == {
        "hold", "infeasible"}
    assert decision.choice == "infeasible"


def test_objective_menu_over_255_options_is_truncated_and_recorded(tmp_path):
    path = tmp_path / "a.jsonl"
    big = {**MENU, "candidates": [MENU["candidates"][1]] * 300}
    decider, fake = _objective_decider(
        {"choice": _choice_answer("candidate_252")}, audit_path=path)
    _run(decider.objective(_obs(), big))
    crit = fake.calls[0]["questions"]["choice"]["criteria"]
    assert len(crit) == 255 and "candidate_252" in crit and "candidate_253" not in crit
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["jev"][-1]["truncated_from"] == 300


def test_objective_gate_runs_before_the_choice():
    decider, fake = _objective_decider(
        _step("claim-a"), _step("stop"), {"choice": _choice_answer("hold")})
    binding, seen = _binding(THREE)
    binding.begin("objective")
    decider.bind_probe(binding)
    _run(decider.objective(_obs(), MENU))
    assert seen == [("claim-a", RG)]
    assert "menu" in fake.calls[0]["state"]
    assert fake.calls[2]["state"]["probe_answers"][0]["service_id"] == "claim-a"


def test_totals_objective_question_gains_only_the_totals_sentence():
    raw, rf = _objective_decider({"choice": _choice_answer("hold")})
    tot, tf = _objective_decider({"choice": _choice_answer("hold")},
                                 include_totals=True)
    _run(raw.objective(_obs(), MENU)), _run(tot.objective(_obs(), MENU))
    rq, tq = rf.calls[0]["questions"]["choice"], tf.calls[0]["questions"]["choice"]
    assert tq["instructions"] == f"{rq['instructions']} {TOTALS_SENTENCE}"
    assert tq["criteria"] == rq["criteria"]
