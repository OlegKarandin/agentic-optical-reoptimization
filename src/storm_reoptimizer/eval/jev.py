"""A System-1 decision arm: TypeSafe's Jev behind the `Decider` protocol
(spec docs/superpowers/specs/2026-09-28-jev-decider-design.md).

Jev answers typed questions (Choice / Score / Noul) with probabilities; it
does not generate text. Every request's `state` is built by the SAME
functions ClaudeDecider uses, so the `raw` arm sees exactly Claude's payload
and the `totals` arm sees that plus `horizon_totals` -- nothing else. The
question wording below is the experiment's independent variable, so it lives
in one block for review (spec §3.2c)."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .agent import (
    TIMING_TOOL,
    ClaudeDecider,
    _peak_capacity_at_risk_gbps,
    project_observation,
)
from .decisions import DecisionError, TimingDecision
from .observation import P_CUT_ENUMERATION_THRESHOLD, Observation
from .probe import ProbeError

# ---------------------------------------------------------------------------
# Constants: model, thresholds, and every word Jev is shown (spec §3.2c).
# ---------------------------------------------------------------------------

# Pinned, never `jev-latest`: two runs must not silently compare different
# models. On 2026-09-28 docs.typesafe.ai/models lists jev-1.13.0 as the only
# version (jev-latest and jev-preview both alias it).
DEFAULT_JEV_MODEL = "jev-1.13.0"
# Spec §3.3 step 2: probing is cheap and a missed probe decides T2/T3, but a
# gate low enough to probe everything stops T2/T3 measuring the choice. Every
# pair's p is audited so the gate's calibration can be read back.
PROBE_GATE_THRESHOLD = 0.5
MAX_CHOICE_OPTIONS = 255   # TypeSafe Choice limit

TOTALS_SENTENCE = (
    "`observation.horizon_totals` gives, per horizon, `sut_ecar_gbps` (the "
    "actionable service's expected capacity at risk), "
    "`largest_restorable_group_ecar_gbps` (the largest co-terminating "
    "group's) and `non_sut_ineligible_ecar_gbps` (services that do not "
    "terminate at the depot).")

TIMING_ACTION_INSTRUCTION = (
    "A storm forecast threatens this network. The depot's spare transponders "
    "(`observation.spares_on_hand`) are shared by every shown service that "
    "terminates there. Decide, for the actionable service `{sut}`, whether "
    "to act this hour or wait for the next forecast issuance "
    "(`observation.next_issuance`, `observation.deadline_hour`). The evidence "
    "is in `observation.exposure`, `observation.co_terminating_groups` and "
    "`observation.probe_answers_this_episode`, plus this hour's own "
    "restorability answers in `probe_answers`.")

TIMING_ACTION_CRITERIA = {
    "act": {
        "what": (
            "Route the actionable service this hour; if the candidate "
            "then chosen needs a spare, that spare leaves the depot now, "
            "before any cut, ahead of every other service."
        ),
        "not_for": (
            "Keeping the spare in the depot for whichever service a "
            "cut actually drops, or buying the next issuance."
        ),
        "examples": [
            (
                "Acting at `observation.deadline_hour` is on time; acting later "
                "lands after the cut."
            ),
        ],
    },
    "wait": {
        "what": (
            "The spare stays in the depot this hour. It is still there "
            "for the actionable service next hour, and it is there for "
            "whichever service a cut actually drops if it is never "
            "committed."
        ),
        "not_for": "Routing the actionable service now.",
        "examples": [
            (
                "No other service can draw on the depot before it is cut, so "
                "waiting cannot lose the spare to a competitor."
            ),
            (
                "Waiting costs lead time and buys the next issuance, if "
                "`observation.next_issuance` says one is still scheduled before "
                "`observation.deadline_hour`."
            ),
        ],
    },
}

CLAIM_SCORE_INSTRUCTION = (
    "How strong is `{service_id}`'s claim on the depot's remaining spare "
    "transponders this hour, if the storm cuts it? Read `{service_id}`'s rows "
    "in `observation.exposure` (`p_cut`, `expected_capacity_at_risk_gbps`), "
    "the group containing `{service_id}` in "
    "`observation.co_terminating_groups`, and any answer about "
    "`{service_id}` in `observation.probe_answers_this_episode` or "
    "`probe_answers`.")

# Situations, not degrees; no numbers (Score docs; spec §3.2c rule 5).
CLAIM_LEVELS = [
    ("The service is not exposed to any forecast cut in this storm, or "
     "cannot use a spare from this depot after its cut: a current probe "
     "answer says no full-restore candidate exists, or its cheapest "
     "restoration needs no spare at the depot."),
    ("The service is exposed only to an unlikely cut, and the capacity it "
     "would lose is small."),
    ("The service is exposed to a cut that is either likely but small in "
     "lost capacity, or large in lost capacity but unlikely."),
    ("The service is exposed to a likely cut with a large expected loss of "
     "capacity, and nothing shown says whether a spare from this depot "
     "could restore it."),
    ("The service is exposed to a likely cut with a large expected loss of "
     "capacity that only a spare from this depot can restore."),
]

CONSTRAINT_INSTRUCTION = (
    "Which risk group in `observation.risk_group_ids` must the reroute of "
    "`{sut}` route around? Every asset in the chosen group becomes unusable "
    "for the reroute; `observation.risk_groups` lists each group's assets.")

OBJECTIVE_INSTRUCTION = (
    "Choose the reroute of `{sut}` to commit this hour from the menu "
    "(`menu.candidates`, each named by its `candidate_label`), `infeasible` "
    "if none is acceptable under these constraints, or `hold` to commit "
    "nothing this hour. Reason about spares from each candidate's "
    "`spares_needed` and `observation.spares_on_hand`; "
    "`observation.decided_this_hour` is the timing decision this choice "
    "executes.")

HOLD_CRITERION = {
    "what": (
        "Commit nothing this hour after seeing the menu: the hour ends "
        "exactly as if the timing decision had been to wait, nothing is "
        "spent, and the spare stays in the depot."
    ),
    "not_for": (
        "A menu on which no candidate is acceptable under the "
        "constraints set -- that is `infeasible`."
    ),
    "examples": [
        (
            "The menu holds nothing consistent with the timing decision made "
            "this hour."
        ),
        (
            "A feasible candidate exists but spending the depot's spare now is "
            "worse than keeping it."
        ),
    ],
}

INFEASIBLE_CRITERION = {
    "what": (
        "None of these candidates is acceptable under the constraints "
        "set; constraints are asked for again and can be loosened."
    ),
    "not_for": (
        "Declining to spend when an acceptable candidate exists -- "
        "that is `hold`."
    ),
    "examples": [
        (
            "No candidate on this menu is acceptable under the avoid set this "
            "reroute was routed under; looser constraints could produce a "
            "different menu."
        ),
    ],
}

PROBE_GATE_INSTRUCTION = (
    "Would knowing whether `{service_id}` can be restored after its cut, "
    "with every asset in `{risk_group_id}` avoided, change the spend-or-hold "
    "decision the depot faces this hour? "
    "`observation.probe_answers_this_episode` lists what is already known.")


def _choice(instructions, criteria: dict) -> dict:
    return {"type": "choice", "instructions": instructions,
            "criteria": criteria}


def _score(instructions, levels) -> dict:
    return {"type": "score", "instructions": instructions,
            "criteria": list(levels)}


def _noul(instructions) -> dict:
    return {"type": "noul", "instructions": instructions}


def _wire(state: dict) -> dict:
    """The state exactly as Claude's prompt renders it (agent.py's
    `json.dumps(..., sort_keys=True, default=str)`), parsed back: the SDK's
    strict pydantic types reject tuples and other non-JSON values."""
    return json.loads(json.dumps(state, sort_keys=True, default=str))


def _answers_record(response, questions: dict) -> dict:
    out = {}
    for qid, q in questions.items():
        if q["type"] == "choice":
            a = response.choices[qid]
            out[qid] = {"choice": a.choice, "confidence": a.confidence,
                        "probabilities": dict(a.probabilities)}
        elif q["type"] == "score":
            a = response.scores[qid]
            out[qid] = {"score": a.score, "confidence": a.confidence,
                        "probabilities": {str(k): v for k, v
                                          in a.probabilities.items()}}
        else:
            out[qid] = {"noul": response.nouls[qid].noul}
    return out


def _summary(model: str, answer, probes: list[dict],
             extra: dict | None = None) -> str:
    """Machine-generated `reasoning` (spec §3.4). No markup can appear, so
    scoring.cites_flip_variable_frac reads ~0 for these arms by
    construction -- stated in the results table, not hidden."""
    top = sorted(answer.probabilities.items(),
                 key=lambda kv: (-kv[1], str(kv[0])))[:4]
    probs = ", ".join(f"{k}:{v:.2f}" for k, v in top)
    text = (f"{model} choice={answer.choice} p={{{probs}}} "
            f"conf={answer.confidence:.2f}")
    if extra:
        text += "; claim " + ", ".join(f"{k}:{v:.2f}" for k, v in extra.items())
    if probes:
        rendered = []
        for r in probes:
            args = r["arguments"]
            outcome = (f"rejected ({r['error']})" if r["error"] is not None
                       else (r["answer"] or {}).get("status"))
            rendered.append(f"{args['service_id']}@{args['risk_group_id']} "
                            f"-> {outcome}")
        text += "; probed [" + ", ".join(rendered) + "]"
    return text


class JevDecider:
    """Standalone replacement decider (spec §3.1). Stateless across calls
    for the same reason as ClaudeDecider (agent.py:746): suite.py builds one
    instance and reuses it across every episode."""

    def __init__(self, model: str = DEFAULT_JEV_MODEL, *,
                 include_totals: bool = False, client=None,
                 audit_path: str | Path | None = None,
                 p_cut_threshold: float = P_CUT_ENUMERATION_THRESHOLD,
                 oms_nodes: dict[str, list[str]] | None = None,
                 lit_runs: Iterable[tuple[str, str]] | None = None) -> None:
        self.model = model
        self.include_totals = include_totals
        self.name = f"jev:{model}+totals" if include_totals else f"jev:{model}"
        self._client = client
        self._p_cut_threshold = p_cut_threshold
        self._audit_path = Path(audit_path) if audit_path else None
        # Write-only telemetry the runner reads (runner.py:1198, 1254, 1280).
        self.last_projection: dict | None = None
        # Set by the runner every hour (runner.py:1074-1082).
        self.oms_nodes = oms_nodes or {}
        self.lit_runs = list(lit_runs) if lit_runs else []
        self._probe = None

    def bind_probe(self, probe) -> None:
        self._probe = probe

    @property
    def _api(self):
        if self._client is None:
            from typesafe_sdk import AsyncTypeSafeClient  # pip install -e ".[jev]"
            self._client = AsyncTypeSafeClient(model=self.model)
        return self._client

    def _project(self, obs: Observation, *,
                 include_risk_group_assets: bool = False) -> dict:
        payload = project_observation(
            obs, p_cut_threshold=self._p_cut_threshold,
            include_risk_group_assets=include_risk_group_assets)
        if self.include_totals:
            # Spec §3.2b: the arm's ONLY difference. Removed from Claude's
            # wire as redundant (agent.py:257-267), not withheld.
            payload["horizon_totals"] = {
                h: dict(v) for h, v in obs.horizon_totals.items()}
        self.last_projection = payload
        return payload

    def _with_totals(self, text: str) -> str:
        return f"{text} {TOTALS_SENTENCE}" if self.include_totals else text

    async def _ask(self, state: dict, questions: dict, log: list,
                   purpose: str):
        response = await self._api.system_one(
            state=_wire(state), questions=questions, model=self.model)
        log.append({
            "purpose": purpose, "questions": questions,
            "answers": _answers_record(response, questions),
            "model": response.model,
            "usage": {"input_tokens": response.usage.input_tokens,
                      "output_tokens": response.usage.output_tokens}})
        return response

    def _validated(self, cls, raw: dict, payload: dict, tool_name: str,
                   obs, probes: list, log: list):
        """Jev can only answer with offered options, so a failure here is a
        mapping bug, not a model error (spec §3.3): no retry, raise loudly."""
        try:
            decision = cls.from_dict(raw)
            ClaudeDecider._check_named_services(decision, payload, tool_name)
        except DecisionError as exc:
            answers = json.dumps([r["answers"] for r in log], default=str)
            raise DecisionError(
                f"{self.name}: mapped answer failed validation ({exc}); "
                f"jev answers: {answers}") from exc
        self._audit(obs, payload, tool_name, decision, probes, log)
        return decision

    def _audit(self, obs, payload, tool_name, decision, probes, log) -> None:
        if self._audit_path is None:
            return
        record = {
            "decider": self.name,
            "scenario_id": obs.scenario_id,
            "hour": obs.hour,
            "iteration": obs.iteration,
            "decision": tool_name,
            "attempts": 1,
            "probes": probes,
            "shown_services": sorted(payload["exposure"]),
            "shown_expected_capacity_at_risk_gbps": {
                svc: round(_peak_capacity_at_risk_gbps(per_horizon), 3)
                for svc, per_horizon in payload["exposure"].items()},
            "omitted_services": payload["omitted_services"],
            "n_services_total": payload["n_services_total"],
            "result": decision.to_dict(),
            "jev": log,
        }
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    async def _probe_gate(self, base: dict, log: list
                          ) -> tuple[list[dict], list[dict]]:
        """Spec §3.3: one batched Noul per (shown service, shown group);
        probe pairs with p >= PROBE_GATE_THRESHOLD, p descending, then
        service id, then group id. The binding's own cap is the cap: a
        ProbeError is recorded and skipped -- the gate never kills a
        decision."""
        payload = base["observation"]
        services = sorted(payload["exposure"])
        groups = sorted(set((payload.get("risk_group_ids") or {}).values()))
        if self._probe is None or not services or not groups:
            return [], []
        pairs = [(s, g) for s in services for g in groups]
        questions = {
            f"probe_{i}": _noul(PROBE_GATE_INSTRUCTION.format(
                service_id=s, risk_group_id=g))
            for i, (s, g) in enumerate(pairs)}
        response = await self._ask(base, questions, log, "probe_gate")
        wanted = sorted(
            ((response.nouls[f"probe_{i}"].noul, s, g)
             for i, (s, g) in enumerate(pairs)
             if response.nouls[f"probe_{i}"].noul >= PROBE_GATE_THRESHOLD),
            key=lambda t: (-t[0], t[1], t[2]))
        answers, probes = [], []
        for p, service_id, risk_group_id in wanted:
            try:
                answer, error = await self._probe(service_id, risk_group_id), None
            except ProbeError as exc:
                answer, error = None, str(exc)
            probes.append({"arguments": {"service_id": service_id,
                                         "risk_group_id": risk_group_id},
                           "p": p, "answer": answer, "error": error})
            if error is None:
                answers.append({"service_id": service_id,
                                "risk_group_id": risk_group_id,
                                "answer": answer})
        return answers, probes

    async def timing(self, obs: Observation) -> TimingDecision:
        payload = self._project(obs)
        base = {"observation": payload}
        log: list = []
        probe_answers, probes = await self._probe_gate(base, log)
        services = sorted(payload["exposure"])
        sut = payload["actionable_service"]
        questions = {"action": _choice(
            self._with_totals(TIMING_ACTION_INSTRUCTION.format(sut=sut)),
            TIMING_ACTION_CRITERIA)}
        for i, svc in enumerate(services):
            questions[f"claim_{i}"] = _score(
                self._with_totals(
                    CLAIM_SCORE_INSTRUCTION.format(service_id=svc)),
                CLAIM_LEVELS)
        response = await self._ask(
            {**base, "probe_answers": probe_answers}, questions, log,
            "decision")
        action = response.choices["action"]
        # Each Score is an absolute, independent judgement (spec §3.3);
        # `score` is already probability-weighted -- not recomputed. Ties are
        # expected; the id tie-break is load-bearing and tested.
        scores = {svc: response.scores[f"claim_{i}"].score
                  for i, svc in enumerate(services)}
        claim_priority = sorted(services, key=lambda s: (-scores[s], s))
        raw = {"action": action.choice, "claim_priority": claim_priority,
               "reasoning": _summary(response.model, action, probes, scores)}
        return self._validated(TimingDecision, raw, payload, TIMING_TOOL,
                               obs, probes, log)
