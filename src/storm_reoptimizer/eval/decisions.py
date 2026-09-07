# src/storm_reoptimizer/eval/decisions.py
"""The three decisions a rollout asks for, and the interface both the
baseline and (in step 6) the agent implement (eval design spec, Decisions
1-3). runner.py talks only to `Decider`, so dropping an LLM in behind this
module is the whole of step 6's integration work.

Decision 1 (timing) is the one no tool produces: the server model has no
concept of time, so ACT-or-WAIT is unmodelled by anything upstream.
Decision 2 (constraints) reshapes the search space -- `avoid` is what
build_layered_graph forbids, so it changes what candidates EXIST.
Decision 3 (objective) only reorders: route_service harvests placements
first and scores them after, so weights reorder a menu that is typically 3-5
entries after dedup. That is why the choice is ordinal-or-holistic and never
a vector of floats: the seven terms are in incommensurate units (a slot
count, 2.0*n_lightpaths, a 0-1 ratio, Gbps, milliseconds, summed dB, a
service count) and total_margin is a BENEFIT subtracted while the other six
are costs. Asking for raw multipliers across those is asking for numerology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .observation import Observation

# In the order evaluate_objective reports them.
COST_TERMS = ("spectrum_used", "transponders", "max_util", "dropped_traffic",
              "added_latency", "total_margin", "services_at_risk")
BENEFIT_TERMS = frozenset({"total_margin"})

_AVOID_KEYS = {"assets", "srlgs", "risk_groups"}
_ACTIONS = {"act", "wait"}
_BASES = {"physical", "srlg", "risk_group"}
# `risk_group` is load-bearing, not decorative: T2a/T3a/T3b's gold
# constraint decision only validates under basis="risk_group"/
# level="risk_group" (the buried-shared-leg mechanic their rehearsal docs
# derive), so leaving it out made 3 of 7 episodes' gold answers
# unrepresentable by anything that goes through from_dict/the JSON schema
# -- i.e. by step 6's LLM decider. Added 2026-08-23 (whole-branch review
# finding C3).
_LEVELS = {"link", "srlg", "node", "risk_group"}


class DecisionError(ValueError):
    """A decision payload that does not satisfy its schema."""


def _reasoning(payload: dict, where: str) -> str:
    text = payload.get("reasoning")
    if not isinstance(text, str) or not text.strip():
        raise DecisionError(f"{where}: `reasoning` is mandatory and non-empty")
    return text


_CLAIM_KEYS = ("service_id", "expected_capacity_at_risk_gbps")


def _contested_claim(payload: dict, where: str) -> dict | None:
    """The strongest competing claim on the shared depot this decision
    weighed, or None for "there is none".

    Elicited, never scored: the graded labels stay exactly what they were, so
    this run's label_correct/pair_solved stay comparable with the control-arm
    findings. What it buys is the decomposition a human currently gets only by
    reading 21 rollouts of prose -- a wrong answer WITH the rival named is a
    judgement failure, the same answer with `null` is an attention failure,
    and the two need different fixes (eval-fairness design, §5.2).

    Deliberately not a spend|conserve enum: that names the axis outright, and
    the scenario files are explicit that enumerating `gold_spare_action`
    "would trivially solve every pair"."""
    claim = payload.get("contested_claim")
    if claim is None:
        return None
    if not isinstance(claim, dict):
        raise DecisionError(
            f"{where}: `contested_claim` must be an object or null")
    missing = [k for k in _CLAIM_KEYS if k not in claim]
    if missing:
        raise DecisionError(
            f"{where}: `contested_claim` is missing {missing}")
    svc = claim["service_id"]
    if not isinstance(svc, str) or not svc.strip():
        raise DecisionError(
            f"{where}: `contested_claim.service_id` must name a service")
    ecar = claim["expected_capacity_at_risk_gbps"]
    if isinstance(ecar, bool) or not isinstance(ecar, (int, float)):
        raise DecisionError(
            f"{where}: `contested_claim.expected_capacity_at_risk_gbps` "
            f"must be a number")
    return {"service_id": svc, "expected_capacity_at_risk_gbps": float(ecar)}


def _claim_priority(payload: dict, where: str) -> tuple[str, ...]:
    """An ordering over the services shown, most deserving of the depot's
    remaining spares first -- the one way a decision acts on behalf of a
    service other than the actionable one (see the agent prompt). Absent
    means "no opinion", not "nobody", so a missing key is `()`, not an
    error.

    Validated here only against SHAPE (a list of strings); that every named
    id actually appeared in the observation shown to the decider is a
    property of the payload, which `from_dict` never sees -- that check
    lives in `agent._check_named_services`, same split as
    `_contested_claim`/`_check_named_services` already draws."""
    items = payload.get("claim_priority", [])
    if not isinstance(items, list) or any(
            not isinstance(item, str) for item in items):
        raise DecisionError(
            f"{where}: `claim_priority` must be a list of service ids")
    return tuple(items)


@dataclass(frozen=True)
class TimingDecision:
    action: str          # "act" | "wait"
    reasoning: str
    # The rival claim on the shared depot this decision weighed. No longer
    # the last field -- `claim_priority` below is now -- but every existing
    # positional construction in baseline.py, assertions.py, tools/ and
    # tests/eval/test_episodes.py names at most these two trailing fields
    # positionally, so both keep working untouched.
    contested_claim: dict | None = None
    # An ordering over the services shown, most deserving first, for the
    # harness's post-cut restoration replay to consume (Task 7). Last field
    # so every positional construction above keeps working untouched.
    claim_priority: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict) -> "TimingDecision":
        reasoning = _reasoning(payload, "timing")
        action = payload.get("action")
        if action not in _ACTIONS:
            raise DecisionError(
                f"timing: `action` must be one of {sorted(_ACTIONS)}, "
                f"got {action!r}")
        return cls(action=action, reasoning=reasoning,
                   contested_claim=_contested_claim(payload, "timing"),
                   claim_priority=_claim_priority(payload, "timing"))

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reasoning": self.reasoning,
                "contested_claim": self.contested_claim,
                "claim_priority": list(self.claim_priority)}


@dataclass(frozen=True)
class ConstraintDecision:
    """The avoid set's HORIZON is the real choice here. Avoiding the full t+6
    cone frequently leaves no feasible path; avoiding only current exposure
    leaves the reroute re-exposed three hours later. Protection posture ships
    alongside it because route_service already takes protected/best_effort/
    basis/level -- deciding when to accept degraded protection is the same
    class of event-state judgement as the horizon choice, and free to expose.
    The baseline pins all four, so the fields cost the comparison nothing."""
    avoid: dict
    reasoning: str
    # The defaults are the POSTURE THIS PROJECT ACTUALLY USES, not the
    # server signature's own. Every real call site on this branch --
    # baseline.ForecastBlindBaseline.constraints, assertions.
    # PLAUSIBLE_ALTERNATIVES, assertions.menu_at_decision_hour, tools/
    # probe_episode.py, tools/build_eval_state.py, and every gold decision in
    # tests/eval/test_episodes.py -- passes protected=False/basis="physical"/
    # level="link" explicitly, and ZERO of them wanted protected=True/srlg.
    # The toy topology carries no static SRLGs (basis="srlg" is a no-op on
    # it) and protected=True populates route_service's `pairs` menu, which
    # plan_from_candidate cannot translate. Defaulting to the unused posture
    # was a pure trap for anything that constructs a ConstraintDecision
    # without naming every field. Flipped 2026-08-23 (whole-branch review
    # finding I5).
    protected: bool = False
    best_effort: bool = False
    basis: str = "physical"
    level: str = "link"
    # The rival claim on the shared depot this decision weighed. Last field so
    # every positional construction in baseline.py, assertions.py, tools/ and
    # tests/eval/test_episodes.py keeps working untouched.
    contested_claim: dict | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "ConstraintDecision":
        reasoning = _reasoning(payload, "constraints")
        avoid = payload.get("avoid")
        if not isinstance(avoid, dict):
            raise DecisionError("constraints: `avoid` must be a mapping")
        unknown = set(avoid) - _AVOID_KEYS
        if unknown:
            raise DecisionError(
                f"constraints: `avoid` has unknown key(s) {sorted(unknown)}; "
                f"allowed: {sorted(_AVOID_KEYS)}")
        basis = payload.get("basis", "physical")
        level = payload.get("level", "link")
        if basis not in _BASES:
            raise DecisionError(f"constraints: unknown basis {basis!r}")
        if level not in _LEVELS:
            raise DecisionError(f"constraints: unknown level {level!r}")
        return cls(avoid=avoid, reasoning=reasoning,
                   protected=bool(payload.get("protected", False)),
                   best_effort=bool(payload.get("best_effort", False)),
                   basis=basis, level=level,
                   contested_claim=_contested_claim(payload, "constraints"))

    def to_dict(self) -> dict[str, Any]:
        return {"avoid": self.avoid, "protected": self.protected,
                "best_effort": self.best_effort, "basis": self.basis,
                "level": self.level, "reasoning": self.reasoning,
                "contested_claim": self.contested_claim}

    def route_service_args(self, service_id: str) -> dict[str, Any]:
        return {"service_id": service_id, "protected": self.protected,
                "basis": self.basis, "level": self.level,
                "best_effort": self.best_effort, "avoid": self.avoid}


@dataclass(frozen=True)
class ObjectiveDecision:
    choice: str                          # "candidate_<i>" | "infeasible"
    priority: tuple[str, ...] | None     # interpretability artifact, optional
    reasoning: str
    # The rival claim on the shared depot this decision weighed. Last field so
    # every positional construction in baseline.py, assertions.py, tools/ and
    # tests/eval/test_episodes.py keeps working untouched.
    contested_claim: dict | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "ObjectiveDecision":
        reasoning = _reasoning(payload, "objective")
        choice = payload.get("choice")
        if not isinstance(choice, str):
            raise DecisionError("objective: `choice` must be a string")
        candidate_index(choice)          # validates the label shape
        priority = payload.get("priority")
        if priority is not None:
            unknown = [t for t in priority if t not in COST_TERMS]
            if unknown:
                raise DecisionError(
                    f"objective: `priority` names non-cost-terms {unknown}; "
                    f"allowed: {list(COST_TERMS)}")
            priority = tuple(priority)
        return cls(choice=choice, priority=priority, reasoning=reasoning,
                   contested_claim=_contested_claim(payload, "objective"))

    def to_dict(self) -> dict[str, Any]:
        return {"choice": self.choice,
                "priority": list(self.priority) if self.priority else None,
                "reasoning": self.reasoning,
                "contested_claim": self.contested_claim}


def candidate_index(choice: str) -> int | None:
    """`"candidate_2"` -> 2; `"infeasible"` -> None."""
    if choice == "infeasible":
        return None
    prefix = "candidate_"
    if not choice.startswith(prefix) or not choice[len(prefix):].isdigit():
        raise DecisionError(
            f"objective: `choice` must be 'candidate_<i>' or 'infeasible', "
            f"got {choice!r}")
    return int(choice[len(prefix):])


def rank_by_priority(candidates: list[dict],
                     priority: tuple[str, ...]) -> list[int]:
    """Candidate indices, best first, sorted lexicographically on `priority`.
    Costs ascend; total_margin is a benefit and descends. Used by the
    baseline's fixed (service_class -> priority ordering) policy, and
    available to a decider that states one."""
    def key(i: int):
        cv = candidates[i]["cost_vector"]
        return tuple(-cv.get(t, 0.0) if t in BENEFIT_TERMS else cv.get(t, 0.0)
                     for t in priority)
    return sorted(range(len(candidates)), key=key)


CONTESTED_CLAIM_SCHEMA = {
    "type": ["object", "null"],
    "additionalProperties": False,
    "required": ["service_id", "expected_capacity_at_risk_gbps"],
    "properties": {
        "service_id": {"type": "string"},
        "expected_capacity_at_risk_gbps": {"type": "number"},
    },
}

TIMING_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["action", "reasoning", "contested_claim", "claim_priority"],
    "properties": {
        "action": {"type": "string", "enum": sorted(_ACTIONS)},
        "reasoning": {"type": "string", "minLength": 1},
        "contested_claim": CONTESTED_CLAIM_SCHEMA,
        "claim_priority": {"type": "array", "items": {"type": "string"}},
    },
}

CONSTRAINT_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["avoid", "reasoning", "contested_claim"],
    "properties": {
        "avoid": {
            "type": "object", "additionalProperties": False,
            "properties": {k: {"type": "array", "items": {"type": "string"}}
                           for k in sorted(_AVOID_KEYS)},
        },
        "protected": {"type": "boolean"},
        "best_effort": {"type": "boolean"},
        "basis": {"type": "string", "enum": sorted(_BASES)},
        "level": {"type": "string", "enum": sorted(_LEVELS)},
        "reasoning": {"type": "string", "minLength": 1},
        "contested_claim": CONTESTED_CLAIM_SCHEMA,
    },
}

OBJECTIVE_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["choice", "priority", "reasoning", "contested_claim"],
    "properties": {
        "choice": {"type": "string"},
        "priority": {"type": ["array", "null"],
                     "items": {"type": "string", "enum": list(COST_TERMS)}},
        "reasoning": {"type": "string", "minLength": 1},
        "contested_claim": CONTESTED_CLAIM_SCHEMA,
    },
}


class Decider(Protocol):
    """Implemented by baseline.ForecastBlindBaseline today and by step 6's
    LLM agent tomorrow. runner.py knows nothing else about either.

    `constraints` takes the menu route_service returns under avoid={} --
    what exists BEFORE this decision narrows it. It mirrors `objective`'s
    `menu` argument rather than riding on the Observation, because an
    Observation field would enter the timing prompt too, and a timing
    decision made against a costed menu is a different experiment
    (remediation spec, W3.3).

    All three are coroutine functions (T2/T3 probe redesign, §5.2): a
    decider may `await` the harness probe it was bound with before
    answering."""

    name: str

    async def timing(self, obs: Observation) -> TimingDecision: ...

    async def constraints(self, obs: Observation,
                          unconstrained_menu: dict | None = None
                          ) -> ConstraintDecision: ...

    async def objective(self, obs: Observation, menu: dict) -> ObjectiveDecision: ...
