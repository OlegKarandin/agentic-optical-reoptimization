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

_AVOID_KEYS = {"assets", "risk_groups"}
_ACTIONS = {"act", "wait"}
# The objective step's third exit: "I looked at this hour's menu and I'd
# rather not spend anything after all" -- distinct from `infeasible`, which
# says "none of these candidates work, give me new constraints and I'll
# retry." `hold` ends the hour outright; see candidate_index and runner.py's
# iteration loop.
HOLD_CHOICE = "hold"


class DecisionError(ValueError):
    """A decision payload that does not satisfy its schema."""


# Tool-call markup a model sometimes leaks into a string argument. Seen
# twice in the 2026-09-09 run (T1a t4, T2b t3): a literal
# `</reasoning><parameter name="contested_claim">` fragment INSIDE the
# reasoning text, with the real argument then null, accepted by the
# validator. Rejecting it rides the existing MAX_ATTEMPTS retry loop in
# agent._decide, which is the only place that can ask for a clean one.
_MARKUP_MARKERS = ("</", "<parameter")


def _reasoning(payload: dict, where: str) -> str:
    text = payload.get("reasoning")
    if not isinstance(text, str) or not text.strip():
        raise DecisionError(f"{where}: `reasoning` is mandatory and non-empty")
    leaked = [m for m in _MARKUP_MARKERS if m in text]
    if leaked:
        raise DecisionError(
            f"{where}: `reasoning` contains tool-call markup {leaked} -- it "
            f"must be prose only. Put each argument in its own field.")
    return text


def _claim_priority(payload: dict, where: str) -> tuple[str, ...]:
    """An ordering over the services shown, most deserving of the depot's
    remaining spares first -- the one way a decision acts on behalf of a
    service other than the actionable one (see the agent prompt). Absent
    means "no opinion", not "nobody", so a missing key is `()`, not an
    error.

    Validated here only against SHAPE (a list of strings); that every named
    id actually appeared in the observation shown to the decider is a
    property of the payload, which `from_dict` never sees -- that check
    lives in `agent._check_named_services`."""
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
    # An ordering over the services shown, most deserving first, for the
    # harness's post-cut restoration replay to consume (Task 7). Every
    # existing positional construction in baseline.py, assertions.py,
    # tools/ and tests/eval/test_episodes.py names at most these three
    # fields positionally, so keeping it last leaves all of them working
    # untouched.
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
                   claim_priority=_claim_priority(payload, "timing"))

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reasoning": self.reasoning,
                "claim_priority": list(self.claim_priority)}


@dataclass(frozen=True)
class ConstraintDecision:
    """What the reroute must route around. The avoid set is the whole
    decision now: it is what `build_layered_graph` forbids, so it changes
    which candidates EXIST.

    The protection posture is DERIVED, not stated. Every real call site on
    this branch already passed the same one (`protected=False`,
    `best_effort=False`, physical/link unless a risk group was named), the
    toy topology carries no static SRLGs, and `protected=True` populates
    route_service's `pairs` menu that `plan_from_candidate` cannot
    translate -- so four fields on the wire bought the model four ways to
    break its own reroute and no way to improve it (spec 6.4).

    A MIXED avoid (named assets AND a named group) is legal and both halves
    bind: `multilayer_optical_network.model.placement_common
    ._forbidden_assets` unions `avoid["assets"]` with the named groups'
    members before `build_layered_graph` ever sees them, and `basis`/`level`
    reach only the disjointness comparison in `exposure.path_basis_keys`."""
    avoid: dict
    reasoning: str

    @property
    def protected(self) -> bool:
        return False

    @property
    def best_effort(self) -> bool:
        return False

    @property
    def basis(self) -> str:
        return "risk_group" if self.avoid.get("risk_groups") else "physical"

    @property
    def level(self) -> str:
        return "risk_group" if self.avoid.get("risk_groups") else "link"

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
        return cls(avoid=avoid, reasoning=reasoning)

    def to_dict(self) -> dict[str, Any]:
        return {"avoid": self.avoid, "protected": self.protected,
                "best_effort": self.best_effort, "basis": self.basis,
                "level": self.level, "reasoning": self.reasoning}

    def route_service_args(self, service_id: str) -> dict[str, Any]:
        return {"service_id": service_id, "protected": self.protected,
                "basis": self.basis, "level": self.level,
                "best_effort": self.best_effort, "avoid": self.avoid}


@dataclass(frozen=True)
class ObjectiveDecision:
    choice: str                          # "candidate_<i>" | "infeasible" | "hold"
    priority: tuple[str, ...] | None     # interpretability artifact, optional
    reasoning: str

    @classmethod
    def from_dict(cls, payload: dict) -> "ObjectiveDecision":
        reasoning = _reasoning(payload, "objective")
        choice = payload.get("choice")
        if not isinstance(choice, str):
            raise DecisionError("objective: `choice` must be a string")
        candidate_index(choice)          # validates the label shape
        # `priority` is no longer on the wire (spec 6.5): every menu in the
        # 2026-09-09 run was eight near-identical candidates and the ordering
        # never separated them. The DATACLASS field stays -- baseline.py's
        # fixed (service_class -> ordering) policy still sets it positionally
        # and rank_by_priority still consumes it -- so a model payload simply
        # always yields None.
        return cls(choice=choice, priority=None, reasoning=reasoning)

    def to_dict(self) -> dict[str, Any]:
        return {"choice": self.choice,
                "priority": list(self.priority) if self.priority else None,
                "reasoning": self.reasoning}


def candidate_index(choice: str) -> int | None:
    """`"candidate_2"` -> 2; `"infeasible"` or `"hold"` -> None."""
    if choice in ("infeasible", HOLD_CHOICE):
        return None
    prefix = "candidate_"
    if not choice.startswith(prefix) or not choice[len(prefix):].isdigit():
        raise DecisionError(
            f"objective: `choice` must be 'candidate_<i>', 'infeasible' or "
            f"'hold', got {choice!r}")
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


TIMING_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["reasoning", "claim_priority", "action"],
    # Property ORDER is load-bearing: strict tool use emits arguments in
    # schema order, so `action` last means the enum is produced AFTER the
    # reasoning that justifies it (spec 6.1).
    "properties": {
        "reasoning": {"type": "string", "minLength": 1},
        "claim_priority": {"type": "array", "items": {"type": "string"}},
        "action": {"type": "string", "enum": sorted(_ACTIONS)},
    },
}

CONSTRAINT_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["reasoning", "avoid"],
    "properties": {
        "reasoning": {"type": "string", "minLength": 1},
        "avoid": {
            "type": "object", "additionalProperties": False,
            "properties": {k: {"type": "array", "items": {"type": "string"}}
                           for k in sorted(_AVOID_KEYS)},
        },
    },
}

OBJECTIVE_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["reasoning", "choice"],
    "properties": {
        "reasoning": {"type": "string", "minLength": 1},
        "choice": {"type": "string"},
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
