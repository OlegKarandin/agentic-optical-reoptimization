# src/storm_reoptimizer/eval/probe.py
"""The one read-only question a decider may ask the harness (T2/T3 probe
redesign spec, 2026-09-06, §5.1): what would the routing tools offer a
service if every asset in one forecast risk group were unusable.

Answered by the SAME `route_service` posture `replay.restore_after_cuts`
uses after a real cut, annotated by `runner.menu_with_path_facts` and
costed by `ledger.spares_needed`, so a probe's answer is what the replay
would actually do -- `tests/eval/test_probe.py`'s live test asserts that
equality. `route_service` is a computation, not a commit; nothing on the
server changes.

Guards mirror `agent.ClaudeDecider._check_named_services`: a probe may name
only a service in the observation's `exposure` and only a group in its
`risk_group_ids`, and at most `MAX_PROBES_PER_DECISION` probes are answered
per decision. Every call -- accepted or rejected -- is recorded, and the
runner writes the records to the hour's trace under `probes`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

from .ledger import spares_needed

PROBE_TOOL = "probe_restorability"
MAX_PROBES_PER_DECISION = 4

PROBE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "service_id": {"type": "string"},
        "risk_group_id": {"type": "string"},
    },
    "required": ["service_id", "risk_group_id"],
    "additionalProperties": False,
}


class ProbeError(ValueError):
    """A probe the harness refuses to answer: unshown service, unknown risk
    group, over the per-decision cap, or outside any decision."""


def probe_scope(risk_group_id: str) -> str:
    """The one sentence that stops a `no_solution` reading as "nothing can
    save this service".

    The probe can only ask about a WHOLE group -- that is what makes its
    answer equal to what `replay.restore_after_cuts` would actually do, and
    the reason there is deliberately no narrower-avoid argument (design §4.3).
    D1 run A read a correct `no_solution` under `rg_D1_t0_t1` as a statement
    about every possible constraint and ended the episode at the timing step
    (2026-09-12 failure analysis, finding 1). One phrasing, built here, so the
    answer and any test of it can never drift apart."""
    return (f"answered while avoiding every asset in {risk_group_id}; "
            f"narrower avoid sets were not evaluated")


@dataclass(frozen=True)
class ProbeAnswer:
    status: str | None
    full_restore_candidates: int
    min_spares_needed_by_site: dict[str, int] | None
    levers: tuple[str, ...]
    scope: str = ""

    def to_dict(self) -> dict:
        return {"status": self.status,
                "full_restore_candidates": self.full_restore_candidates,
                "min_spares_needed_by_site": (
                    dict(self.min_spares_needed_by_site)
                    if self.min_spares_needed_by_site is not None else None),
                "levers": list(self.levers),
                "scope": self.scope}


def _spare_cost(needed: dict[str, int]) -> tuple[int, list[tuple[str, int]]]:
    return sum(needed.values()), sorted(needed.items())


def unrestorable_under(answers: Iterable[dict],
                       current_group_ids: Iterable[str]) -> frozenset[str]:
    """Services whose LATEST answer under a CURRENT group is `no_solution`
    (spec §4.7). `answers` is typically the episode's carried
    `probe_answers_this_episode` plus this hour's own new answers; only
    entries whose `risk_group_id` is in `current_group_ids` are considered --
    a stale answer, from a superseded group, never enters the result, so the
    "superseded group -> rejected" case is handled by this filter alone, with
    no separate staleness check needed.

    Only the agent's OWN probes count: a harness-side restorability check
    here would make `ranking_conflict`'s accept/reject decision itself reveal
    which way a T2-shaped pair flips, for a claimant the agent never asked
    about."""
    current = set(current_group_ids)
    latest: dict[str, str | None] = {}
    for a in answers:
        if a["risk_group_id"] in current:
            latest[a["service_id"]] = a.get("status")
    return frozenset(s for s, status in latest.items() if status == "no_solution")


async def answer_probe(call, *, service_id: str, risk_group_id: str,
                       geometry, issuance, damage_radius_km: float,
                       demands: dict[str, float],
                       lit_runs: Iterable[tuple[str, str]] = ()) -> ProbeAnswer:
    """`route_service` for `service_id` avoiding `risk_group_id`, in the
    replay's own posture, reduced to the facts a decider may see. `call` is
    an `async call(name, arguments) -> dict`.

    `lit_runs` is this rollout's own `SpareLedger.lit_runs` at the moment
    of the probe -- threaded in from the caller (`ProbeBinding`'s binder in
    runner.py), never a whole ledger, so this module stays read-only. Load-
    bearing for the module docstring's own promise: `spares_needed`'s
    mate-pairing rule must see the SAME lit runs the replay would, or a
    probe answer and the replay's real spend silently diverge."""
    from .runner import menu_with_path_facts   # lazy: runner imports this module

    if service_id not in demands:
        raise ProbeError(
            f"no demand is known for service {service_id!r}; it is not on "
            f"this hour's roster")
    menu = await call("route_service", {
        "service_id": service_id, "protected": False, "basis": "physical",
        "level": "link", "best_effort": False,
        "avoid": {"risk_groups": [risk_group_id]}})
    menu = menu_with_path_facts(
        menu, geometry, service_id, issuance=issuance,
        damage_radius_km=damage_radius_km, demand_gbps=demands[service_id])
    workable = [c for c in (menu.get("candidates") or [])
                if c.get("shortfall_gbps") == 0
                and c.get("path_delta", {}).get("changes_working_path")]
    costs = [spares_needed(c, geometry.oms_nodes, lit_runs=lit_runs)
            for c in workable]
    cheapest = min(costs, key=_spare_cost) if costs else None
    return ProbeAnswer(
        status=menu.get("status"),
        full_restore_candidates=len(workable),
        min_spares_needed_by_site=cheapest,
        levers=tuple(sorted({c["lever"] for c in workable})),
        scope=probe_scope(risk_group_id))


class ProbeBinding:
    """One per hour of one rollout. `begin(decision)` opens a decision and
    resets its cap; `await binding(service_id, risk_group_id)` answers or
    raises `ProbeError`; `records` is what the trace stores."""

    def __init__(self, *, answer: Callable[[str, str], Awaitable[ProbeAnswer]],
                 service_ids, risk_group_ids,
                 max_per_decision: int = MAX_PROBES_PER_DECISION) -> None:
        self._answer = answer
        self.service_ids = set(service_ids)
        self.risk_group_ids = set(risk_group_ids)
        self.max_per_decision = max_per_decision
        self.decision: str | None = None
        self._count = 0
        self.records: list[dict[str, Any]] = []

    def begin(self, decision: str) -> None:
        self.decision = decision
        self._count = 0

    async def __call__(self, service_id: str, risk_group_id: str) -> dict:
        record = {"decision": self.decision, "service_id": service_id,
                  "risk_group_id": risk_group_id, "answer": None, "error": None}
        self.records.append(record)
        try:
            if self.decision is None:
                raise ProbeError("no decision is in progress; the probe is "
                                 "only answered while a decision is being made")
            if self._count >= self.max_per_decision:
                raise ProbeError(
                    f"probe cap reached: at most {self.max_per_decision} "
                    f"probes are answered per decision")
            if service_id not in self.service_ids:
                raise ProbeError(
                    f"service_id {service_id!r} is not in this observation's "
                    f"`exposure`; name a service you were shown")
            if risk_group_id not in self.risk_group_ids:
                raise ProbeError(
                    f"risk_group_id {risk_group_id!r} is not one of this "
                    f"observation's `risk_group_ids`")
        except ProbeError as exc:
            record["error"] = str(exc)
            raise
        self._count += 1
        answer = await self._answer(service_id, risk_group_id)
        record["answer"] = answer.to_dict()
        return record["answer"]
