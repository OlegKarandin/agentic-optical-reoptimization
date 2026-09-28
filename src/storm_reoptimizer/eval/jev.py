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

from .agent import project_observation
from .observation import P_CUT_ENUMERATION_THRESHOLD, Observation

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
