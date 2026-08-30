# src/storm_reoptimizer/eval/agent.py
"""The LLM decider (agent decider design spec, 2026-08-24, build order step
6). One class behind the `Decider` protocol decisions.py already defines, so
runner.py needs no change at all -- that was the whole point of the
interface.

Two things this module owns that the spec did not cost out, and which are the
difference between a suite run costing single-digit dollars and one costing
hundreds:

  * `Observation.to_dict()` is the intended prompt payload, but against
    eval/states/loaded-s17.json it carries 573 services (97,654 JSON chars
    for the roster alone) plus one `exposure` entry per service PER HORIZON.
    Raw, that is ~50-70K tokens on each of the up-to-11 calls a single
    acting hour makes -- and nearly all of it is services hundreds of
    kilometres from the cone. `project_observation` trims it.
  * Trimming silently would be worse than not trimming. Every projected
    payload therefore carries `n_services_total` and an `omitted_services`
    summary (how many, the largest p_cut among them, their summed expected
    capacity at risk), so the decider can see the shape of what it was not
    shown rather than being quietly misled about the network's scale.

The threshold is not invented here. scenarios/T3a.yaml's own authoring note
records that its three named claimants "are the only non-SUT services above
p_cut 0.005 at this horizon in EITHER half" -- 0.005 is the enumeration
cut-off the episode authors already reasoned against, so the projection's
notion of "relevant" is the gold rationale's notion of "relevant"."""
from __future__ import annotations

import json
from pathlib import Path

from .cone import expected_capacity_at_risk_gbps
from .decisions import (
    CONSTRAINT_JSON_SCHEMA, ConstraintDecision, DecisionError,
    OBJECTIVE_JSON_SCHEMA, ObjectiveDecision, TIMING_JSON_SCHEMA,
    TimingDecision,
)
from .observation import Observation
from .runner import menu_for_prompt as _menu_for_prompt

# See the module docstring: scenarios/T3a.yaml's t3 cone comment. Keyed on
# p_cut rather than cone containment because D1 is built to punish the
# containment test -- its SUT sits at offset 58.4 km against a 7.5 km
# half-width (outside the polygon) with p_cut 0.976.
P_CUT_ENUMERATION_THRESHOLD = 0.005


def _peak_p_cut(per_horizon: dict) -> float:
    """The largest cut probability this service carries at any horizon of the
    current issuance. Peak, not sum: the horizons are alternative futures of
    one storm, not independent events."""
    return max((float(e["p_cut"]) for e in per_horizon.values()), default=0.0)


def _peak_capacity_at_risk_gbps(per_horizon: dict) -> float:
    return max((expected_capacity_at_risk_gbps(float(e["p_cut"]),
                                               float(e["demand_gbps"]))
                for e in per_horizon.values()), default=0.0)


def _depot_eligible(svc: str, obs: Observation) -> bool:
    """Whether `svc` terminates at this episode's depot site, so a spare
    spent there is a real claim against the same inventory `spares_on_hand`
    counts.

    Read off `restorable_groups` rather than a second, independent notion of
    "terminates at depot_site": Task 10's co-terminating grouping already
    excludes every service that does not (`observation._restorable_groups`'s
    own `depot_site not in sites: continue`), and membership does not depend
    on the horizon or the magnitude of exposure -- a service present in
    `exposure` at all gets a per-horizon entry (and so a group) at EVERY
    horizon the current issuance publishes if it is eligible at all. Reusing
    it, instead of adding a parallel eligibility test, is what guarantees the
    projection's notion of "eligible" and `horizon_totals`' own
    `largest_restorable_group_ecar_gbps` can never disagree about who is
    competing for this depot."""
    return any(svc in group["members"]
              for groups in obs.restorable_groups.values()
              for group in groups)


def project_observation(
    obs: Observation, *,
    p_cut_threshold: float = P_CUT_ENUMERATION_THRESHOLD,
) -> dict:
    """`obs.to_dict()` reduced to decision-relevant content, plus an explicit
    account of what was dropped.

    Kept: the service under test always, and every service that is BOTH
    depot-eligible (`_depot_eligible`: terminates at this episode's depot
    site, so it can actually draw on the inventory `spares_on_hand` counts)
    AND whose peak p_cut reaches `p_cut_threshold`. `services` is trimmed to
    the same set -- `exposure` already carries each service's `demand_gbps`,
    so the roster is nearly redundant with it for decision purposes.
    `restorable_groups` is trimmed too, to groups whose members are ALL kept
    -- otherwise a service's id would leak through that field even though it
    was dropped everywhere else.

    Everything dropped is accounted for, split into TWO buckets rather than
    one (design spec §4.2): `below_threshold` for depot-eligible services
    that are simply quiet, and `ineligible_for_depot` for services that may
    be badly exposed but cannot draw on this depot at all, so they are no
    part of the contest this episode's spare decides. Folding the two
    together would hand the decider a large `max_p_cut` it cannot interpret
    -- an ineligible service's own exposure is not a competing claim, no
    matter how high -- which is exactly the overstatement D2 found in the
    gold rationales. Each bucket's `count` is over the server's full roster
    (a service with no representative point has no exposure entry at all,
    and is counted here as `below_threshold` -- no exposure data means no
    basis to call it ineligible either); the two risk figures are over the
    dropped services that do have exposure, and contribute 0.0 for the
    rest."""
    payload = obs.to_dict()
    exposure = payload["exposure"]

    keep = {obs.service_under_test}
    keep |= {svc for svc, per_horizon in exposure.items()
             if _depot_eligible(svc, obs)
             and _peak_p_cut(per_horizon) >= p_cut_threshold}
    dropped = [svc for svc in exposure if svc not in keep]
    below_threshold = [svc for svc in dropped if _depot_eligible(svc, obs)]
    ineligible_for_depot = [svc for svc in dropped
                            if not _depot_eligible(svc, obs)]

    def _bucket(svcs: list[str], *, extra_count: int = 0) -> dict:
        return {
            "count": len(svcs) + extra_count,
            "max_p_cut": round(
                max((_peak_p_cut(exposure[svc]) for svc in svcs),
                   default=0.0), 4),
            "summed_expected_capacity_at_risk_gbps": round(
                sum(_peak_capacity_at_risk_gbps(exposure[svc])
                    for svc in svcs), 3),
        }

    all_services = payload["services"]
    payload["exposure"] = {svc: per_horizon
                           for svc, per_horizon in exposure.items()
                           if svc in keep}
    payload["services"] = [s for s in all_services if s["id"] in keep]
    payload["restorable_groups"] = {
        horizon: tuple(g for g in groups
                      if all(m in keep for m in g["members"]))
        for horizon, groups in payload["restorable_groups"].items()}
    payload["n_services_total"] = len(all_services)
    # Roster entries with no exposure data at all (no storm-cuttable span --
    # see build_observation) are neither in `exposure` nor `dropped` above;
    # fold them into `below_threshold`, since zero exposure is quiet by
    # definition and there is no evidence to call them ineligible.
    no_exposure_data = (len(all_services) - len(payload["services"])
                        - len(dropped))
    payload["omitted_services"] = {
        "p_cut_threshold": p_cut_threshold,
        "below_threshold": _bucket(below_threshold,
                                   extra_count=no_exposure_data),
        "ineligible_for_depot": _bucket(ineligible_for_depot),
    }
    return payload


# Keywords strict tool use does not accept. decisions.py's schemas are the
# canonical contract and stay exactly as they are (they are also what the
# trace and the unit tests read); this adapter is what goes on the wire.
_UNSUPPORTED_KEYWORDS = ("minLength", "maxLength", "minimum", "maximum",
                         "multipleOf", "minItems", "maxItems")
_ARRAY_KEYWORDS = ("items",)
_OBJECT_KEYWORDS = ("properties", "required", "additionalProperties")


def strict_tool_schema(node):
    """A JSON Schema from decisions.py, rewritten into the subset strict tool
    use accepts: unsupported constraints dropped, and a type-union list
    (`{"type": ["array", "null"]}`, which OBJECTIVE_JSON_SCHEMA's `priority`
    uses) rewritten as `anyOf`. Never mutates its argument.

    Dropping `minLength: 1` from `reasoning` loses nothing: `_reasoning()` in
    decisions.py already rejects an empty or whitespace-only string, and
    every payload goes through `from_dict` regardless."""
    if not isinstance(node, dict):
        return node
    out = {k: v for k, v in node.items() if k not in _UNSUPPORTED_KEYWORDS}
    if "properties" in out:
        out["properties"] = {k: strict_tool_schema(v)
                             for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = strict_tool_schema(out["items"])
    types = out.get("type")
    if not isinstance(types, list):
        return out
    rest = {k: v for k, v in out.items() if k != "type"}
    branches = []
    for one in types:
        branch = {"type": one}
        for key, value in rest.items():
            if key in _ARRAY_KEYWORDS and one != "array":
                continue
            if key in _OBJECT_KEYWORDS and one != "object":
                continue
            branch[key] = value
        branches.append(branch)
    return {"anyOf": branches}


DEFAULT_MODEL = "claude-sonnet-5"
# Thinking tokens count against max_tokens. A generous cap costs nothing
# (billing is on tokens actually produced) and avoids a truncated tool call.
MAX_TOKENS = 16000
# Self-correction rounds for a schema/validation failure. Transient API
# failures are the SDK's own max_retries' concern, not this loop's.
MAX_ATTEMPTS = 3

TIMING_TOOL = "submit_timing_decision"
CONSTRAINT_TOOL = "submit_constraint_decision"
OBJECTIVE_TOOL = "submit_objective_decision"

# The prompt is assembled from three literals -- not re-flowed into one --
# because `_RIVAL_TOTALS_BULLET` has to slot into the "What you can see"
# list below as its own bullet, right after `omitted_services` and before
# the "## The three decisions" section starts. Concatenated, HEAD +
# _RIVAL_TOTALS_BULLET + TAIL is byte-identical to what the now-removed
# rival-totals arm used to send -- diff against that before believing
# otherwise if either seam is ever touched again.
_SYSTEM_PROMPT_HEAD = """\
You are the restoration decision-maker for a multi-layer IP-over-optical \
network during a tropical storm.

## The situation

A storm cone advances across the network's region. A forecast issuance \
publishes, for each future hour (a "horizon"), a cone with a cross-track \
diameter `width_km` and a centre. The convention is that the realized track \
falls inside the cone about two thirds of the time, so an asset sitting \
`offset_km` off the cone's centre has a cut probability `p_cut` that falls \
off smoothly with distance -- it is not a hard in-or-out test.

The hazard this creates is specific. A service's working path and its \
protection path were certified disjoint at design time, against buried \
conduits and shared amplifiers. Aerial fibre spans on both paths can \
nevertheless sit inside the same emerging storm cone. No static check flags \
that, because the correlation did not exist when the paths were certified. \
Restoring the service means routing around a risk group synthesized from the \
forecast, not one that was in the design.

You are not asked to find routes, compute optical performance, or check \
disjointness. Deterministic tools own all of that and are better at it than \
you are. You are asked for judgement in three places where the tools have \
nothing to say.

## What you can see

Each request carries one observation:

- `hour`, `hours_remaining` -- where you are in the event.
- `issued_at`, `cones` -- the most recent forecast issuance and its \
horizons. You never see a future issuance. Waiting is what buys the next \
one; that is the entire cost-benefit of waiting.
- `exposure` -- per service, per horizon: `hours_ahead`, `offset_km`, \
`width_km`, `p_cut`, `demand_gbps`, and `expected_capacity_at_risk_gbps` -- \
the product `p_cut * demand_gbps`, already computed for you. It is the \
quantity that makes two competing claims on one resource comparable, and it \
is in Gbps.
- `services` -- the roster for the services shown.
- `spares_on_hand` -- spare transponders held at `depot_site`, the one site \
whose inventory (the scenario's own `spare_inventory`, held per site) is \
scarce in this episode. Lighting a new lightpath consumes one transponder at \
each of its two endpoint sites; an `ip_reroute` consumes none. Every other \
site is stocked well enough that only `depot_site` ever binds. This \
inventory is invisible to the routing tools: they will happily propose a \
candidate the depot cannot fulfil, and the harness will reject that choice.
\n\
  That depot is SHARED -- with the other services that terminate at the same \
site. A transponder you spend there is not available to them. Order matters: \
a service cut in an EARLIER hour reaches the depot before one cut later, and \
among services cut in the same hour the larger demand has the stronger \
claim. You are not asked to restore them and their restoration is not \
simulated, but their claim on this site's inventory is real.
\n\
  Services that do not terminate at `depot_site` are summarized under \
`omitted_services.ineligible_for_depot`. They may be badly exposed; they are \
not competing for this inventory, because restoring them draws on their own \
sites' depots.
- `actions_taken` and `spares_spent` -- what YOU have already committed \
earlier in this episode: per action its hour, its lever, the spare pairs it \
cost, the `avoid` set it was routed under, and `effective_at_hour` -- the \
HOUR LABEL (not a raw index; this episode's `hours` may not be positional, \
e.g. `[t0, t1, t2, t6]`) from which it is effective, or `null` if that lands \
past the episode's last hour; plus the running total of pairs already spent. \
A reroute you committed in an earlier hour has already moved the service, so \
the `exposure` above describes its CURRENT path, not the path it had when \
you acted.
- `lead_time_hours` -- hours between issuing an action on a lever and it \
being effective, per lever. An `ip_reroute` is a config change and lands \
immediately. Lighting a new optical path is provisioning and takes the \
scenario's lead time. Acting later than (exposure hour - lead time) means \
the action lands after the cut.
- `risk_group_ids` -- horizon hour -> the id of the risk group defined for \
that cone. These ids are what you name when you constrain routing.
- `unconstrained_menu` -- present on the constraints request only. The \
routing menu as it stands with nothing avoided: each entry's \
`candidate_label`, its `lever`, and `spares_needed`, its own cost in spare \
transponders PER SITE. Cost vectors are not shown here; they belong to the \
objective decision. Constraining removes entries from this list -- an entry \
that reuses a path your `avoid` set forbids will not survive into the menu \
you are given at the next step. Labels here are positions within THIS list \
only -- the menu you receive at the objective step is renumbered from its \
own surviving entries, so a label you see here does not necessarily name \
the same candidate there.
- `iteration`, `last_rejection` -- within one hour you may get up to five \
attempts. `last_rejection` tells you why the previous attempt failed.
- `n_services_total` and `omitted_services` -- the observation shows you the \
actionable service plus every other DEPOT-ELIGIBLE service whose cut \
probability is high enough to be shown individually. The rest are \
summarized, split into two accounts so a large number in one is never \
misread as the other: one bucket for depot-eligible services that are \
simply quiet, and `omitted_services.ineligible_for_depot` for services that \
may be badly exposed but do not terminate at `depot_site` at all, so they \
are no part of this contest. Each bucket carries `count`, `max_p_cut`, and \
`summed_expected_capacity_at_risk_gbps` -- the services in that bucket taken \
at their own worst horizon, then summed. Read both before assuming the \
network is as small as the list you were given. The actionable service is \
shown unconditionally, even at zero exposure, because it is the only one \
these tools can act on -- not because it has the stronger claim. Every \
other service listed is a real competing claim on the same depot.
"""

# W3.2's measured arm. Purely descriptive: what the field contains, over
# which set, and in what unit. It states no threshold, names no episode, and
# says nothing about which way the comparison should come out -- the
# comparison is the judgement being measured.
_RIVAL_TOTALS_BULLET = """\
- `horizon_totals` -- per horizon, three already-summed figures: \
`sut_ecar_gbps`, the actionable service's own expected capacity at risk; \
`largest_restorable_group_ecar_gbps`, the LARGEST expected capacity at risk \
among the co-terminating groups sharing this depot; and \
`non_sut_ineligible_ecar_gbps`, the summed expected capacity at risk of \
every service that cannot draw on this depot at all. One spare buys one \
lightpath: services that co-terminate (share both endpoints) are jointly \
restored by it and their figures add within a group, but across groups only \
the MAXIMUM is an honest competing claim, never a sum. \
`non_sut_ineligible_ecar_gbps` is kept for scale, not as a claim: it is over \
the whole network, not only the services listed above, and none of it can \
be restored from this depot regardless of how large it reads.
- `restorable_groups` -- per horizon, the co-terminating groups themselves \
that `largest_restorable_group_ecar_gbps` maxes over: each with `endpoints`, \
`members`, and `ecar_gbps`, so you can see WHICH services would share one \
restoring lightpath, not just the total.
"""

_SYSTEM_PROMPT_TAIL = """\

## The three decisions

1. **Timing** (`""" + TIMING_TOOL + """`). Act now, or wait for the next \
issuance? Waiting buys a sharper forecast and costs lead time, and possibly \
the spare inventory if another service claims it first. Acting now on a \
wide, uncertain cone can spend a scarce spare on a service that was never \
going to be cut. No tool computes this: the network model has no concept of \
time at all.

2. **Constraints** (`""" + CONSTRAINT_TOOL + """`). What must the reroute \
route around? `avoid` takes lists of asset ids, SRLG ids, or risk-group ids. \
The real choice is the HORIZON: avoiding the full far-horizon cone often \
leaves no feasible path at all, while avoiding only the current exposure \
leaves the reroute re-exposed a few hours later. `protected`, `best_effort`, \
`basis` and `level` set the protection posture. On this network \
`basis="srlg"` is a no-op (it carries no static SRLGs), and \
`protected=True` produces a paired menu the harness cannot translate into a \
plan -- so `protected=False`, `basis="physical"`, `level="link"` is the \
working posture, with `basis="risk_group"`/`level="risk_group"` available \
when the constraint you mean is a forecast risk group rather than a physical \
link. This decision changes which candidates EXIST.

3. **Objective** (`""" + OBJECTIVE_TOOL + """`). Which candidate from the \
routing menu? Answer with the `candidate_label` of the entry you want, or \
`infeasible` if none is acceptable. This only reorders a menu that already \
exists; it does not create options. `priority`, when you state one, is an \
ordering over these seven cost terms:

- `spectrum_used` -- slots consumed. Cost.
- `transponders` -- **network-wide**: 2.0 x the count of every lightpath in \
the whole model after the candidate is materialized on a clone. It is NOT \
this candidate's own transponder cost, and comparing candidates on it is \
close to meaningless. Each candidate carries a precomputed `spares_needed` \
field, which IS its own cost in spare transponders -- now a PER-SITE dict, \
one entry per site where this candidate's own new lightpaths charge a \
transponder (empty for an `ip_reroute`). Reason about spares from \
`spares_needed` and `spares_on_hand`, never from `transponders`.
- `max_util` -- worst link utilization, 0-1. Cost.
- `dropped_traffic` -- Gbps that stays unrestored. Cost.
- `added_latency` -- milliseconds added. Cost.
- `total_margin` -- summed optical margin in dB. This one is a BENEFIT: more \
is better, and it is subtracted where the others are added.
- `services_at_risk` -- count of services left exposed. Cost.

The terms are in incommensurate units, which is why you order them rather \
than weight them.

## Your reasoning

`reasoning` is mandatory on all three decisions and it is read. Write the \
actual judgement: which numbers in this observation you compared, which way \
the comparison came out, and what you gave up. If you waited, say what you \
expect the next issuance to resolve. If you spent a spare, say what claim \
you preferred over the claims you did not serve. If you held one, say what \
you held it for and what you accepted on the service you can act on in \
exchange.

`contested_claim` names the strongest competing claim on the depot you are \
aware of and its `expected_capacity_at_risk_gbps` from the observation, or \
null if you judge there is none. Name a service that appears in `exposure`.

Do not pad it with a checklist of terms from this prompt. A paragraph naming \
every concept above while explaining no decision is worse than two sentences \
that state the comparison you actually made.
"""

SYSTEM_PROMPT = (
    _SYSTEM_PROMPT_HEAD + _RIVAL_TOTALS_BULLET + _SYSTEM_PROMPT_TAIL)


def _tool(name: str, description: str, schema: dict) -> dict:
    return {"name": name, "description": description, "strict": True,
            "input_schema": strict_tool_schema(schema)}


# Order is fixed and identical on every request: tool definitions are the
# outermost cache tier, and reordering them invalidates everything after.
TOOLS = [
    _tool(TIMING_TOOL,
          "Submit the act-or-wait decision for this hour. Call this exactly "
          "once, after weighing what the next forecast issuance would "
          "resolve against the lead time and spare inventory that waiting "
          "puts at risk.",
          TIMING_JSON_SCHEMA),
    _tool(CONSTRAINT_TOOL,
          "Submit the routing constraints for this attempt: what the reroute "
          "must avoid, and the protection posture to route under. Call this "
          "exactly once, after choosing which horizon's exposure the reroute "
          "has to survive.",
          CONSTRAINT_JSON_SCHEMA),
    _tool(OBJECTIVE_TOOL,
          "Submit which candidate from the routing menu to commit, or "
          "`infeasible` if none is acceptable. Call this exactly once, after "
          "comparing the candidates on the terms that matter for this "
          "service and this event state.",
          OBJECTIVE_JSON_SCHEMA),
]

TIMING_INSTRUCTION = (
    f"Decide whether to act this hour or wait for the next issuance, then "
    f"call `{TIMING_TOOL}`.")
CONSTRAINT_INSTRUCTION = (
    f"Decide what this reroute must route around and under what protection "
    f"posture, then call `{CONSTRAINT_TOOL}`.")
OBJECTIVE_INSTRUCTION = (
    f"Choose one candidate from the menu above by its `candidate_label`, or "
    f"`infeasible`, then call `{OBJECTIVE_TOOL}`.")


class ClaudeDecider:
    """A `Decider` (decisions.py) backed by the Claude API.

    Stateless by construction: suite.py builds ONE decider and reuses it
    across every episode and every run, so per-instance conversation memory
    would leak T1a's context into T2b's prompt. Each of the three methods is
    a fresh single-turn request; everything that must persist across a
    rollout already travels on the Observation runner.py builds (`hour`,
    `iteration`, `last_rejection`).

    `last_projection` is the one piece of per-call state, and it is
    write-only: nothing in this class ever reads it back, so it cannot leak
    one episode's context into another's prompt. It exists because the runner
    must record exactly what went on the wire, and only the decider knows --
    the projection is decider-owned and a runner-side recomputation would
    silently diverge the day it changes (run-viewer design, §5.1).

    `client` exists so tests can inject a fake. The real client is built
    lazily on first use, not in __init__, so constructing a decider -- which
    suite.py does before any rollout, and every unit test does -- never
    requires the optional `anthropic` extra."""

    def __init__(self, model: str = DEFAULT_MODEL, *, client=None,
                 p_cut_threshold: float = P_CUT_ENUMERATION_THRESHOLD,
                 audit_path: str | Path | None = None,
                 oms_nodes: dict[str, list[str]] | None = None) -> None:
        self.model = model
        # suite.run_suite keys its results dict AND the trace filename on
        # `name`.
        self.name = f"agent:{model}"
        self._client = client
        self._p_cut_threshold = p_cut_threshold
        self._audit_path = Path(audit_path) if audit_path else None
        self._system_prompt = SYSTEM_PROMPT
        # Write-only telemetry; see the class docstring.
        self.last_projection: dict | None = None
        # oms_id -> [src_node_id, dst_node_id], the static optical adjacency
        # menu_for_prompt needs to resolve a candidate's new_lightpaths to
        # endpoint SITES (ledger.spares_needed). Unchanged across an episode
        # (it is the topology's own OMS graph, not scenario state), so
        # settable once rather than threaded through `objective`'s signature
        # -- decisions.py's `Decider` protocol is shared by every decider,
        # including baselines that never touch spares_needed at all.
        self.oms_nodes: dict[str, list[str]] = oms_nodes or {}

    @property
    def _api(self):
        if self._client is None:
            import anthropic      # optional extra: pip install -e ".[agent]"
            # Credentials come from the environment the SDK already reads
            # (ANTHROPIC_API_KEY, or an `ant auth login` profile). This repo
            # has no secrets convention and must not invent one.
            self._client = anthropic.Anthropic()
        return self._client

    def _user_content(self, payload: dict, instruction: str, *,
                      menu: dict | None = None,
                      unconstrained_menu: dict | None = None) -> str:
        body = {"observation": payload}
        if unconstrained_menu is not None:
            body["unconstrained_menu"] = unconstrained_menu
        if menu is not None:
            body["menu"] = _menu_for_prompt(menu, self.oms_nodes)
        rendered = json.dumps(body, indent=2, sort_keys=True, default=str)
        return f"{rendered}\n\n{instruction}"

    def _create(self, messages: list[dict], tool_name: str):
        return self._api.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": self._system_prompt,
                     "cache_control": {"type": "ephemeral"}}],
            # Sonnet 5 accepts no budget_tokens and no temperature/top_p/
            # top_k -- any of them is a 400.
            thinking={"type": "adaptive"},
            tools=TOOLS,
            tool_choice={"type": "tool", "name": tool_name,
                         "disable_parallel_tool_use": True},
            messages=messages,
        )

    @staticmethod
    def _check_contested_claim(decision, payload, tool_name) -> None:
        """A claim must name a service the model was actually shown.

        Same class of guard as the hallucinated risk-group id that burned 3 of
        5 iterations in a real T3a rollout (control-arm findings, root cause
        #3): the generated JSON disagreeing with the payload it was generated
        from. It rides the existing MAX_ATTEMPTS retry loop rather than a new
        mechanism -- from_dict cannot do this because it never sees the
        observation."""
        claim = getattr(decision, "contested_claim", None)
        if claim is None:
            return
        if claim["service_id"] not in payload["exposure"]:
            raise DecisionError(
                f"{tool_name}: `contested_claim.service_id` "
                f"{claim['service_id']!r} is not in this observation's "
                f"`exposure`. Name a service you were shown, or null.")

    def _decide(self, tool_name, decision_cls, obs, payload, user_content):
        """One decision, with up to MAX_ATTEMPTS self-correction rounds.

        Only schema/validation failures are retried here. Transient API
        failures (429, 5xx, connection errors) belong to the SDK's own
        `max_retries` and are deliberately not caught: this loop exists for
        the class of failure the SDK cannot see.

        On the final failure the DecisionError propagates. run_episode never
        catches a decider exception, so a decider that cannot produce a valid
        decision kills the rollout visibly instead of silently substituting a
        degraded default that would then be scored as if it were a
        judgement."""
        messages: list[dict] = [{"role": "user", "content": user_content}]
        failure: DecisionError | None = None
        for attempt in range(MAX_ATTEMPTS):
            response = self._create(messages, tool_name)
            block = next((b for b in response.content
                          if getattr(b, "type", None) == "tool_use"
                          and b.name == tool_name), None)
            if block is None:
                failure = DecisionError(
                    f"{tool_name}: no tool_use block in response")
                # No tool_use id to answer, so the correction cannot be a
                # tool_result; it has to be a plain user turn.
                messages.append(
                    {"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": (
                    f"That response contained no `{tool_name}` tool call. "
                    f"Call `{tool_name}` with your decision.")})
                continue
            try:
                decision = decision_cls.from_dict(block.input)
                self._check_contested_claim(decision, payload, tool_name)
            except DecisionError as exc:
                failure = exc
                # Echo the assistant content back unchanged -- with adaptive
                # thinking it carries a thinking block that must survive the
                # turn -- then answer its tool_use with the validation error.
                messages.append(
                    {"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": block.id,
                    "is_error": True, "content": str(exc)}]})
                continue
            self._audit(obs, payload, tool_name, decision, attempt + 1)
            return decision
        raise failure

    def _audit(self, obs, payload, tool_name, decision, attempts) -> None:
        """Write-only telemetry: exactly what this call showed the model.

        It cannot ride inside the decision itself -- tests/eval/
        test_decisions.py asserts *Decision.to_dict() equals an exact dict,
        so an extra field would break that round-trip -- so it lands in a
        JSONL sidecar instead. This is what makes the projection auditable:
        a reader can check that every claimant an episode names was actually
        shown, and that omitted_services never hid something material."""
        if self._audit_path is None:
            return
        record = {
            "decider": self.name,
            "scenario_id": obs.scenario_id,
            "hour": obs.hour,
            "iteration": obs.iteration,
            "decision": tool_name,
            "attempts": attempts,
            "shown_services": sorted(payload["exposure"]),
            # Not just WHICH services were shown but at what magnitude --
            # otherwise the sidecar cannot answer "was the claimant this
            # episode's gold names actually visible, and how big was it?"
            # Peak over horizons, matching omitted_services' rollup so the
            # two halves of the account are in the same unit (W3.1).
            "shown_expected_capacity_at_risk_gbps": {
                svc: round(_peak_capacity_at_risk_gbps(per_horizon), 3)
                for svc, per_horizon in payload["exposure"].items()},
            "omitted_services": payload["omitted_services"],
            "n_services_total": payload["n_services_total"],
            "result": decision.to_dict(),
        }
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def _project(self, obs: Observation) -> dict:
        self.last_projection = project_observation(
            obs, p_cut_threshold=self._p_cut_threshold)
        return self.last_projection

    def timing(self, obs: Observation) -> TimingDecision:
        payload = self._project(obs)
        return self._decide(
            TIMING_TOOL, TimingDecision, obs, payload,
            self._user_content(payload, TIMING_INSTRUCTION))

    def constraints(self, obs: Observation,
                    unconstrained_menu: dict | None = None
                    ) -> ConstraintDecision:
        payload = self._project(obs)
        return self._decide(
            CONSTRAINT_TOOL, ConstraintDecision, obs, payload,
            self._user_content(payload, CONSTRAINT_INSTRUCTION,
                               unconstrained_menu=unconstrained_menu))

    def objective(self, obs: Observation, menu: dict) -> ObjectiveDecision:
        payload = self._project(obs)
        return self._decide(
            OBJECTIVE_TOOL, ObjectiveDecision, obs, payload,
            self._user_content(payload, OBJECTIVE_INSTRUCTION, menu=menu))
