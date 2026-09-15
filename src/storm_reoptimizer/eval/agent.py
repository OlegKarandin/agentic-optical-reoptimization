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
from .probe import (
    MAX_PROBES_PER_DECISION, PROBE_JSON_SCHEMA, PROBE_TOOL, ProbeError,
)
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


def _project_exposure_entry(entry: dict) -> dict:
    """One exposure entry, trimmed to the four decision-relevant fields and
    rebuilt as a FRESH dict -- never mutate `entry` itself, since
    `to_dict()["exposure"]` IS `obs.exposure` (observation.py:253) and the
    runner writes the Observation to the trace after projection runs
    (runner.py:926/937).

    `expected_capacity_at_risk_gbps` is recomputed here rather than read off
    `entry`, so a hand-built entry that omits it (as the unit tests' fixture
    observations do) projects identically to a real one: both derive it from
    `p_cut`/`demand_gbps` the same way `observation.py` does."""
    p_cut = float(entry["p_cut"])
    demand_gbps = float(entry["demand_gbps"])
    projected = {
        "hours_ahead": entry["hours_ahead"],
        "p_cut": entry["p_cut"],
        "demand_gbps": entry["demand_gbps"],
        "expected_capacity_at_risk_gbps": round(
            expected_capacity_at_risk_gbps(p_cut, demand_gbps), 3),
    }
    # Copied through rather than recomputed: `build_observation` owns the
    # band (it has the scenario's revision scale and the spans), and a second
    # computation here could silently disagree with the one the trace, the
    # viewer and the t0-equality gate all read.
    band = entry.get("p_cut_if_track_revised")
    if band is not None:
        projected["p_cut_if_track_revised"] = band
    return projected


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
    include_risk_group_assets: bool = False,
) -> dict:
    """`obs.to_dict()` reduced to decision-relevant content, plus an explicit
    account of what was dropped.

    Kept: the service under test always, and every service that is BOTH
    depot-eligible (`_depot_eligible`: terminates at this episode's depot
    site, so it can actually draw on the inventory `spares_on_hand` counts)
    AND whose peak p_cut reaches `p_cut_threshold`. `services` is dropped
    from the payload outright, for every service, not just the trimmed set
    (spec 5.3) -- `exposure` already carries each kept service's
    `demand_gbps` and `restorable_groups` carries its endpoints, so the
    roster was wholly redundant for decision purposes, and it was the single
    largest block in the payload.

    Each kept exposure entry is itself trimmed to `hours_ahead`, `p_cut`,
    `demand_gbps`, `expected_capacity_at_risk_gbps` (`_project_exposure_
    entry`); `offset_km` and `width_km` are dropped, `damage_radius_km` is
    dropped from the payload entirely, and `cones` is replaced with
    `horizons` -- a plain list of the horizon hours this issuance publishes,
    in the same ascending-hours order `risk_group_ids` is built in (dict
    iteration order, NOT a lexicographic sort -- "t10" would sort before
    "t2"), not the raw cone objects. `p_cut` already integrates all of this
    geometry, so surfacing it separately added nothing a decider could act
    on; in practice it was misread as forecast uncertainty rather than
    corroborating detail behind a number already shown.

    `restorable_groups` keeps a group WHOLE -- every member, and `ecar_gbps`
    unchanged -- if it contains AT LEAST ONE kept member, and drops the group
    entirely otherwise. Two things this is NOT: it does not drop a member
    from an otherwise-kept group just because that member alone is below
    threshold (a mixed group would then show a smaller `ecar_gbps` than the
    one `horizon_totals.largest_restorable_group_ecar_gbps` maxes over, and a
    service already visible in `exposure` for being individually above
    threshold could vanish from the group view because a groupmate is not);
    and it does not partially trim a group's members while keeping its full
    summed `ecar_gbps` (same disagreement, the other way). Showing the real
    group intact is what keeps the group total honest against the totals
    bullet. A group with NO kept member is dropped entirely -- otherwise a
    below-threshold or ineligible service's id would leak through this field
    even though it was dropped everywhere else.

    Everything dropped is accounted for, split into TWO buckets rather than
    one (design spec §4.2): `below_threshold` for depot-eligible services
    that are simply quiet, and `ineligible_for_depot` for services that may
    be badly exposed but cannot draw on this depot at all, so they are no
    part of the contest this episode's spare decides. Folding the two
    together would hand the decider a large `max_p_cut` it cannot interpret
    -- an ineligible service's own exposure is not a competing claim, no
    matter how high -- which is exactly the overstatement D2 found in the
    gold rationales. `ineligible_for_depot` is reduced to `count` alone
    (spec 5.3): a service that cannot draw on this depot is not a competing
    claim however exposed it is, so its summed risk is a number with no
    decision attached, and the 2026-08-29 gold-rationale review found
    exactly that number being read as one anyway. `below_threshold` keeps
    its full account -- a large number THERE is something the decider can
    act on (wait for it to clear the threshold, or not). Each bucket's
    `count` is over the server's full roster (a service with no
    representative point has no exposure entry at all, and is counted here
    as `below_threshold` -- no exposure data means no basis to call it
    ineligible either).

    `include_risk_group_assets` (spec 5.1) additionally surfaces
    `obs.risk_group_assets` as `risk_groups` -- each horizon's group, named
    down to the asset, with that span's own `p_cut` and whether it lies on
    the actionable service's own working/protection corridor. It is what
    makes `avoid.assets` nameable rather than all-or-nothing, and it is
    large, so only the caller deciding constraints passes it."""
    payload = obs.to_dict()
    payload.pop("damage_radius_km")
    # `list(...)`, NOT `sorted(...)` -- horizon-hour labels ("t2", "t10", ...)
    # sort lexicographically, which would put "t10" before "t2". Dict
    # iteration order is insertion order, and `observation.py` builds `cones`
    # by iterating `issuance.horizons` in the same ascending-hours order
    # `risk_group_ids` is built in, so plain `list()` is what actually
    # matches `risk_group_ids`' own key order.
    payload["horizons"] = list(payload.pop("cones"))
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
                   default=0.0), 3),
            "summed_expected_capacity_at_risk_gbps": round(
                sum(_peak_capacity_at_risk_gbps(exposure[svc])
                    for svc in svcs), 3),
        }

    all_services = payload["services"]
    payload["exposure"] = {
        svc: {horizon: _project_exposure_entry(entry)
             for horizon, entry in per_horizon.items()}
        for svc, per_horizon in exposure.items() if svc in keep}
    # The roster and the per-horizon totals leave the WIRE, not the
    # Observation (spec 5.3). `exposure` already carries each kept service's
    # demand and `restorable_groups` carries its endpoints, so the roster was
    # the payload's largest redundant block; and every figure in
    # `horizon_totals` is re-derivable from a row still shown --
    # `sut_ecar_gbps` IS the actionable service's own row,
    # `largest_restorable_group_ecar_gbps` IS the first group, and the prompt
    # itself told the model to ignore `non_sut_ineligible_ecar_gbps`.
    # `observation_record` (runner.py) still writes both to the trace.
    payload.pop("services")
    payload.pop("horizon_totals")
    payload["restorable_groups"] = {
        horizon: tuple(g for g in groups
                      if any(m in keep for m in g["members"]))
        for horizon, groups in payload["restorable_groups"].items()}
    payload["n_services_total"] = len(all_services)
    # Roster entries with no exposure data at all (no storm-cuttable span --
    # see build_observation) are neither in `exposure` nor `dropped` above;
    # fold them into `below_threshold`, since zero exposure is quiet by
    # definition and there is no evidence to call them ineligible.
    no_exposure_data = (len(all_services) - len(payload["exposure"])
                        - len(dropped))
    payload["omitted_services"] = {
        "p_cut_threshold": p_cut_threshold,
        "below_threshold": _bucket(below_threshold,
                                   extra_count=no_exposure_data),
        # COUNT ONLY (spec 5.3). A service that cannot draw on this depot is
        # not a competing claim however exposed it is, so its summed risk is
        # a number with no decision attached -- and the 2026-08-29 gold-
        # rationale review found exactly that number being read as one.
        "ineligible_for_depot": {"count": len(ineligible_for_depot)},
    }
    # The groups' CONTENTS, only where they are decidable (spec 5.1). The
    # constraints decision is the one that names assets; the timing and
    # objective steps have no use for a list that can run to dozens of
    # fibres, and the payload is the scarce resource here.
    payload.pop("risk_group_assets", None)
    if include_risk_group_assets:
        # Each entry names the GROUP it belongs to, not only the horizon.
        # The constraints paragraph below tells the model to name asset ids
        # out of this block and risk-group ids out of `risk_group_ids`; with
        # no id here, connecting the two was a cross-reference the model had
        # to perform, and a wrong mental model of the structure is the most
        # likely origin of D1 run B seed 2's `{"risk_groups": [""]}`
        # (2026-09-12 failure analysis, the cross-episode finding).
        # Fresh dicts, not `list(obs.risk_group_assets)`: the tuple's own
        # dicts are a frozen Observation's, and `_project_exposure_entry`
        # already establishes that the projection never hands those out.
        payload["risk_groups"] = [
            {"horizon": entry["horizon"],
             "risk_group_id": obs.risk_group_ids.get(entry["horizon"]),
             "assets": entry["assets"]}
            for entry in obs.risk_group_assets]
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
# because `_RESTORABLE_GROUPS_BULLET` has to slot into the "What you can
# see" list below as its own bullet, right after `omitted_services` and
# before the "## The three decisions" section starts.
_SYSTEM_PROMPT_HEAD = """\
You are the restoration decision-maker for a multi-layer IP-over-optical \
network during a tropical storm.

## The situation

A storm cone advances across the network's region. A forecast issuance \
publishes, for each future hour (a "horizon"), a cone. For every service \
the harness has already turned that cone into `p_cut`, the probability the \
storm cuts the service at that horizon. It is a smooth probability, not a \
hard in-or-out test: a service near the track but outside it still carries \
risk.

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
- `issued_at`, `horizons` -- the most recent forecast issuance and the \
horizon hours it covers. You never see the CONTENT of a future issuance; \
`issuance_schedule` lists the hours at which issuances arrive, so you know \
whether one is still coming. Waiting is what buys the next one.
- `exposure` -- per service, per horizon: `hours_ahead`, `p_cut`, \
`demand_gbps`, and `expected_capacity_at_risk_gbps` -- \
the product `p_cut * demand_gbps`, already computed for you. It is the \
quantity that makes two competing claims on one resource comparable, and it \
is in Gbps. For a protected service `p_cut` is the probability BOTH its \
working and protection paths are cut -- the probability it actually goes \
down. While another issuance is still scheduled, each row also carries \
`p_cut_if_track_revised`: the smallest, largest and mean `p_cut` this \
service would show if the next issuance moved the cone centre by \
`revision_radius_km` in any direction. `p_cut` says how likely the cut is \
if this issuance is right; the band says how much that number can change \
when the issuance is revised.
- `spares_on_hand` -- spare transponders held at `depot_site`, the one site \
whose inventory (the scenario's own `spare_inventory`, held per site) is \
scarce in this episode. Lighting a new lightpath consumes one transponder at \
each of its two endpoint sites; an `ip_reroute` consumes none. Every other \
site is stocked well enough that only `depot_site` ever binds. This \
inventory is invisible to the routing tools: they will happily propose a \
candidate the depot cannot fulfil, and the harness will reject that choice.
\n\
  That depot is SHARED -- with the other services that terminate at the same \
site. A transponder used there is not available to them. Order matters: \
a service cut in an EARLIER hour reaches the depot before one cut later, and \
among services cut in the same hour the larger demand has the stronger \
claim. You are not asked to restore them and their restoration is not \
simulated, but their claim on this site's inventory is real.
\n\
  Services that do not terminate at `depot_site` are summarized under \
`omitted_services.ineligible_for_depot`. They may be badly exposed; they are \
not competing for this inventory, because restoring them draws on their own \
sites' depots.
- `claim_priority` -- an ordering of the services shown, most deserving of \
this depot's spares first, INCLUDING the actionable service. A cut, if the \
storm makes one, happens at a horizon hour. At that hour you give a timing \
decision first; the cut is injected after it; the harness then restores \
the services the cut dropped, in the ranking in force at that moment, with \
whatever spares REMAIN, each restoration effective after its lever's lead \
time. The ranking in force is the last one you stated: \
`standing_claim_priority` shows it, and an empty `claim_priority` keeps \
it. Ranking another service above the actionable one and then committing \
the depot's last spare to the actionable one is a contradiction, and the \
harness rejects that commit.
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
- `deadline_hour` -- per lever, the LAST hour at which an action on that \
lever still lands at or before the latest published horizon. Acting AT \
that hour is on time. Acting earlier buys nothing unless no issuance is \
scheduled in between; `next_issuance.hour` tells you whether one is.
- `risk_group_ids` -- horizon hour -> the id of the risk group defined for \
that cone. These ids are what you name when you constrain routing.
- `iteration`, `last_rejection` -- within one hour you may get up to five \
attempts. `last_rejection` tells you why the previous attempt failed.
- `decided_this_hour` -- on the constraints and menu requests only: the \
timing decision you already made this hour, with its reasoning, \
`contested_claim`, `claim_priority`, and the probe answers you obtained. \
The constraints and the menu choice EXECUTE that decision. If the menu \
holds nothing consistent with it, answer `hold`.
- `attempts_this_hour` -- on the constraints and menu requests: every \
avoid set already tried this hour, the menu status and size it produced, \
and what you answered. Repeating an avoid set that produced no menu \
cannot produce one.
- `n_services_total` and `omitted_services` -- the observation shows you the \
actionable service plus every other service that is BOTH known to terminate \
at `depot_site` AND whose cut probability is high enough to be shown \
individually. The rest are summarized, split into two accounts so a large \
number in one is never misread as the other: one bucket for services simply \
not shown individually here (too quiet to clear the bar, or with no \
exposure data at all to judge against `depot_site` in the first place -- \
this bucket does NOT assert they all terminate at `depot_site`, only that \
none is confirmed not to), and `omitted_services.ineligible_for_depot` for \
services CONFIRMED not to terminate at `depot_site`, so they are no part of \
this contest. Each bucket carries `count`, `max_p_cut`, and \
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
_RESTORABLE_GROUPS_BULLET = """\
- `restorable_groups` -- per horizon, the co-terminating groups that would \
share one restoring lightpath: each with `endpoints`, `members`, and \
`ecar_gbps`, so you can see WHICH services would share it, not just a \
summed figure. One spare buys one lightpath: services that co-terminate \
(share both endpoints) are jointly restored by it and their figures add \
within a group, but across groups only the MAXIMUM is an honest competing \
claim, never a sum.
"""

_SYSTEM_PROMPT_TAIL = """\

## One question you may ask

`""" + PROBE_TOOL + """` takes a `service_id` from `exposure` and a \
`risk_group_id` from `risk_group_ids`, and returns what the routing tools \
would offer that service if every asset in that risk group were unusable: \
`status`, `full_restore_candidates` (how many candidates restore its full \
demand on a genuinely different path), `min_spares_needed_by_site` (the \
cheapest such candidate's spare transponders per site, or null if there is \
none) and `levers`. It computes and changes nothing. You may call it up to \
""" + str(MAX_PROBES_PER_DECISION) + """ times per decision, before the \
decision tool; the answer comes back as a tool result. You may call it at \
any of the three decisions, ahead of that decision's tool call.

What the answer means for the spare: a service with no full-restore \
candidate under the group that cuts it cannot use the spare after its \
cut, however exposed it is. A service whose cheapest candidate charges \
nothing at `depot_site` does not need it either.
"""
# The third sentence is a documented NUDGE, kept as an experiment for the
# first measured run and reported as such (spec 2, and README's "What to
# read afterwards"). Every probe in the 2026-09-09 run named the SUT; the
# two sentences before it say what an answer MEANS, and this one says who
# to ask about. If the run shows the nudge is what produced a claimant
# probe rather than the semantics, say so and drop it.
_SYSTEM_PROMPT_TAIL += "The answer is as relevant to the\n" \
    "services you would keep the spare for as to the one you can act on."
_SYSTEM_PROMPT_TAIL += """

## The three decisions

1. **Timing** (`""" + TIMING_TOOL + """`). This decision is what allocates \
the depot's last spare. `act` means: route the actionable service this \
hour, and if the candidate you then choose needs a spare, that spare leaves
the depot now, before any cut, ahead of every other service. `wait` means: \
the spare stays in the depot this hour. It is still there for the \
actionable service next hour, and it is there for whichever service a cut \
actually drops if you never commit it. No other service can draw on the \
depot before it is cut, so waiting cannot lose the spare to a competitor. \
What waiting costs is lead time. What it buys is the next issuance, if \
`next_issuance` says one is still scheduled before `deadline_hour`.

2. **Constraints** (`""" + CONSTRAINT_TOOL + """`). What must the reroute \
route around? `avoid` takes risk-group ids from `risk_group_ids`, and asset \
ids from `risk_groups` -- a LIST with one entry per horizon, each entry \
naming its `horizon`, its `risk_group_id`, and its `assets`: one row per \
asset with `asset_id`, that asset's own `p_cut`, and `on` (whether it sits \
on the actionable service's working path, its protection path, or neither). \
A risk group contains every span the cone touches, including spans that are \
not on the actionable service's path, and a group naming all of a site's \
spans leaves that site unroutable. Avoiding the whole group is the safest \
reroute and sometimes has no path; avoiding only the spans that matter keeps \
a path open at some residual exposure, which the menu then reports per \
candidate. If `probe_restorability` reports no solution under a group, that \
avoid set has no path and repeating it cannot produce one. This decision \
changes which candidates EXIST.

3. **Objective** (`""" + OBJECTIVE_TOOL + """`). Which candidate from the \
routing menu? Answer with the `candidate_label` of the entry you want, \
`infeasible` if none is acceptable, or `hold` to commit nothing this hour.

Two answers commit nothing. `infeasible` says none of these candidates is \
acceptable under the constraints you set; you will be asked for constraints \
again and can loosen them. `hold` says you decline to commit anything this \
hour after seeing the menu: the hour ends exactly as if you had waited, \
nothing is spent, and the spare stays in the depot.

Each candidate carries `spares_needed` (its own cost in spare \
transponders, per site), `residual_exposure` (its own `p_cut`/`ecar_gbps` \
per horizon, computed as `exposure` is), `path_delta` and \
`collides_with_protection`, and a `cost_vector` for the record. Reason \
about spares from `spares_needed` and `spares_on_hand`.

Each candidate also carries three facts about what committing it would \
actually do to the actionable service's OWN path, computed against the \
current working and protection corridors: `path_delta.changes_working_path` \
-- whether the service ends up on a different set of OMS at all; \
`path_delta.oms_retained_cuttable` -- which storm-cuttable spans you are on \
now you would STILL be on; `residual_exposure` -- this candidate's own \
`p_cut`/`ecar_gbps` per horizon, computed the same way `exposure` \
above is; and `collides_with_protection` -- whether this candidate would put \
the service on its own protection corridor, which would remove the \
automatic 1:1 switchover that protects it today. Surviving `avoid` is not \
the same claim as reducing this service's exposure: `avoid` prunes on hard \
polygon containment, while `p_cut` falls off smoothly, and the two \
disagree exactly at the offset band some episodes are built at -- a \
candidate can clear `avoid` while leaving `changes_working_path` false and \
`residual_exposure` identical to what you already carry. Committing that is \
sometimes the right call when spares are scarce and the exposure is \
tolerable; make it because you decided so, not because nothing told you the \
service never moved.

A committed candidate that changes no path and costs no spare is recorded as \
a wait, not an action.

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
null if you judge there is none. Name a service that appears in `exposure`. \
If you stated a `claim_priority`, say why that order.

Do not pad it with a checklist of terms from this prompt. A paragraph naming \
every concept above while explaining no decision is worse than two sentences \
that state the comparison you actually made.
"""

SYSTEM_PROMPT = (
    _SYSTEM_PROMPT_HEAD + _RESTORABLE_GROUPS_BULLET + _SYSTEM_PROMPT_TAIL)


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
          "must avoid. Call this exactly once, after choosing which "
          "horizon's exposure the reroute has to survive.",
          CONSTRAINT_JSON_SCHEMA),
    _tool(OBJECTIVE_TOOL,
          "Submit which candidate from the routing menu to commit, or "
          "`infeasible` if none is acceptable under these constraints, or "
          "`hold` to commit nothing this hour. Call this exactly once, "
          "after comparing the candidates on the terms that matter for "
          "this service and this event state.",
          OBJECTIVE_JSON_SCHEMA),
    _tool(PROBE_TOOL,
          "Ask what the routing tools would offer one service shown in "
          "`exposure` if every asset in one of `risk_group_ids` were "
          "unusable. Read-only: it computes and changes nothing on the "
          "network. Returns `status`, `full_restore_candidates` (how many "
          "candidates restore the service's full demand on a genuinely "
          "different path), `min_spares_needed_by_site` (the cheapest such "
          "candidate's spare transponders per site, null if there is none) "
          f"and `levers`. At most {MAX_PROBES_PER_DECISION} calls per "
          "decision.",
          PROBE_JSON_SCHEMA),
]

TIMING_INSTRUCTION = (
    f"Decide whether to act this hour or wait for the next issuance, then "
    f"call `{TIMING_TOOL}`.")
CONSTRAINT_INSTRUCTION = (
    f"Decide what this reroute must route around, then call "
    f"`{CONSTRAINT_TOOL}`.")
OBJECTIVE_INSTRUCTION = (
    f"Choose one candidate from the menu above by its `candidate_label`, "
    f"`infeasible` if none is acceptable under these constraints, or "
    f"`hold` to commit nothing this hour, then call `{OBJECTIVE_TOOL}`.")


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
        # Bound by runner.run_episode per hour; see bind_probe below.
        self._probe = None

    def bind_probe(self, probe) -> None:
        """runner.run_episode binds one ProbeBinding per hour before
        `timing` and unbinds (None) after the hour; outside a rollout there
        is nothing to call and a probe is answered with an error result."""
        self._probe = probe

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
                      menu: dict | None = None) -> str:
        body = {"observation": payload}
        if menu is not None:
            body["menu"] = _menu_for_prompt(menu, self.oms_nodes)
        rendered = json.dumps(body, indent=2, sort_keys=True, default=str)
        return f"{rendered}\n\n{instruction}"

    def _create(self, messages: list[dict]):
        return self._api.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": self._system_prompt,
                     "cache_control": {"type": "ephemeral"}}],
            # Sonnet 5 accepts no budget_tokens and no temperature/top_p/
            # top_k -- any of them is a 400.
            thinking={"type": "adaptive"},
            tools=TOOLS,
            tool_choice={"type": "any", "disable_parallel_tool_use": True},
            messages=messages,
        )

    @staticmethod
    def _check_named_services(decision, payload, tool_name) -> None:
        """Every service id a decision names must be one the model was
        actually shown -- `contested_claim.service_id`, and every id in
        `claim_priority` -- and every id an `avoid` names must be one this
        observation carries.

        Same class of guard as the hallucinated risk-group id that burned 3 of
        5 iterations in a real T3a rollout (control-arm findings, root cause
        #3): the generated JSON disagreeing with the payload it was generated
        from. It rides the existing MAX_ATTEMPTS retry loop rather than a new
        mechanism -- from_dict cannot do this because it never sees the
        observation.

        The `avoid` half also rejects an avoid that binds NOTHING, and that
        forbids a legitimate answer: "I judge no constraint is needed here."
        The trade is deliberate. A vacuous avoid is not recorded as that
        judgement -- `route_service` returns the full menu including the
        service's own current path as a zero-spare `ip_reroute`, the objective
        step takes it, and the trace shows an inert commit, which reads as an
        action rather than as a decision not to act (D1 run B seeds 1 and 2,
        2026-09-12 failure analysis, finding 2). The same answer remains
        expressible one step later as `hold` or `infeasible` at the objective
        step, which is where it is RECORDED as a decision."""
        claim = getattr(decision, "contested_claim", None)
        if claim is not None and claim["service_id"] not in payload["exposure"]:
            raise DecisionError(
                f"{tool_name}: `contested_claim.service_id` "
                f"{claim['service_id']!r} is not in this observation's "
                f"`exposure`. Name a service you were shown, or null.")
        for svc in getattr(decision, "claim_priority", ()):
            if svc not in payload["exposure"]:
                raise DecisionError(
                    f"{tool_name}: `claim_priority` names {svc!r}, which is "
                    f"not in this observation's `exposure`. Name only "
                    f"services you were shown.")
        if (tool_name == TIMING_TOOL
                and not getattr(decision, "claim_priority", ())
                and not payload.get("standing_claim_priority")):
            raise DecisionError(
                f"{tool_name}: `claim_priority` is empty and no ranking is "
                f"standing yet, so there is none to keep. State an ordering "
                f"of the services shown -- most deserving of this depot's "
                f"spares first, INCLUDING the actionable service.")
        if tool_name == CONSTRAINT_TOOL:
            known_assets = {row["asset_id"]
                            for group in payload.get("risk_groups") or ()
                            for row in group.get("assets") or ()}
            known_groups = {gid for gid in
                            (payload.get("risk_group_ids") or {}).values()
                            if gid}
            if not known_assets and not known_groups:
                # Nothing was shown that an avoid could legally name. Not
                # reachable on the shipped suite -- a decidable hour always
                # has a risk group defined for its exposed horizon -- but
                # three rejected attempts kill a rollout, so never trap a
                # model with an unanswerable request.
                return
            avoid = getattr(decision, "avoid", None) or {}
            for asset in avoid.get("assets") or ():
                if asset not in known_assets:
                    raise DecisionError(
                        f"{tool_name}: `avoid.assets` names {asset!r}, which "
                        f"is not an `asset_id` in any entry of this "
                        f"observation's `risk_groups`. Name only assets you "
                        f"were shown.")
            for group_id in avoid.get("risk_groups") or ():
                if group_id not in known_groups:
                    raise DecisionError(
                        f"{tool_name}: `avoid.risk_groups` names "
                        f"{group_id!r}, which is not one of this "
                        f"observation's `risk_group_ids`. Name only groups "
                        f"you were shown.")
            if not (avoid.get("assets") or avoid.get("risk_groups")):
                raise DecisionError(
                    f"{tool_name}: `avoid` names no asset and no risk group, "
                    f"so it binds nothing -- `route_service` would return the "
                    f"menu it returns with no constraint at all, including "
                    f"this service's CURRENT path as a zero-cost candidate. "
                    f"Name the assets or the group the reroute must route "
                    f"around. If your judgement is that no reroute is worth "
                    f"making, answer `hold` at the objective step, where that "
                    f"is recorded as a decision.")

    async def _decide(self, tool_name, decision_cls, obs, payload, user_content):
        """One decision, with up to MAX_ATTEMPTS self-correction rounds and
        any number of ACCEPTED probe rounds in between.

        `tool_choice: any` lets the model call `probe_restorability` before
        the decision tool. An accepted probe is answered as a tool_result and
        costs no attempt; a REJECTED probe (unshown service, unknown group,
        over the cap, nothing bound) is answered as an error tool_result and
        costs one attempt, the same class of failure as a hallucinated
        claim_priority id -- otherwise a model that keeps probing past the
        cap could loop forever. A tool_use naming the wrong decision tool is
        answered with an error tool_result (an unanswered tool_use is an API
        error, so the correction cannot be a plain user turn any more); a
        response with no tool_use at all still gets a plain user turn.

        Only schema/validation failures are retried here. Transient API
        failures belong to the SDK's own `max_retries`. On the final failure
        the DecisionError propagates: run_episode never catches a decider
        exception, so a decider that cannot decide kills the rollout visibly."""
        messages: list[dict] = [{"role": "user", "content": user_content}]
        failure: DecisionError | None = None
        probes: list[dict] = []
        attempt = 0
        while attempt < MAX_ATTEMPTS:
            response = self._create(messages)
            block = next((b for b in response.content
                          if getattr(b, "type", None) == "tool_use"), None)
            if block is None:
                attempt += 1
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
            if block.name == PROBE_TOOL:
                answer, error = await self._run_probe(dict(block.input))
                probes.append({"arguments": dict(block.input),
                               "answer": answer, "error": error})
                # Echo the assistant content back unchanged -- with adaptive
                # thinking it carries a thinking block that must survive the
                # turn -- then answer its tool_use with the probe's result.
                messages.append(
                    {"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": block.id,
                    "is_error": error is not None,
                    "content": json.dumps(answer) if error is None else error}]})
                if error is not None:
                    attempt += 1
                    failure = DecisionError(
                        f"{tool_name}: probe rejected -- {error}")
                continue
            if block.name != tool_name:
                attempt += 1
                failure = DecisionError(
                    f"{tool_name}: the model called {block.name!r} instead")
                messages.append(
                    {"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": block.id,
                    "is_error": True,
                    "content": (f"This request needs `{tool_name}`, not "
                                f"`{block.name}`. Call `{tool_name}` with "
                                f"your decision.")}]})
                continue
            try:
                decision = decision_cls.from_dict(block.input)
                self._check_named_services(decision, payload, tool_name)
            except DecisionError as exc:
                attempt += 1
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
            self._audit(obs, payload, tool_name, decision, attempt + 1, probes)
            return decision
        raise failure

    async def _run_probe(self, arguments: dict) -> tuple[dict | None, str | None]:
        """`(answer, None)` or `(None, error_text)`; never raises."""
        if self._probe is None:
            return None, (f"`{PROBE_TOOL}` is not available in this context: "
                          f"no rollout is in progress")
        try:
            answer = await self._probe(arguments.get("service_id"),
                                       arguments.get("risk_group_id"))
        except ProbeError as exc:
            return None, str(exc)
        return answer, None

    def _audit(self, obs, payload, tool_name, decision, attempts, probes) -> None:
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
            "probes": probes,
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

    def _project(self, obs: Observation, *,
                 include_risk_group_assets: bool = False) -> dict:
        self.last_projection = project_observation(
            obs, p_cut_threshold=self._p_cut_threshold,
            include_risk_group_assets=include_risk_group_assets)
        return self.last_projection

    async def timing(self, obs: Observation) -> TimingDecision:
        payload = self._project(obs)
        return await self._decide(
            TIMING_TOOL, TimingDecision, obs, payload,
            self._user_content(payload, TIMING_INSTRUCTION))

    async def constraints(self, obs: Observation,
                          unconstrained_menu: dict | None = None
                          ) -> ConstraintDecision:
        # Accepted and IGNORED. The model no longer sees this menu -- see
        # `_user_content` -- but the Decider protocol (decisions.py) and
        # runner.py still pass it positionally.
        payload = self._project(obs, include_risk_group_assets=True)
        return await self._decide(
            CONSTRAINT_TOOL, ConstraintDecision, obs, payload,
            self._user_content(payload, CONSTRAINT_INSTRUCTION))

    async def objective(self, obs: Observation, menu: dict) -> ObjectiveDecision:
        payload = self._project(obs)
        return await self._decide(
            OBJECTIVE_TOOL, ObjectiveDecision, obs, payload,
            self._user_content(payload, OBJECTIVE_INSTRUCTION, menu=menu))
