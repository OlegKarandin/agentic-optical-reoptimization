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

from .cone import expected_capacity_at_risk_gbps
from .observation import Observation

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


def project_observation(
    obs: Observation, *,
    p_cut_threshold: float = P_CUT_ENUMERATION_THRESHOLD,
) -> dict:
    """`obs.to_dict()` reduced to decision-relevant content, plus an explicit
    account of what was dropped.

    Kept: the service under test always, and every service whose peak p_cut
    reaches `p_cut_threshold`. `services` is trimmed to the same set --
    `exposure` already carries each service's `demand_gbps`, so the roster is
    nearly redundant with it for decision purposes.

    `omitted_services["count"]` is over the server's full roster (a service
    with no representative point has no exposure entry at all and is counted
    here); the two risk figures are over the omitted services that do have
    exposure, and contribute 0.0 for the rest."""
    payload = obs.to_dict()
    exposure = payload["exposure"]

    keep = {obs.service_under_test}
    keep |= {svc for svc, per_horizon in exposure.items()
             if _peak_p_cut(per_horizon) >= p_cut_threshold}
    omitted = [svc for svc in exposure if svc not in keep]

    all_services = payload["services"]
    payload["exposure"] = {svc: per_horizon
                           for svc, per_horizon in exposure.items()
                           if svc in keep}
    payload["services"] = [s for s in all_services if s["id"] in keep]
    payload["n_services_total"] = len(all_services)
    payload["omitted_services"] = {
        "count": len(all_services) - len(payload["services"]),
        "p_cut_threshold": p_cut_threshold,
        "max_p_cut": round(
            max((_peak_p_cut(exposure[svc]) for svc in omitted), default=0.0),
            4),
        "summed_expected_capacity_at_risk_gbps": round(
            sum(_peak_capacity_at_risk_gbps(exposure[svc])
                for svc in omitted), 3),
    }
    return payload
