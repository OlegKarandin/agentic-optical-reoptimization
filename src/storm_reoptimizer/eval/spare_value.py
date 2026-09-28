# src/storm_reoptimizer/eval/spare_value.py
"""Decision-hour facts for a spend-or-hold call, and the two things gold's
hold branch and a live check (spec 2026-09-27-t2-correlated-claims-and-
honest-scope-design.md §3.3/§4.6) are built on: the best possible ranking of
who a held spare should go to, and the actual expected value, in Gbps-hours,
of spending it now versus holding it.

`decision_facts` gathers everything the calculation needs from one live
server: the joint cut-outcome table at the decision-hour issuance's own
latest horizon (`oracle.spend_risk_group` -- the SAME risk group `oracle.
spend_decider` avoids), which of the services ever shown to a decider by the
time that horizon's cut lands are restorable under that group and at what
depot cost, and how long a cut there stays down before the harness's own
post-cut replay would restore it (`observation.build_observation`'s
`hours_down_if_cut`).

Deliberately does NOT import `.assertions` or `.gold`: this module ITSELF
imports `.oracle`/`.probe`/`.observation`/`.agent`/`.runner` (see below), and
it is `.gold`/`.assertions` that consume THIS module, not the other way
around (both DO import this module -- Task 5's own wiring)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mcp.client import Client

from ..geo_mapper import load_edges
from ..mcp_client import call_tool_json
from .agent import project_observation
from .observation import Observation, build_observation, cut_outcome_rows
from .oracle import spend_risk_group
from .probe import answer_probe
from .runner import (
    EVENT_TYPE, horizon_risk_group_asset_ids, service_geometry,
)
from .scenario_file import ScenarioFile
from ..events.filters import get_filter


@dataclass(frozen=True)
class DecisionFacts:
    """Every fact a "spend the depot's held spare on `sut` now, or hold it"
    decision at `scenario.decision_hour` depends on, gathered as of `rg_id`
    (`oracle.spend_risk_group`'s own `(horizon, rg_id)` for the decision-hour
    issuance's latest horizon).

    `rows`: the joint cut-outcome table at `horizon`, RAW -- unmerged (no
    `merge_below` folding) and unrounded (no `ndigits`) -- over exactly
    `shown_through_cut`, as `observation.cut_outcome_rows(d_obs,
    shown_through_cut, merge_below=0.0, ndigits=None)[horizon]` returns it,
    each row's `down` set turned into a `frozenset` for hashability. Rows
    are mutually exclusive and their `p` sums to 1.0.

    `demands`/`restorable`/`depot_spares` are keyed over exactly
    `shown_through_cut`: `demands[s]` is `s`'s `demand_gbps`; `restorable[s]`
    is whether a probe of `s` under `rg_id` found `full_restore_candidates >
    0` (the same reading `assertions.probe_reading`'s `"restorable"` kind
    uses); `depot_spares[s]` is that probe's cheapest full-restore
    candidate's own spend at `depot_site` (0 when `s` restores for free
    there, or is not restorable at all)."""
    sut: str
    depot_site: str
    spares_on_hand: int
    horizon: str
    rg_id: str
    rows: tuple[tuple[frozenset[str], float], ...]
    demands: dict[str, float]
    restorable: dict[str, bool]
    depot_spares: dict[str, int]
    unrestored: int
    restored_after_cut: int
    shown_through_cut: frozenset[str]


async def probe_answers_under_decision_group(
    client: Client, scenario: ScenarioFile, service_ids: Iterable[str], *,
    topology_path: str | Path,
) -> dict[str, dict]:
    """Every named `service_ids`'s probe answer under the decision-hour
    issuance's own latest-horizon risk group, defined EXACTLY as
    `oracle.spend_risk_group` + `runner.horizon_risk_group_asset_ids` define
    it. Pre-decision realized cuts are replayed first, as
    `menu_at_decision_hour` does.

    Moved here, verbatim apart from taking `service_ids` as a parameter
    instead of reading `scenario.metadata["claimant_services"]`, from
    `assertions.claimant_probe_answers` (2026-09-27 plan, Task 4) -- that
    function is now a thin wrapper over this one, so there is one probing
    code path, not two. Raises a plain `ValueError` (not `assertions.
    PairInvalid`) when `spend_risk_group` cannot compute a risk group;
    `claimant_probe_answers` re-raises that as `PairInvalid` itself, since
    this module does not import `.assertions`."""
    import functools

    edges = load_edges(topology_path)
    d = scenario.decision_hour
    for hour in scenario.hours[:scenario.hours.index(d)]:
        cuts = scenario.realized.get(hour, ())
        if cuts:
            await call_tool_json(client, "inject_failure",
                                 {"asset_ids": list(cuts)})
    horizon, rg_id = spend_risk_group(scenario)
    issuance = scenario.forecast[d]
    geometry = await service_geometry(client, topology_path, edges=edges)
    topo = await call_tool_json(client, "get_topology", {"layer": "optical"})
    await call_tool_json(client, "define_risk_group", {
        "rg_id": rg_id,
        "asset_ids": horizon_risk_group_asset_ids(
            issuance.horizons[horizon], scenario.damage_radius_km,
            edges=edges, oms=topo["oms"], filter_fn=get_filter(EVENT_TYPE)),
        "metadata": {"event_type": EVENT_TYPE, "scenario": scenario.id,
                     "issued_at": d, "horizon": horizon}})
    services = (await call_tool_json(client, "get_services"))["services"]
    demands = {s["id"]: float(s["demand_gbps"]) for s in services}
    call = functools.partial(call_tool_json, client)
    answers = {}
    for service_id in service_ids:
        answer = await answer_probe(
            call, service_id=service_id, risk_group_id=rg_id, geometry=geometry,
            issuance=issuance, damage_radius_km=scenario.damage_radius_km,
            demands=demands)
        answers[service_id] = answer.to_dict()
    return answers


async def decision_facts(
    client: Client, scenario: ScenarioFile, *, topology_path: str | Path,
) -> DecisionFacts:
    """Gather a `DecisionFacts` for `scenario` from one connected `client`.

    `shown_through_cut` is the union, across every hour from the episode's
    first hour through `horizon` INCLUSIVE, of `agent.project_observation
    (obs)["exposure"]`'s keys at that hour -- every service ever shown to a
    decider by the time the cut lands. Hold never acts, so no commit moves a
    path between these hours: each hour's observation is built from the SAME
    `service_geometry`/`get_services` read (fetched once, not once per
    hour), purely from the scenario's own forecast -- no server call and no
    realized-cut replay happens in this loop.

    The joint-cut `rows` and `hours_down_if_cut` come from the observation
    built at `scenario.decision_hour` itself (`d_obs` below) -- the hour
    whose own issuance IS the one `horizon`/`rg_id` name (`spend_risk_group`'s
    own contract: `d`'s issuance is read before anything else could differ).
    `d_obs` is always produced by the loop above: `scenario_file.load_scenario`
    requires an issuance's horizons to lie strictly after its own issue hour,
    so `scenario.hours.index(d) < scenario.hours.index(horizon)`.

    Restorability and depot cost come from probing every `shown_through_cut`
    service under `rg_id` (`probe_answers_under_decision_group`) -- real
    server calls, replaying pre-decision realized cuts first, exactly as a
    decider's own probe would.

    Raises the same `ValueError` `oracle.spend_risk_group` raises when
    `scenario.decision_hour` has no forecast issuance of its own (or that
    issuance has no horizon)."""
    edges = load_edges(topology_path)
    geometry = await service_geometry(client, topology_path, edges=edges)
    services = tuple((await call_tool_json(client, "get_services"))["services"])

    horizon, rg_id = spend_risk_group(scenario)
    horizon_index = scenario.hours.index(horizon)
    d = scenario.decision_hour

    shown_through_cut: set[str] = set()
    d_obs: Observation | None = None
    for hour in scenario.hours[:horizon_index + 1]:
        obs = build_observation(
            scenario, hour,
            service_spans=geometry.cuttable_spans,
            services=services,
            spares_on_hand=scenario.spares_on_hand,
            endpoint_sites=geometry.endpoint_sites,
            depot_site=scenario.depot_site,
            protection_spans=geometry.protection_cuttable_spans)
        shown_through_cut |= set(project_observation(obs)["exposure"])
        if hour == d:
            d_obs = obs
    assert d_obs is not None, (
        f"{scenario.id}: decision hour {d!r} was not visited while walking "
        f"hours up to horizon {horizon!r} -- scenario_file.load_scenario "
        f"guarantees an issuance's horizons lie strictly after its own "
        f"issue hour, so this should be unreachable")

    shown = frozenset(shown_through_cut)
    demands = {s["id"]: float(s["demand_gbps"])
              for s in services if s["id"] in shown}
    rows = tuple(
        (frozenset(row["down"]), row["p"])
        for row in cut_outcome_rows(
            d_obs, shown, merge_below=0.0, ndigits=None)[horizon])

    answers = await probe_answers_under_decision_group(
        client, scenario, shown, topology_path=topology_path)
    restorable = {s: a["full_restore_candidates"] > 0
                 for s, a in answers.items()}
    depot_spares = {
        s: (a["min_spares_needed_by_site"] or {}).get(scenario.depot_site, 0)
        for s, a in answers.items()}

    hours_down = d_obs.hours_down_if_cut[horizon]

    return DecisionFacts(
        sut=scenario.service_under_test,
        depot_site=scenario.depot_site,
        spares_on_hand=scenario.spares_on_hand,
        horizon=horizon,
        rg_id=rg_id,
        rows=rows,
        demands=demands,
        restorable=restorable,
        depot_spares=depot_spares,
        unrestored=hours_down["unrestored"],
        restored_after_cut=hours_down["restored_after_cut"],
        shown_through_cut=shown)


def best_hold_ranking(facts: DecisionFacts) -> tuple[str, ...]:
    """Who a held spare SHOULD go to, best case (spec §4.6): every
    restorable service in `facts.shown_through_cut`, ordered by
    `demand x (unrestored - restored_after_cut)` -- the Gbps-hours a real
    restore of it actually saves -- descending, ties broken by id, then
    `facts.sut` appended if it is not already present (so a decider reading
    this as a priority order is never told to ignore the service under
    test outright, even when it is not itself restorable)."""
    delay = facts.unrestored - facts.restored_after_cut
    candidates = [s for s in facts.shown_through_cut
                 if facts.restorable.get(s, False)]
    ranked = tuple(sorted(
        candidates, key=lambda s: (-facts.demands[s] * delay, s)))
    if facts.sut not in ranked:
        ranked = ranked + (facts.sut,)
    return ranked


def expected_spare_value(facts: DecisionFacts) -> dict[str, float]:
    """The real expected value, in Gbps-hours, of spending the depot's one
    held spare on `facts.sut` right now versus holding it (spec §3.3's
    worked method).

    `spend`: the spare escapes the SUT before the cut, so it saves the
    SUT's own demand for every hour it would otherwise stay down
    (`facts.unrestored`), weighted by the probability the SUT is actually
    among the storm's `down` outcomes: `P(sut in down) x unrestored x
    demand[sut]`.

    `hold`: for each joint-cut row, the held spare is offered, in
    `best_hold_ranking`'s order, to the first service that is (a) actually
    down in that row, (b) restorable under `facts.rg_id`, and (c) needs a
    real spare to do it (`facts.depot_spares[s] > 0`) -- a service whose own
    restoration is free at the depot is restored either way and does not
    claim the spare, matching the replay's own cheapest-first `can_afford`
    walk and the prompt's "does not need it either". That service is
    restored `facts.restored_after_cut` hours late instead of staying down
    the full `facts.unrestored`, so the row contributes
    `p x (unrestored - restored_after_cut) x demand[s]`; a row with no such
    service (nobody eligible is actually down in it) contributes 0.

    Single-spare only: raises `ValueError` if `facts.spares_on_hand != 1`.
    A `down` row may name more than one restorable, spare-needing claimant
    at once, and this formula gives the ONE held spare to only the first of
    them in ranked order -- it does not model splitting spares across
    simultaneous claims, so it is not meaningful for more than one spare."""
    if facts.spares_on_hand != 1:
        raise ValueError(
            f"expected_spare_value is single-spare by construction: it "
            f"gives the ONE held spare to the first eligible claimant in "
            f"each row and does not model splitting several spares across "
            f"simultaneous claims. facts.spares_on_hand={facts.spares_on_hand!r}")
    ranking = best_hold_ranking(facts)
    delay = facts.unrestored - facts.restored_after_cut
    p_sut_down = sum(p for down, p in facts.rows if facts.sut in down)
    spend = p_sut_down * facts.unrestored * facts.demands[facts.sut]
    hold = 0.0
    for down, p in facts.rows:
        for s in ranking:
            if (s in down and facts.restorable.get(s, False)
                    and facts.depot_spares.get(s, 0) > 0):
                hold += p * delay * facts.demands[s]
                break
    return {"spend": spend, "hold": hold}
