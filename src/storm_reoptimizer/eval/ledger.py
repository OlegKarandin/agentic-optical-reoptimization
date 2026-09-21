# src/storm_reoptimizer/eval/ledger.py
"""The harness-owned spare-transponder ledger (eval design spec, build order
item 4; per-site depot, exposure-and-depot design §4.1).

Why it lives here and not on the server: `spare_inventory` is not in the
model. It exists only as an argument to solve_allocation/_pack and to the
operating-network builder -- not persisted in the state file, not part of
NetworkModel, invisible to route_service, and dropped outright from the MCP
layer's evaluate_objective signature. "Spares on hand" is not readable from
the server at all.

The split mirrors how operators actually run: the optical controller knows
spectrum and topology, while spare pluggables live in an inventory system the
controller cannot see -- and restoration tooling that trusts the controller's
view alone routinely proposes plans the depot cannot fulfil. It yields a
retry signal for free: route_service can legitimately return a candidate the
ledger cannot afford, and the harness rejects that choice with a typed
`insufficient_spares` result that feeds the capped loop.

THE UNIT IS TRANSPONDERS PER SITE -- one transponder at EACH endpoint site of
each new lightpath. A bare integer had a real hole: a "hybrid" candidate can
groom satna -> X onto an existing lightpath and light a new X -> allahabad
leg that never touches satna at all, paying nothing from satna's depot even
though a real transponder pair. So the ledger is scoped per site: a
lightpath's cost is charged at BOTH of its endpoint sites, and only the sites
a candidate actually touches can ever bind.

A COUNTER-PROPAGATING PAIR SHARES ITS TRANSPONDERS. A real pluggable is
TX+RX: one per end serves both directions, so a lightpath's reverse mate
over the SAME two endpoint sites -- keyed on the unordered site pair, not
the OMS route, since a transponder's two ends need not share a fibre route
-- is free once one direction is already lit. Per site S and unordered
endpoint pair P = {S, far}:

    charge(S, P) = max(new_fwd + lit_fwd, new_bwd + lit_bwd)
                 - max(lit_fwd, lit_bwd)

`lit_fwd`/`lit_bwd` count only runs LIT BY THIS ROLLOUT (`SpareLedger.
lit_runs`, grown by `debit()`) -- never the lightpaths already present in
the seeded state, which the offline builder already charged against its
own PIN_SPARE_INVENTORY (`model/allocation.py`'s own "one transponder per
new-run endpoint, gated on spare inventory"); pairing them now would
double-count in the opposite direction and make large parts of the network
restorable for zero spares, a far bigger behavioural change than this fix
intends. Documented as a known modeling boundary, not a bug, in the same
register `network.ip_link_capacity_gbps` uses for its own.

Do NOT derive this from cost_vector["transponders"]: score_candidate
materializes the candidate on a WHOLE-MODEL clone and evaluate_objective's
term is `2.0 * len(model.list_lightpaths())`, i.e. the entire network's
count. The candidate's own cost is exactly one transponder per lightpath per
endpoint site. The server's own evaluate_objective transponder term
(`2.0 * len(model.list_lightpaths())`) carries the SAME per-direction
double-count this fix corrects here -- deliberately NOT fixed to match: it
is a whole-network constant offset among candidates that light the same
number of lightpaths (so it cannot change the menu's ranking, the only
thing this eval reads it for), and it lives in the sibling repo, out of
this module's reach."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


class InsufficientSpares(RuntimeError):
    """A chosen candidate the depot cannot fulfil, naming the site that binds."""

    def __init__(self, needed: dict[str, int], inventory: dict[str, int],
                 site: str) -> None:
        super().__init__(
            f"candidate needs {needed[site]} spare transponder(s) at {site!r}; "
            f"{inventory.get(site, 0)} on hand there")
        self.needed = dict(needed)
        self.inventory = dict(inventory)
        self.site = site


def _lightpath_endpoints(lightpath: dict, oms_nodes: dict) -> tuple[str, str]:
    """The two SITES a lightpath terminates at.

    Taken as the nodes appearing exactly ONCE across its legs' endpoints, not
    from the first and last leg: leg order is not guaranteed head-to-tail, and
    an interior junction appears once per adjacent leg."""
    seen: dict[str, int] = {}
    for oms_id in lightpath.get("oms_sequence") or ():
        for node in oms_nodes.get(oms_id, ()):
            seen[node] = seen.get(node, 0) + 1
    ends = [node for node, count in seen.items() if count == 1]
    if len(ends) != 2:
        raise ValueError(
            f"lightpath {lightpath.get('oms_sequence')!r} does not resolve to "
            f"exactly two endpoint sites (got {sorted(ends)}); its OMS "
            f"sequence is not a simple path over {sorted(seen)}")
    return ends[0], ends[1]


def spares_needed(candidate: dict, oms_nodes: dict, *,
                  lit_runs: Iterable[tuple[str, str]] = ()) -> dict[str, int]:
    """Transponders this candidate consumes, PER SITE, net of any
    counter-propagating run this ROLLOUT already lit.

    `lit_runs` is an iterable of `(src_site, dst_site)` tuples for
    lightpaths already lit earlier in this SAME rollout -- typically
    `SpareLedger.lit_runs`. Defaulted to `()` so every caller that wants a
    candidate's standalone, rollout-independent cost (an authoring-time
    check, an offline tool with no ledger) keeps today's arithmetic
    exactly, and so a candidate whose OWN `new_lightpaths` never contains
    two counter-propagating runs of the same pair is unaffected regardless.

    An `ip_reroute` lights nothing and so costs nothing anywhere, which is
    precisely what makes the lever label a trustworthy scoring target.

    Do NOT derive this from cost_vector["transponders"]: score_candidate
    materializes the candidate on a WHOLE-MODEL clone and evaluate_objective's
    term is `2.0 * len(model.list_lightpaths())`, i.e. the entire network's
    count. See the module docstring for the mate-pairing rule and its
    §3.4-numbered scope boundary."""
    new_by_pair: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    for lightpath in candidate.get("new_lightpaths") or ():
        run = _lightpath_endpoints(lightpath, oms_nodes)
        pair = tuple(sorted(run))
        by_dir = new_by_pair.setdefault(pair, {})
        by_dir[run] = by_dir.get(run, 0) + 1

    lit_by_pair: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    for run in lit_runs:
        pair = tuple(sorted(run))
        by_dir = lit_by_pair.setdefault(pair, {})
        by_dir[run] = by_dir.get(run, 0) + 1

    needed: dict[str, int] = {}
    for pair, by_dir in new_by_pair.items():
        fwd, bwd = pair, (pair[1], pair[0])
        lit = lit_by_pair.get(pair, {})
        new_fwd, new_bwd = by_dir.get(fwd, 0), by_dir.get(bwd, 0)
        lit_fwd, lit_bwd = lit.get(fwd, 0), lit.get(bwd, 0)
        charge = (max(new_fwd + lit_fwd, new_bwd + lit_bwd)
                 - max(lit_fwd, lit_bwd))
        if charge <= 0:
            continue
        for site in pair:
            needed[site] = needed.get(site, 0) + charge
    return needed


@dataclass
class SpareLedger:
    """Mutable, one per episode rollout. Reported to the decider as part of
    each hour's observation and decremented on every successful commit.

    `on_hand` and `spent` are properties over `depot_site` alone, so
    `Observation.spares_on_hand` and `metadata.spares_on_hand` keep their
    existing meaning -- "spares at depot_site" -- and `rules.OBSERVABLE_VARS`
    needs no change. Any site not named in `inventory` starts at
    `default_spares_per_site`: a generous, deliberately non-binding stock,
    since this harness models scarcity only at the depot; a candidate is
    refused only when the DEPOT can't cover its own charge, never because some
    unrelated far-end site's assumed stock ran out.

    `lit_runs` is every NEW lightpath's own `(src_site, dst_site)` this
    ledger has ever debited, oldest first -- grown by `debit()`, read by
    `spares_needed`'s mate-pairing rule. Starts empty regardless of what the
    seeded server state already has lit (spec §3.4)."""
    inventory: dict[str, int]
    depot_site: str
    oms_nodes: dict[str, list[str]]
    default_spares_per_site: int = 8
    debits: list[dict] = field(default_factory=list)
    lit_runs: list[tuple[str, str]] = field(default_factory=list)

    def _stock(self, site: str) -> int:
        return self.inventory.get(site, self.default_spares_per_site)

    @property
    def on_hand(self) -> int:
        return self._stock(self.depot_site)

    @property
    def spent(self) -> int:
        return sum(d["spares"].get(self.depot_site, 0) for d in self.debits)

    def can_afford(self, candidate: dict) -> bool:
        needed = spares_needed(candidate, self.oms_nodes, lit_runs=self.lit_runs)
        return all(count <= self._stock(site)
                   for site, count in needed.items())

    def _binding_site(self, needed: dict[str, int]) -> str | None:
        """The first site that can't cover its own charge, preferring
        `depot_site` when several sites bind -- it is the one the decider's
        retry loop can actually act on."""
        if self.depot_site in needed and needed[self.depot_site] > self._stock(
                self.depot_site):
            return self.depot_site
        for site, count in needed.items():
            if count > self._stock(site):
                return site
        return None

    def rejection(self, candidate: dict) -> dict:
        """The typed result the runner feeds back into the capped loop."""
        needed = spares_needed(candidate, self.oms_nodes, lit_runs=self.lit_runs)
        site = self._binding_site(needed) or self.depot_site
        return {"type": "insufficient_spares", "site": site,
                "needed": needed, "inventory": dict(self.inventory)}

    def debit(self, candidate: dict, *, hour: str,
              service_id: str, origin: str = "decider") -> dict[str, int]:
        """Consume this candidate's spares at every site it charges. Raises
        InsufficientSpares and changes nothing if any charged site cannot
        cover it.

        `origin` distinguishes spares spent by the decider's own choices from
        spares spent by the harness's own deterministic restoration logic
        ("harness") -- later scoring needs to tell the two apart.

        On success, every NEW lightpath's own endpoints (in the direction
        `_lightpath_endpoints` resolves them) are appended to `lit_runs` --
        even a candidate that netted a zero charge because it mated against
        an already-lit run still lights a real, directed lightpath, and a
        THIRD run over the same pair must still see both prior directions."""
        needed = spares_needed(candidate, self.oms_nodes, lit_runs=self.lit_runs)
        site = self._binding_site(needed)
        if site is not None:
            raise InsufficientSpares(needed, self.inventory, site)
        for charged_site, count in needed.items():
            self.inventory[charged_site] = self._stock(charged_site) - count
        if needed:
            self.debits.append(
                {"hour": hour, "service_id": service_id, "spares": needed,
                 "origin": origin})
        for lightpath in candidate.get("new_lightpaths") or ():
            self.lit_runs.append(_lightpath_endpoints(lightpath, self.oms_nodes))
        return needed
