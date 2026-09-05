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

Do NOT derive this from cost_vector["transponders"]: score_candidate
materializes the candidate on a WHOLE-MODEL clone and evaluate_objective's
term is `2.0 * len(model.list_lightpaths())`, i.e. the entire network's
count. The candidate's own cost is exactly one transponder per lightpath per
endpoint site."""
from __future__ import annotations

from dataclasses import dataclass, field


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


def spares_needed(candidate: dict, oms_nodes: dict) -> dict[str, int]:
    """Transponders this candidate consumes, PER SITE: one at each endpoint of
    each new lightpath.

    An `ip_reroute` lights nothing and so costs nothing anywhere, which is
    precisely what makes the lever label a trustworthy scoring target.

    Do NOT derive this from cost_vector["transponders"]: score_candidate
    materializes the candidate on a WHOLE-MODEL clone and evaluate_objective's
    term is `2.0 * len(model.list_lightpaths())`, i.e. the entire network's
    count."""
    needed: dict[str, int] = {}
    for lightpath in candidate.get("new_lightpaths") or ():
        for site in _lightpath_endpoints(lightpath, oms_nodes):
            needed[site] = needed.get(site, 0) + 1
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
    unrelated far-end site's assumed stock ran out."""
    inventory: dict[str, int]
    depot_site: str
    oms_nodes: dict[str, list[str]]
    default_spares_per_site: int = 8
    debits: list[dict] = field(default_factory=list)

    def _stock(self, site: str) -> int:
        return self.inventory.get(site, self.default_spares_per_site)

    @property
    def on_hand(self) -> int:
        return self._stock(self.depot_site)

    @property
    def spent(self) -> int:
        return sum(d["spares"].get(self.depot_site, 0) for d in self.debits)

    def can_afford(self, candidate: dict) -> bool:
        needed = spares_needed(candidate, self.oms_nodes)
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
        needed = spares_needed(candidate, self.oms_nodes)
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
        ("harness") -- later scoring needs to tell the two apart."""
        needed = spares_needed(candidate, self.oms_nodes)
        site = self._binding_site(needed)
        if site is not None:
            raise InsufficientSpares(needed, self.inventory, site)
        for charged_site, count in needed.items():
            self.inventory[charged_site] = self._stock(charged_site) - count
        if needed:
            self.debits.append(
                {"hour": hour, "service_id": service_id, "spares": needed,
                 "origin": origin})
        return needed
