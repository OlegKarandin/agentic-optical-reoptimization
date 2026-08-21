# src/storm_reoptimizer/eval/ledger.py
"""The harness-owned spare-transponder ledger (eval design spec, build order
item 4).

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

THE UNIT IS TRANSPONDER PAIRS -- one pair per new lightpath. Do not derive it
from cost_vector["transponders"]: score_candidate materializes the candidate
on a WHOLE-MODEL clone and evaluate_objective's term is
`2.0 * len(model.list_lightpaths())`, i.e. the entire network's count. The
candidate's own cost is exactly len(new_lightpaths)."""
from __future__ import annotations

from dataclasses import dataclass, field


class InsufficientSpares(RuntimeError):
    """A chosen candidate the depot cannot fulfil."""

    def __init__(self, needed: int, on_hand: int) -> None:
        super().__init__(
            f"candidate needs {needed} spare transponder pair(s); "
            f"{on_hand} on hand")
        self.needed = needed
        self.on_hand = on_hand


def pairs_needed(candidate: dict) -> int:
    """Transponder pairs this candidate consumes: one per new lightpath. An
    ip_reroute costs none, which is precisely what makes `ip_reroute` a
    trustworthy scoring target -- the lever label means exactly "this
    recovery costs no transponders"."""
    return len(candidate["new_lightpaths"])


@dataclass
class SpareLedger:
    """Mutable, one per episode rollout. Reported to the decider as part of
    each hour's observation and decremented on every successful commit."""
    on_hand: int
    spent: int = 0
    debits: list[dict] = field(default_factory=list)

    def can_afford(self, candidate: dict) -> bool:
        return pairs_needed(candidate) <= self.on_hand

    def rejection(self, candidate: dict) -> dict:
        """The typed result the runner feeds back into the capped loop."""
        return {"type": "insufficient_spares",
                "needed": pairs_needed(candidate), "on_hand": self.on_hand}

    def debit(self, candidate: dict, *, hour: str, service_id: str) -> int:
        """Consume this candidate's spares. Raises InsufficientSpares and
        changes nothing if the depot cannot cover it."""
        needed = pairs_needed(candidate)
        if needed > self.on_hand:
            raise InsufficientSpares(needed, self.on_hand)
        if needed:
            self.on_hand -= needed
            self.spent += needed
            self.debits.append(
                {"hour": hour, "service_id": service_id, "pairs": needed})
        return needed
