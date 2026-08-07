# src/storm_reoptimizer/events/filters.py
"""(event_type -> vulnerability-filter) table -- CLAUDE.md: "Structured
feeds with known types go through a plain (event_type -> filter) table --
no agent." Only storm is real today: a storm threatens aerial fiber, not
buried. Flood's inverted filter (buried spans -- handholes, splice
closures, low-lying crossings -- are exposed; aerial is safe) is step 7's
job and needs topology attributes this toy dataset doesn't carry yet."""
from __future__ import annotations

from typing import Callable

from ..geo_mapper import Edge

FILTERS: dict[str, Callable[[Edge], bool]] = {
    "storm": lambda edge: edge.mount_type == "aerial",
}


def get_filter(event_type: str) -> Callable[[Edge], bool]:
    try:
        return FILTERS[event_type]
    except KeyError:
        raise KeyError(
            f"no vulnerability filter defined for event_type {event_type!r} "
            f"(known: {sorted(FILTERS)})"
        ) from None
