# src/storm_reoptimizer/events/contract.py
"""The internal event contract every upstream adapter normalizes to
(CLAUDE.md's "Event input contract & upstream adapters"):
{geometry, event_type, valid_at, attributes}. A moving event (a storm cone
advancing) is a sequence of these, one per forecast hour."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HazardEvent:
    geometry: dict  # GeoJSON geometry object (polygon / track cone / perimeter)
    event_type: str
    valid_at: str  # ISO 8601 timestamp, UTC
    attributes: dict[str, Any] = field(default_factory=dict)
