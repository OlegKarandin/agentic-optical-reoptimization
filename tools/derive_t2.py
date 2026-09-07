# tools/derive_t2.py
"""T2 -- restorable at all (spec 2026-09-06 §4.1): derive_t1.derive with
T2's defaults. Every derive_t1 flag still applies; the bearings and radii
are the authoring inputs (docs/superpowers/plans/notes/2026-09-06-t2-t3-
authoring.md records the ones used)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive_t1  # noqa: E402

T2_DEFAULTS = dict(
    pair="T2", width_km=60.0, damage_radius_km=50.0,
    hours="t0,t1,t2,t3,t4,t5,t6,t7", decision_hour="t1", lead_time=2,
    state="eval/states/t2-jalgaon-s17.json",
    sut="t2-svc-jalgaon-nagpur", depot="jalgaon",
    claimants="t2-claimant-jalgaon-khandwa", escape_node="aurangabad",
    sut_posture="protected", half_a_cuts="claimants", half_b_cuts="claimants,sut",
    alt_span=["khandwa:dhar:out:in"],
    probe_flip=("A:restorable:t2-claimant-jalgaon-khandwa:restorable,"
                "B:restorable:t2-claimant-jalgaon-khandwa:not_restorable"),
)


def build_parser():
    return derive_t1._build_parser(defaults=T2_DEFAULTS)


def main() -> None:
    asyncio.run(derive_t1.derive(build_parser().parse_args()))


if __name__ == "__main__":
    main()
