# tools/derive_t3.py
"""T3 -- the two-corridor restorability variant (spec 2026-09-06 §4.2,
FALLBACK design): derive_t1.derive with T3's defaults.

**Why the fallback, not the originally-drafted survivor-groom design.**
Task 8 (`docs/superpowers/plans/notes/2026-09-06-t2-t3-authoring.md`, "Step
2: T3 groom feasibility") checked live, against a real server, whether
`t3-claimant-jalgaon-dhulia` gets a zero-spare `ip_reroute` once its own
aerial corridor is avoided, grooming onto four single-hop "survivor"
lightpaths pinned along the buried `jalgaon-aurangabad-ahmednagar-nasik-
dhulia` alternative. It does not, at 100 G or 50 G survivor demand:
`solve_allocation_model` never lights a dedicated lightpath on any of the
four direct survivor spans, so no lightpath with spare capacity ever sits
on that path for the claimant to reuse -- confirmed by inspecting the built
state directly, not guessed. Per the design spec's own documented fallback
text, the survivor pins were dropped (`tools/build_eval_state.py`'s
`T3_PINS` is now SUT + two claimants only) and T3 became the two-corridor
restorability variant instead: half A's cone toward dhulia, half B's toward
khandwa (with `khandwa-dhar` IN the footprint there), and `probe_flip` kind
`restorable` (not the primary design's `spares_needed`) -- both corridors
are restorable at the SAME real cost (one spare pair each), so the flip is
which corridor's spare is actually held, not one being free.

Every derive_t1 flag still applies; the bearings and radii are the
authoring inputs (docs/superpowers/plans/notes/2026-09-06-t2-t3-authoring.md
records the ones used)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive_t1  # noqa: E402

T3_DEFAULTS = dict(
    pair="T3", width_km=60.0, damage_radius_km=50.0,
    hours="t0,t1,t2,t3,t4,t5,t6,t7", decision_hour="t1", lead_time=2,
    state="eval/states/t3-jalgaon-s17.json",
    sut="t3-svc-jalgaon-nagpur", depot="jalgaon",
    claimants="t3-claimant-jalgaon-khandwa,t3-claimant-jalgaon-dhulia",
    escape_node="surat", sut_posture="unprotected",
    half_a_cuts="claimant:t3-claimant-jalgaon-dhulia",
    half_b_cuts="claimant:t3-claimant-jalgaon-khandwa,sut",
    alt_span=["khandwa:dhar:out:in"],
    pcut_match_claimant="t3-claimant-jalgaon-dhulia:t3-claimant-jalgaon-khandwa",
    require_flip_tie=True,
    probe_flip=("A:restorable:t3-claimant-jalgaon-dhulia:restorable,"
                "B:restorable:t3-claimant-jalgaon-khandwa:not_restorable"),
)


def build_parser():
    return derive_t1._build_parser(defaults=T3_DEFAULTS)


def main() -> None:
    asyncio.run(derive_t1.derive(build_parser().parse_args()))


if __name__ == "__main__":
    main()
