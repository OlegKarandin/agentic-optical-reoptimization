# tools/derive_t3.py
"""T3 -- the two-corridor restorability variant (spec 2026-09-06 §4.2,
FALLBACK design): derive_t1.derive with T3's defaults.

**Why the fallback, not the originally-drafted survivor-groom design.**
Task 8 (`docs/superpowers/plans/notes/2026-09-06-t2-t3-authoring.md`, "Step
2: T3 groom feasibility") checked live, against a real server, whether a
claimant gets a zero-spare `ip_reroute` once its own aerial corridor is
avoided, grooming onto four single-hop "survivor" lightpaths pinned along
the buried `jalgaon-aurangabad-ahmednagar-nasik-dhulia` alternative. It does
not, at 100 G or 50 G survivor demand: `solve_allocation_model` never lights
a dedicated lightpath on any of the four direct survivor spans, so no
lightpath with spare capacity ever sits on that path for the claimant to
reuse -- confirmed by inspecting the built state directly, not guessed. Per
the design spec's own documented fallback text, the survivor pins were
dropped and T3 became the two-corridor restorability variant instead: each
half's cone points at a DIFFERENT claimant corridor, and `probe_flip` kind
`restorable` (not the primary design's `spares_needed`) carries the flip --
which corridor can be reached at all once the whole storm risk group is
avoided, not one being free.

**The corridors are `khandwa` and `buldhana`, not `khandwa` and `dhulia`
(2026-09-06, Task 11, found live).** Spec §4.2 assumed the unprotected SUT
`t3-svc-jalgaon-nagpur` leaves jalgaon via `buldhana` and therefore left
`dhulia` free for the second claimant. The live solve assigns it the other
way round -- the SAME reversal Task 10 found for T2's protected SUT -- so
the SUT's ONLY near-depot aerial span IS `jalgaon<->dhulia`. Consequences,
both measured rather than assumed:

  * `p_cut(SUT) == p_cut(a dhulia claimant)` to the last Sobol point at
    every cone near jalgaon: a dhulia claimant's exposure is perfectly
    correlated with the SUT's, the confound CLAUDE.md records for the old
    satna-homed claimant family.
  * `derive_t1._realized_for("claimant:<dhulia claimant>")` and
    `_realized_for("sut")` resolve to the SAME fibre, so "half A cuts the
    dhulia corridor and leaves S untouched" cannot be realized at all:
    cutting it cuts the SUT, both halves then grade `spend`, and
    `assert_gold_choices_differ` fails.

`buldhana` is structurally khandwa's twin (exactly two links,
`jalgaon<->buldhana` and `buldhana<->amravati`, both aerial;
`amravati<->nagpur` buried), so it carries the same restorable/
not-restorable flip. See `tools/build_eval_state.py`'s `T3_PINS`.

**Half A / half B, as shipped:**

  * half A (`gold_spare_action: conserve`) -- cone toward `khandwa` with
    `khandwa<->dhar` OUT of the damage footprint; realized cut is the
    khandwa corridor only, the SUT is untouched. The probe reads
    `restorable` for the khandwa claimant, so holding the spare buys its
    restoration and gold is HOLD.
  * half B (`gold_spare_action: spend`) -- cone toward `buldhana` with
    `buldhana<->amravati` IN the footprint; realized cuts are the buldhana
    corridor AND the SUT's own working first hop. The probe reads
    `not_restorable` for the buldhana claimant (both of its links are in
    the group), so holding the spare buys nothing and gold is SPEND.

`--pcut-match-claimant` solves half B's bearing so the BULDHANA corridor's
p_cut in B equals the KHANDWA corridor's p_cut in A **exactly**. That is
T3's own question: the two halves put an identically-sized claim in front of
the decider (`claimed_competing_ecar_gbps` is the same number in both), so
the observation carries no signal at all and only the probe separates them.

**`--require-flip-tie` is deliberately NOT set** (it was, before this pair
was first derived live). Four of the five `derived.FLIP_VARS` are the
NETWORK-WIDE claimant ECAR at the exposure horizon, summed over every
non-SUT service; two cones that differ at all -- and they must differ, or
the probe cannot flip -- move that sum by hundreds of Gbps, so a bit-
identical tie across all of FLIP_VARS is unreachable by construction. It is
also incompatible with `assertions.assert_flip_dominates`, which requires
the LARGEST inter-half FLIP_VARS difference to exceed the SUT's own
equal-in-both-halves ECAR: a full tie makes that difference exactly 0 and
fails. The tie T3 actually contributes to
`assert_no_global_policy_solves_the_suite` is on
`largest_restorable_group_ecar_gbps`, which the claimant p_cut match above
holds equal to the float.

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
    claimants="t3-claimant-jalgaon-khandwa,t3-claimant-jalgaon-buldhana",
    # `aurangabad`, jalgaon's OTHER buried spur, not T2's `surat`:
    # `assert_escape_route_survives` (suite pre-flight) is a HARD gate on
    # this field -- it requires an `optical_reroute` candidate whose new
    # lightpath's OMS route TOUCHES the named node, in the unconstrained
    # (`reference_avoid`) menu, and requires it to validate. Measured live
    # for `t3-svc-jalgaon-nagpur`: of the 14 candidates that menu offers,
    # none touches `surat`; candidates 2 and 4 route via `aurangabad`, and
    # the assertion (menu + `validate_plan`) passes on it in BOTH halves.
    # T1's own SUT does have a surat candidate, which is why `surat` works
    # there. NOTE for the controller: T2a/T2b declare `surat` and FAIL this
    # same assertion today (reproduced at HEAD, before this task's changes)
    # -- that is a pre-existing T2 defect, not one T3 introduces.
    escape_node="aurangabad", sut_posture="unprotected",
    half_a_cuts="claimant:t3-claimant-jalgaon-khandwa",
    half_b_cuts="claimant:t3-claimant-jalgaon-buldhana,sut",
    # khandwa<->dhar OUT in A is what makes the khandwa claimant restorable
    # there; buldhana<->amravati IN in B is what isolates the buldhana
    # claimant there. The other two containments are incidental to the flip
    # but asserted so a retuned cone cannot move them unnoticed.
    alt_span=["khandwa:dhar:out:out", "buldhana:amravati:in:in"],
    pcut_match_claimant=("t3-claimant-jalgaon-khandwa:"
                         "t3-claimant-jalgaon-buldhana"),
    probe_flip=("A:restorable:t3-claimant-jalgaon-khandwa:restorable,"
                "B:restorable:t3-claimant-jalgaon-buldhana:not_restorable"),
)


def build_parser():
    return derive_t1._build_parser(defaults=T3_DEFAULTS)


def main() -> None:
    asyncio.run(derive_t1.derive(build_parser().parse_args()))


if __name__ == "__main__":
    main()
