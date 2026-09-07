# tools/build_eval_state.py
"""Offline two-stage builder for the eval harness's loaded operating network
(eval design spec, "Loaded network"). NOT part of the storm_reoptimizer
package (lives outside src/ and tests/) and imports multilayer_optical_network
directly -- the one narrow, documented exception to CLAUDE.md's "no
server-internals imports" rule. Run under the multilayer-optical-mcp conda
env's python; never imported by this app's own runtime process.

Two stages, because neither alone gives what the eval needs:

  1. A population-weighted gravity load (generate_demands + solve_allocation_
     model, at a single empirically-calibrated DEMAND_SCALE_GBPS -- see that
     constant's docstring for why this bypasses build_operating_network()'s
     own scale search) lays down realistic background traffic at
     ~target_mean_util. This is what supplies the spectrum/transponder
     grooming contention and the existing lightpaths an ip_reroute candidate
     needs to groom onto. It does NOT supply protected-service spare
     contention: background demands are never protected here
     (protected_fraction=0.0 -- see its docstring for why), so
     PROTECTION_CONSTRAINTS currently binds nothing in this stage.
     storm-svc-1 (stage 2) is the only protected service the model itself
     carries; the harness's own SpareLedger (a separate, later task) is
     what synthesizes protected-service spare scarcity for scored episodes,
     independent of the network model's real reservations (eval design
     spec: "spare_inventory is not in the model"). A fraction of background
     demands going unplaced (for ordinary routing/capacity reasons, not a
     disjoint-pair search -- none runs when nothing is protected) is an
     expected, realistic outcome, not a build failure (only a total
     NO_SOLUTION is).
  2. storm-svc-1 (satna<->allahabad, 300G, protected) is then pinned onto the
     result via solve_allocation_model, preserving the verified
     both-halves-aerial ring property step 4 established. Gravity load alone
     would leave us hunting for a both-legs-aerial service and hoping one
     exists at that seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from statistics import mean

from multilayer_optical_network.gnpy_adapter.adapter import (
    recompute_qot_under_loading,
)
from multilayer_optical_network.model.allocation import (
    make_adapter_evaluator, solve_allocation_model,
)
from multilayer_optical_network.model.modes import default_modes
from multilayer_optical_network.model.qot_results import QoTCache, QoTResultStore
from multilayer_optical_network.model.scenario import (
    generate_demands, simulate_ip_routing,
)
from multilayer_optical_network.model.solvers import SolverStatus
from multilayer_optical_network.model.whatif import loading_from_model
from multilayer_optical_network.state_file import (
    dump_state, load_model_from_state_file, topology_fingerprint,
)
from multilayer_optical_network.topology_loader import load_model_from_topology_file

SERVICE_UNDER_TEST = {
    "id": "storm-svc-1", "src": "satna", "dst": "allahabad",
    "demand_gbps": 300.0, "protected": True,
}
# The satna-homed claimant family (exposure-and-depot design, §3.3). Both
# DIRECTIONS of one satna <-> X corridor per group, so a single new lightpath
# genuinely restores the whole group and the claimant aggregate the gold
# rationales bill is honestly what one transponder buys. Before this, every
# claimant the rationales named terminated somewhere else entirely -- a satna
# line card cannot restore kolkata <-> mumbai -- and the contention the eval
# scored was fictional.
#
# Destination and demand come from tools/probe_claimants.py's output; see
# docs/superpowers/plans/notes/2026-08-30-claimant-family.md for the five
# properties this corridor satisfies and the probe trail behind it (jabalpur
# is satna's only aerial direction whose span, ('jabalpur', 'satna'), is
# disjoint from storm-svc-1's own aerial legs, ('rewa', 'satna') and
# ('jhansi', 'satna')).
CLAIMANT_SERVICES = [
    {"id": "claimant-satna-jabalpur-fwd", "src": "satna", "dst": "jabalpur",
     "demand_gbps": 100.0, "protected": False},
    {"id": "claimant-satna-jabalpur-rev", "src": "jabalpur", "dst": "satna",
     "demand_gbps": 100.0, "protected": False},
]
# Generous, and deliberately local to the pins' endpoints: scarcity in this
# eval is the HARNESS ledger's concern (eval design spec, "spare_inventory is
# not in the model"), not the builder's. A build that failed to place a
# stage-2 pin for want of inventory would be a setup artifact. Widened to
# cover jabalpur alongside satna/allahabad for the claimant pair above, and
# (2026-09-05, Task 15) jalgaon/indore/dhulia for the T1 pins below.
PIN_SPARE_INVENTORY = {"satna": 4, "allahabad": 4, "jabalpur": 4,
                       "jalgaon": 4, "indore": 4, "dhulia": 4,
                       "nagpur": 4, "khandwa": 4, "aurangabad": 4,
                       "ahmednagar": 4, "nasik": 4}
# The REDESIGNED T1 pair's own service under test and claimant corridor
# (T1 spend-or-hold redesign spec 2026-09-05 §4.1-4.2; plan Task 15). The old
# T1 graded storm-svc-1, whose protection leg (satna<->jhansi) sat outside
# every T1 cone -- so "the storm threatens this service" was true only on a
# technicality and the spend/hold comparison tested nothing. This SUT is
# CLAUDE.md's canonical case for real: both legs leave jalgaon on AERIAL
# spans (working via khandwa, protection via buldhana -- confirmed live), so
# 1:1 switchover does not save it, and its only way out is a new lightpath
# over one of jalgaon's BURIED spurs (aurangabad/surat).
#
# Chosen by tools/find_sut.py (--prefilter shortlist, then live --evaluate
# runs against this very state); every acceptance number is recorded in
# docs/superpowers/plans/notes/2026-09-05-t1-authoring.md. NO mount-type
# change was needed: jalgaon already has four aerial neighbours (akola,
# buldhana, dhulia, khandwa) plus two buried ones (aurangabad, surat), so the
# claimant corridor below shares no aerial span with either SUT leg.
#
# `dhulia` is the claimant's far end for a reason the first candidate
# (jalgaon<->khandwa) failed on: the harness's post-cut restoration replay
# (replay.py) has to be able to REACH the far end after the corridor is cut
# AND the whole storm risk group is avoided, or "hold the spare for the
# claimant" buys nothing and the two halves tie at identical loss (measured:
# both rollouts 20100.0 Gbps-h, margin 0.000). khandwa's only other link is
# aerial (dhar<->khandwa) and sits inside the same cone, so the avoid set
# disconnects it; dhulia's is BURIED (dhulia<->nasik), which a storm filter
# can never admit, so the replay always has a route home.
#
# D1/T2/T3 still name storm-svc-1 and are untouched by these three extra
# stage-2 pins: stage 1 (the gravity load) runs before any pin and is not
# re-routed by them, and the pin list is solved in order, so storm-svc-1 and
# the satna claimants are placed exactly as before (asserted by
# test_the_rebuild_preserves_every_background_service_geometry).
T1_PINS: list[dict] = [
    {"id": "t1-svc-jalgaon-indore", "src": "jalgaon", "dst": "indore",
     "demand_gbps": 300.0, "protected": True},
    {"id": "t1-claimant-jalgaon-dhulia-fwd", "src": "jalgaon",
     "dst": "dhulia", "demand_gbps": 100.0, "protected": False},
    {"id": "t1-claimant-jalgaon-dhulia-rev", "src": "dhulia",
     "dst": "jalgaon", "demand_gbps": 100.0, "protected": False},
]
# The T2/T3 probe redesign's pins (spec 2026-09-06, §4.1/§4.2; plan Task 7).
# Each pair gets its OWN state file built from loaded-s17.json with
# --base-state, so T1's state, menus, gold and frozen scalars are untouched
# and T3's survivor lightpaths never appear in T2's or T1's menus as free
# grooms. jalgaon still needs no mount-type change: both SUTs' first hops
# (buldhana/dhulia, read back live in the authoring note) and the claimant
# corridors (khandwa, dhulia) are aerial, the escape (surat/aurangabad) is
# buried.
T2_PINS: list[dict] = [
    {"id": "t2-svc-jalgaon-nagpur", "src": "jalgaon", "dst": "nagpur",
     "demand_gbps": 300.0, "protected": True},
    {"id": "t2-claimant-jalgaon-khandwa", "src": "jalgaon", "dst": "khandwa",
     "demand_gbps": 200.0, "protected": False},
]
# T3's original design (spec 4.2) added single-hop unprotected "survivor"
# services along dhulia's BURIED alternative path (jalgaon-aurangabad,
# aurangabad-ahmednagar, ahmednagar-nasik, nasik-dhulia), sized to leave
# headroom, so route_service would offer the dhulia claimant a zero-spare
# ip_reroute groomed onto them. Checked live 2026-09-06 (plan Task 8) via
# tools/probe_restorability.py against this pin set at both 100 G (as
# originally committed) and 50 G survivor demand: NOT offered either time,
# and not a headroom problem -- `solve_allocation_model` never lit a
# dedicated lightpath on the direct jalgaon-aurangabad (etc.) spans at all,
# instead grooming each survivor demand onto FOUR hops of unrelated
# pre-existing IP links, so no lightpath with spare capacity ever sat on
# the buried path for the claimant to reuse.
# route_service's menu for t3-claimant-jalgaon-dhulia under a group
# containing its own corridor came back `optical_reroute`-only both times
# (min_spares_needed_by_site {"jalgaon": 1, "dhulia": 1}, never {}). Per
# spec 4.2's own documented fallback, the survivor pins are dropped and T3
# is the two-corridor restorability variant instead (SUT + two claimants
# only; see the spec for the corresponding cone/footprint change).
#
# Order matters: SUT, then claimants. Pins solve in order.
T3_PINS: list[dict] = [
    {"id": "t3-svc-jalgaon-nagpur", "src": "jalgaon", "dst": "nagpur",
     "demand_gbps": 300.0, "protected": False},
    {"id": "t3-claimant-jalgaon-khandwa", "src": "jalgaon", "dst": "khandwa",
     "demand_gbps": 200.0, "protected": False},
    {"id": "t3-claimant-jalgaon-dhulia", "src": "jalgaon", "dst": "dhulia",
     "demand_gbps": 200.0, "protected": False},
]
PIN_SETS: dict[str, list[dict]] = {"t2": T2_PINS, "t3": T3_PINS}
# NOT "srlg": the toy topology has zero static SRLGs (confirmed:
# toy_india_topology.json's "srlgs" is 0), and generate_demands'/
# solve_allocation_model's own docs say srlg-basis disjointness is a NO-OP
# when no SRLGs are defined. Passing it here silently let background
# "protected" demands' working/protection legs share physical links --
# confirmed empirically: with basis=srlg, validating even a completely EMPTY
# plan against basis=physical/level=link on the resulting state reported 161
# pre-existing disjointness violations across background services (found
# while investigating why Task 5's route_service candidates for storm-svc-1
# all failed validation -- they didn't cause the violations, the state
# already had them). storm-svc-1's own pin below never receives this
# constant at all (it's a hand-built dict with no "constraints" key), so its
# disjoint-pair search already falls back to basis=physical/level=link via
# solve_allocation_model's own default -- this constant now matches that,
# making the whole network consistent under one real, enforced basis instead
# of one real (storm-svc-1) and one vacuous (everything else).
PROTECTION_CONSTRAINTS = {"basis": "physical", "level": "link"}
# Currently dead for the background load specifically: at
# protected_fraction=0.0 (build()'s default -- see its docstring) no
# background demand is ever marked protected, so this never gets attached
# to one. Kept, not deleted, for interface stability (the Task 1 brief's
# own "Produces" contract names this constant) and because it takes effect
# again the moment protected_fraction is raised above 0.

# generate_demands' default node_mass is graph degree, which on this sparse
# 143-node/~190-edge toy topology puts most weight on a handful of very
# SHORT, adjacent city pairs (e.g. delhi<->sonipat 44km, dhar<->indore 56km)
# because gravity weight is mass[u]*mass[v]/distance_km. Those pairs each
# accumulate several quantized 100Gbps demand units onto one dedicated IP
# link and saturate it (max_util -> 0.875-1.0) while the network-wide MEAN
# stays around 0.15-0.30 -- far below target_mean_util=0.6 -- because most of
# the ~20k possible pairs never cross the quantization threshold at all.
# Verified empirically (storm-reoptimizer eval-harness Task 1 investigation,
# 2026-08-21): raising max_util_cap doesn't help (the same plateau recurs),
# and neither does a finer unit_gbps or a flatter alpha. Real population is
# the correct mass proxy for a traffic gravity model (this is the standard
# formulation in the TE literature) and empirically fixes it: at this
# topology/seed, population-weighted mass reaches mean_util 0.50 (scale
# 102400) to 0.68 (scale 204800), squarely inside build()'s [0.4, 0.8]
# target band, instead of plateauing at ~0.17-0.30.
#
# 2011 Census of India town/city-proper population figures (not urban
# agglomeration -- kept to one consistent methodology across all 143 nodes),
# compiled from Wikipedia's "List of cities in India by population" for the
# major metros and citypopulation.de / census2011.co.in for the smaller
# district towns. A few names in this toy topology are old/alternate
# spellings resolved by nearest lat/lon match to the real place: "hadiagarh"
# -> Haidergarh (Barabanki, UP); "callicut"/"kozhikode" are the same city
# (Kozhikode, Kerala); "goa" is the state total, "panjim" the city of Panaji
# proper (the two share identical topology coordinates); "talwandi_bahi" and
# "torangallu" (Toranagallu, an industrial township with no reliable recent
# census count) are order-of-magnitude estimates, not sourced counts.
NODE_POPULATION = {
    "varanasi": 1198491, "udaipur": 451100, "hadiagarh": 17200,
    "sitapur": 177234, "dehradun": 569578, "lucknow": 2817105,
    "kanpur": 2765348, "fatehpur": 193193, "jaunpur": 180362,
    "allahabad": 1112544, "patna": 1684222, "bokaro": 414820,
    "hazaribagh": 142489, "gaya": 468614, "kolkata": 4496694,
    "satna": 282977, "dhanbad": 1162472, "asansol": 563917,
    "rewa": 235654, "jhansi": 505693, "torangallu": 18000,
    "bellary": 410445, "goa": 1458545, "hubli": 943788,
    "solapur": 951558, "belgaum": 488157, "raichur": 234073,
    "gulbarga": 533587, "chitradurg": 180282, "panjim": 70991,
    "ranchi": 1073427, "jamshedpur": 631364, "coimbatore": 1050721,
    "thirussur": 315957, "rourkela": 272721, "kharagpur": 207604,
    "cannonore": 330832, "mangalore": 488968, "palghat": 288259,
    "kozhikode": 431560, "rohtak": 374292, "gurgaon": 876969,
    "bhatinda": 285788, "kot_kapura": 91979, "noida": 637272,
    "meerut": 1305429, "delhi": 11034555, "sonipat": 289333,
    "moradabad": 887871, "bareilly": 903668, "chennai": 6748026,
    "kanchipuram": 216091, "bangalore": 8443675, "kolar": 141959,
    "hassan": 207694, "mysore": 893062, "ongole": 213526,
    "vijayavada": 1034358, "tirupati": 287482, "nellore": 499575,
    "raipur": 1010433, "bhandara": 91845, "nagpur": 2405665,
    "wardha": 106444, "amravati": 647057, "buldhana": 67431,
    "akola": 425817, "khandwa": 200738, "bhubaneshwar": 843402,
    "dhenkanal": 67414, "jabalpur": 1055525, "damoh": 139561,
    "sagar": 274556, "callicut": 431560, "nanded": 550439,
    "ahmednagar": 350859, "pune": 3124458, "satara": 120195,
    "visakhapatnam": 1728128, "chandrapur": 320379, "hyderabad": 6993262,
    "sangareddy": 72344, "chandigarh": 961587, "kolhapur": 549236,
    "sangli": 502793, "ambala": 195153, "dhar": 93917,
    "ujjain": 515215, "gandhinagar": 292797, "himmatnagar": 81137,
    "ahmedabad": 5577940, "anand": 209410, "bhopal": 1798218,
    "ratlam": 264914, "indore": 1964086, "vidisha": 155951,
    "aurangabad": 1175116, "jalgaon": 460228, "nasik": 1486053,
    "dhulia": 375559, "valsad": 139764, "mumbai": 12442373,
    "bharuch": 169007, "surat": 4467797, "godhra": 143644,
    "baroda": 1670806, "karnal": 302140, "talwandi_bahi": 17285,
    "kollam": 367107, "ernakulam": 602046, "thiruvalla": 52883,
    "kottayem": 55374, "tirunelveli": 473637, "tiruchendur": 32171,
    "kanyakumari": 22453, "trivandrum": 743691, "allepey": 240991,
    "gwalior": 1054420, "rajgarh": 29726, "ajmer": 542321,
    "agra": 1585704, "mathura": 349909, "ghaziabad": 1648643,
    "kota": 1001694, "bhilwara": 359483, "tonk": 165294,
    "jaipur": 3046163, "sivakasi": 71040, "trichy": 847387,
    "tirupur": 877778, "erode": 360604, "salem": 829267,
    "palladam": 42225, "chidambaram": 62153, "pondicherry": 244377,
    "pathankot": 156306, "hoshiarpur": 168653, "amritsar": 1132383,
    "jalandhar": 862886, "ludhiana": 1618879, "patiala": 406192,
    "ramanathapuram": 61440, "madural": 1017865,
}

# build_operating_network()'s own exponential-climb/bisection search assumes
# achieved utilization grows monotonically with offered scale; on this
# topology it doesn't (route/grooming choices shift non-smoothly as scale
# grows), so the search latches onto whichever scale it first samples as
# infeasible and bisects into a narrow band near it -- landing at
# achieved_mean_util ~0.31 regardless of max_util_cap (0.95 or 0.98), well
# under the [0.4, 0.8] target band. Bypassing the search and generating
# demand at a single, empirically-calibrated scale sidesteps this: verified
# directly against generate_demands+solve_allocation_model (storm-reoptimizer
# eval-harness Task 1 investigation, 2026-08-21), scale 102400 Gbps gives
# achieved_mean_util=0.526 at protected_fraction=0.0 (see its docstring) --
# comfortably inside the target band with headroom on both sides. A small
# fraction of demands can still go unplaced at this scale for ordinary
# routing/capacity reasons (no disjoint-pair search runs when nothing is
# protected); that's expected under real load, not a build failure.
DEMAND_SCALE_GBPS = 102400.0


def _default_pin_inventory(pins: list[dict]) -> dict:
    """PIN_SPARE_INVENTORY, widened with 4 at every pin's endpoints not
    already covered -- generous and local to the pins' own sites, per
    PIN_SPARE_INVENTORY's own docstring (scarcity is the harness ledger's
    concern, not the builder's)."""
    inventory = dict(PIN_SPARE_INVENTORY)
    for pin in pins:
        inventory.setdefault(pin["src"], 4)
        inventory.setdefault(pin["dst"], 4)
    return inventory


def _pin_and_dump(model, store: QoTResultStore, pins: list[dict],
                   pin_inventory: dict, topology_path: str, out_path: str,
                   meta: dict, err_context: str, log_note: str) -> dict:
    """The one shared pin/solve/fatal-check/dump/write/print tail for BOTH
    the default two-stage build and --base-state mode -- kept as a single
    copy so a later change to solve_allocation_model's call shape or the
    fatal-check condition can't silently drift between the two call sites.
    `err_context` names what's being pinned onto, for the fatal message;
    `log_note` is appended verbatim to the "wrote <out>: N services" line.
    Returns the written doc."""
    pin_qot = make_adapter_evaluator(model, store, cache=QoTCache())
    pin_result, work = solve_allocation_model(model, pin_qot, pins, pin_inventory)
    if pin_result.status is not SolverStatus.SOLUTION or pin_result.unplaced:
        raise SystemExit(
            f"build_eval_state: could not pin all of "
            f"{[p['id'] for p in pins]} onto {err_context} "
            f"(status={pin_result.status}, unplaced={pin_result.unplaced})")

    raw = json.loads(Path(topology_path).read_text(encoding="utf-8-sig"))
    doc = dump_state(work, fingerprint=topology_fingerprint(raw), meta=meta)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {out_path}: {len(doc['services'])} services{log_note}",
          flush=True)
    return doc


def build(topology_path: str, out_path: str, *, seed: int,
          target_mean_util: float = 0.6, max_util_cap: float = 0.95,
          protected_fraction: float = 0.0,
          scale_gbps: float = DEMAND_SCALE_GBPS,
          pins: list[dict] | None = None,
          pin_inventory: dict | None = None,
          base_state: str | None = None) -> None:
    """`target_mean_util` no longer calibrates the offered scale (that's now
    the fixed, pre-calibrated `scale_gbps` -- see DEMAND_SCALE_GBPS's
    docstring for why); it only sets the +/-0.2 acceptance tolerance below.
    `max_util_cap` is advisory, not enforced: at the calibrated scale,
    achieved_max_util is 1.0 (one IP-layer candidate link fully subscribed)
    -- that's a normal outcome of a loaded network, not an invalid one, so
    exceeding the cap prints a warning instead of raising.

    `protected_fraction` lowered from the plan's illustrative 0.3 to 0.0:
    background demands are never protected here. storm-svc-1 (below) is
    still protected -- this only concerns the population-weighted gravity
    load's own demands.

    Reason: with any nonzero protected_fraction tried (0.3, 0.15, 0.1, 0.05,
    0.02), some background protected demands' working paths ended up
    sharing spare/protection capacity on a common candidate link whose
    combined worst-case reserved need exceeded that link's real capacity --
    a real "protection_oversubscribed" condition surfaced by validate_plan,
    not a translation bug (found while investigating why every route_service
    candidate for storm-svc-1 failed validation; a completely EMPTY plan
    reported the same violations, proving they were pre-existing in the
    network, not caused by storm-svc-1's candidate). validate_plan reports
    violations for the WHOLE network on every call -- it is not scoped to
    the service being validated -- so ANY pre-existing violation anywhere
    blocks every later validate_plan call regardless of what's being
    checked. Verified empirically (2026-08-21) at scale_gbps=102400:
    protected_fraction 0.3/0.15/0.1/0.05/0.02 left 161/26/8/4/2 pre-existing
    violations respectively (all traced to the same few highest-gravity-
    weight background pairs, which stayed in the protected set at every
    fraction tried down to 0.02); only 0.0 reaches zero. Confirmed end to
    end at 0.0: an empty plan validates clean (0 violations), storm-svc-1's
    first route_service candidate validates and commits successfully, and
    the next hour's menu changes as a result -- exactly what Task 5's test
    requires. mean_util stays in the target band throughout this range
    (0.50-0.53), since the background load's overall volume doesn't depend
    on whether its demands are flagged protected.

    What this trades away: at protected_fraction=0.0, PROTECTION_CONSTRAINTS
    binds nothing here (no background demand is ever marked protected, so
    no disjoint-pair search ever runs for one) and the network model itself
    holds no protected-service spare RESERVATION other than storm-svc-1's
    own -- there is no other protected service in the model for storm-svc-1
    to contend with over spares. Unprotected background demands still
    consume spectrum/transponders and still create grooming contention (an
    ip_reroute candidate still has real lightpaths to groom onto), but not
    spare-reservation contention. If a later task needs the network MODEL
    itself (not the harness's SpareLedger) to show multiple protected
    services genuinely contending for spares, this ruling would need
    revisiting; per the eval design spec ("spare_inventory is not in the
    model"), scored spare scarcity is synthesized by the harness's own
    ledger, decoupled from the network model's real reservations, so no
    task in the current plan appears to need this.

    `pins` defaults to SERVICE_UNDER_TEST + CLAIMANT_SERVICES + T1_PINS (the
    original hardcoded stage-2 pin set, unchanged); passing an explicit list
    replaces it wholesale rather than extending it. `pin_inventory` defaults
    to PIN_SPARE_INVENTORY widened with 4 at every pin endpoint not already
    covered (see _default_pin_inventory).

    `base_state`, when given, skips stages 1 and 2 (the gravity-load build)
    entirely: it loads `base_state` as the starting model via
    `load_model_from_state_file` (same pattern as
    `tools/probe_claimants.py`) and pins only `pins` onto it as a new stage,
    reusing the identical stage-2 tail (`_pin_and_dump`, also used by the
    default two-stage build below). This is what makes authoring a new T1
    SUT/pair cheap: pin it onto the already-built loaded-s17.json instead
    of paying for a full rebuild. `seed`, `target_mean_util`, `max_util_cap`,
    `protected_fraction` and `scale_gbps` are unused in this mode.
    `--base-state` is designed to always pair with at least one `--pin`:
    used alone it falls through to the default pin set
    (SERVICE_UNDER_TEST + CLAIMANT_SERVICES + T1_PINS), which would likely
    collide with services the base state already contains (storm-svc-1 and
    the claimant pair are already pinned into loaded-s17.json)."""
    if pins is None:
        pins = [SERVICE_UNDER_TEST, *CLAIMANT_SERVICES, *T1_PINS]
    if pin_inventory is None:
        pin_inventory = _default_pin_inventory(pins)
    modes = default_modes()

    if base_state is not None:
        model = load_model_from_state_file(topology_path, base_state, modes=modes)
        store = QoTResultStore()
        base_meta = json.loads(
            Path(base_state).read_text(encoding="utf-8-sig"))["meta"]
        _pin_and_dump(
            model, store, pins, pin_inventory, topology_path, out_path,
            meta={**base_meta, "pins": [p["id"] for p in pins]},
            err_context=f"base_state={base_state}",
            log_note=(f" (base-state mode, base={base_state}, "
                       f"pins={[p['id'] for p in pins]})"))
        return

    model = load_model_from_topology_file(topology_path, modes=modes)
    store = QoTResultStore()
    qot = make_adapter_evaluator(model, store, cache=QoTCache())
    routers = list(model.list_routers())

    node_mass = {r.site: float(NODE_POPULATION[r.site]) for r in routers}
    demands = generate_demands(
        model, seed=seed, scale=scale_gbps, protected_fraction=protected_fraction,
        node_mass=node_mass, protection_constraints=PROTECTION_CONSTRAINTS)
    spare_inventory = {r.site: 10 ** 6 for r in routers}
    result, loaded = solve_allocation_model(model, qot, demands, spare_inventory)
    if result.status is SolverStatus.NO_SOLUTION:
        raise SystemExit(
            f"build_eval_state: operating-network build placed nothing "
            f"(0 of {len(demands)} demands)")
    if result.unplaced:
        print(f"build_eval_state: {len(result.unplaced)} of {len(demands)} "
              f"background demands went unplaced (no feasible route or "
              f"insufficient transponders under load -- at "
              f"protected_fraction=0.0 no demand here ever enters the "
              f"disjoint-pair search) -- expected under real load, not a "
              f"build failure", flush=True)
    recompute_qot_under_loading(
        model=loaded, store=store, loading=loading_from_model(loaded))

    utils = [u.utilization for u in simulate_ip_routing(loaded).utilizations
             if u.utilization is not None]
    achieved_mean_util = mean(utils) if utils else 0.0
    achieved_max_util = max(utils) if utils else 0.0
    if not (target_mean_util - 0.2 <= achieved_mean_util <= target_mean_util + 0.2):
        raise SystemExit(
            f"build_eval_state: achieved_mean_util={achieved_mean_util:.3f} is "
            f"far from target_mean_util={target_mean_util}; re-calibrate "
            f"scale_gbps (see DEMAND_SCALE_GBPS's docstring)")
    if achieved_max_util > max_util_cap:
        print(f"build_eval_state: WARNING achieved_max_util="
              f"{achieved_max_util:.3f} exceeds max_util_cap={max_util_cap} "
              f"-- one or more IP-layer candidate links are fully "
              f"subscribed; not treated as fatal (see build()'s docstring)",
              flush=True)

    _pin_and_dump(
        loaded, store, pins, pin_inventory, topology_path, out_path,
        meta={
            "note": "storm-reoptimizer eval loaded operating network",
            "seed": seed,
            "target_mean_util": target_mean_util,
            "achieved_mean_util": achieved_mean_util,
            "achieved_max_util": achieved_max_util,
            "n_background_demands": len(demands),
            "service_under_test": SERVICE_UNDER_TEST["id"],
        },
        err_context="the loaded network",
        log_note=f", mean_util={achieved_mean_util:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="build_eval_state")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--target-mean-util", type=float, default=0.6,
        help="Acceptance tolerance center (+/-0.2) for the build's achieved "
             "mean utilization. Does NOT calibrate offered traffic scale -- "
             "that's the fixed DEMAND_SCALE_GBPS; see its docstring.")
    parser.add_argument(
        "--max-util-cap", type=float, default=0.95,
        help="Advisory only: prints a warning (not fatal) if the achieved "
             "max utilization exceeds this.")
    parser.add_argument(
        "--pin", action="append", default=None, metavar="JSON",
        help='One pin, as JSON: {"id","src","dst","demand_gbps","protected"}. '
             "Repeatable. Replaces the default pin set (SERVICE_UNDER_TEST + "
             "CLAIMANT_SERVICES + T1_PINS) wholesale, rather than extending "
             "it.")
    parser.add_argument(
        "--pin-set", default=None, choices=sorted(PIN_SETS),
        help="A named pin set (tools/build_eval_state.PIN_SETS) to pin, "
             "e.g. `t2` or `t3`. Combined with any --pin given (named set "
             "first). Meant for --base-state mode.")
    parser.add_argument(
        "--pin-inventory", default=None, metavar="JSON",
        help="Spare inventory for the pin stage, as a JSON object of "
             "site -> count. Defaults to PIN_SPARE_INVENTORY widened with 4 "
             "at every pin endpoint not already covered.")
    parser.add_argument(
        "--base-state", default=None, metavar="PATH",
        help="Skip stages 1-2 (the gravity-load build) and instead load "
             "this existing state file as the starting model, pinning only "
             "--pin(s) onto it as a new stage. Makes adding a new SUT/pair "
             "cheap: no full rebuild needed.")
    args = parser.parse_args()
    pins = [*PIN_SETS[args.pin_set]] if args.pin_set else []
    pins += [json.loads(p) for p in args.pin or []]
    pins = pins or None
    pin_inventory = (json.loads(args.pin_inventory)
                      if args.pin_inventory is not None else None)
    build(args.topology, args.out, seed=args.seed,
          target_mean_util=args.target_mean_util,
          max_util_cap=args.max_util_cap,
          pins=pins, pin_inventory=pin_inventory, base_state=args.base_state)


if __name__ == "__main__":
    main()
