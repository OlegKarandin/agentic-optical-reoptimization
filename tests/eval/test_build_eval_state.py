"""The loaded operating network must be reproducible: same seed, same
target_mean_util, same service count and mean utilization within tolerance
(eval design spec, "Testing"). This is a real two-stage build against the
143-node toy topology under the sibling conda env -- slow, and cached on disk
by the loaded_state_path fixture so the rest of the suite pays for it once."""
import asyncio
import json
import subprocess
from pathlib import Path

from storm_reoptimizer.geo_mapper import load_edges
from storm_reoptimizer.eval.runner import service_geometry
from storm_reoptimizer.mcp_client import connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)
TOY_INDIA_TOPOLOGY_PATH = TOPOLOGY_PATH
BUILDER = Path(__file__).parent.parent.parent / "tools" / "build_eval_state.py"


def test_loaded_state_has_the_service_under_test_and_a_real_background_load(
    loaded_state_path,
):
    doc = json.loads(loaded_state_path.read_text(encoding="utf-8"))
    service_ids = {s["id"] for s in doc["services"]}
    assert "storm-svc-1" in service_ids
    # Background load: enough services for spare contention, and enough
    # lit lightpaths for an ip_reroute candidate to exist at all.
    assert len(service_ids) >= 10
    assert len(doc["lightpaths"]) >= 10
    assert doc["meta"]["seed"] == 17
    assert 0.4 <= doc["meta"]["achieved_mean_util"] <= 0.8


def test_loaded_state_carries_the_t1_pins(loaded_state_path):
    """Task 15 (T1 spend-or-hold redesign): the redesigned T1 pair's own SUT
    and claimant corridor are stage-2 pins in `build_eval_state.T1_PINS`, so
    the shipped `eval/states/loaded-s17.json` must actually carry all three.
    Named here rather than read back from T1_PINS on purpose -- this is the
    check that a rebuild really placed them, not a restatement of the
    constant."""
    doc = json.loads(loaded_state_path.read_text(encoding="utf-8"))
    service_ids = {s["id"] for s in doc["services"]}
    assert {"t1-svc-jalgaon-indore",
            "t1-claimant-jalgaon-dhulia-fwd",
            "t1-claimant-jalgaon-dhulia-rev"} <= service_ids
    # The old SUT and the old (satna-homed) claimant family stay: D1/T2/T3
    # are still authored against them.
    assert {"storm-svc-1", "claimant-satna-jabalpur-fwd",
            "claimant-satna-jabalpur-rev"} <= service_ids


def test_jalgaon_hosts_the_t1_pair_without_a_mount_type_change():
    """The T1 site's own precondition, asserted rather than assumed (Task 15;
    same discipline as `test_satna_has_three_independent_aerial_directions`
    above).

    `jalgaon` was chosen because it needs NO topology edit: it already has
    four aerial neighbours and two buried ones, so the SUT's two legs
    (`khandwa` working, `buldhana` protection), the claimant corridor
    (`dhulia`) and the buried escape (`surat`/`aurangabad`) are four
    independent directions out of one site. `dhulia` in particular is the
    claimant far end because its OTHER link is BURIED (`dhulia <-> nasik`) --
    a storm filter never admits it, so the harness's post-cut restoration
    replay always has a route home. The first candidate tried, `khandwa`, has
    only aerial links (`dhar <-> khandwa`), the storm avoid disconnected it,
    and both gold rollouts tied at identical loss."""
    edges = load_edges(TOY_INDIA_TOPOLOGY_PATH)
    by_site = {}
    for edge in edges:
        for near, far in ((edge.src, edge.dst), (edge.dst, edge.src)):
            by_site.setdefault(near, {}).setdefault(edge.mount_type, set()).add(far)
    jalgaon = by_site["jalgaon"]
    assert jalgaon["aerial"] == {"akola", "buldhana", "dhulia", "khandwa"}
    assert jalgaon["buried"] == {"aurangabad", "surat"}
    assert by_site["dhulia"]["buried"] == {"nasik"}
    assert by_site["khandwa"].get("buried", set()) == set()


def test_build_is_reproducible_at_the_same_seed(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    first = json.loads(loaded_state_path.read_text(encoding="utf-8"))
    out = tmp_path / "rebuild-s17.json"
    proc = subprocess.run(
        [local_server_command[0], str(BUILDER),
         "--topology", str(TOPOLOGY_PATH), "--out", str(out), "--seed", "17"],
        env=local_server_env, capture_output=True, text=True, timeout=1800,
        check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    second = json.loads(out.read_text(encoding="utf-8"))

    assert [s["id"] for s in second["services"]] == [s["id"] for s in first["services"]]
    assert abs(second["meta"]["achieved_mean_util"]
               - first["meta"]["achieved_mean_util"]) < 0.02


def test_base_state_mode_pins_a_new_service_onto_an_existing_state(
        tmp_path, loaded_state_path, local_server_command, local_server_env):
    """Task 12: `--base-state` skips stages 1-2 entirely and pins only the
    given `--pin`(s) onto the already-built loaded-s17.json, so a later task
    can add a brand-new SUT/pair without paying for a full network rebuild.
    tirupati<->nellore is a real adjacent (single-hop) aerial span in the toy
    topology (confirmed against toy_india_topology.json's edge list), chosen
    so the pin has an obvious feasible placement and the test exercises the
    base-state mechanism rather than routing feasibility."""
    base = json.loads(loaded_state_path.read_text(encoding="utf-8"))
    base_ids = {s["id"] for s in base["services"]}
    out = tmp_path / "pinned.json"
    pin = json.dumps({"id": "probe-x", "src": "tirupati", "dst": "nellore",
                       "demand_gbps": 100.0, "protected": False})
    proc = subprocess.run(
        [local_server_command[0], str(BUILDER),
         "--topology", str(TOPOLOGY_PATH), "--out", str(out), "--seed", "17",
         "--base-state", str(loaded_state_path), "--pin", pin],
        env=local_server_env, capture_output=True, text=True, timeout=120,
        check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    doc = json.loads(out.read_text(encoding="utf-8"))
    service_ids = {s["id"] for s in doc["services"]}
    assert "probe-x" in service_ids
    assert base_ids <= service_ids


def test_satna_has_three_independent_aerial_directions():
    """The claimant family's precondition, asserted rather than assumed
    (exposure-and-depot design, §3.2 Option B).

    A satna-homed claimant needs an aerial corridor that shares no aerial span
    with storm-svc-1's working (satna<->rewa) or protection (satna<->jhansi)
    legs -- otherwise its exposure is perfectly correlated with the SUT's and
    the "competing claim" is the same claim counted twice. satna<->jabalpur is
    that third direction. Changing an edge's mount_type is safe and
    bit-identical in the rebuilt state because `mount_type` appears ZERO times
    in multilayer_optical_network: it is read only by this repo's geo_mapper,
    events/filters and the viewer, and affects no routing, QoT, spectrum or
    allocation."""
    edges = load_edges(TOY_INDIA_TOPOLOGY_PATH)
    aerial = {tuple(sorted((e.src, e.dst))) for e in edges
              if e.mount_type == "aerial"}
    satna_aerial = {pair for pair in aerial if "satna" in pair}
    assert satna_aerial == {("rewa", "satna"), ("jhansi", "satna"),
                            ("jabalpur", "satna")}
    assert len(aerial) == 73


# Exact working-path node lists, read off the CURRENT (pre-Task-7, pre-
# claimant-pin) eval/states/loaded-s17.json via service_geometry before the
# claimant pins were added -- a real before/after comparison, not a
# restatement of whatever the rebuild happens to produce.
_BACKGROUND_WORKING_PATHS = {
    "d0029": ["allahabad", "fatehpur", "kanpur"],
    "d0348": ["kanpur", "fatehpur", "allahabad"],
    "d0462": [
        "nagpur", "wardha", "chandrapur", "hyderabad", "raichur",
        "torangallu", "bangalore", "kolar", "tirupati", "nellore", "ongole",
        "visakhapatnam", "dhenkanal", "bhubaneshwar", "kharagpur", "kolkata",
        "raipur", "jabalpur", "satna", "jhansi", "gwalior", "agra",
        "mathura", "delhi",
    ],
    "d0212": [
        "delhi", "jaipur", "bhilwara", "ratlam", "ujjain", "indore",
        "jabalpur", "raipur", "bhandara", "nagpur",
    ],
    "d0363": [
        "kolkata", "kharagpur", "bhubaneshwar", "dhenkanal", "raipur",
        "jabalpur", "satna", "jhansi", "gwalior", "agra", "mathura", "delhi",
        "jaipur", "bhilwara", "ratlam", "ujjain", "dhar", "khandwa",
        "jalgaon", "dhulia", "nasik", "mumbai",
    ],
}


def test_the_rebuild_preserves_every_background_service_geometry(
        loaded_state_path, local_server_command, local_server_env):
    """Stage 1 (gravity load) runs BEFORE the stage-2 pins and is not
    re-routed by them, so adding claimant pins must not move any pre-existing
    service. Verified empirically before the rebuild: storm-svc-1's
    representative point came back as (24.855553333333333,
    81.32777666666667) -- bit-identical to the value T3a.yaml documents.

    These five background services are named because every gold rationale in
    the suite is arithmetic over them."""
    async def _geometry():
        async with connect_server(
                str(TOPOLOGY_PATH), server_command=local_server_command,
                env=local_server_env,
                extra_args=["--state", str(loaded_state_path)]) as client:
            return await service_geometry(client, TOPOLOGY_PATH)

    geometry = asyncio.run(_geometry())

    assert geometry.points["storm-svc-1"] == (24.855553333333333,
                                              81.32777666666667)
    for svc, expected_working in _BACKGROUND_WORKING_PATHS.items():
        assert svc in geometry.paths, svc
        assert geometry.paths[svc]["working"] == expected_working, svc


import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))
import build_eval_state  # noqa: E402


def test_the_t2_and_t3_pin_sets_are_the_specs():
    """Spec 4.1/4.2 (T2/T3 probe redesign): the pins, verbatim. T3's SUT is
    UNPROTECTED. T3's original design added survivor pins along dhulia's
    buried alternative so route_service would offer a zero-spare ip_reroute
    groom; Task 8 checked that live (twice, at two survivor demand sizes)
    and it was never offered -- solve_allocation_model never lit a dedicated
    lightpath on the direct buried spans at all, so no lightpath with spare
    capacity ever sat there. Per the spec's documented fallback (recorded in
    `docs/superpowers/plans/notes/2026-09-06-t2-t3-authoring.md`), T3 is now
    the two-corridor restorability variant: SUT + two claimants, NO
    survivor pins."""
    t2 = {p["id"]: p for p in build_eval_state.T2_PINS}
    assert t2["t2-svc-jalgaon-nagpur"] == {
        "id": "t2-svc-jalgaon-nagpur", "src": "jalgaon", "dst": "nagpur",
        "demand_gbps": 300.0, "protected": True}
    assert t2["t2-claimant-jalgaon-khandwa"]["demand_gbps"] == 200.0
    assert t2["t2-claimant-jalgaon-khandwa"]["protected"] is False
    t3 = {p["id"]: p for p in build_eval_state.T3_PINS}
    assert t3["t3-svc-jalgaon-nagpur"]["protected"] is False
    # khandwa + buldhana, NOT spec 4.2's khandwa + dhulia: the live solve
    # gives the unprotected SUT a working leg whose only near-depot aerial
    # span IS `jalgaon<->dhulia`, so a dhulia claimant would be perfectly
    # correlated with the SUT and its realized cut WOULD BE the SUT's own
    # first-hop cut (2026-09-06 plan, Task 11 -- see T3_PINS' own comment and
    # tools/derive_t3.py's docstring). buldhana is khandwa's structural twin:
    # exactly two links, both aerial, with a BURIED onward link past the far
    # end so the restoration replay can reach it when the alternative span is
    # outside the footprint.
    assert {t3["t3-claimant-jalgaon-khandwa"]["dst"],
            t3["t3-claimant-jalgaon-buldhana"]["dst"]} == {"khandwa",
                                                           "buldhana"}
    assert not hasattr(build_eval_state, "T3_SURVIVOR_PINS")
    assert {p["id"] for p in build_eval_state.T3_PINS} == {
        "t3-svc-jalgaon-nagpur", "t3-claimant-jalgaon-khandwa",
        "t3-claimant-jalgaon-buldhana"}
    assert build_eval_state.PIN_SETS == {
        "t2": build_eval_state.T2_PINS, "t3": build_eval_state.T3_PINS}
    # Pins solve in order: SUT, then claimants.
    ids = [p["id"] for p in build_eval_state.T3_PINS]
    assert ids.index("t3-svc-jalgaon-nagpur") < ids.index(
        "t3-claimant-jalgaon-buldhana")


def test_t2_and_t3_pins_contain_no_mate_pair():
    """Transponder-pairing spec (2026-09-21), §3.6: T2's and T3's own pins
    must not contain a mate pair (same unordered endpoint pair, opposite
    direction) -- after Task 1, a claimant restoring for free because its
    reverse mate is already lit would collapse the spare contention these
    pairs are built to test, exactly the failure the pre-2026-08-30 satna
    claimants had for a different reason (CLAUDE.md). Enforced here rather
    than merely believed."""
    for name, pins in (("T2", build_eval_state.T2_PINS),
                       ("T3", build_eval_state.T3_PINS)):
        directions: dict[tuple[str, str], tuple[str, str]] = {}
        for pin in pins:
            direction = (pin["src"], pin["dst"])
            pair = tuple(sorted(direction))
            prior = directions.get(pair)
            assert prior is None or prior == direction, (
                f"{name}: {pin['id']!r} ({direction}) is the reverse mate "
                f"of an earlier pin over the same endpoint pair {pair} "
                f"({prior}) -- the ledger's mate-pairing rule would make "
                f"its restoration free, collapsing the spare contention "
                f"this pair is built to test")
            directions.setdefault(pair, direction)


def test_the_pair_state_files_carry_their_pins_and_the_base(eval_state_paths):
    base = {s["id"] for s in json.loads(
        eval_state_paths["eval/states/loaded-s17.json"].read_text(encoding="utf-8"))["services"]}
    for state_file, pin_set in (("eval/states/t2-jalgaon-s17.json", "t2"),
                                ("eval/states/t3-jalgaon-s17.json", "t3")):
        doc = json.loads(eval_state_paths[state_file].read_text(encoding="utf-8"))
        ids = {s["id"] for s in doc["services"]}
        assert base <= ids
        assert {p["id"] for p in build_eval_state.PIN_SETS[pin_set]} <= ids
        assert doc["meta"]["pins"] == [p["id"] for p in build_eval_state.PIN_SETS[pin_set]]
