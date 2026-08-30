"""Rollout mechanics against a real server (eval design spec, "Testing").
Uses a throwaway two-hour scenario written into tmp_path -- the seven real
episodes come later and must not be needed to prove the loop works."""
import asyncio
import dataclasses
import json
import textwrap
from pathlib import Path

import pytest

from storm_reoptimizer.eval import runner
from storm_reoptimizer.eval.baseline import ForecastBlindBaseline, ScriptedDecider
from storm_reoptimizer.eval.decisions import (
    ConstraintDecision,
    ObjectiveDecision,
    TimingDecision,
)
from storm_reoptimizer.eval.observation import Observation
from storm_reoptimizer.eval.runner import (
    MAX_ITERATIONS,
    Action,
    action_payloads,
    run_episode,
    service_geometry,
    unconstrained_menu_projection,
)
from storm_reoptimizer.eval.scenario_file import ConeAtHorizon, Issuance, load_scenario
from storm_reoptimizer.mcp_client import connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)

GEOMETRY_TOPOLOGY = {"graph": {"nodes": [
    {"id": "satna", "lat": 24.6, "lon": 80.8},
    {"id": "rewa", "lat": 24.5, "lon": 81.3},
    {"id": "allahabad", "lat": 25.4, "lon": 81.8},
    {"id": "jhansi", "lat": 25.4, "lon": 78.6},
], "edges": []}, "srlgs": []}

GEOMETRY_REPLIES = {
    ("get_topology", "ip"): {"ip_links": [
        {"id": "l_sr", "lightpath_id": "lp_sr"},
        {"id": "l_ra", "lightpath_id": "lp_ra"},
        {"id": "l_sj", "lightpath_id": "lp_sj"}]},
    ("get_topology", "optical"): {"oms": [
        {"id": "oms_sr", "src_node_id": "satna", "dst_node_id": "rewa",
         "elements": ["fiber_1"]},
        {"id": "oms_ra", "src_node_id": "rewa", "dst_node_id": "allahabad",
         "elements": ["fiber_2"]},
        {"id": "oms_sj", "src_node_id": "satna", "dst_node_id": "jhansi",
         "elements": ["fiber_3"]},
        {"id": "oms_gap", "src_node_id": "satna", "dst_node_id": "nowhere",
         "elements": ["fiber_4"]}]},
    ("get_lightpaths", None): [
        {"id": "lp_sr", "oms_sequence": ["oms_sr"]},
        {"id": "lp_ra", "oms_sequence": ["oms_ra"]},
        {"id": "lp_sj", "oms_sequence": ["oms_sj", "oms_gap"]}],
    ("get_services", None): {"services": [
        {"id": "storm-svc-1", "demand_gbps": 300.0,
         "src_router": "router_satna", "dst_router": "router_allahabad",
         "working_path": ["l_sr", "l_ra"], "protection_path": ["l_sj"]}]},
}


def _geometry_call(counter):
    async def call(name, arguments=None, **kw):
        counter.append(name)
        layer = (arguments or {}).get("layer")
        return GEOMETRY_REPLIES[(name, layer)]
    return call


def _write_geometry_topology(tmp_path):
    path = tmp_path / "topo.json"
    path.write_text(json.dumps(GEOMETRY_TOPOLOGY), encoding="utf-8")
    return path

PROBE_MENU = {
    "status": "solution",
    "candidates": [
        {"lever": "ip_reroute", "reused_lightpaths": ["lp-x"],
         "new_lightpaths": [], "restored_gbps": 300.0, "shortfall_gbps": 0.0,
         "cost_vector": {"dropped_traffic": 0.0, "transponders": 38.0,
                         "services_at_risk": 0, "total_margin": 5.0,
                         "added_latency": 4.0, "spectrum_used": 18,
                         "max_util": 0.8}},
        {"lever": "optical_reroute", "reused_lightpaths": [],
         "new_lightpaths": [{"oms_sequence": ["oms_1"], "lam": 0,
                             "mode_id": "300G@4.8dB", "gsnr_db": 12.0,
                             "bitrate_gbps": 300.0}],
         "restored_gbps": 300.0, "shortfall_gbps": 0.0,
         "cost_vector": {"dropped_traffic": 0.0, "transponders": 40.0,
                         "services_at_risk": 0, "total_margin": 5.0,
                         "added_latency": 9.0, "spectrum_used": 20,
                         "max_util": 0.7}},
    ],
    "pairs": [],
}

# oms_1's two endpoint SITES, for spares_needed to resolve PROBE_MENU's one
# real new lightpath against.
PROBE_OMS_NODES = {"oms_1": ["satna", "rewa"]}


def test_the_probe_projection_shows_what_exists_and_what_it_costs_in_spares():
    projected = unconstrained_menu_projection(PROBE_MENU, PROBE_OMS_NODES)
    assert projected["status"] == "solution"
    assert projected["candidates"] == [
        {"candidate_label": "candidate_0", "lever": "ip_reroute",
         "spares_needed": {}},
        {"candidate_label": "candidate_1", "lever": "optical_reroute",
         "spares_needed": {"satna": 1, "rewa": 1}},
    ]


def test_the_probe_projection_withholds_the_cost_vector():
    # F3's fix must not collapse decisions 2 and 3 into one. Weighing the
    # cost vector IS decision 3's job; showing it at decision 2 would make
    # the objective step a rubber stamp (remediation spec, W3.3).
    projected = unconstrained_menu_projection(PROBE_MENU, PROBE_OMS_NODES)
    for candidate in projected["candidates"]:
        assert "cost_vector" not in candidate
        assert "new_lightpaths" not in candidate
        assert "restored_gbps" not in candidate


def test_a_menu_with_no_solution_projects_to_an_empty_candidate_list():
    projected = unconstrained_menu_projection({"status": "NO_SOLUTION"})
    assert projected == {"status": "NO_SOLUTION", "candidates": []}


# A minimal, deliberately non-discriminating episode: two hours, one issuance,
# one realized cut. Its job is to exercise the loop, not to score anyone.
#
# `center`/`width_km` (what build_observation's p_cut arithmetic reads) sit
# ON storm-svc-1's real working aerial span (satna<->rewa), so its exposure
# is genuinely nonzero under the region model -- Task 3 of the exposure-and-
# depot plan scores exposure off the real span, not an averaged midpoint, so
# a cone that no longer touches the real corridor at all (the old, pre-Task-3
# geometry below) reads zero exposure and the baseline never acts.
#
# `cone` (the GeoJSON polygon `_define_horizon_risk_groups` intersects
# against the LOCAL topology's edges to build the avoid set) is DELIBERATELY
# a different, disjoint box, near allahabad, that intersects no aerial edge
# at all -- confirmed live. Avoiding satna-rewa for real is not a free
# no-op: it is a shared backbone trunk carrying hundreds of other services,
# and route_service's cheapest alternative for storm-svc-1 then runs
# straight through protection's own corridor (satna<->jhansi<->allahabad),
# which is exactly the disjointness_collapse `_WidensOnDisjointnessRejection`
# exists to recover from -- not something a plain ForecastBlindBaseline
# ever will.
#
# `ConeAtHorizon` never cross-checks its `cone` polygon against its
# `center`/`width_km` (scenario_file.py), so this split is legal -- but it is
# a NEW incoherence, not an inherited one: the OLD, pre-Task-3 fixture's
# `center`/`width_km` (25.1, 81.8 / 120) was itself roughly the centroid and
# extent of ITS OWN polygon (the same box below, near allahabad) -- the two
# fields agreed, and its "empty risk group" property fell out of the polygon
# alone sitting off every aerial edge. Task 3 moved exposure scoring onto the
# real span, which is nowhere near that polygon, so THIS fixture is the first
# one in the suite to deliberately point `center`/`width_km` at a location
# its own `cone` polygon doesn't cover. That's a real, new fiction, kept
# because the alternative -- a `cone` that actually reaches the real span --
# makes the avoid set non-empty and forces the genuine disjointness_collapse
# above, which is exactly what SMOKE exists NOT to exercise (that's
# EXPOSURE_SMOKE's job, paired with `_WidensOnDisjointnessRejection`, below).
SMOKE = textwrap.dedent("""
    id: SMOKE
    seed: 17
    state_file: eval/states/loaded-s17.json
    service_under_test: storm-svc-1
    track: hudhud
    hours: [t0, t1]
    decision_hour: t0
    lead_time_hours: 1
    spares_on_hand: 2
    depot_site: satna
    spare_inventory: {satna: 2}
    damage_radius_km: 74
    reference_avoid: {}
    forecast:
      t0:
        t1: {cone: {type: Polygon, coordinates: [[[81.3, 24.8], [82.3, 24.8], [82.3, 25.4], [81.3, 25.4], [81.3, 24.8]]]}, width_km: 90, center: {lat: 24.55, lon: 81.15}}
    realized:
      t1: []
    gold:
      survived: [storm-svc-1]
      max_spares_wasted: 2
      decision_at_t0: act
      label: act
      rationale: smoke episode; not scored
    flip_variable: [smoke]
    metadata:
      cone_width_km: 90
      cone_motion_kmh: 20
      n_future_claimants: 0
      exposure_horizon_hours: 1
      spares_on_hand: 2
""")

# Originally the t0 cone was T2a.yaml's t1:t6 (far) cone, copied verbatim --
# a ~100km-radius circle centred ON storm-svc-1's own point (24.855553,
# 81.327777), 200km wide. That circle excluded both satna<->rewa (working)
# and satna<->jhansi (protection), the wide avoid group that funnelled every
# real reroute candidate through jhansi<->allahabad (protection's own
# segment), producing the genuine disjointness_collapse
# `test_t2a_carries_a_real_validate_plan_rejection` already proves recovers
# via widen-and-retry.
#
# Retuned 2026-08-30 (exposure-and-depot design, satna<->jabalpur going
# aerial, §3.2 Option B). satna has degree 3 and, once jabalpur is aerial
# too, that same circle also swept up satna<->jabalpur (its nearest point to
# the circle's centre is ~65km, well inside the ~100km radius) -- fully
# enclosing satna. With NO edge left outside the avoid set, "avoid the risk
# group" and "reach satna" become mutually exclusive and the menu comes back
# no_solution even after widening: there was no longer an escape route left
# to find. jabalpur is the escape route this scenario needs to prove exists
# (that's the whole point of "exposure follows the service after it
# reroutes" -- a reroute has to actually be possible), so the fix is
# geometric, not a parameter tweak: the circle can't simply shrink, because
# jhansi and jabalpur's spans are BOTH ~65km from this centre along nearly
# the same distance band (satna is their common near endpoint), so no radius
# alone separates "expose jhansi" from "expose jabalpur" -- confirmed by
# sweeping every radius from 20-125km, which flips both together every time
# (scratchpad/sweep_exposure_smoke_radius.py). What DOES separate them is
# direction: jhansi sits at bearing ~291 deg from satna, rewa at ~96 deg,
# jabalpur at ~212 deg -- three very different directions. The cone below is
# the same technique Task 3 used for `SMOKE` (the `cone` polygon that
# `map_geo_event_to_assets` actually routes against and `center`/`width_km`,
# which only feed `p_cut_region`/`nearest_span_offset_km`'s scoring, are
# never cross-checked against each other -- confirmed in `scenario_file.py`)
# taken one step further: `cone` is now a hand-built "keyhole" polygon --
# the same ~90km-radius disc, centred at satna itself this time, with a
# small hole directly over satna's own point (so all three of satna's edges
# don't trivially touch it at their shared origin) and an 80 deg wide wedge
# cut out toward jabalpur's bearing (212 +/- 40 deg, comfortably clear of
# rewa's 96 deg and jhansi's 291 deg) removed. `center`/`width_km` are left
# exactly as they were (still the real storm-svc-1 point, 200km) since they
# don't drive routing and this fixture's own p_cut/offset_km assertions
# still hold under them (rewa's offset from that centre, ~36km, is smaller
# than jabalpur's, ~65-100km, so p_cut still drops after the reroute onto
# jabalpur, as the test requires). Verified against the real geo_mapper:
# exposed == {(satna,rewa), (satna,jhansi)} only, both before and after
# rounding coordinates to 5 decimals (scratchpad/build_exposure_smoke_cone.py,
# scratchpad/finalize_exposure_smoke_cone.py). A plain
# ForecastBlindBaseline("immediate") still never recovers from the
# rewa+jhansi disjointness_collapse (it hits the cap with zero commits), so
# this fixture remains reserved for tests that pair it with
# `_WidensOnDisjointnessRejection`.
EXPOSURE_SMOKE = textwrap.dedent("""
    id: EXPOSURE_SMOKE
    seed: 17
    state_file: eval/states/loaded-s17.json
    service_under_test: storm-svc-1
    track: hudhud
    hours: [t0, t1]
    decision_hour: t0
    lead_time_hours: 1
    spares_on_hand: 2
    depot_site: satna
    spare_inventory: {satna: 2}
    damage_radius_km: 74
    reference_avoid: {}
    forecast:
      t0:
        t1: {cone: {type: Polygon, coordinates: [[[81.62627533988763, 24.425603365240285], [81.6069971776491, 24.34864062744429], [81.58026817756033, 24.27393807660027], [81.5463457545037, 24.202215139109416], [81.50555659995715, 24.134162546067515], [81.45829353577656, 24.07043568114689], [81.40501173110663, 24.011648268893378], [81.34622431885312, 23.95836646422344], [81.28249745393249, 23.911103400042858], [81.2144448608906, 23.8703142454963], [81.14272192339973, 23.836391822439673], [81.06801937255571, 23.80966282235089], [80.99105663475972, 23.790384660112373], [80.9471105934481, 23.783865880676014], [80.85224219210275, 24.450446012092105], [80.85961777245996, 24.45117244335206], [80.88489532056663, 24.458840303739944], [80.90819124232209, 24.471292233340478], [80.92861028851777, 24.48804971148223], [80.94536776665953, 24.50846875767792], [80.95781969626006, 24.53176467943338], [80.96548755664794, 24.557042227540048], [80.96807667624866, 24.58333], [80.96548755664794, 24.609617772459952], [80.95781969626006, 24.63489532056662], [80.94536776665953, 24.65819124232208], [80.92861028851777, 24.67861028851777], [80.90819124232209, 24.695367766659523], [80.88489532056663, 24.707819696260056], [80.85961777245996, 24.71548755664794], [80.83333, 24.718076676248653], [80.80704222754005, 24.71548755664794], [80.78176467943338, 24.707819696260056], [80.75846875767792, 24.695367766659523], [80.73804971148223, 24.67861028851777], [80.72129223334048, 24.65819124232208], [80.70884030373995, 24.63489532056662], [80.70117244335206, 24.609617772459952], [80.69858332375135, 24.58333], [80.70117244335206, 24.557042227540048], [80.70583876097633, 24.541659439870248], [80.06548793506103, 24.332360528040812], [80.0596628223509, 24.34864062744429], [80.04038466011238, 24.425603365240285], [80.02874299496499, 24.504085096751165], [80.02484994250808, 24.58333], [80.02874299496499, 24.662574903248835], [80.04038466011238, 24.741056634759715], [80.0596628223509, 24.81801937255571], [80.08639182243968, 24.89272192339973], [80.1203142454963, 24.964444860890584], [80.16110340004286, 25.032497453932486], [80.20836646422345, 25.09622431885311], [80.26164826889338, 25.155011731106622], [80.32043568114689, 25.20829353577656], [80.38416254606751, 25.255556599957142], [80.45221513910941, 25.2963457545037], [80.52393807660027, 25.330268177560328], [80.5986406274443, 25.35699717764911], [80.67560336524029, 25.376275339887627], [80.75408509675117, 25.387917005035014], [80.83333, 25.391810057491917], [80.91257490324884, 25.387917005035014], [80.99105663475972, 25.376275339887627], [81.06801937255571, 25.35699717764911], [81.14272192339973, 25.330268177560328], [81.21444486089058, 25.2963457545037], [81.28249745393249, 25.255556599957142], [81.34622431885312, 25.20829353577656], [81.40501173110663, 25.155011731106622], [81.45829353577656, 25.09622431885311], [81.50555659995715, 25.032497453932486], [81.5463457545037, 24.964444860890584], [81.58026817756033, 24.892721923399733], [81.6069971776491, 24.81801937255571], [81.62627533988763, 24.741056634759715], [81.63791700503502, 24.662574903248835], [81.64181005749192, 24.58333], [81.63791700503502, 24.504085096751165], [81.62627533988763, 24.425603365240285]]]}, width_km: 200, center: {lat: 24.855553333333333, lon: 81.32777666666667}}
    realized:
      t1: []
    gold:
      survived: [storm-svc-1]
      max_spares_wasted: 2
      decision_at_t0: act
      label: act
      rationale: smoke episode; not scored
    flip_variable: [smoke]
    metadata:
      cone_width_km: 200
      cone_motion_kmh: 20
      n_future_claimants: 0
      exposure_horizon_hours: 1
      spares_on_hand: 2
""")


def test_action_payloads_resolves_the_real_hour_label_not_a_synthesized_one():
    # T2a/T2b/T3a/T3b's actual hours: [t0, t1, t2, t6] -- non-positional.
    # A naive f"t{index}" reading of effective_at_index=3 would guess "t3",
    # a real hour that never appears in this list. This is exactly the
    # off-by-four-hours misreading the final-review finding is about: an
    # optical_reroute committed at t2 (index 2) with lead time 1 records
    # effective_at_index=3, and index 3 in this list names t6, not t3.
    t2a_hours = ["t0", "t1", "t2", "t6"]
    action = Action(hour="t2", hour_index=2, lever="optical_reroute",
                    effective_at_index=3, spares={"satna": 1},
                    service_id="storm-svc-1")
    [payload] = action_payloads((action,), hours=t2a_hours)
    assert payload["effective_at_index"] == 3
    assert payload["effective_at_hour"] == "t6"
    assert payload["effective_at_hour"] != "t3"


def test_action_payloads_resolves_none_past_the_last_hour():
    # Acting on the episode's last hour with lead time 1 puts the index one
    # past the end of `hours` -- must not raise, must not synthesize a label.
    hours = ["t0", "t1"]
    action = Action(hour="t1", hour_index=1, lever="optical_reroute",
                    effective_at_index=2, spares={"satna": 1},
                    service_id="storm-svc-1")
    [payload] = action_payloads((action,), hours=hours)
    assert payload["effective_at_hour"] is None


def test_action_payloads_defaults_hours_to_empty_for_positional_callers():
    # tests/eval/test_scoring.py constructs Action directly and never calls
    # action_payloads with a `hours` argument; the default must not raise.
    action = Action(hour="t0", hour_index=0, lever="ip_reroute",
                    effective_at_index=0, spares={}, service_id="storm-svc-1")
    [payload] = action_payloads((action,))
    assert payload["effective_at_hour"] is None


def _scenario(tmp_path):
    path = tmp_path / "SMOKE.yaml"
    path.write_text(SMOKE, encoding="utf-8")
    return load_scenario(path)


def _exposure_scenario(tmp_path):
    path = tmp_path / "EXPOSURE_SMOKE.yaml"
    path.write_text(EXPOSURE_SMOKE, encoding="utf-8")
    return load_scenario(path)


async def _run(scenario, decider, state_path, server_command, server_env):
    async with connect_server(
        TOPOLOGY_PATH, server_command=server_command, env=server_env,
        extra_args=["--state", str(state_path)],
    ) as client:
        return await run_episode(client, scenario, decider,
                                 topology_path=TOPOLOGY_PATH)


class _RecordingDecider:
    """Wraps a real decider and keeps every observation the harness handed it.
    Lets a test assert what the rollout SHOWED the decider, which the trace
    does not record -- only the decisions come back in EpisodeTrace."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.observations = []          # one per hour: the timing observation
        self.probes = []                # one per constraints() call

    def timing(self, obs):
        self.observations.append(obs)
        return self.inner.timing(obs)

    def constraints(self, obs, unconstrained_menu=None):
        self.probes.append(unconstrained_menu)
        return self.inner.constraints(obs, unconstrained_menu)

    def objective(self, obs, menu):
        return self.inner.objective(obs, menu)


def test_the_constraints_step_is_shown_the_menu_it_is_about_to_narrow(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # F3 (remediation spec lines 96-112). decisions.py: decision 2 changes
    # which candidates EXIST. It was made blind to what constraining costs.
    decider = _RecordingDecider(ForecastBlindBaseline("immediate"))
    trace = asyncio.run(_run(
        _scenario(tmp_path), decider,
        loaded_state_path, local_server_command, local_server_env))

    assert decider.probes, "the acting hour never called constraints()"
    probe = decider.probes[0]
    assert probe["status"] == "solution"
    assert probe["candidates"], "a real unconstrained menu is never empty here"
    for candidate in probe["candidates"]:
        assert set(candidate) == {"candidate_label", "lever", "spares_needed"}
    # The probe is the menu BEFORE the agent's avoid set, so it is the same
    # object for every iteration of ONE hour -- it does not depend on the
    # answer, and re-probing per iteration would be a wasted tool call.
    # SMOKE's immediate baseline commits at BOTH of its hours here
    # (spares_on_hand=2 is sized for exactly that: storm-svc-1's offset from
    # the cone centre never crosses outside the half-width even after t0's
    # reroute), so decider.probes spans two DIFFERENT acting hours with two
    # different real route_service results -- check same-object reuse within
    # each hour's own iterations, not across the whole episode.
    offset = 0
    for hour_record in trace.hours:
        n_iterations = len(hour_record.get("iterations", []))
        if n_iterations == 0:
            continue
        group = decider.probes[offset:offset + n_iterations]
        assert all(p is group[0] for p in group), (
            f"hour {hour_record['hour']!r} re-probed across its own "
            f"iterations")
        offset += n_iterations
    assert offset == len(decider.probes)
    # And it lands in the trace, so a reader can see what the constraints
    # decision was looking at without re-running the episode.
    acting = next(h for h in trace.hours if h.get("committed"))
    assert acting["unconstrained_menu"] == probe


class _WidensOnDisjointnessRejection:
    """Wraps ForecastBlindBaseline('immediate') -- same timing/objective --
    but reacts to a genuine validate_plan disjointness_collapse rejection by
    adding the violation's OWN `shared_assets` to the avoid set and retrying.

    Local to this module (test_runner.py's own docstring: the seven real
    episodes must not be needed to prove the loop works, so this stays
    self-contained rather than importing from test_episodes.py). Same
    mechanic, first proven against a real server, as
    tests/eval/test_episodes.py's `_WidensOnDisjointnessRejection` /
    `test_t2a_carries_a_real_validate_plan_rejection`: storm-svc-1's own
    static protection lightpath (satna<->jhansi<->allahabad) is the cheapest
    way into allahabad from anywhere else, so a plain forecast-blind avoid
    that also excludes satna<->rewa (working) funnels every real reroute
    candidate through jhansi<->allahabad -- a real, first-try
    disjointness_collapse under basis=physical/level=link, not fabricated.
    Widening with the violation's own shared_assets (naming the specific
    jhansi<->allahabad fiber/amp/roadm ids) finds a genuinely disjoint route
    that validates on a later iteration."""

    def __init__(self) -> None:
        self._inner = ForecastBlindBaseline("immediate")
        self.name = "widens-on-disjointness-rejection"

    def timing(self, obs):
        return self._inner.timing(obs)

    def constraints(self, obs, unconstrained_menu=None):
        base = self._inner.constraints(obs, unconstrained_menu)
        extra: set[str] = set()
        if obs.last_rejection and obs.last_rejection.get("type") == "validation_violations":
            for violation in obs.last_rejection.get("violations", []):
                if violation.get("type") == "disjointness_collapse":
                    extra.update(violation.get("shared_assets", []))
        if not extra:
            return base
        avoid = dict(base.avoid)
        avoid["assets"] = sorted(set(avoid.get("assets", [])) | extra)
        return ConstraintDecision(
            avoid=avoid,
            reasoning=base.reasoning + "; widened after disjointness rejection",
            protected=base.protected, best_effort=base.best_effort,
            basis=base.basis, level=base.level)

    def objective(self, obs, menu):
        return self._inner.objective(obs, menu)


def test_exposure_follows_the_service_after_it_reroutes(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # F4's stale-geometry half (remediation spec lines 116-121).
    # EXPOSURE_SMOKE publishes ONE issuance at t0, so both hours read the
    # same cone at the same horizon -- offset_km is then a function of the
    # service's representative point ALONE, and can only move if the
    # harness re-read the working path after the t0 commit. storm-svc-1 has
    # no valid alternate route that both escapes a real exposure and stays
    # clear of its own protection leg without a widen-and-retry
    # (task-1-report.md); a plain ForecastBlindBaseline("immediate") never
    # moves the node sequence at all, so this uses
    # _WidensOnDisjointnessRejection instead. This is why the test uses its
    # own EXPOSURE_SMOKE scenario rather than the shared SMOKE fixture:
    # SMOKE's cone is deliberately kept clear of storm-svc-1's corridor so
    # every OTHER test in this file, which pairs it with a plain
    # ForecastBlindBaseline, keeps committing cleanly at t0.
    decider = _RecordingDecider(_WidensOnDisjointnessRejection())
    trace = asyncio.run(_run(
        _exposure_scenario(tmp_path), decider,
        loaded_state_path, local_server_command, local_server_env))

    assert trace.actions and trace.actions[0].hour == "t0", (
        "this test needs the t0 commit to have happened")
    assert len(decider.observations) == 2
    before = decider.observations[0].exposure["storm-svc-1"]["t1"]
    after = decider.observations[1].exposure["storm-svc-1"]["t1"]
    assert after["offset_km"] != before["offset_km"], (
        "the service moved but the harness re-used the pre-commit point")
    # The baseline avoids the exposed risk group, so the committed path
    # routes around the cone's assets and the representative point has to
    # move AWAY from the cone centre.
    assert after["p_cut"] < before["p_cut"]


def test_a_baseline_rollout_completes_and_records_every_hour(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    trace = asyncio.run(_run(
        _scenario(tmp_path), ForecastBlindBaseline("immediate"),
        loaded_state_path, local_server_command, local_server_env))
    assert [h["hour"] for h in trace.hours] == ["t0", "t1"]
    assert trace.terminal_status in {"converged", "declared_infeasible"}
    assert trace.tool_calls > 0
    assert all("timing" in h for h in trace.hours)


def test_the_next_hour_is_told_what_was_committed_in_the_previous_one(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # F4's no-memory half (remediation spec lines 122-132). The decider is
    # asked to decide again at t1 with no hint that it already acted at t0
    # unless the harness tells it.
    decider = _RecordingDecider(ForecastBlindBaseline("immediate"))
    trace = asyncio.run(_run(
        _scenario(tmp_path), decider,
        loaded_state_path, local_server_command, local_server_env))

    assert trace.actions and trace.actions[0].hour == "t0"
    committed = trace.actions[0]
    assert decider.observations[0].actions_taken == ()
    assert decider.observations[0].spares_spent == 0

    at_t1 = decider.observations[1]
    assert len(at_t1.actions_taken) == 1
    remembered = at_t1.actions_taken[0]
    assert remembered["hour"] == "t0"
    assert remembered["lever"] == committed.lever
    assert remembered["spares"] == committed.spares
    assert remembered["effective_at_index"] == committed.effective_at_index
    # SMOKE's hours are positional (["t0", "t1"]), so the label happens to
    # equal what a naive f"t{index}" scheme would guess here -- the real
    # non-positional case (T2a-style [t0, t1, t2, t6]) is covered directly
    # in tests/eval/test_observation.py, since that's a property of
    # action_payloads' own resolution, not of the rollout.
    scenario_hours = ["t0", "t1"]
    assert remembered["effective_at_hour"] == (
        scenario_hours[committed.effective_at_index]
        if committed.effective_at_index < len(scenario_hours) else None)
    # The avoid set is what makes the memory usable: "I already routed
    # around that risk group" is the premise the agent got wrong.
    assert "risk_groups" in remembered["avoid"]
    # spares_spent is scoped to the depot site alone (ledger.py's own
    # contract) -- satna, since that's every real reroute's own home site.
    assert at_t1.spares_spent == committed.spares.get("satna", 0)


def test_a_committed_action_debits_the_ledger_and_records_its_lead_time(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    trace = asyncio.run(_run(
        _scenario(tmp_path), ForecastBlindBaseline("immediate"),
        loaded_state_path, local_server_command, local_server_env))
    assert trace.actions, "the immediate baseline should have acted at t0"
    action = trace.actions[0]
    assert action.hour == "t0"
    # ip_reroute lands at once; anything else costs the scenario's lead time.
    expected = 0 if action.lever == "ip_reroute" else 1
    assert action.effective_at_index == action.hour_index + expected
    # spares_remaining is on_hand at the depot site (satna) alone.
    assert trace.spares_remaining == 2 - sum(
        a.spares.get("satna", 0) for a in trace.actions)


def test_an_unaffordable_choice_is_rejected_and_the_loop_retries(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # run_episode builds the SpareLedger from `scenario.spare_inventory`
    # (Task 9), not `spares_on_hand` -- both must be zeroed for the ledger
    # to actually be empty. `depot_site` is left alone: it already comes
    # through as "satna" from SMOKE, which is what `spare_inventory` here
    # needs to key on.
    broke = dataclasses.replace(
        _scenario(tmp_path), spares_on_hand=0, spare_inventory={"satna": 0})
    trace = asyncio.run(_run(
        broke, ForecastBlindBaseline("immediate"),
        loaded_state_path, local_server_command, local_server_env))
    rejections_by_hour = [h.get("rejections", []) for h in trace.hours]
    # With no spares, any optical candidate must be refused by the ledger --
    # assert this actually happened rather than silently no-op'ing if it
    # didn't (found live: a stale `spares_on_hand=0`-only replace() left
    # `spare_inventory` at SMOKE's real {"satna": 2}, so the ledger was never
    # actually broke and this whole test passed for the wrong reason).
    insufficient = [r for hour_rejections in rejections_by_hour
                    for r in hour_rejections
                    if r["type"] == "insufficient_spares"]
    assert insufficient, (
        f"expected at least one insufficient_spares rejection with an "
        f"empty ledger; rejections_by_hour={rejections_by_hour}")
    # The retry cap is per ACTING HOUR (a global constraint), not per
    # episode -- both t0 and t1 are exposed here, so each independently
    # exhausts its own cap; the total across the episode can exceed
    # MAX_ITERATIONS even though no single hour ever does.
    assert max(len(hr) for hr in rejections_by_hour) <= MAX_ITERATIONS
    assert trace.spares_remaining == 0


def test_the_loop_caps_at_five_iterations(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # A decider that always picks an out-of-range candidate can never
    # converge; the harness must stop it, not spin.
    always_bad = ScriptedDecider(
        "always-bad",
        default_timing=TimingDecision("act", "always act"),
        default_constraints=ConstraintDecision(avoid={}, reasoning="none"),
        default_objective=ObjectiveDecision("candidate_999", None, "bad index"))
    trace = asyncio.run(_run(
        _scenario(tmp_path), always_bad,
        loaded_state_path, local_server_command, local_server_env))
    assert trace.terminal_status == "hit_cap"
    assert max(len(h.get("iterations", [])) for h in trace.hours) == MAX_ITERATIONS


class _LoosenAfterInfeasible:
    """Declares the menu infeasible under a tight avoid on iteration 0, then
    loosens and commits on iteration 1 -- the correction shape MAX_ITERATIONS
    and last_rejection exist for, and the one the runner had no path back
    from (remediation spec, W2.3)."""

    name = "loosen-after-infeasible"

    def __init__(self):
        self.rejections_seen = []

    def timing(self, obs):
        return TimingDecision("act", "act: exposed at the next horizon")

    def constraints(self, obs, unconstrained_menu=None):
        self.rejections_seen.append(obs.last_rejection)
        if obs.iteration == 0:
            return ConstraintDecision(
                avoid={"risk_groups": sorted(obs.risk_group_ids.values())},
                reasoning="tight: avoid every forecast risk group")
        return ConstraintDecision(
            avoid={}, reasoning="loosened after declaring the menu infeasible")

    def objective(self, obs, menu):
        if obs.iteration == 0:
            return ObjectiveDecision("infeasible", None,
                                     "nothing on this menu is acceptable")
        return ObjectiveDecision("candidate_0", None, "take the first entry")


def test_declaring_infeasible_feeds_back_a_rejection_and_the_loop_retries(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    decider = _LoosenAfterInfeasible()
    trace = asyncio.run(_run(
        _scenario(tmp_path), decider,
        loaded_state_path, local_server_command, local_server_env))

    first_hour = trace.hours[0]
    outcomes = [step["outcome"] for step in first_hour["iterations"]]
    assert outcomes[0] == "declared_infeasible"
    assert "committed" in outcomes, (
        "the loosened second attempt must have been given a chance to commit")
    # The declaration is fed back as a typed rejection, exactly like an
    # invalid choice or an unaffordable candidate.
    assert first_hour["rejections"][0]["type"] == "declared_infeasible"
    assert "menu_size" in first_hour["rejections"][0]
    assert decider.rejections_seen[1]["type"] == "declared_infeasible"
    assert trace.terminal_status == "converged"
    assert trace.actions


def test_a_decider_that_only_ever_declares_infeasible_still_terminates(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    always_infeasible = ScriptedDecider(
        "always-infeasible",
        default_timing=TimingDecision("act", "always act"),
        default_constraints=ConstraintDecision(avoid={}, reasoning="none"),
        default_objective=ObjectiveDecision("infeasible", None, "never happy"))
    trace = asyncio.run(_run(
        _scenario(tmp_path), always_infeasible,
        loaded_state_path, local_server_command, local_server_env))

    assert trace.terminal_status == "declared_infeasible"
    assert not trace.actions
    assert max(len(h.get("iterations", []))
               for h in trace.hours) == MAX_ITERATIONS


def test_lead_time_marks_an_optical_reroute_at_the_cut_hour_as_late(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # The service is cut at t1 and the baseline acts at t1 (not t0), so an
    # optical_reroute cannot have landed. This is scored in Task 10; here we
    # only assert the trace carries the arithmetic to score it.
    scenario = _scenario(tmp_path)
    trace = asyncio.run(_run(
        scenario, ForecastBlindBaseline("at_deadline"),
        loaded_state_path, local_server_command, local_server_env))
    for action in trace.actions:
        assert action.effective_at_index >= action.hour_index


async def _geometry(state_path, server_command, server_env):
    async with connect_server(
        TOPOLOGY_PATH, server_command=server_command, env=server_env,
        extra_args=["--state", str(state_path)],
    ) as client:
        return await service_geometry(client, TOPOLOGY_PATH)


def test_service_geometry_keeps_only_the_storm_cuttable_spans(
        loaded_state_path, local_server_command, local_server_env):
    geometry = asyncio.run(_geometry(
        loaded_state_path, local_server_command, local_server_env))

    # storm-svc-1's working path is oms_satna_rewa (AERIAL) then
    # oms_rewa_allahabad (buried). Exactly one span survives the storm filter,
    # and it is the one that starts at satna.
    spans = geometry.cuttable_spans["storm-svc-1"]
    assert len(spans) == 1
    satna = (24.58333, 80.83333)
    assert spans[0][0] == pytest.approx(satna, abs=1e-5)

    # The full working path is still two legs -- the filter trims exposure,
    # not geometry.
    assert len(geometry.paths["storm-svc-1"]["working"]) == 3

    # And the endpoint SITES are what the per-site ledger will charge.
    assert geometry.endpoint_sites["storm-svc-1"] == ("satna", "allahabad")


def test_a_service_on_wholly_buried_fibre_has_no_cuttable_span(
        loaded_state_path, local_server_command, local_server_env):
    # d0029 routes allahabad -> fatehpur -> kanpur; both spans are buried, so
    # its true storm exposure is ZERO. The shipped midpoint model scored it
    # 0.4715 and that number is what made T1a's gold label `wait`
    # (exposure-and-depot design, D4).
    geometry = asyncio.run(_geometry(
        loaded_state_path, local_server_command, local_server_env))
    assert geometry.cuttable_spans["d0029"] == ()


def test_service_geometry_keeps_the_polyline_the_midpoint_came_from(tmp_path):
    calls = []
    geo = asyncio.run(runner.service_geometry(
        None, _write_geometry_topology(tmp_path),
        call=_geometry_call(calls)))
    assert geo.paths["storm-svc-1"]["working"] == [
        "satna", "rewa", "allahabad"]
    assert geo.paths["storm-svc-1"]["protection"] == [
        "satna", "jhansi", "nowhere"]


def test_service_geometry_records_nodes_the_local_topology_lacks(tmp_path):
    # runner.py:189's `if n in coords` guard drops unmappable nodes silently.
    # The viewer must render such a gap rather than draw a shorter line
    # (run-viewer design, open item 3).
    geo = asyncio.run(runner.service_geometry(
        None, _write_geometry_topology(tmp_path),
        call=_geometry_call([])))
    assert geo.unmapped_nodes["storm-svc-1"] == ["nowhere"]


def test_service_geometry_maps_every_oms_to_its_two_endpoints(tmp_path):
    # Resolved from the server's explicit src_node_id/dst_node_id, NEVER from
    # the `oms_satna_rewa` name convention -- that convention happens to hold
    # on this topology and is not a contract (run-viewer design, §5.1).
    geo = asyncio.run(runner.service_geometry(
        None, _write_geometry_topology(tmp_path),
        call=_geometry_call([])))
    assert geo.oms_nodes["oms_sr"] == ["satna", "rewa"]
    assert len(geo.oms_nodes) == 4


def test_service_geometry_strips_the_router_prefix_for_endpoint_sites(tmp_path):
    # endpoint_sites names the SITE a new lightpath would charge a
    # transponder to, not the router id get_services reports.
    geo = asyncio.run(runner.service_geometry(
        None, _write_geometry_topology(tmp_path),
        call=_geometry_call([])))
    assert geo.endpoint_sites["storm-svc-1"] == ("satna", "allahabad")


def test_service_geometry_reports_no_cuttable_spans_without_local_edges(
        tmp_path):
    # GEOMETRY_TOPOLOGY declares no edges at all, so no mount_type is known
    # for any leg -- cuttable_spans must come back empty rather than raise,
    # and every service still gets a (possibly empty) entry.
    geo = asyncio.run(runner.service_geometry(
        None, _write_geometry_topology(tmp_path),
        call=_geometry_call([])))
    assert geo.cuttable_spans["storm-svc-1"] == ()


def test_service_points_is_unchanged_and_still_costs_four_calls(tmp_path):
    path = _write_geometry_topology(tmp_path)
    calls = []
    points = asyncio.run(runner.service_points(
        None, path, call=_geometry_call(calls)))
    # Midpoint over the WORKING path's deduped nodes only -- protection must
    # not move it, or every recorded p_cut in the suite shifts.
    assert points["storm-svc-1"] == pytest.approx(
        ((24.6 + 24.5 + 25.4) / 3, (80.8 + 81.3 + 81.8) / 3))
    assert calls == ["get_topology", "get_topology", "get_lightpaths",
                     "get_services"]


def test_menu_for_prompt_adds_the_label_and_the_spare_cost_and_keeps_the_rest():
    menu = {"status": "solution", "candidates": [
        {"lever": "ip_reroute", "reused_lightpaths": ["lp_1"],
         "new_lightpaths": [], "restored_gbps": 300.0, "shortfall_gbps": 0.0,
         "cost_vector": {"transponders": 418.0}},
        {"lever": "optical_reroute", "reused_lightpaths": [],
         "new_lightpaths": [{"oms_sequence": ["oms_sj"], "lam": 3,
                             "mode_id": "m1", "gsnr_db": 17.0,
                             "bitrate_gbps": 400.0}],
         "restored_gbps": 300.0, "shortfall_gbps": 0.0,
         "cost_vector": {"transponders": 420.0}}]}
    out = runner.menu_for_prompt(menu, {"oms_sj": ["satna", "jhansi"]})
    assert out["status"] == "solution"
    assert [c["candidate_label"] for c in out["candidates"]] == [
        "candidate_0", "candidate_1"]
    assert [c["spares_needed"] for c in out["candidates"]] == [
        {}, {"satna": 1, "jhansi": 1}]
    # Unlike unconstrained_menu_projection, this one keeps the cost vector --
    # it is what decision 3 weighs.
    assert out["candidates"][0]["cost_vector"] == {"transponders": 418.0}
    assert menu["candidates"][0] == {
        "lever": "ip_reroute", "reused_lightpaths": ["lp_1"],
        "new_lightpaths": [], "restored_gbps": 300.0, "shortfall_gbps": 0.0,
        "cost_vector": {"transponders": 418.0}}     # no mutation


# Measured, not computed, against SMOKE + ForecastBlindBaseline("immediate")
# BEFORE task A4's recording changes landed (task-A4-brief.md, Step 1): run
# the pre-change suite and read off the number it actually reports. Pinning
# this is what test_recording_costs_no_extra_server_calls below is for --
# recording the hour's observation/menu/geometry must add zero calls to the
# already-counted call sites.
EXPECTED_TOOL_CALLS_IMMEDIATE_BASELINE = 22


def _observation_with_exposure(exposure: dict, *, sut: str = "storm-svc-1",
                                horizon_totals: dict | None = None
                                ) -> Observation:
    """A bare Observation carrying exactly the exposure map the caller gives
    -- test_agent.py's `_obs` pattern, but the exposure dict is the whole
    point of the call rather than incidental setup. `_visible_services` and
    `observation_record` only ever read `p_cut` off each per-horizon entry
    and pass the rest through verbatim, so entries need no other keys.

    `horizon_totals` defaults to an obviously-synthetic sentinel, deliberately
    NOT derivable from `exposure` -- so a test asserting the trimmed record's
    `horizon_totals` still equals `obs.horizon_totals` is proof the value
    survived untouched, not proof two independent computations happened to
    agree."""
    horizons = sorted({h for per_horizon in exposure.values()
                       for h in per_horizon})
    services = tuple({"id": svc, "demand_gbps": 100.0} for svc in exposure)
    return Observation(
        scenario_id="TEST", service_under_test=sut, hour="t0",
        hour_index=0, hours_remaining=1,
        issuance=Issuance(issued_at="t0", horizons={
            h: ConeAtHorizon(cone={"type": "Polygon", "coordinates": []},
                             width_km=100.0, center={"lat": 0.0, "lon": 0.0})
            for h in horizons}),
        exposure=exposure, services=services, spares_on_hand=1,
        lead_time_hours=1, risk_group_ids={h: f"rg_{h}" for h in horizons},
        horizon_totals=(horizon_totals if horizon_totals is not None else
                        {"sentinel_horizon": {"sut_ecar_gbps": 42.0,
                                              "non_sut_total_ecar_gbps": 99.0}}))


def _scripted_acting_decider() -> ScriptedDecider:
    """Always acts, constrains nothing, and takes the menu's first candidate
    -- the shape test_the_loop_caps_at_five_iterations already proves finds
    real candidates against SMOKE under avoid={}. Used where a test needs a
    committed hour with a real, full routing menu on record, not just a
    baseline's fixed policy."""
    return ScriptedDecider(
        "scripted-acting",
        default_timing=TimingDecision("act", "always act"),
        default_constraints=ConstraintDecision(avoid={}, reasoning="none"),
        default_objective=ObjectiveDecision(
            "candidate_0", None, "take the first entry"))


async def _run_episode_with(decider, *, scenario, state_path, server_command,
                            server_env):
    """`_run` above, generalized over decider and scenario: every A4 test
    needs a real rollout against SMOKE (or a variant of it) with a decider
    the existing fixtures don't already parametrize `_run` for."""
    async with connect_server(
        TOPOLOGY_PATH, server_command=server_command, env=server_env,
        extra_args=["--state", str(state_path)],
    ) as client:
        return await run_episode(client, scenario, decider,
                                 topology_path=TOPOLOGY_PATH)


def test_visible_services_are_those_with_any_nonzero_cut_probability():
    obs = _observation_with_exposure({
        "storm-svc-1": {"t2": {"p_cut": 0.0}, "t6": {"p_cut": 0.52}},
        "d0004":       {"t2": {"p_cut": 0.12}, "t6": {"p_cut": 0.0}},
        "d9999":       {"t2": {"p_cut": 0.0},  "t6": {"p_cut": 0.0}}})
    assert runner._visible_services(obs) == ["d0004", "storm-svc-1"]


def test_visible_services_keeps_the_actionable_service_even_at_zero_p_cut():
    # final-review fix (2026-08-29): a successful reroute is exactly what
    # drives the actionable service's own p_cut to 0 -- dropping it here
    # would silently erase the "reroute worked, midpoint settled" case
    # from service_points/service_paths/observation.exposure for the hour,
    # disagreeing with project_observation's `keep` set (agent.py), which
    # keeps obs.service_under_test unconditionally.
    obs = _observation_with_exposure({
        "storm-svc-1": {"t2": {"p_cut": 0.0}, "t6": {"p_cut": 0.0}},
        "d9999":       {"t2": {"p_cut": 0.0}, "t6": {"p_cut": 0.0}}},
        sut="storm-svc-1")
    assert runner._visible_services(obs) == ["storm-svc-1"]


def test_the_hour_record_keeps_the_totals_whole_while_trimming_the_rows():
    obs = _observation_with_exposure({
        "storm-svc-1": {"t2": {"p_cut": 0.4}},
        "unexposed-svc": {"t2": {"p_cut": 0.0}}})
    geo = runner.ServiceGeometry(
        points={"storm-svc-1": (24.6, 80.8)},
        paths={"storm-svc-1": {"working": ["satna", "rewa"],
                               "protection": ["satna", "jhansi"]}},
        oms_nodes={}, unmapped_nodes={},
        cuttable_spans={"storm-svc-1": (((24.6, 80.8), (24.5, 81.3)),)},
        endpoint_sites={"storm-svc-1": ("satna", "allahabad")})
    rec = runner.observation_record(obs, geo)
    assert sorted(rec["observation"]["exposure"]) == ["storm-svc-1"]
    assert rec["observation"]["horizon_totals"] == obs.horizon_totals
    assert rec["service_points"]["storm-svc-1"] == [24.6, 80.8]
    assert rec["service_paths"]["storm-svc-1"]["protection"] == [
        "satna", "jhansi"]
    # The viewer must be able to draw exactly what was scored, not just where
    # the deprecated midpoint sat.
    assert rec["cuttable_spans"]["storm-svc-1"] == [
        [[24.6, 80.8], [24.5, 81.3]]]


def test_the_trace_records_the_projection_the_decider_reported(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # Baselines have no projection and honestly record null; getattr gets
    # that without extending the Decider protocol (run-viewer design, §5.1).
    trace = asyncio.run(_run_episode_with(
        ForecastBlindBaseline("immediate"), scenario=_scenario(tmp_path),
        state_path=loaded_state_path, server_command=local_server_command,
        server_env=local_server_env))
    assert all(h["projected"] is None for h in trace.hours)


def test_the_trace_records_the_full_menu_decision_three_weighed(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    trace = asyncio.run(_run_episode_with(
        _scripted_acting_decider(), scenario=_scenario(tmp_path),
        state_path=loaded_state_path, server_command=local_server_command,
        server_env=local_server_env))
    acting = next(h for h in trace.hours if h.get("committed") is not None)
    menu = acting["iterations"][0]["menu"]
    assert "cost_vector" in menu["candidates"][0]
    assert menu["candidates"][0]["candidate_label"] == "candidate_0"
    # unconstrained_menu still strips it -- the two projections are different
    # on purpose.
    assert "cost_vector" not in acting["unconstrained_menu"]["candidates"][0]


def test_the_trace_root_carries_the_oms_endpoint_map(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    trace = asyncio.run(_run_episode_with(
        ForecastBlindBaseline("immediate"), scenario=_scenario(tmp_path),
        state_path=loaded_state_path, server_command=local_server_command,
        server_env=local_server_env))
    assert trace.oms_nodes
    assert all(len(v) == 2 for v in trace.oms_nodes.values())


def test_recording_costs_no_extra_server_calls(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # Verification item 7 of the run-viewer design, and the one that would
    # quietly invalidate a comparison if it failed.
    trace = asyncio.run(_run_episode_with(
        ForecastBlindBaseline("immediate"), scenario=_scenario(tmp_path),
        state_path=loaded_state_path, server_command=local_server_command,
        server_env=local_server_env))
    assert trace.tool_calls == EXPECTED_TOOL_CALLS_IMMEDIATE_BASELINE
