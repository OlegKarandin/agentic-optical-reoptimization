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
from storm_reoptimizer.mcp_client import call_tool_json, connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)

GEOMETRY_TOPOLOGY = {"graph": {"nodes": [
    {"id": "satna", "lat": 24.6, "lon": 80.8},
    {"id": "rewa", "lat": 24.5, "lon": 81.3},
    {"id": "allahabad", "lat": 25.4, "lon": 81.8},
    {"id": "jhansi", "lat": 25.4, "lon": 78.6},
], "edges": [
    {"src": "satna", "dst": "rewa", "mount_type": "aerial"},
    {"src": "rewa", "dst": "allahabad", "mount_type": "buried"},
    {"src": "satna", "dst": "jhansi", "mount_type": "aerial"},
]}, "srlgs": []}

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


def _write_geometry_topology_without_edges(tmp_path):
    # Same nodes as GEOMETRY_TOPOLOGY, but with the edges list emptied -- for
    # the two tests below whose whole point is "no local edges means no
    # mount_type is known for any leg".
    topo = {"graph": {"nodes": GEOMETRY_TOPOLOGY["graph"]["nodes"],
                      "edges": []}, "srlgs": []}
    path = tmp_path / "topo_no_edges.json"
    path.write_text(json.dumps(topo), encoding="utf-8")
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
# no realized cut. Its job is to exercise the loop, not to score anyone.
#
# COHERENT GEOMETRY (retuned 2026-09-01, hazard-footprint seam fix). Before
# that fix the risk group came from the `cone` POLYGON while exposure came
# from `center`/`width_km`, and this fixture exploited the gap: its polygon
# was a box near allahabad touching no aerial edge at all (empty avoid set,
# so every reroute was the inert same-corridor one), while `center`/
# `width_km` sat on the real working span so exposure still read nonzero.
# That decoupling IS the seam defect the fix closes -- `avoid` now derives
# from `events.geo.damage_footprint(center, width_km, damage_radius_km)`,
# a disc of radius `width_km/2 + damage_radius_km` about the SAME centre the
# p_cut arithmetic reads -- so the split is no longer expressible and the
# three fields below have to be chosen together.
#
# What the real topology allows, measured with `geo_mapper.load_edges` +
# Shapely against src/storm_reoptimizer/data/toy_india_topology.json:
#
#   * storm-svc-1 is satna->rewa->allahabad working, satna->jhansi->allahabad
#     protection. Its ONLY storm-cuttable (aerial) working span is
#     satna<->rewa, so ANY nonzero exposure means the footprint disc reaches
#     that span, which means the risk group always names it. "Exposed but
#     satna<->rewa not avoided" is arithmetically impossible now.
#   * satna's three aerial spans (rewa ~96 deg, jabalpur ~212 deg, jhansi
#     ~291 deg) all START AT SATNA, so for any centre off those bearings the
#     nearest point of the jhansi span and of the jabalpur span is satna
#     itself -- identical distances. A DISC therefore cannot separate them:
#     the only two reachable risk groups are {satna<->rewa} (disc short of
#     satna) and all three spans (disc reaching satna, which encloses the
#     depot and leaves no egress at all). The keyhole polygon EXPOSURE_SMOKE
#     used to cut jabalpur out of the group is dead weight now, for the same
#     reason: `damage_footprint` never looks at `cone.cone`.
#
# So this fixture takes the only usable option: centre the cone ON the
# satna<->rewa corridor but PAST rewa, at (24.50333, 81.58) -- 1.6 span
# lengths out along satna->rewa -- with width_km 90 and damage_radius_km 20.
# Then W = width_km/2 = 45 and R = W + damage_radius_km = 65, against
# measured offsets of 28.5 km to satna<->rewa (so 28.5 <= 45: the service is
# INSIDE the cone and the forecast-blind baseline acts, p_cut 0.1659) and
# 76.1 km to both the jhansi and the jabalpur span (so 76.1 > 65: neither is
# in the group -- 11.1 km of clearance). Risk group == {satna<->rewa},
# verified live via `map_geo_event_to_assets(damage_footprint(...))`.
#
# CONSEQUENCE, and why the deciders below changed: avoiding satna<->rewa for
# real is not a free no-op. rewa is a stub behind that span (its only other
# edge is the buried rewa<->allahabad), and allahabad's three accesses --
# fatehpur, rewa, jhansi -- are ALL BURIED, so no storm risk group can ever
# name jhansi<->allahabad. route_service's cheapest alternative is therefore
# satna->jhansi->allahabad, which IS the protection path: a real, first-try
# `disjointness_collapse` on `oms_jhansi_allahabad` under basis=physical/
# level=link. A plain ForecastBlindBaseline re-proposes it every iteration
# and hits the cap with zero commits; `_WidensOnDisjointnessRejection` adds
# the violation's own shared_assets and commits an optical_reroute on the
# next iteration (satna->jabalpur->...->fatehpur->allahabad, the one route
# to allahabad that is genuinely disjoint from protection). Every test in
# this file that needs SMOKE to COMMIT is therefore paired with that decider;
# the ones that only need the loop to run, or need a rejection rather than a
# commit, still use the plain baseline.
#
# After the t0 reroute the service's nearest aerial span is satna<->jabalpur
# at 76.1 km -- outside the 45 km half-width -- so the baseline waits at t1
# and the episode commits exactly once. (Nothing can change that: t1 exposure
# would need the disc to reach satna, which is the enclose-the-depot case.)
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
    damage_radius_km: 20
    reference_avoid: {}
    forecast:
      t0:
        t1: {cone: {type: Polygon, coordinates: [[[81.58, 24.90802], [81.69511, 24.89424], [81.80238, 24.85381], [81.89449, 24.78949], [81.96516, 24.70568], [82.0096, 24.60807], [82.02475, 24.50333], [82.0096, 24.39859], [81.96516, 24.30098], [81.89449, 24.21717], [81.80238, 24.15285], [81.69511, 24.11242], [81.58, 24.09864], [81.46489, 24.11242], [81.35762, 24.15285], [81.26551, 24.21717], [81.19484, 24.30098], [81.1504, 24.39859], [81.13525, 24.50333], [81.1504, 24.60807], [81.19484, 24.70568], [81.26551, 24.78949], [81.35762, 24.85381], [81.46489, 24.89424], [81.58, 24.90802]]]}, width_km: 90, center: {lat: 24.50333, lon: 81.58}}
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
      claimant_services: []
""")

# The fixture for "exposure follows the service after it reroutes": the
# rollout must actually MOVE storm-svc-1, and both the pre- and post-commit
# exposure readings must be real numbers off the real corridor, so that
# `p_cut` dropping is a measurement and not a degeneration to zero.
#
# History, kept because it is the reason this fixture exists separately from
# SMOKE. The t0 cone started as T2a.yaml's t1:t6 circle, ~100km radius
# centred on storm-svc-1's own point. When satna<->jabalpur went aerial
# (2026-08-30, exposure-and-depot design §3.2 Option B) that circle swept up
# all three of satna's aerial spans, enclosing the depot: "avoid the risk
# group" and "reach satna" became mutually exclusive and no widening could
# find an escape. The 2026-08-30 answer was a hand-built "keyhole" polygon --
# a disc with an 80 deg wedge cut out toward jabalpur's bearing -- which
# worked only because `map_geo_event_to_assets` was fed `cone.cone` while
# `p_cut` was computed from `center`/`width_km`, two independently
# controllable geometries.
#
# The 2026-09-01 hazard-footprint seam fix removes that degree of freedom on
# purpose: the risk group is now `damage_footprint(center, width_km,
# damage_radius_km)`, a DISC, and it never looks at `cone.cone` at all. The
# keyhole cannot come back -- and no disc can reproduce it, because satna is
# the common near endpoint of the jhansi and jabalpur spans, so any centre
# off those two bearings is exactly equidistant from both (verified live:
# identical offsets to the tenth of a km, and the 2026-08-30 radius sweep
# already reported both flipping together at every radius from 20 to 125km).
#
# So this fixture uses the one shape that leaves an escape route, the same
# one SMOKE uses -- a disc on the satna<->rewa corridor that stops short of
# satna -- but pushed further out and made wider, which is what keeps this
# fixture distinct: centre (24.48333, 81.76667), exactly two span lengths
# along satna->rewa, width_km 120, damage_radius_km 25. Measured against the
# real topology (geo_mapper.load_edges + Shapely): W = 60, R = 85; offset to
# satna<->rewa 47.6 km (inside the half-width, so the service is exposed and
# the forecast-blind timing rule fires, p_cut 0.1143); offset to the jhansi
# and jabalpur spans 95.1 km each (10.1 km outside R, so the group is
# {satna<->rewa} alone and jabalpur stays open as the egress). After the
# commit the service rides satna->jabalpur->...->fatehpur->allahabad, whose
# nearest aerial span sits at 95.1 km: offset_km moves 47.6 -> 95.1 and
# p_cut 0.1143 -> 0.0196 -- still nonzero, which is the point. The wider cone
# is what keeps that residual reading measurable rather than a floor at 0.
#
# As before, a plain ForecastBlindBaseline("immediate") never recovers from
# the resulting disjointness_collapse on jhansi<->allahabad (it hits the cap
# with zero commits -- allahabad's three accesses are all BURIED, so no storm
# risk group can ever name the protection corridor's own segment), so this
# fixture stays reserved for tests that pair it with
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
    damage_radius_km: 25
    reference_avoid: {}
    forecast:
      t0:
        t1: {cone: {type: Polygon, coordinates: [[[81.76667, 25.02292], [81.92013, 25.00454], [82.06312, 24.95063], [82.18592, 24.86488], [82.28014, 24.75313], [82.33937, 24.62299], [82.35958, 24.48333], [82.33937, 24.34367], [82.28014, 24.21353], [82.18592, 24.10178], [82.06312, 24.01603], [81.92013, 23.96212], [81.76667, 23.94374], [81.61321, 23.96212], [81.47022, 24.01603], [81.34742, 24.10178], [81.2532, 24.21353], [81.19397, 24.34367], [81.17376, 24.48333], [81.19397, 24.62299], [81.2532, 24.75313], [81.34742, 24.86488], [81.47022, 24.95063], [81.61321, 25.00454], [81.76667, 25.02292]]]}, width_km: 120, center: {lat: 24.48333, lon: 81.76667}}
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
      cone_width_km: 120
      cone_motion_kmh: 20
      n_future_claimants: 0
      exposure_horizon_hours: 1
      spares_on_hand: 2
      claimant_services: []
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


def test_action_payloads_carry_origin_and_service():
    a = runner.Action(hour="t1", hour_index=1, lever="optical_reroute",
                      effective_at_index=3, spares={"x": 1}, service_id="s",
                      origin="harness")
    (p,) = runner.action_payloads([a], hours=("t0", "t1", "t2", "t3"))
    assert p["origin"] == "harness" and p["service_id"] == "s"


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

    async def timing(self, obs):
        self.observations.append(obs)
        return await self.inner.timing(obs)

    async def constraints(self, obs, unconstrained_menu=None):
        self.probes.append(unconstrained_menu)
        return await self.inner.constraints(obs, unconstrained_menu)

    async def objective(self, obs, menu):
        return await self.inner.objective(obs, menu)


class _ProbingDecider:
    """Wraps a decider; at every timing call probes the first non-SUT service
    shown, under the latest horizon's group, through whatever the runner
    bound. Proves the binding is live during `timing` and torn down after
    the hour."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.answers = []
        self.bound = []

    def bind_probe(self, probe):
        self.bound.append(probe)

    async def timing(self, obs):
        probe = self.bound[-1]
        others = [s for s in obs.exposure if s != obs.service_under_test]
        if others and obs.risk_group_ids:
            horizon = list(obs.risk_group_ids)[-1]
            self.answers.append(await probe(others[0], obs.risk_group_ids[horizon]))
        return await self.inner.timing(obs)

    async def constraints(self, obs, unconstrained_menu=None):
        return await self.inner.constraints(obs, unconstrained_menu)

    async def objective(self, obs, menu):
        return await self.inner.objective(obs, menu)


def test_the_runner_binds_a_probe_per_hour_and_records_every_call(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    decider = _ProbingDecider(ScriptedDecider("hold"))
    trace = asyncio.run(_run(
        _scenario(tmp_path), decider,
        loaded_state_path, local_server_command, local_server_env))
    # Bound once per hour, unbound (None) after each: [b, None, b, None, ...]
    assert decider.bound[-1] is None
    assert len([b for b in decider.bound if b is not None]) == len(trace.hours)
    probed_hours = [h for h in trace.hours if h.get("probes")]
    assert probed_hours, "no hour showed a second exposed service to probe"
    record = probed_hours[0]["probes"][0]
    assert record["decision"] == "timing"
    assert record["error"] is None
    assert set(record["answer"]) == {"status", "full_restore_candidates",
                                     "min_spares_needed_by_site", "levers"}
    assert record["answer"] == decider.answers[0]
    # The probe's route_service call is counted like every other tool call.
    assert trace.tool_calls > 0


def test_the_constraints_step_is_shown_the_menu_it_is_about_to_narrow(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # F3 (remediation spec lines 96-112). decisions.py: decision 2 changes
    # which candidates EXIST. It was made blind to what constraining costs.
    # Paired with the widening decider because a plain ForecastBlindBaseline
    # can no longer commit on SMOKE at all (see SMOKE's own note: avoiding the
    # exposed working span funnels every cheap reroute through the service's
    # own protection corridor), and this test needs a committed hour to read
    # `unconstrained_menu` back off the trace.
    decider = _RecordingDecider(_WidensOnDisjointnessRejection())
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
    # SMOKE acts at t0 only (after the t0 reroute storm-svc-1's nearest aerial
    # span is 76.1km out, past the 45km half-width, so t1 waits), and t0 runs
    # TWO iterations -- the first-try disjointness_collapse and the widened
    # retry -- which is exactly the case this check exists for: same-object
    # reuse WITHIN one hour's iterations, walked per hour rather than across
    # the whole episode so a second acting hour would not be assumed away.
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
    """Wraps a ForecastBlindBaseline variant -- same timing/objective -- but
    reacts to a genuine validate_plan disjointness_collapse rejection by
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
    that validates on a later iteration --
    satna->jabalpur->...->fatehpur->allahabad, the only way into allahabad
    that touches neither the avoided working span nor the protection leg
    (allahabad's other two accesses are rewa, a stub behind satna<->rewa, and
    jhansi, protection's own).

    Since the 2026-09-01 hazard-footprint seam fix this is the ONLY decider in
    this file that can commit against SMOKE or EXPOSURE_SMOKE while actually
    honouring the exposed risk group: that group now always names the exposed
    working span, so every reroute is real and every first try collides with
    protection. (The scripted deciders still commit, but they route under
    avoid={} and so never face the collision.) `variant` is forwarded to the
    wrapped baseline so a test about at_deadline timing still gets a real
    committed action to assert on."""

    def __init__(self, variant: str = "immediate") -> None:
        self._inner = ForecastBlindBaseline(variant)
        self.name = "widens-on-disjointness-rejection"

    async def timing(self, obs):
        return await self._inner.timing(obs)

    async def constraints(self, obs, unconstrained_menu=None):
        base = await self._inner.constraints(obs, unconstrained_menu)
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

    async def objective(self, obs, menu):
        return await self._inner.objective(obs, menu)


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
    # _WidensOnDisjointnessRejection instead. EXPOSURE_SMOKE rather than the
    # shared SMOKE fixture because this test needs BOTH readings to be real
    # numbers off the real corridor: its wider, further-out cone leaves the
    # post-commit path measurably (not vanishingly) exposed, so p_cut
    # 0.1143 -> 0.0196 is a measurement rather than a collapse to the floor.
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
    # The baseline avoids the exposed risk group, so the committed path's
    # WORKING leg genuinely moves away from the cone -- its own offset more
    # than doubles (47.6 -> 95.1 km) and it exits the damage footprint
    # outright. `in_footprint` (Task 4's per-leg categorical fact) is the
    # cleanest way to see that: it flips True -> False on exactly the leg
    # that moved.
    assert after["offset_km"] > before["offset_km"]
    assert before["legs"]["working"]["in_footprint"] is True
    assert after["legs"]["working"]["in_footprint"] is False
    # `p_cut` is the JOINT (both-legs-cut) probability (Task 4), not the
    # working leg's own, and it does NOT reliably fall here -- verified NOT
    # to be a fixture accident (Task 4 investigation, 2026-09-05): satna's
    # three real aerial spans fan out from ONE shared node, with rewa at
    # bearing 96.7 deg from satna, jhansi (protection, fixed) at 292.6 deg,
    # and jabalpur (the only route disjoint from protection) at 209.6 deg --
    # 83 deg from jhansi versus rewa's 164 deg. Moving the working leg off
    # rewa and onto jabalpur therefore moves it CLOSER in bearing to the
    # fixed protection leg, which can raise the Gaussian mass in the
    # intersection of their two buffered regions even as the working leg's
    # own far endpoint moves twice as far from the storm. A grid search over
    # 9791 valid (cone centre x width x damage_radius) combinations that
    # still expose rewa while excluding both jhansi and jabalpur from the
    # risk group found no configuration with a real, unambiguous decrease --
    # the best margin found was +6e-5, below this model's own ~2.56e-4
    # sampling-noise floor. So this is a structural property of the joint
    # metric under this real topology, not something a different
    # EXPOSURE_SMOKE cone could fix. `p_cut` still changing at all (rather
    # than being reused stale) is exactly what the offset/in_footprint
    # assertions above already prove.


def test_a_baseline_rollout_completes_and_records_every_hour(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # Still a forecast-blind baseline (same timing rule, same fixed objective
    # ordering); the widening wrapper is what lets it reach a terminal state
    # other than the retry cap on SMOKE now that the risk group really names
    # the exposed working span. Keeping the plain baseline here would make the
    # terminal_status assertion below accept every status there is, which is
    # not a check any more.
    trace = asyncio.run(_run(
        _scenario(tmp_path), _WidensOnDisjointnessRejection(),
        loaded_state_path, local_server_command, local_server_env))
    assert [h["hour"] for h in trace.hours] == ["t0", "t1"]
    assert trace.terminal_status in {"converged", "declared_infeasible"}
    assert trace.tool_calls > 0
    assert all("timing" in h for h in trace.hours)


def test_committed_steps_and_hours_carry_inert_and_timing_effective(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # Task 6: the harness could not previously tell a real commit (one that
    # moves the working path or spends a spare) from a hollow one (a
    # candidate validated and committed that changes neither) -- a free
    # inert reroute must never count as "the agent acted". Every committed
    # step now carries `inert`, and every hour (act or wait) carries
    # `timing_effective` derived from whether any NON-inert commit landed.
    trace = asyncio.run(_run(
        _scenario(tmp_path), _WidensOnDisjointnessRejection(),
        loaded_state_path, local_server_command, local_server_env))
    assert trace.hours, "nothing to check the new fields against"
    for hour_record in trace.hours:
        assert hour_record["timing_effective"] in {"act", "wait"}
        for step in hour_record.get("iterations", []):
            if step.get("outcome") == "committed":
                assert "inert" in step


def test_the_next_hour_is_told_what_was_committed_in_the_previous_one(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # F4's no-memory half (remediation spec lines 122-132). The decider is
    # asked to decide again at t1 with no hint that it already acted at t0
    # unless the harness tells it. Widening wrapper because SMOKE's t0 commit
    # only happens after the disjointness retry (see SMOKE's own note); the
    # memory being asserted is the harness's, not the decider's.
    decider = _RecordingDecider(_WidensOnDisjointnessRejection())
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
        _scenario(tmp_path), _WidensOnDisjointnessRejection(),
        loaded_state_path, local_server_command, local_server_env))
    assert trace.actions, (
        "the immediate baseline, widened past the disjointness collapse, "
        "should have acted at t0")
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

    async def timing(self, obs):
        return TimingDecision("act", "act: exposed at the next horizon")

    async def constraints(self, obs, unconstrained_menu=None):
        self.rejections_seen.append(obs.last_rejection)
        if obs.iteration == 0:
            return ConstraintDecision(
                avoid={"risk_groups": sorted(obs.risk_group_ids.values())},
                reasoning="tight: avoid every forecast risk group")
        return ConstraintDecision(
            avoid={}, reasoning="loosened after declaring the menu infeasible")

    async def objective(self, obs, menu):
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
    # The at_deadline variant acts when the hours-to-exposure meets an
    # optical_reroute's lead time, so an optical_reroute committed there
    # lands strictly LATER than the hour it was decided in. This is scored in
    # Task 10; here we only assert the trace carries the arithmetic to score
    # it -- so the run has to produce an action, which on SMOKE means going
    # through the widening retry (see SMOKE's own note).
    scenario = _scenario(tmp_path)
    trace = asyncio.run(_run(
        scenario, _WidensOnDisjointnessRejection("at_deadline"),
        loaded_state_path, local_server_command, local_server_env))
    assert trace.actions, "nothing to check the lead-time arithmetic against"
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
    # A topology with no edges at all means no mount_type is known for any
    # leg -- cuttable_spans must come back empty rather than raise, and every
    # service still gets a (possibly empty) entry.
    geo = asyncio.run(runner.service_geometry(
        None, _write_geometry_topology_without_edges(tmp_path),
        call=_geometry_call([])))
    assert geo.cuttable_spans["storm-svc-1"] == ()


def test_service_geometry_maps_lightpath_ids_to_their_oms_sequence(tmp_path):
    # menu_with_path_facts resolves a candidate's `reused_lightpaths` ids
    # through this map -- it is oms_seq_by_lp (already built and discarded
    # every hour), just exposed rather than thrown away.
    geo = asyncio.run(runner.service_geometry(
        None, _write_geometry_topology(tmp_path),
        call=_geometry_call([])))
    assert geo.oms_sequences["lp_sr"] == ("oms_sr",)
    assert geo.oms_sequences["lp_sj"] == ("oms_sj", "oms_gap")


def test_service_geometry_maps_working_and_protection_paths_to_oms_ids(
        tmp_path):
    # Unlike `paths` (node ids, deduped, for the viewer's midpoint), this is
    # the ordered OMS-id walk menu_with_path_facts diffs a candidate against
    # -- one entry per leg, not deduped, since two legs never share an OMS.
    geo = asyncio.run(runner.service_geometry(
        None, _write_geometry_topology(tmp_path),
        call=_geometry_call([])))
    assert geo.path_oms["storm-svc-1"]["working"] == ("oms_sr", "oms_ra")
    assert geo.path_oms["storm-svc-1"]["protection"] == ("oms_sj", "oms_gap")


def test_service_geometry_reports_no_cuttable_span_by_oms_without_local_edges(
        tmp_path):
    # Same rule as cuttable_spans above, keyed by OMS instead of by service.
    geo = asyncio.run(runner.service_geometry(
        None, _write_geometry_topology_without_edges(tmp_path),
        call=_geometry_call([])))
    assert geo.cuttable_span_by_oms == {}


def test_service_geometry_exposes_protection_cuttable_spans(tmp_path):
    # storm-svc-1's protection path (satna -> jhansi -> nowhere) rides
    # oms_sj (satna<->jhansi, AERIAL) then oms_gap (satna<->nowhere,
    # unmappable) -- exactly one protection-leg span is storm-cuttable.
    geo = asyncio.run(runner.service_geometry(
        None, _write_geometry_topology(tmp_path), call=_geometry_call([])))
    assert "storm-svc-1" in geo.protection_cuttable_spans
    assert len(geo.protection_cuttable_spans["storm-svc-1"]) == 1


def test_service_geometry_indexes_cuttable_spans_by_oms_id(
        loaded_state_path, local_server_command, local_server_env):
    # storm-svc-1's working path is oms_satna_rewa (AERIAL) then
    # oms_rewa_allahabad (buried) -- the same fact
    # test_service_geometry_keeps_only_the_storm_cuttable_spans establishes
    # per-service. This is the SAME edge_mount/filter_fn walk, keyed by OMS
    # id instead, so a span menu_with_path_facts scores for a candidate and
    # a span the exposure row scores for the service can never disagree.
    geometry = asyncio.run(_geometry(
        loaded_state_path, local_server_command, local_server_env))
    working_oms = geometry.path_oms["storm-svc-1"]["working"]
    assert len(working_oms) == 2
    assert working_oms[0] in geometry.cuttable_span_by_oms
    assert working_oms[1] not in geometry.cuttable_span_by_oms
    assert geometry.cuttable_span_by_oms[working_oms[0]] == (
        geometry.cuttable_spans["storm-svc-1"][0])


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


# T1 inert-reroute finding (2026-08-31-t1-inert-reroute-finding.md): a
# candidate reusing storm-svc-1's OWN working lightpath survived `avoid` and
# out-scored a genuinely different corridor by a 1-count services_at_risk
# margin, because nothing in the menu said the two candidates differed in
# whether they moved the service at all. `_path_fact_geometry` mirrors that
# exact shape -- one cuttable leg (satna<->rewa) on the working path, one
# buried leg beyond it, and a protection path riding a disjoint corridor
# entirely (satna<->jhansi<->allahabad) -- so candidate_0 (reuse working) and
# candidate_2 (reuse protection) in the real trace map onto the two
# candidates these tests build.
def _path_fact_geometry():
    return runner.ServiceGeometry(
        points={}, paths={}, oms_nodes={}, unmapped_nodes={},
        cuttable_spans={}, endpoint_sites={},
        oms_sequences={
            "lp-cand-storm-svc-1-0": ("oms_satna_rewa", "oms_rewa_allahabad"),
            "lp-prot-storm-svc-1-0": ("oms_satna_jhansi",
                                     "oms_jhansi_allahabad")},
        cuttable_span_by_oms={
            "oms_satna_rewa": ((24.6, 80.8), (24.5, 81.3))},
        path_oms={"storm-svc-1": {
            "working": ("oms_satna_rewa", "oms_rewa_allahabad"),
            "protection": ("oms_satna_jhansi", "oms_jhansi_allahabad")}})


def _path_fact_issuance():
    # Cone centred exactly on the cuttable span's own endpoint (offset 0) at
    # a width/damage-radius generous enough that p_cut is unambiguously
    # large -- the test only needs "clearly exposed" vs "clearly not",
    # not a pinned scalar (that contract belongs to test_exposure_model.py).
    return Issuance(issued_at="t0", horizons={"t1": ConeAtHorizon(
        cone={"type": "Polygon", "coordinates": []}, width_km=90.0,
        center={"lat": 24.6, "lon": 80.8})})


def test_reusing_the_own_working_lightpath_reports_an_unchanged_path():
    menu = {"status": "solution", "candidates": [
        {"lever": "ip_reroute",
         "reused_lightpaths": ["lp-cand-storm-svc-1-0"], "new_lightpaths": []}]}
    out = runner.menu_with_path_facts(
        menu, _path_fact_geometry(), "storm-svc-1",
        issuance=_path_fact_issuance(), damage_radius_km=50.0,
        demand_gbps=300.0)
    candidate = out["candidates"][0]
    assert candidate["path_delta"] == {
        "changes_working_path": False, "oms_added": [], "oms_removed": [],
        "oms_retained_cuttable": ["oms_satna_rewa"]}
    assert candidate["residual_exposure"]["t1"]["p_cut"] > 0.5
    assert candidate["collides_with_protection"] == {
        "collides": False, "oms_shared_with_protection": []}


def test_reusing_the_protection_lightpath_reports_a_changed_path_and_a_collision():
    menu = {"status": "solution", "candidates": [
        {"lever": "ip_reroute",
         "reused_lightpaths": ["lp-prot-storm-svc-1-0"], "new_lightpaths": []}]}
    out = runner.menu_with_path_facts(
        menu, _path_fact_geometry(), "storm-svc-1",
        issuance=_path_fact_issuance(), damage_radius_km=50.0,
        demand_gbps=300.0)
    candidate = out["candidates"][0]
    assert candidate["path_delta"] == {
        "changes_working_path": True,
        "oms_added": ["oms_jhansi_allahabad", "oms_satna_jhansi"],
        "oms_removed": ["oms_rewa_allahabad", "oms_satna_rewa"],
        "oms_retained_cuttable": []}
    assert candidate["residual_exposure"]["t1"]["p_cut"] == 0.0
    assert candidate["collides_with_protection"] == {
        "collides": True,
        "oms_shared_with_protection": ["oms_jhansi_allahabad",
                                      "oms_satna_jhansi"]}


def test_a_reused_lightpath_with_no_known_oms_sequence_surfaces_unresolved():
    # An id menu_with_path_facts cannot look up must be reported, not
    # silently treated as zero exposure -- same policy as
    # ServiceGeometry.unmapped_nodes.
    menu = {"status": "solution", "candidates": [
        {"lever": "ip_reroute", "reused_lightpaths": ["lp-unknown"],
         "new_lightpaths": []}]}
    out = runner.menu_with_path_facts(
        menu, _path_fact_geometry(), "storm-svc-1",
        issuance=_path_fact_issuance(), damage_radius_km=50.0,
        demand_gbps=300.0)
    candidate = out["candidates"][0]
    assert candidate["residual_exposure_unresolved"] == ["lp-unknown"]
    assert candidate["residual_exposure"]["t1"]["p_cut"] == 0.0


def test_a_new_lightpath_only_candidate_resolves_from_its_own_oms_sequence():
    menu = {"status": "solution", "candidates": [
        {"lever": "optical_reroute", "reused_lightpaths": [],
         "new_lightpaths": [{"oms_sequence": ["oms_satna_rewa"], "lam": 0,
                             "mode_id": "m", "gsnr_db": 1.0,
                             "bitrate_gbps": 1.0}]}]}
    out = runner.menu_with_path_facts(
        menu, _path_fact_geometry(), "storm-svc-1",
        issuance=_path_fact_issuance(), damage_radius_km=50.0,
        demand_gbps=300.0)
    candidate = out["candidates"][0]
    assert candidate["path_delta"]["oms_retained_cuttable"] == [
        "oms_satna_rewa"]
    assert candidate["residual_exposure"]["t1"]["p_cut"] > 0.5


def test_menu_with_path_facts_does_not_mutate_the_input_menu():
    menu = {"status": "solution", "candidates": [
        {"lever": "ip_reroute", "reused_lightpaths": ["lp-cand-storm-svc-1-0"],
         "new_lightpaths": []}]}
    runner.menu_with_path_facts(
        menu, _path_fact_geometry(), "storm-svc-1",
        issuance=_path_fact_issuance(), damage_radius_km=50.0,
        demand_gbps=300.0)
    assert menu["candidates"][0] == {
        "lever": "ip_reroute", "reused_lightpaths": ["lp-cand-storm-svc-1-0"],
        "new_lightpaths": []}


# Pinning this is what test_recording_costs_no_extra_server_calls below is
# for -- recording the hour's observation/menu/geometry must add zero calls
# to the already-counted call sites.
#
# Originally 22, measured against SMOKE + ForecastBlindBaseline("immediate")
# BEFORE task A4's recording changes landed (task-A4-brief.md, Step 1).
# Re-measured 2026-09-01 for the hazard-footprint seam fix, which changed
# both SMOKE's geometry and the decider this test can use: the property under
# test is untouched, only the rollout it is measured over moved. Written out
# so it is derivable rather than magic --
#
#   every hour:  4 (service_geometry's get_topology x2 / get_lightpaths /
#                   get_services) + 1 (the hour's own get_services)
#                + 1 (get_topology for the risk groups)          = 6
#   t0:          + 1 define_risk_group (one issuance, one horizon)
#                + 1 unconstrained probe route_service
#                + 2 iteration 0 (route_service, validate_plan -> rejected)
#                + 3 iteration 1 (route_service, validate_plan, commit_plan)
#                                                                 = 13
#   t1:          waits -- the t0 reroute put storm-svc-1 outside the cone   6
#   episode end: + 1 simulate_ip_routing                                    1
#                                                                    total 20
EXPECTED_TOOL_CALLS_SMOKE_ROLLOUT = 20


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


def test_the_trace_records_path_facts_on_every_candidate_decision_3_sees(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # menu_with_path_facts must run against the SAME menu the decider's own
    # objective() call and the trace's recorded menu are built from -- one
    # annotation, both readers, per the T1 inert-reroute finding.
    trace = asyncio.run(_run_episode_with(
        _scripted_acting_decider(), scenario=_scenario(tmp_path),
        state_path=loaded_state_path, server_command=local_server_command,
        server_env=local_server_env))
    acting = next(h for h in trace.hours if h.get("committed") is not None)
    candidate = acting["iterations"][0]["menu"]["candidates"][0]
    assert "path_delta" in candidate
    assert "residual_exposure" in candidate
    assert "collides_with_protection" in candidate
    # unconstrained_menu is decision 2's deliberately cost-blind projection --
    # these facts are exactly the kind of judgement-relevant content F3 kept
    # out of it, so they must not leak in.
    assert "path_delta" not in acting["unconstrained_menu"]["candidates"][0]


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
    # quietly invalidate a comparison if it failed. The widening decider so
    # the counted rollout covers the commit path too, not just an hour that
    # spins on rejections.
    trace = asyncio.run(_run_episode_with(
        _WidensOnDisjointnessRejection(), scenario=_scenario(tmp_path),
        state_path=loaded_state_path, server_command=local_server_command,
        server_env=local_server_env))
    assert trace.tool_calls == EXPECTED_TOOL_CALLS_SMOKE_ROLLOUT


def test_the_risk_group_walk_names_fibers_of_filtered_edges_only():
    """horizon_risk_group_asset_ids resolves the edges a hazard geometry
    touches to the FIBER element ids of the OMS between the same two nodes,
    in either node order, and drops non-fiber elements."""
    from shapely.geometry import LineString

    from storm_reoptimizer.eval.runner import horizon_risk_group_asset_ids
    from storm_reoptimizer.eval.scenario_file import ConeAtHorizon
    from storm_reoptimizer.events.geo import circle_polygon
    from storm_reoptimizer.geo_mapper import Edge

    inside = Edge(src="a", dst="b", mount_type="aerial",
                  geometry=LineString([(81.0, 25.0), (81.1, 25.0)]))
    buried = Edge(src="a", dst="c", mount_type="buried",
                  geometry=LineString([(81.0, 25.0), (81.1, 25.05)]))
    far = Edge(src="d", dst="e", mount_type="aerial",
               geometry=LineString([(90.0, 25.0), (90.1, 25.0)]))
    oms = [
        # dst/src reversed relative to the Edge, to pin the both-orders walk.
        {"id": "oms_ab", "src_node_id": "b", "dst_node_id": "a",
         "elements": ["fiber_a_b_0", "edfa_a_b_0"]},
        {"id": "oms_ac", "src_node_id": "a", "dst_node_id": "c",
         "elements": ["fiber_a_c_0"]},
        {"id": "oms_de", "src_node_id": "d", "dst_node_id": "e",
         "elements": ["fiber_d_e_0"]},
    ]
    cone = ConeAtHorizon(cone=circle_polygon(25.0, 81.05, 30.0),
                         width_km=60.0, center={"lat": 25.0, "lon": 81.05})

    assert horizon_risk_group_asset_ids(
        cone, 0.1, edges=[inside, buried, far], oms=oms,
        filter_fn=lambda e: e.mount_type == "aerial") == ["fiber_a_b_0"]


# Same cone/geometry as EXPOSURE_SMOKE (see its own long comment above for
# why this exact disc: it names risk group {satna<->rewa} alone, leaving
# satna<->jabalpur -- and so the satna->jabalpur->...->fatehpur->allahabad
# escape route -- open), stretched to four hours so a realized cut at t1
# leaves t2/t3 to show nothing further happens. `claimant_services: []`:
# this test is about the replay mechanism, not claimant contention.
_REPLAY_SCENARIO_TEMPLATE = textwrap.dedent("""
    id: {id}
    seed: 17
    state_file: eval/states/loaded-s17.json
    service_under_test: storm-svc-1
    track: hudhud
    hours: [t0, t1, t2, t3]
    decision_hour: t0
    lead_time_hours: 1
    spares_on_hand: 2
    depot_site: satna
    spare_inventory: {{satna: 2}}
    damage_radius_km: 25
    reference_avoid: {{}}
    forecast:
      t0:
        t1: {{cone: {{type: Polygon, coordinates: [[[81.76667, 25.02292], [81.92013, 25.00454], [82.06312, 24.95063], [82.18592, 24.86488], [82.28014, 24.75313], [82.33937, 24.62299], [82.35958, 24.48333], [82.33937, 24.34367], [82.28014, 24.21353], [82.18592, 24.10178], [82.06312, 24.01603], [81.92013, 23.96212], [81.76667, 23.94374], [81.61321, 23.96212], [81.47022, 24.01603], [81.34742, 24.10178], [81.2532, 24.21353], [81.19397, 24.34367], [81.17376, 24.48333], [81.19397, 24.62299], [81.2532, 24.75313], [81.34742, 24.86488], [81.47022, 24.95063], [81.61321, 25.00454], [81.76667, 25.02292]]]}}, width_km: 120, center: {{lat: 24.48333, lon: 81.76667}}}}
    realized:
      t1: [{cuts}]
    gold:
      survived: [storm-svc-1]
      max_spares_wasted: 2
      decision_at_t0: wait
      label: wait
      rationale: replay live test fixture; not scored
    flip_variable: [smoke]
    metadata:
      cone_width_km: 120
      cone_motion_kmh: 20
      n_future_claimants: 0
      exposure_horizon_hours: 1
      spares_on_hand: 2
      claimant_services: []
""")


def _replay_scenario(tmp_path, scenario_id: str, cuts: list[str]):
    path = tmp_path / f"{scenario_id}.yaml"
    path.write_text(_REPLAY_SCENARIO_TEMPLATE.format(
        id=scenario_id, cuts=", ".join(cuts)), encoding="utf-8")
    return load_scenario(path)


def test_post_cut_restoration_replay(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    # Fiber ids are server-minted (per-state, deterministic for a given seed
    # and topology, not typed by hand elsewhere) -- confirm both are real
    # OMS elements on THIS state before trusting them in `realized` below.
    async def _optical_topology():
        async with connect_server(
            TOPOLOGY_PATH, server_command=local_server_command,
            env=local_server_env,
            extra_args=["--state", str(loaded_state_path)],
        ) as client:
            return await call_tool_json(client, "get_topology", {"layer": "optical"})

    optical = asyncio.run(_optical_topology())
    elements = {e for o in optical["oms"] for e in o["elements"]}
    assert "fiber_satna_rewa_0" in elements
    assert "fiber_satna_jhansi_0" in elements

    # Cutting only the WORKING leg: storm-svc-1's own 1:1 protection absorbs
    # it, so simulate_ip_routing never reports IT dropped, and it is the only
    # scope member this scenario has (claimant_services: []) -- so the
    # replay is never attempted for anything, even though the SAME real
    # fibre carries other, unrelated, unprotected services that DO go down
    # collaterally (this fixture makes no claim about them; only storm-svc-1
    # and the replay's own scope are this test's business).
    working_only = _replay_scenario(
        tmp_path, "REPLAY_WORKING_ONLY", ["fiber_satna_rewa_0"])
    trace = asyncio.run(_run(
        working_only, ScriptedDecider("hold"),
        loaded_state_path, local_server_command, local_server_env))
    assert isinstance(trace.hours[1]["dropped_after_cut"], list)
    assert "storm-svc-1" not in trace.hours[1]["dropped_after_cut"]
    assert trace.restorations == ()

    # Cutting BOTH legs: the service actually goes down, and the harness's
    # own deterministic replay -- not the decider, which only ever waits --
    # is what gets a chance to restore it.
    both_legs = _replay_scenario(
        tmp_path, "REPLAY_BOTH_LEGS",
        ["fiber_satna_rewa_0", "fiber_satna_jhansi_0"])
    trace2 = asyncio.run(_run(
        both_legs, ScriptedDecider("hold"),
        loaded_state_path, local_server_command, local_server_env))
    assert "storm-svc-1" in trace2.hours[1]["dropped_after_cut"]
    assert trace2.restorations, "expected the replay to attempt storm-svc-1"
    outcome = trace2.restorations[0]["outcome"]
    # Measured live: the menu's first structurally-workable candidate
    # (shortfall 0, real path change, affordable) is satna->jhansi->allahabad
    # -- storm-svc-1's OWN now-degraded protection corridor, cheapest because
    # it is already provisioned -- which validate_plan genuinely rejects
    # (disjointness_collapse against storm-svc-1's own still-standing
    # protection assets, and mode_infeasible on the dark leg). Unlike the
    # hourly decision loop, this replay never retries a rejection with a
    # widened avoid set (the spec states a single pick-and-commit, not a
    # capped retry loop), so "rejected" is a real, expected outcome here --
    # not a bug in either the candidate selection or try_commit.
    assert outcome in {"restored", "no_candidate", "rejected"}, (
        f"expected the live escape-route menu to restore storm-svc-1, "
        f"report no workable candidate, or have its one pick rejected; "
        f"got {outcome!r} (record: {trace2.restorations[0]!r})")
