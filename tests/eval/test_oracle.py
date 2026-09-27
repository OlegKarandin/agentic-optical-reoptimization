"""The scripted oracle deciders (T1 spend-or-hold redesign, Task 9). Pure:
no server. `escape_objective` is exercised directly against hand-built,
already-annotated menus (the exact shape `runner.menu_with_path_facts` /
`menu_for_prompt` produce -- see test_runner.py/test_agent.py for real
examples); `spend_decider`/`hold_decider` are exercised through their public
`Decider` interface (`.timing`/`.constraints`), never through their private
attributes, mirroring test_baseline.py's own style."""
import asyncio
import textwrap

from storm_reoptimizer.eval.oracle import escape_objective, hold_decider, spend_decider
from storm_reoptimizer.eval.observation import Observation
from storm_reoptimizer.eval.scenario_file import ConeAtHorizon, Issuance, load_scenario


def _run(coro):
    return asyncio.run(coro)


ORACLE_SCENARIO_YAML = textwrap.dedent("""
    id: ORACLE_TEST
    seed: 1
    state_file: eval/states/loaded-s17.json
    service_under_test: storm-svc-1
    track: hudhud
    hours: [t0, t1, t2, t3]
    decision_hour: t1
    lead_time_hours: 1
    spares_on_hand: 1
    depot_site: satna
    spare_inventory: {satna: 1}
    damage_radius_km: 74
    track_revision_km_per_hour_ahead: 30
    reference_avoid: {}
    forecast:
      t0:
        t3: {cone: {type: Polygon, coordinates: []}, width_km: 90, center: {lat: 25.0, lon: 81.0}}
      t1:
        t2: {cone: {type: Polygon, coordinates: []}, width_km: 90, center: {lat: 25.1, lon: 81.0}}
        t3: {cone: {type: Polygon, coordinates: []}, width_km: 90, center: {lat: 25.2, lon: 81.0}}
    realized: {}
    gold:
      survived: [storm-svc-1]
      max_spares_wasted: 0
      decision_at_t0: wait
      label: wait
      rationale: oracle test fixture; not scored
    flip_variable: [test]
    metadata:
      cone_width_km: 90
      cone_motion_kmh: 20
      n_future_claimants: 2
      exposure_horizon_hours: 2
      spares_on_hand: 1
      claimant_services: [claim-a, claim-b]
      claimed_competing_ecar_gbps: 50.0
      claimed_competing_ecar_at: t3
""")


def _scenario(write_scenario):
    return load_scenario(write_scenario(ORACLE_SCENARIO_YAML, "ORACLE_TEST.yaml"))


def _dummy_obs(hour: str) -> Observation:
    """Only `.hour` is read by `ScriptedDecider.timing`/`.constraints` --
    every other field is a structurally-valid placeholder."""
    return Observation(
        scenario_id="ORACLE_TEST", service_under_test="storm-svc-1", hour=hour,
        hour_index=0, hours_remaining=0,
        issuance=Issuance(issued_at=hour, horizons={}),
        exposure={}, services=(), spares_on_hand=1, lead_time_hours=1)


# -- escape_objective ---------------------------------------------------

def _obs_with_horizon(horizon: str = "t3") -> Observation:
    return Observation(
        scenario_id="X", service_under_test="storm-svc-1", hour="t1",
        hour_index=1, hours_remaining=2,
        issuance=Issuance(issued_at="t1", horizons={horizon: ConeAtHorizon(
            cone={"type": "Polygon", "coordinates": []}, width_km=90.0,
            center={"lat": 25.0, "lon": 81.0})}),
        exposure={}, services=(), spares_on_hand=1, lead_time_hours=1)


def _candidate(*, changes_path, collides, spares_needed, shortfall, p_cut,
              horizon="t3"):
    return {
        "path_delta": {"changes_working_path": changes_path, "oms_added": [],
                       "oms_removed": [], "oms_retained_cuttable": []},
        "collides_with_protection": {"collides": collides,
                                     "oms_shared_with_protection": []},
        "spares_needed": spares_needed,
        "shortfall_gbps": shortfall,
        "residual_exposure": {horizon: {"p_cut": p_cut, "ecar_gbps": 1.0}},
    }


# The exact three-candidate scenario the task brief specifies.
INERT_REUSE = _candidate(changes_path=False, collides=False, spares_needed={},
                         shortfall=0.0, p_cut=0.5)
PROTECTION_COLLIDING = _candidate(
    changes_path=True, collides=True, spares_needed={"satna": 1, "jhansi": 1},
    shortfall=0.0, p_cut=0.05)
CLEAN_ESCAPE = _candidate(
    changes_path=True, collides=False, spares_needed={"d": 1, "far": 1},
    shortfall=0.0, p_cut=0.01)


def test_escape_objective_picks_the_real_escape():
    menu = {"status": "solution",
           "candidates": [INERT_REUSE, PROTECTION_COLLIDING, CLEAN_ESCAPE]}
    decision = escape_objective(_obs_with_horizon(), menu)
    assert decision.choice == "candidate_2"


def test_escape_objective_is_infeasible_with_no_real_escape():
    menu = {"status": "solution",
           "candidates": [INERT_REUSE, PROTECTION_COLLIDING]}
    decision = escape_objective(_obs_with_horizon(), menu)
    assert decision.choice == "infeasible"


def test_escape_objective_is_infeasible_on_an_empty_menu():
    decision = escape_objective(_obs_with_horizon(), {"candidates": []})
    assert decision.choice == "infeasible"


def test_escape_objective_rejects_a_shortfall_candidate():
    partial = _candidate(changes_path=True, collides=False,
                         spares_needed={"d": 1, "far": 1}, shortfall=50.0,
                         p_cut=0.01)
    decision = escape_objective(_obs_with_horizon(),
                                {"candidates": [partial]})
    assert decision.choice == "infeasible"


def test_escape_objective_rejects_a_zero_spare_candidate():
    """A candidate that changes the working path with NO new lightpath --
    an ip_reroute onto an existing lightpath -- is not a real SPEND, even
    though it clears the other three predicates."""
    zero_spare = _candidate(changes_path=True, collides=False,
                            spares_needed={}, shortfall=0.0, p_cut=0.01)
    decision = escape_objective(_obs_with_horizon(),
                                {"candidates": [zero_spare]})
    assert decision.choice == "infeasible"


def test_escape_objective_picks_the_lowest_residual_among_two_escapes():
    worse = _candidate(changes_path=True, collides=False,
                       spares_needed={"d": 1, "far": 1}, shortfall=0.0,
                       p_cut=0.2)
    better = _candidate(changes_path=True, collides=False,
                        spares_needed={"d": 1, "far": 1}, shortfall=0.0,
                        p_cut=0.05)
    decision = escape_objective(_obs_with_horizon(),
                                {"candidates": [worse, better]})
    assert decision.choice == "candidate_1"


def test_escape_objective_scores_at_the_latest_horizon_not_the_first():
    """Two horizons; the residual at the EARLIER one favours candidate_0, at
    the LATER (the one that matters) it favours candidate_1."""
    obs = Observation(
        scenario_id="X", service_under_test="storm-svc-1", hour="t1",
        hour_index=1, hours_remaining=2,
        issuance=Issuance(issued_at="t1", horizons={
            "t2": ConeAtHorizon(cone={"type": "Polygon", "coordinates": []},
                                width_km=90.0, center={"lat": 25.0, "lon": 81.0}),
            "t3": ConeAtHorizon(cone={"type": "Polygon", "coordinates": []},
                                width_km=90.0, center={"lat": 25.2, "lon": 81.0})}),
        exposure={}, services=(), spares_on_hand=1, lead_time_hours=1)
    candidate_0 = {
        "path_delta": {"changes_working_path": True}, "shortfall_gbps": 0.0,
        "collides_with_protection": {"collides": False},
        "spares_needed": {"d": 1, "far": 1},
        "residual_exposure": {"t2": {"p_cut": 0.01}, "t3": {"p_cut": 0.9}}}
    candidate_1 = {
        "path_delta": {"changes_working_path": True}, "shortfall_gbps": 0.0,
        "collides_with_protection": {"collides": False},
        "spares_needed": {"d": 1, "far": 1},
        "residual_exposure": {"t2": {"p_cut": 0.9}, "t3": {"p_cut": 0.01}}}
    decision = escape_objective(obs, {"candidates": [candidate_0, candidate_1]})
    assert decision.choice == "candidate_1"


# -- spend_decider / hold_decider ----------------------------------------

def test_spend_decider_acts_only_at_the_decision_hour(write_scenario):
    decider = spend_decider(_scenario(write_scenario))
    assert _run(decider.timing(_dummy_obs("t0"))).action == "wait"
    assert _run(decider.timing(_dummy_obs("t1"))).action == "act"
    assert _run(decider.timing(_dummy_obs("t2"))).action == "wait"
    assert _run(decider.timing(_dummy_obs("t3"))).action == "wait"


def test_hold_decider_waits_at_every_hour(write_scenario):
    decider = hold_decider(_scenario(write_scenario))
    for hour in ("t0", "t1", "t2", "t3"):
        assert _run(decider.timing(_dummy_obs(hour))).action == "wait"


def test_spend_decider_claim_priority_puts_the_sut_first(write_scenario):
    decider = spend_decider(_scenario(write_scenario))
    expected = ("storm-svc-1", "claim-a", "claim-b")
    assert _run(decider.timing(_dummy_obs("t1"))).claim_priority == expected
    # The bias is stated the same way even on an hour it waits, since
    # runner.run_episode reads THIS hour's own timing decision for restore
    # ordering, not only the decision-hour one.
    assert _run(decider.timing(_dummy_obs("t0"))).claim_priority == expected


def test_hold_decider_claim_priority_puts_claimants_first(write_scenario):
    decider = hold_decider(_scenario(write_scenario))
    expected = ("claim-a", "claim-b", "storm-svc-1")
    for hour in ("t0", "t1", "t2", "t3"):
        assert _run(decider.timing(_dummy_obs(hour))).claim_priority == expected


def test_hold_decider_states_the_given_ranking(write_scenario):
    d = hold_decider(_scenario(write_scenario), claim_priority=("x", "y"))
    assert d._default_timing.claim_priority == ("x", "y")   # ScriptedDecider stores it privately
    for hour in ("t0", "t1", "t2", "t3"):
        timing = _run(d.timing(_dummy_obs(hour)))
        assert timing.action == "wait"
        assert timing.claim_priority == ("x", "y")


def test_spend_decider_constraints_avoid_the_latest_horizon_risk_group(
        write_scenario):
    decider = spend_decider(_scenario(write_scenario))
    decision = _run(decider.constraints(_dummy_obs("t1")))
    # t1's own issuance publishes t2 AND t3 -- the LATEST (t3) is the one
    # avoided, not the nearer t2.
    assert decision.avoid == {"risk_groups": ["rg_ORACLE_TEST_t1_t3"]}
    assert (decision.protected, decision.best_effort, decision.basis,
           decision.level) == (False, False, "risk_group", "risk_group")


def test_spend_decider_objective_is_the_escape_objective(write_scenario):
    decider = spend_decider(_scenario(write_scenario))
    decider.oms_nodes = {"oms_d": ["satna", "d"], "oms_far": ["d", "far"]}
    menu = {"status": "solution", "candidates": [
        {"lever": "ip_reroute", "reused_lightpaths": [], "new_lightpaths": [],
         "shortfall_gbps": 0.0,
         "path_delta": {"changes_working_path": False},
         "collides_with_protection": {"collides": False},
         "residual_exposure": {"t3": {"p_cut": 0.5}}},
        {"lever": "optical_reroute", "reused_lightpaths": [],
         "new_lightpaths": [{"oms_sequence": ["oms_d", "oms_far"], "lam": 0,
                             "mode_id": "M", "gsnr_db": 10.0,
                             "bitrate_gbps": 300.0}],
         "shortfall_gbps": 0.0,
         "path_delta": {"changes_working_path": True},
         "collides_with_protection": {"collides": False},
         "residual_exposure": {"t3": {"p_cut": 0.01}}},
    ]}
    decision = _run(decider.objective(_obs_with_horizon(), menu))
    assert decision.choice == "candidate_1"
