"""The twin-pair validity checks (eval design spec, "Twin-pair discipline")
and their own tests -- "a deliberately invalid pair (same gold both halves)
must be rejected by assertions.py"."""
import asyncio
import dataclasses
import textwrap
from pathlib import Path

import pytest

from storm_reoptimizer.eval.assertions import (
    PairInvalid, _binding_site_violations, _check_no_free_escape,
    _claimant_depot_violations, _claimant_exposure_violations,
    _jabalpur_optical_reroute_candidate, assert_both_legs_exposed,
    assert_claim_is_one_lightpath, assert_claimants_depot_eligible,
    assert_claimants_have_filterable_exposure, assert_flip_dominates,
    assert_gold_choices_differ, assert_gold_matches_outcomes,
    assert_group_fits_one_lightpath, assert_issuance_prefix_shared,
    assert_no_global_policy_solves_the_suite,
    assert_realized_cuts_pass_the_event_filter, assert_sampling_error_within_margin,
    assert_shared_scalars_equal, assert_spend_is_real, claimant_service_ids,
)
from storm_reoptimizer.eval.derived import FlipScalars
from storm_reoptimizer.eval.scenario_file import (
    SCENARIOS_DIR, load_scenario,
)
from storm_reoptimizer.mcp_client import connect_server

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)

# The PRE-REDESIGN T1 pair (storm-svc-1, depot satna), moved here by Task 15
# when T1 was re-authored on the jalgaon-homed SUT. Kept, and still exercised
# below, because the two negative cases Task 9 established -- a protection leg
# that sits outside the cone, and a service with no real escape at all -- are
# genuine, live-confirmed properties of THAT geometry, and they are the
# evidence that `assert_both_legs_exposed`/`assert_spend_is_real` can actually
# fail. `storm-svc-1` and the satna claimant family are still pinned into
# `eval/states/loaded-s17.json` (D1/T2/T3 need them), so these files still
# load and route against the same live state.
DISCARDED_DIR = Path(__file__).parent / "fixtures" / "discarded"

TWIN = textwrap.dedent("""
    id: {id}
    pair: P
    seed: 17
    state_file: eval/states/loaded-s17.json
    service_under_test: storm-svc-1
    track: hudhud
    hours: [t0, t1, t2, t3]
    decision_hour: t1
    lead_time_hours: 1
    spares_on_hand: 1
    depot_site: satna
    spare_inventory: {{satna: 1}}
    damage_radius_km: 74
    track_revision_km_per_hour_ahead: 30
    reference_avoid: {{risk_groups: [rg_ref]}}
    forecast:
      t0:
        t3: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: {t0_lat}, lon: 81.0}}}}
      t1:
        t3: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: {t1_lat}, lon: 81.0}}}}
    realized:
      # ids denote SPANS: load_scenario adds each fibre's reverse-direction mate
      t3: [fiber_fatehpur_allahabad_0]
    gold:
      survived: [storm-svc-1]
      max_spares_wasted: 0
      decision_at_t0: wait
      label: {label}
      rationale: fixture
    flip_variable: [svc-b]
    metadata:
      label_rule: timing_at_decision_hour
      cone_width_km: 90
      cone_motion_kmh: 20
      n_future_claimants: 1
      exposure_horizon_hours: 2
      spares_on_hand: 1
      widest_avoid_feasible: true
      claimant_services: []
""")


def _half(tmp_path, sid, *, label="wait", t0_lat=25.0, t1_lat=25.0,
         metadata=None):
    path = tmp_path / f"{sid}.yaml"
    path.write_text(
        TWIN.format(id=sid, label=label, t0_lat=t0_lat, t1_lat=t1_lat),
        encoding="utf-8")
    scenario = load_scenario(path)
    if metadata:
        scenario = dataclasses.replace(
            scenario, metadata={**scenario.metadata, **metadata})
    return scenario


def test_a_valid_pair_passes_the_static_checks(tmp_path):
    a = _half(tmp_path, "Pa", label="wait", t1_lat=25.0)
    b = _half(tmp_path, "Pb", label="act", t1_lat=25.6)
    assert_gold_choices_differ(a, b)
    assert_issuance_prefix_shared(a, b)
    assert_shared_scalars_equal(a, b)


def test_same_gold_in_both_halves_is_rejected(tmp_path):
    a = _half(tmp_path, "Pa", label="wait")
    b = _half(tmp_path, "Pb", label="wait")
    with pytest.raises(PairInvalid, match="same gold"):
        assert_gold_choices_differ(a, b)


def test_a_pair_whose_issued_t0_blocks_differ_is_rejected(tmp_path):
    a = _half(tmp_path, "Pa", label="wait", t0_lat=25.0)
    b = _half(tmp_path, "Pb", label="act", t0_lat=25.9)
    with pytest.raises(PairInvalid, match="strictly before"):
        assert_issuance_prefix_shared(a, b)


def test_sharing_the_issuance_AT_the_decision_hour_is_also_rejected(tmp_path):
    # Identical information cannot carry two opposite correct answers: such a
    # pair is ill-posed, not hard.
    a = _half(tmp_path, "Pa", label="wait", t1_lat=25.0)
    b = _half(tmp_path, "Pb", label="act", t1_lat=25.0)
    with pytest.raises(PairInvalid, match="identical"):
        assert_issuance_prefix_shared(a, b)


def test_an_enumerated_scalar_differing_across_halves_is_rejected(tmp_path):
    a = _half(tmp_path, "Pa", label="wait", t1_lat=25.0)
    b = _half(tmp_path, "Pb", label="act", t1_lat=25.6)
    b.metadata["cone_width_km"] = 180
    with pytest.raises(PairInvalid, match="cone_width_km"):
        assert_shared_scalars_equal(a, b)


# W1.2's whole-suite check (Gate A): no ONE fixed threshold, on ONE
# claimant-side scalar, under ONE fixed orientation, answers every twin half.
# These fixtures reuse TWIN's own minimal contract (the geometry is
# irrelevant to this check -- it reads only `pair`, `id`, and
# `metadata.gold_spare_action`) but need three distinct pairs instead of
# TWIN's single hardcoded "P", so they format their own pair token.
_SPARE_TWIN = textwrap.dedent("""
    id: {id}
    pair: {pair}
    seed: 17
    state_file: eval/states/loaded-s17.json
    service_under_test: storm-svc-1
    track: hudhud
    hours: [t0, t1, t2, t3]
    decision_hour: t1
    lead_time_hours: 1
    spares_on_hand: 1
    depot_site: satna
    spare_inventory: {{satna: 1}}
    damage_radius_km: 74
    track_revision_km_per_hour_ahead: 30
    reference_avoid: {{risk_groups: [rg_ref]}}
    forecast:
      t0:
        t3: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: 25.0, lon: 81.0}}}}
      t1:
        t3: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: 25.6, lon: 81.0}}}}
    realized:
      # ids denote SPANS: load_scenario adds each fibre's reverse-direction mate
      t3: [fiber_fatehpur_allahabad_0]
    gold:
      survived: [storm-svc-1]
      max_spares_wasted: 0
      decision_at_t0: wait
      label: {label}
      rationale: fixture
    flip_variable: [svc-b]
    metadata:
      label_rule: timing_at_decision_hour
      cone_width_km: 90
      cone_motion_kmh: 20
      n_future_claimants: 1
      exposure_horizon_hours: 2
      spares_on_hand: 1
      widest_avoid_feasible: true
      gold_spare_action: {spare_action}
      claimant_services: []
""")


def _spare_half(tmp_path, sid, *, pair, label, spare_action):
    path = tmp_path / f"{sid}.yaml"
    path.write_text(
        _SPARE_TWIN.format(id=sid, pair=pair, label=label,
                           spare_action=spare_action),
        encoding="utf-8")
    return load_scenario(path)


_SIX_HALVES_SPEC = (
    ("T1a", "T1", "wait", "conserve"),
    ("T1b", "T1", "act", "spend"),
    ("T2a", "T2", "wait", "spend"),
    ("T2b", "T2", "act", "conserve"),
    ("T3a", "T3", "wait", "spend"),
    ("T3b", "T3", "act", "conserve"),
)


@pytest.fixture
def six_halves(tmp_path):
    return [_spare_half(tmp_path, sid, pair=pair, label=label,
                        spare_action=action)
            for sid, pair, label, action in _SIX_HALVES_SPEC]


@pytest.fixture
def six_halves_same_spare_action(six_halves):
    for half in six_halves:
        if half.id == "T1b":
            half.metadata["gold_spare_action"] = "conserve"
    return six_halves


# One threshold at 89.4 with orientation lo->spend answers all six.
_SOLVED = {
    "T1b": 21.5, "T3a": 51.0, "T2a": 71.1,      # spend
    "T1a": 107.6, "T3b": 182.8, "T2b": 347.4,   # conserve
}
# T2a moved above T1a's conserve value: the values interleave, so no single
# threshold separates the labels under either global orientation.
#
# This stopped being hypothetical on 2026-08-26. These two dicts are stylised
# (their conserve tail is not any one real column), but the SHAPE is exactly
# what the shipped suite did: `derived.claimant_ecar_min_over_horizons` really
# did solve all six halves at 89.35 G with T2a at 71.1 G, and T2a's near-cone
# retune really did fix it by lifting that one value past a CONSERVE half's --
# to 114.6 G, clear of T1a's 107.6 G. See derived.py's module docstring.
_INTERLEAVED = {**_SOLVED, "T2a": 114.6}


def test_a_global_threshold_that_answers_every_half_is_rejected(six_halves):
    flip_values = {sid: {"claimant_ecar_at_exposure_horizon": v}
                   for sid, v in _SOLVED.items()}
    with pytest.raises(PairInvalid, match="one global threshold"):
        assert_no_global_policy_solves_the_suite(six_halves, flip_values)


def test_interleaved_values_defeat_every_global_threshold(six_halves):
    flip_values = {sid: {"claimant_ecar_at_exposure_horizon": v}
                   for sid, v in _INTERLEAVED.items()}
    assert_no_global_policy_solves_the_suite(six_halves, flip_values)


def test_per_pair_orientation_is_NOT_granted(six_halves):
    """rules.score_rule grants per-pair orientation deliberately -- correct
    for a confound check, wrong here: a policy an operator could deploy has
    ONE orientation. With per-pair orientation these values would pass."""
    flip_values = {sid: {"claimant_ecar_at_exposure_horizon": v} for sid, v in {
        "T1b": 10.0, "T1a": 20.0,      # spend below conserve
        "T2a": 40.0, "T2b": 30.0,      # conserve below spend -- opposite sense
        "T3a": 50.0, "T3b": 60.0,
    }.items()}
    assert_no_global_policy_solves_the_suite(six_halves, flip_values)


def test_a_pair_whose_halves_declare_the_same_spare_action_is_rejected(
        six_halves_same_spare_action):
    with pytest.raises(PairInvalid, match="gold_spare_action"):
        assert_no_global_policy_solves_the_suite(
            six_halves_same_spare_action, {})


def test_global_policy_report_names_how_each_variable_is_blocked(six_halves):
    """`_SOLVED`/`_INTERLEAVED` above only ever populate ONE
    `FLIP_VARS` member (`claimant_ecar_at_exposure_horizon`) -- every
    existing test above wraps them the same way. So this test wraps them
    identically and restricts its assertions to that one variable, rather
    than asserting `set(report) == set(FLIP_VARS)`: a var nobody supplied
    any value for is omitted from the report entirely (same silent skip
    `assert_no_global_policy_solves_the_suite` always had for a partial
    sweep), not populated with a placeholder entry."""
    from storm_reoptimizer.eval.assertions import global_policy_report
    from storm_reoptimizer.eval.derived import FLIP_VARS
    var = FLIP_VARS[0]

    interleaved = {sid: {var: v} for sid, v in _INTERLEAVED.items()}
    report = global_policy_report(six_halves, interleaved)
    assert var in report
    assert report[var]["n"] == 6
    assert report[var]["best"] < 6
    assert report[var]["blocked_by"] in ("tie", "reversal", "interleave")

    solved = {sid: {var: v} for sid, v in _SOLVED.items()}
    solved_report = global_policy_report(six_halves, solved)
    assert solved_report[var]["best"] == 6
    assert solved_report[var]["blocked_by"] == "solved"


# GATE B (`assert_flip_dominates`, F2): the largest EQUAL-in-both-halves
# signal about the service under test must not outweigh the flip itself.
@pytest.fixture
def t1a():
    return load_scenario(SCENARIOS_DIR / "T1a.yaml")


@pytest.fixture
def t1b():
    return load_scenario(SCENARIOS_DIR / "T1b.yaml")


def _flip(scenario_id, *, at, sut_by_horizon, before=0.0, exposure="t3"):
    earlier = tuple(h for h in sut_by_horizon if h != exposure)
    # peak/min are over the PUBLISHED horizons' claimant values, so with no
    # earlier horizon there is only one value and both collapse onto `at`.
    # `before` is 0.0 in that case because it sums an empty tuple, not because
    # a real horizon carries no claim, so `min(at, before)` would report 0.0
    # for a claimant aggregate that is nowhere near zero.
    #
    # NOT a load-bearing guard for the three tests below: they all pass
    # `before` at its 0.0 default, and `assert_flip_dominates` takes a max()
    # over FLIP_VARS, so a spurious 0.0 in one variable could never lower the
    # flip magnitude those tests assert on. It is here so this helper builds a
    # FlipScalars that means what `flip_scalars_from_spans` would mean --
    # a future test that reads min/peak off it gets a truthful object rather
    # than one that happens not to matter yet.
    return FlipScalars(
        scenario_id=scenario_id, exposure_horizon=exposure,
        earlier_horizons=earlier,
        claimant_ecar_at_exposure_horizon=at,
        claimant_ecar_before_exposure_horizon=before,
        claimant_ecar_peak_over_horizons=max(at, before) if earlier else at,
        claimant_ecar_min_over_horizons=min(at, before) if earlier else at,
        # 0.0 in every call site below: `assert_flip_dominates` takes a max()
        # over FLIP_VARS' inter-half DIFFERENCES, and this helper's two
        # `_flip(...)` calls in each test always pass the SAME 0.0 for this
        # field on both halves, so its difference is 0.0 and can never be the
        # variable that decides the max -- not a load-bearing zero, the same
        # way `before`'s own comment above explains. No dataclass default is
        # added for it: a real (non-test) caller must always name a value.
        largest_restorable_group_ecar_gbps=0.0,
        sut_ecar_by_horizon=dict(sut_by_horizon))


def test_a_distractor_larger_than_the_flip_is_rejected(t1a, t1b):
    """T1 before W1.4: an equal-in-both-halves nowcast worth 265.0 G against
    a flip magnitude of 86.1 G. Every expected-value reasoner answers the same
    thing in both halves, and every equality assertion passes."""
    a = _flip("T1a", at=107.6, sut_by_horizon={"t2": 265.0, "t3": 48.4})
    b = _flip("T1b", at=21.5, sut_by_horizon={"t2": 265.0, "t3": 48.4})
    with pytest.raises(PairInvalid, match="dominates the flip"):
        assert_flip_dominates(t1a, t1b, a, b)


def test_a_flip_larger_than_every_equal_signal_is_accepted(t1a, t1b):
    """T1 after W1.4: the nowcast is gone, so the largest equal SUT signal is
    48.4 G against a flip magnitude of 86.1 G."""
    a = _flip("T1a", at=107.6, sut_by_horizon={"t3": 48.4})
    b = _flip("T1b", at=21.5, sut_by_horizon={"t3": 48.4})
    assert_flip_dominates(t1a, t1b, a, b)


def test_an_unequal_sut_signal_is_not_this_checks_business(t1a, t1b):
    """A SUT figure that DIFFERS across the halves is
    assert_pair_derived_geometry_is_equal's job, not this one. This check only
    weighs signals that are equal in both halves -- those are the ones every
    reasoner reads identically."""
    a = _flip("T1a", at=107.6, sut_by_horizon={"t3": 300.0})
    b = _flip("T1b", at=21.5, sut_by_horizon={"t3": 48.4})
    assert_flip_dominates(t1a, t1b, a, b)


# W1.6 (`_check_no_free_escape`, F2's secondary half): gold's reasoning is
# always "spending the pair isn't worth it", but "act" and "spend" are
# different events. A zero-pair candidate that MOVES the service is a free
# lever the label rule never grades -- exactly how the agent beat T1a's gold
# label while satisfying every other scoring criterion (it committed an
# ip_reroute with pairs_needed = 0, so gold.survived and
# gold.max_spares_wasted: 0 both held).
#
# Finding #5 (2026-08-26 re-review): the predicate now ALSO requires that
# committing the free candidate would flip the graded label away from gold's
# -- "moves the service" alone is necessary but not sufficient. Two of the
# then-shipped 2026-08-31 conserve halves were false positives under the old,
# broader check on the two label rules retired 2026-09-06 (see
# `test_label_if_committed_rejects_a_retired_rule` below); `_T1_KW`/etc.
# below fix the label_rule/gold_label a T1a-shaped half needs.
_T1_KW = dict(label_rule="timing_at_decision_hour", gold_label="wait",
             reference_avoid={})


def test_a_free_candidate_that_moves_the_service_defeats_a_conserve_gold():
    """The T1a failure mode: a 0-pair candidate the agent can take to improve
    its own position while keeping the spare. Gold says "don't spend"; the
    label rule reads "acted"; both are satisfiable at once. Task 5: for
    `timing_at_decision_hour`, a commit reads "act" unless it is INERT (moves
    nothing and needs no spare) -- but `_check_no_free_escape` already
    filters out every candidate that reuses everything currently working
    (see the `current <= reused` `continue` above), so every candidate that
    reaches `_label_if_committed` from here already moves the service, and
    this remains a genuine escape regardless of lever. See
    `test_label_if_committed_reads_wait_for_an_inert_commit` below for the
    direct case where an inert commit reads "wait"."""
    menu = {"status": "solution", "candidates": [
        {"lever": "ip_reroute", "reused_lightpaths": ["lp-somewhere-else"],
         "new_lightpaths": [], "restored_gbps": 300.0,
         "shortfall_gbps": 0.0, "cost_vector": {}}]}
    with pytest.raises(PairInvalid, match="free lever"):
        _check_no_free_escape("T1a", menu, current={"lp-current-working-0"},
                              **_T1_KW)


def test_a_free_candidate_that_only_stays_put_is_safe():
    menu = {"status": "solution", "candidates": [
        {"lever": "ip_reroute", "reused_lightpaths": ["lp-current-working-0"],
         "new_lightpaths": [], "restored_gbps": 300.0,
         "shortfall_gbps": 0.0, "cost_vector": {}}]}
    _check_no_free_escape("T1a", menu, current={"lp-current-working-0"},
                          **_T1_KW)


def test_candidates_that_cost_a_pair_are_not_this_checks_business():
    menu = {"status": "solution", "candidates": [
        {"lever": "optical_reroute", "reused_lightpaths": [],
         "new_lightpaths": [{"oms_sequence": ["oms_1"]}],
         "restored_gbps": 300.0, "shortfall_gbps": 0.0, "cost_vector": {}}]}
    _check_no_free_escape("T1a", menu, current={"lp-current-working-0"},
                          oms_nodes={"oms_1": ["site_a", "site_b"]},
                          **_T1_KW)


def test_the_raised_message_names_the_specific_lightpath_and_reason():
    """Finding #3's more precise ask: the live xfail test relies on
    `raises=PairInvalid` alone to identify the known T1a finding, so this
    pure unit test pins the message content that finding's own report quotes
    ("costs ZERO pairs" and the specific lightpath id) -- a regression that
    changed the WRONG PairInvalid message would still satisfy `raises=
    PairInvalid` on the live test, but would fail this one."""
    menu = {"status": "solution", "candidates": [
        {"lever": "ip_reroute", "reused_lightpaths": ["lp-prot-storm-svc-1-0"],
         "new_lightpaths": [], "restored_gbps": 300.0,
         "shortfall_gbps": 0.0, "cost_vector": {}}]}
    with pytest.raises(PairInvalid, match="costs ZERO.*lp-prot-storm-svc-1-0"):
        _check_no_free_escape(
            "T1a", menu, current={"lp-cand-storm-svc-1-0"}, **_T1_KW)


def test_label_if_committed_rejects_a_retired_rule():
    # Spec 7 (T2/T3 probe redesign): avoid_horizon_at_decision_hour and
    # chosen_lever_at_decision_hour were retired with the pairs that used
    # them.
    from storm_reoptimizer.eval.assertions import _label_if_committed
    with pytest.raises(ValueError, match="unknown label_rule"):
        _label_if_committed(label_rule="avoid_horizon_at_decision_hour",
                            candidate={}, avoid_used={})


def test_label_if_committed_reads_wait_for_an_inert_commit():
    """Task 5: `timing_at_decision_hour` is no longer unconditionally "act".
    A commit that neither moves the working path nor needs a depot spare is
    INERT, and mirrors `runner.py`'s own `step["inert"]` -- reads "wait", the
    same as a genuine no-op hour, per `scoring.decision_label`'s own
    docstring: "an act that commits nothing (or commits inertly) is a
    wait"."""
    from storm_reoptimizer.eval.assertions import _label_if_committed
    assert _label_if_committed(
        label_rule="timing_at_decision_hour", candidate={}, avoid_used={},
        changes_working_path=False, depot_spares_needed=0) == "wait"


def test_label_if_committed_reads_act_for_a_substantive_commit():
    """The other half of the same flip: a commit that DOES move the working
    path (regardless of whether it also spends a depot spare) is substantive,
    not inert, and still reads "act"."""
    from storm_reoptimizer.eval.assertions import _label_if_committed
    assert _label_if_committed(
        label_rule="timing_at_decision_hour", candidate={}, avoid_used={},
        changes_working_path=True, depot_spares_needed=0) == "act"
    assert _label_if_committed(
        label_rule="timing_at_decision_hour", candidate={}, avoid_used={},
        changes_working_path=False, depot_spares_needed=1) == "act"


# --------------------------------------------------------------------------
# Task 12 (exposure-and-depot plan, dimensional-coherence invariants 1-8).
# --------------------------------------------------------------------------


# claimant_service_ids: the shared parsing note (invariants 2-4).
def test_claimant_service_ids_returns_the_declared_list():
    scenario = load_scenario(SCENARIOS_DIR / "T1a.yaml")
    # The REAL jalgaon-homed claimant pair (Task 15, T1 spend-or-hold
    # redesign, 2026-09-05). Was the satna-homed pair until T1 was rebuilt
    # on a new SUT; before that, the pre-Task-14 placeholder ids (d0029,
    # d0348), which had no real aerial exposure at all.
    assert claimant_service_ids(scenario) == (
        "t1-claimant-jalgaon-dhulia-fwd", "t1-claimant-jalgaon-dhulia-rev")


def test_claimant_service_ids_is_empty_when_none_are_declared(tmp_path):
    # D1.yaml is no longer the suite's "no claimants" example (Task 14: it
    # now honestly declares the real satna-homed pair -- see D1.yaml's own
    # gold.rationale for why). Use the synthetic fixture instead.
    scenario = _scenario_claiming(tmp_path, 0.0, at="t6", claimants=())
    assert claimant_service_ids(scenario) == ()


def test_claimant_service_ids_rejects_an_id_not_in_the_rationale():
    """A declared id that does not appear literally in gold.rationale has
    drifted from the prose it is supposed to summarize."""
    scenario = load_scenario(SCENARIOS_DIR / "T1a.yaml")
    scenario.metadata["claimant_services"] = ["d9999"]
    with pytest.raises(PairInvalid, match="d9999"):
        claimant_service_ids(scenario)


# Invariant 1 (`assert_realized_cuts_pass_the_event_filter`). T1a's own
# shipped `realized` block used to inject fiber_allahabad_fatehpur_0/_1 and
# fiber_fatehpur_allahabad_0 -- confirmed BURIED in the toy topology -- which
# is exactly the defect this invariant exists to catch; Task 14 fixed T1a's
# own episode data (it now realizes a cut on the real, aerial
# satna<->jabalpur claimant span instead), so this test uses the TWIN
# fixture's own synthetic `realized: {{t3: [fiber_fatehpur_allahabad_0]}}`
# against a hand-built BURIED oms_by_id instead of depending on a real
# episode's (now-fixed) content. T1b's realized cut (fiber_satna_rewa_0,
# checked below) is aerial and correct. The filter check raises on the
# FIRST realized asset it names -- `load_scenario` expands the fixture's one
# named id to include its reverse-direction mate too (Task 1, 2026-09-27:
# realized cuts denote whole spans), but that mate is never reached here
# because the original direction alone already fails the filter.
def test_a_realized_cut_on_buried_fibre_fails_the_build(tmp_path):
    # realized: {t3: [fiber_fatehpur_allahabad_0]}
    scenario = _half(tmp_path, "Pa", label="wait")
    oms_by_id = {
        "oms_fatehpur_allahabad": {
            "id": "oms_fatehpur_allahabad", "src_node_id": "fatehpur",
            "dst_node_id": "allahabad", "elements": ["fiber_fatehpur_allahabad_0"]},
    }
    with pytest.raises(PairInvalid, match="buried"):
        assert_realized_cuts_pass_the_event_filter(
            scenario, topology_path=TOPOLOGY_PATH, oms_by_id=oms_by_id)


def test_a_realized_cut_on_aerial_fibre_passes():
    # T1b's own realized cut, since Task 15 rebuilt the pair on the
    # jalgaon-homed SUT: the two AERIAL spans its working and protection
    # legs leave the depot on. `load_scenario` now expands each named fibre
    # to include its reverse-direction mate (Task 1, 2026-09-27), so
    # oms_by_id's elements carry both directions of each physical span --
    # exactly how a real OMS bundles them.
    scenario = load_scenario(SCENARIOS_DIR / "T1b.yaml")
    oms_by_id = {
        "oms_jalgaon_buldhana": {
            "id": "oms_jalgaon_buldhana", "src_node_id": "jalgaon",
            "dst_node_id": "buldhana", "elements": ["fiber_jalgaon_buldhana_0",
                                                     "fiber_buldhana_jalgaon_0"]},
        "oms_jalgaon_khandwa": {
            "id": "oms_jalgaon_khandwa", "src_node_id": "jalgaon",
            "dst_node_id": "khandwa", "elements": ["fiber_jalgaon_khandwa_0",
                                                   "fiber_jalgaon_khandwa_1",
                                                   "fiber_khandwa_jalgaon_0",
                                                   "fiber_khandwa_jalgaon_1"]},
    }
    assert_realized_cuts_pass_the_event_filter(
        scenario, topology_path=TOPOLOGY_PATH, oms_by_id=oms_by_id)


def test_a_realized_cut_naming_no_known_fiber_is_rejected():
    scenario = load_scenario(SCENARIOS_DIR / "T1b.yaml")
    with pytest.raises(PairInvalid, match="names no fiber"):
        assert_realized_cuts_pass_the_event_filter(
            scenario, topology_path=TOPOLOGY_PATH, oms_by_id={})


# Invariants 2 and 3 (`assert_claimants_have_filterable_exposure`,
# `assert_claimants_depot_eligible`). The async wrappers no-op for an empty
# claimant list WITHOUT ever awaiting the client, so a scenario declaring
# claimant_services: [] exercises that path for real with no server at all
# (`None` is passed as the client -- a real one would error if awaited); the
# discriminating logic itself lives in the pure helpers, unit-tested
# directly. D1.yaml is no longer this suite's "no claimants" example (Task
# 14: it now honestly declares the real satna-homed pair), so this uses the
# synthetic `_scenario_claiming` fixture with an empty claimant list instead.
def test_claimants_have_filterable_exposure_is_a_noop_with_no_claimants(
        tmp_path):
    scenario = _scenario_claiming(tmp_path, 0.0, at="t6", claimants=())
    asyncio.run(assert_claimants_have_filterable_exposure(
        None, scenario, topology_path=TOPOLOGY_PATH))


def test_claimants_depot_eligible_is_a_noop_with_no_claimants(tmp_path):
    scenario = _scenario_claiming(tmp_path, 0.0, at="t6", claimants=())
    asyncio.run(assert_claimants_depot_eligible(
        None, scenario, topology_path=TOPOLOGY_PATH))


def test_a_claimant_with_no_cuttable_span_is_a_violation():
    violations = _claimant_exposure_violations(
        ("d0001", "d0002"), {"d0001": ((( 25.0, 81.0), (25.1, 81.1)),)})
    assert violations == ["d0002"]


def test_a_claimant_with_a_cuttable_span_is_not_a_violation():
    violations = _claimant_exposure_violations(
        ("d0001",), {"d0001": (((25.0, 81.0), (25.1, 81.1)),)})
    assert violations == []


def test_a_claimant_not_terminating_at_the_depot_is_a_violation():
    violations = _claimant_depot_violations(
        ("d0001", "d0002"),
        {"d0001": ("satna", "allahabad"), "d0002": ("kolkata", "mumbai")},
        "satna")
    assert violations == ["d0002"]


def test_a_claimant_terminating_at_the_depot_is_not_a_violation():
    violations = _claimant_depot_violations(
        ("d0001",), {"d0001": ("allahabad", "satna")}, "satna")
    assert violations == []


# Invariant 4 (`assert_claim_is_one_lightpath`). `_CLAIM_SCENARIO` is a
# minimal, otherwise-inert scenario that declares a non-empty
# `claimant_services` (making `claimed_competing_ecar_gbps`/`_at`
# CONDITIONALLY required, scenario_file.py) and its own claim horizon
# EXPLICITLY via `claimed_competing_ecar_at` -- 2026-08-30 review fix,
# round 1, finding 1: the claim's horizon is declared, never inferred from
# the SUT's own `exposure_horizon_hours` (which is a different hour for
# T2/T3-shaped episodes; see `assert_claim_is_one_lightpath`'s own
# docstring). `claimants` parameterizes WHICH services the claim names --
# needed since round 2's fix restricts the group comparison to only the
# groups the claim's OWN named services belong to (see the group-scoping
# tests below), so different tests need different claimant/group overlaps.
# The rationale text names every one of `claimants` literally, satisfying
# `claimant_service_ids`'s own cross-check against `gold.rationale`.
_CLAIM_SCENARIO = textwrap.dedent("""
    id: CLAIM
    seed: 17
    state_file: eval/states/loaded-s17.json
    service_under_test: storm-svc-1
    track: hudhud
    hours: [t0, t1, t2, t6]
    decision_hour: t1
    lead_time_hours: 1
    spares_on_hand: 1
    depot_site: satna
    spare_inventory: {{satna: 1}}
    damage_radius_km: 74
    track_revision_km_per_hour_ahead: 30
    reference_avoid: {{}}
    forecast:
      t0:
        t6: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: 25.0, lon: 81.0}}}}
      t1:
        t6: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: 25.0, lon: 81.0}}}}
    realized: {{}}
    gold:
      survived: [storm-svc-1]
      max_spares_wasted: 0
      decision_at_t0: wait
      label: wait
      rationale: fixture naming {rationale_names} in its arithmetic
    flip_variable: [claimant]
    metadata:
      label_rule: timing_at_decision_hour
      cone_width_km: 90
      cone_motion_kmh: 20
      n_future_claimants: 1
      exposure_horizon_hours: 2
      spares_on_hand: 1
      claimant_services: [{claimants_yaml}]
      claimed_competing_ecar_gbps: {claimed}
      claimed_competing_ecar_at: {at}
""")


def _scenario_claiming(tmp_path, claimed, at="t6",
                       claimants=("claim-fixture-service",)):
    path = tmp_path / "CLAIM.yaml"
    path.write_text(_CLAIM_SCENARIO.format(
        claimed=claimed, at=at, claimants_yaml=", ".join(claimants),
        rationale_names=" ".join(claimants)),
        encoding="utf-8")
    return load_scenario(path)


def test_a_claimant_aggregate_above_the_largest_group_fails_the_build(
        tmp_path):
    """Would have caught D2. The rationale bills 108.8 G across THREE
    co-terminating groups it names members of (agra<->gwalior 24.0,
    allahabad<->delhi 29.5, delhi<->kanpur 55.3); one spare buys one
    lightpath, so its honest figure is the largest of THOSE (the claim's
    OWN groups), 55.3 G."""
    groups = {"t6": ({"endpoints": ("agra", "gwalior"), "members": ("x",),
                      "ecar_gbps": 24.0},
                     {"endpoints": ("allahabad", "delhi"),
                      "members": ("w",), "ecar_gbps": 29.5},
                     {"endpoints": ("delhi", "kanpur"), "members": ("y", "z"),
                      "ecar_gbps": 55.3})}
    with pytest.raises(PairInvalid, match="55.3"):
        assert_claim_is_one_lightpath(
            _scenario_claiming(tmp_path, 108.8, at="t6",
                               claimants=("x", "w", "y", "z")), groups)


def test_a_claimant_aggregate_at_or_below_the_largest_group_passes(tmp_path):
    groups = {"t6": ({"endpoints": ("delhi", "kanpur"), "members": ("y", "z"),
                      "ecar_gbps": 55.3},)}
    assert_claim_is_one_lightpath(
        _scenario_claiming(tmp_path, 55.3, at="t6", claimants=("y", "z")),
        groups)


def test_the_claim_is_checked_against_its_OWN_declared_horizon(tmp_path):
    """2026-08-30 review fix, round 1, finding 1: T2a/T3a-shaped defect.
    This scenario's SUT exposure horizon (`exposure_horizon_hours: 2` from
    decision_hour `t1`) resolves to `t6`, but the claim is declared at `t2`
    -- exactly the T2/T3 shape, where the competing claim is billed at the
    NEAR horizon while the SUT's own exposure is graded at the FAR one. If
    the check used the SUT's horizon instead of the declared one, it would
    compare 60.0 against t6's 999.0 group and WRONGLY pass; using the
    declared `t2` horizon, it correctly compares against t2's 55.3 (the
    claim's own group there) and fails."""
    groups = {
        "t2": ({"endpoints": ("delhi", "kanpur"), "members": ("y", "z"),
               "ecar_gbps": 55.3},),
        "t6": ({"endpoints": ("agra", "gwalior"), "members": ("w",),
               "ecar_gbps": 999.0},),
    }
    with pytest.raises(PairInvalid, match="55.3"):
        assert_claim_is_one_lightpath(
            _scenario_claiming(tmp_path, 60.0, at="t2",
                               claimants=("y", "z")), groups)


def test_the_claim_is_checked_against_its_OWN_group_not_an_unrelated_one(
        tmp_path):
    """2026-08-30 review fix, round 2: T3a-shaped defect. The claim's own
    named services (y, z) sit in a SMALL group (12.1 G) while an unrelated
    service (w, not named by this claim) sits in a much LARGER group
    (999.0 G) at the SAME horizon -- exactly T3a's live shape (its own
    claimants are not members of the one depot-eligible group that exists
    today). Comparing against the largest group ANYWHERE in the network (the
    original implementation) would let a real 20.0 G over-claim pass for
    reasons that have nothing to do with the claim itself; comparing only
    against the claim's OWN group (12.1 G) correctly rejects it."""
    groups = {"t6": (
        {"endpoints": ("delhi", "kanpur"), "members": ("y", "z"),
         "ecar_gbps": 12.1},
        {"endpoints": ("agra", "gwalior"), "members": ("w",),
         "ecar_gbps": 999.0},
    )}
    with pytest.raises(PairInvalid, match="12.1"):
        assert_claim_is_one_lightpath(
            _scenario_claiming(tmp_path, 20.0, at="t6",
                               claimants=("y", "z")), groups)


def test_a_claim_within_its_own_group_passes_despite_a_larger_unrelated_one(
        tmp_path):
    groups = {"t6": (
        {"endpoints": ("delhi", "kanpur"), "members": ("y", "z"),
         "ecar_gbps": 12.1},
        {"endpoints": ("agra", "gwalior"), "members": ("w",),
         "ecar_gbps": 999.0},
    )}
    assert_claim_is_one_lightpath(
        _scenario_claiming(tmp_path, 12.1, at="t6", claimants=("y", "z")),
        groups)


def test_claim_is_one_lightpath_is_a_noop_when_no_claimants_are_declared(
        tmp_path):
    """A scenario declaring metadata.claimant_services: [] and,
    consistently, no claimed_competing_ecar_gbps has no claim to check.
    (D1.yaml no longer serves as this suite's own empty-claimant example --
    Task 14 gave it the real satna-homed pair, since both storm-svc-1's own
    corridor and the real claimant corridor originate at the same node D1's
    cone is centred on. Every shipped episode now declares a real, current
    claim -- see the "second red window" table in the Task 12 report, and
    D1.yaml's own gold.rationale, for what checking it against live data
    finds.) Uses the synthetic `_scenario_claiming` fixture with an empty
    claimant list instead."""
    scenario = _scenario_claiming(tmp_path, 0.0, at="t6", claimants=())
    assert_claim_is_one_lightpath(scenario, {})   # must not raise


def test_claim_is_one_lightpath_is_a_noop_even_with_a_stray_declared_value(
        tmp_path):
    """2026-08-30 review fix, round 2, small fix: the no-op decision is
    gated on `claimant_service_ids(scenario)` (the PARSED, cross-checked
    list), not on the raw `claimed_competing_ecar_gbps` metadata key.
    scenario_file.py only makes the two claim keys required TOGETHER when
    claimant_services is non-empty -- it does not forbid a stray
    claimed_competing_ecar_gbps on an episode whose claimant list is EMPTY,
    so an empty-claimant scenario with such a stray value must still no-op
    cleanly rather than raise a bare KeyError reading the missing horizon
    key. Uses the synthetic fixture (D1.yaml is no longer empty-claimant,
    see the test above)."""
    scenario = _scenario_claiming(tmp_path, 0.0, at="t6", claimants=())
    scenario.metadata["claimed_competing_ecar_gbps"] = 999.0
    assert_claim_is_one_lightpath(scenario, {})   # must not raise


# Invariant 5 (`assert_group_fits_one_lightpath`).
def test_a_group_exceeding_one_lightpaths_capacity_fails_the_build(tmp_path):
    scenario = _scenario_claiming(tmp_path, 0.0)
    groups = {"t6": ({"endpoints": ("a", "b"), "members": ("x",),
                      "ecar_gbps": 900.0},)}
    with pytest.raises(PairInvalid, match="800"):
        assert_group_fits_one_lightpath(scenario, groups)


def test_a_group_within_one_lightpaths_capacity_passes(tmp_path):
    scenario = _scenario_claiming(tmp_path, 0.0)
    groups = {"t6": ({"endpoints": ("a", "b"), "members": ("x",),
                      "ecar_gbps": 300.0},)}
    assert_group_fits_one_lightpath(scenario, groups)


# Invariant 6 (`assert_depot_is_the_binding_site`).
def test_a_non_depot_site_that_cannot_cover_its_charge_is_a_violation():
    candidates = [{"lever": "optical_reroute",
                   "new_lightpaths": [{"oms_sequence": ["oms_1"]}]}]
    oms_nodes = {"oms_1": ["satna", "farsite"]}
    violations = _binding_site_violations(
        candidates, oms_nodes, depot_site="satna",
        spare_inventory={"satna": 1, "farsite": 0},
        default_spares_per_site=8)
    assert violations == [(0, "farsite", 1, 0)]


def test_a_non_depot_site_covered_by_the_ledger_default_is_not_a_violation():
    candidates = [{"lever": "optical_reroute",
                   "new_lightpaths": [{"oms_sequence": ["oms_1"]}]}]
    oms_nodes = {"oms_1": ["satna", "farsite"]}
    violations = _binding_site_violations(
        candidates, oms_nodes, depot_site="satna",
        spare_inventory={"satna": 1}, default_spares_per_site=8)
    assert violations == []


def test_the_depot_site_itself_is_never_flagged():
    """The depot is SUPPOSED to be the binding site -- its own scarcity is
    not a violation of this invariant."""
    candidates = [{"lever": "optical_reroute",
                   "new_lightpaths": [{"oms_sequence": ["oms_1"]}]}]
    oms_nodes = {"oms_1": ["satna", "farsite"]}
    violations = _binding_site_violations(
        candidates, oms_nodes, depot_site="satna",
        spare_inventory={"satna": 0, "farsite": 8},
        default_spares_per_site=8)
    assert violations == []


# Invariant 7 (`assert_escape_route_survives`).
def test_finds_the_jabalpur_optical_reroute_candidate():
    candidates = [
        {"lever": "ip_reroute", "new_lightpaths": []},
        {"lever": "optical_reroute",
         "new_lightpaths": [{"oms_sequence": ["oms_a", "oms_b"]}]},
    ]
    oms_nodes = {"oms_a": ["satna", "jabalpur"],
                "oms_b": ["jabalpur", "damoh"]}
    index, candidate = _jabalpur_optical_reroute_candidate(
        candidates, oms_nodes)
    assert index == 1
    assert candidate["lever"] == "optical_reroute"


def test_an_ip_reroute_via_jabalpur_does_not_count():
    """The escape route must be an optical_reroute -- an ip_reroute lights
    nothing and cannot be the escape corridor T2's/T3's gold spend halves
    rely on."""
    candidates = [{"lever": "ip_reroute",
                   "new_lightpaths": [{"oms_sequence": ["oms_a"]}]}]
    oms_nodes = {"oms_a": ["satna", "jabalpur"]}
    index, candidate = _jabalpur_optical_reroute_candidate(
        candidates, oms_nodes)
    assert (index, candidate) == (None, None)


def test_no_jabalpur_candidate_returns_none():
    candidates = [{"lever": "optical_reroute",
                   "new_lightpaths": [{"oms_sequence": ["oms_c"]}]}]
    oms_nodes = {"oms_c": ["satna", "rewa"]}
    index, candidate = _jabalpur_optical_reroute_candidate(
        candidates, oms_nodes)
    assert (index, candidate) == (None, None)


# Invariant 8 (`assert_sampling_error_within_margin`).
def test_a_flip_margin_within_10x_of_the_sampling_error_fails_the_build():
    # The gap must clear DERIVED_TOLERANCE (1e-3, spec 5.3) or
    # `_distinct_within_tolerance` collapses the two values into one and no
    # margin is computed at all -- so 0.0015, not the pre-redesign 0.0002,
    # is the smallest gap that is both a real (non-noise) distinction and
    # still within 10x of measured_error (2.56e-4 * 10 = 2.56e-3).
    flip_values = {
        "T1a": {"largest_restorable_group_ecar_gbps": 100.0},
        "T1b": {"largest_restorable_group_ecar_gbps": 100.0015},
    }
    with pytest.raises(PairInvalid, match="sampling error"):
        assert_sampling_error_within_margin(
            flip_values, measured_error=2.56e-4)


def test_a_flip_margin_10x_the_sampling_error_passes():
    flip_values = {
        "T1a": {"largest_restorable_group_ecar_gbps": 100.0},
        "T1b": {"largest_restorable_group_ecar_gbps": 100.01},
    }
    assert_sampling_error_within_margin(flip_values, measured_error=2.56e-4)


def test_the_sampling_error_check_is_a_noop_with_no_flip_values():
    assert_sampling_error_within_margin({}, measured_error=2.56e-4)


# Invariant 9 (hazard-footprint plan, 2026-09-01).
def test_an_empty_group_at_a_horizon_with_measurable_exposure_is_a_violation():
    from storm_reoptimizer.eval.assertions import (
        _risk_group_coverage_violations,
    )

    exposure = {
        "storm-svc-1": {"t3": {"p_cut": 0.1349}},
        "quiet-svc": {"t3": {"p_cut": 0.0001}},
    }
    assert _risk_group_coverage_violations(
        exposure, {"t3": []}, threshold=0.005) == [
            ("t3", "storm-svc-1", 0.1349)]


def test_a_nonempty_group_is_never_a_violation_however_exposed():
    from storm_reoptimizer.eval.assertions import (
        _risk_group_coverage_violations,
    )

    exposure = {"storm-svc-1": {"t3": {"p_cut": 0.99}}}
    assert _risk_group_coverage_violations(
        exposure, {"t3": ["fiber_x_0"]}, threshold=0.005) == []


def test_an_empty_group_with_nothing_measurably_at_risk_is_fine():
    """The invariant is one-directional on purpose: a horizon at which
    nothing is at risk is ENTITLED to an empty group."""
    from storm_reoptimizer.eval.assertions import (
        _risk_group_coverage_violations,
    )

    exposure = {"storm-svc-1": {"t3": {"p_cut": 0.001}}}
    assert _risk_group_coverage_violations(
        exposure, {"t3": []}, threshold=0.005) == []


# `assert_gold_matches_outcomes` (T1 spend-or-hold redesign, Task 9). Pure --
# the two policies' own simulated outcomes are handed in, never computed
# here (a later task's oracle enumerator, run for real, produces them).
# Sentinel, so `min_margin_gbps_h=None` means "declare NO floor" rather than
# "leave the scenario's own floor alone". It used to be safe to conflate the
# two only because the shipped T1a carried no floor at all; Task 15's
# enumerated gold gives it one (175.0), which made the no-floor test below
# silently assert against a real floor.
_KEEP = object()


def _with_gold(scenario, *, label=None, min_margin_gbps_h=_KEEP):
    gold = scenario.gold
    if label is not None:
        gold = dataclasses.replace(gold, label=label)
    if min_margin_gbps_h is not _KEEP:
        gold = dataclasses.replace(gold, min_margin_gbps_h=min_margin_gbps_h)
    return dataclasses.replace(scenario, gold=gold)


def test_gold_matches_outcomes_when_the_label_is_the_argmin_by_enough(t1a):
    scenario = _with_gold(t1a, label="spend", min_margin_gbps_h=100.0)
    assert_gold_matches_outcomes(scenario, {"spend": 0.0, "hold": 600.0})


def test_gold_matches_outcomes_rejects_a_label_that_is_not_the_argmin(t1a):
    scenario = _with_gold(t1a, label="hold", min_margin_gbps_h=100.0)
    with pytest.raises(PairInvalid, match="argmin"):
        assert_gold_matches_outcomes(scenario, {"spend": 0.0, "hold": 600.0})


def test_gold_matches_outcomes_rejects_a_margin_below_the_floor(t1a):
    scenario = _with_gold(t1a, label="spend", min_margin_gbps_h=100.0)
    with pytest.raises(PairInvalid, match="margin"):
        assert_gold_matches_outcomes(scenario, {"spend": 0.0, "hold": 50.0})


def test_gold_matches_outcomes_skips_the_margin_check_with_no_floor_declared(
        t1a):
    scenario = _with_gold(t1a, label="spend", min_margin_gbps_h=None)
    assert_gold_matches_outcomes(scenario, {"spend": 0.0, "hold": 1.0})


def test_gold_matches_outcomes_rejects_an_empty_outcomes_dict(t1a):
    with pytest.raises(PairInvalid, match="no outcomes"):
        assert_gold_matches_outcomes(t1a, {})


# `assert_both_legs_exposed` / `assert_spend_is_real` (T1 spend-or-hold
# redesign, Task 9) -- live, against the real server. Both pass/fail paths
# are exercised against REAL geometry, not a forced/synthetic threshold.
#
# UPDATED by Task 15 (2026-09-05), which rebuilt T1 on the jalgaon-homed SUT
# `t1-svc-jalgaon-indore`: the two FAIL cases below now read the DISCARDED
# pre-redesign fixtures rather than the shipped scenarios, because the whole
# point of the rebuild was to make the shipped pair pass them.
#
# `assert_both_legs_exposed`: the shipped T1a passes (working 0.1172,
# protection 0.1218 -- confirmed live), and so does the shipped T1b (working
# 0.1172, protection 0.6947); the PRE-REDESIGN T1b does not (working 0.0969,
# protection 0.0176 -- storm-svc-1's protection leg, satna<->jhansi, sits
# outside every T1 cone). That was a REAL finding, not a test artifact, and
# it is exactly the gap the rebuild closed.
#
# `assert_spend_is_real`: neither half of the PRE-REDESIGN T1 had a real
# escape for storm-svc-1 under the spend decider's own risk-group avoid
# (confirmed live: T1a's wide, corridor-facing cone sweeps every aerial
# direction out of satna, leaving `route_service` zero candidates; T1b's
# narrower cone leaves candidates, but EVERY alternative route into
# allahabad funnels through oms_jhansi_allahabad -- storm-svc-1's own
# protection leg's last hop -- so every one collides with protection). The
# rebuilt pair has 8 real escape candidates in both halves; the shipped
# PASS side is checked by `test_episodes.py::test_t1_spend_is_real`. The
# local PASS path below is exercised against a hand-built scenario reusing
# the pre-redesign T1b's
# OWN real cone (Task 14's real `claimant-satna-jabalpur-fwd`, UNPROTECTED --
# `protection_path: []`, live-confirmed -- so `collides_with_protection` can
# never bind for it) rerouting off its own direct satna<->jabalpur span: a
# real, live, single-lightpath, non-inert escape exists and is picked.
def _connected(loaded_state_path, local_server_command, local_server_env):
    return connect_server(
        TOPOLOGY_PATH, server_command=local_server_command,
        env=local_server_env,
        extra_args=["--state", str(loaded_state_path)])


def test_both_legs_exposed_passes_on_t1a(
        loaded_state_path, local_server_command, local_server_env):
    scenario = load_scenario(SCENARIOS_DIR / "T1a.yaml")

    async def _run():
        async with _connected(
                loaded_state_path, local_server_command, local_server_env
        ) as client:
            await assert_both_legs_exposed(
                client, scenario, topology_path=TOPOLOGY_PATH)

    asyncio.run(_run())


def test_both_legs_exposed_rejects_the_old_t1b_whose_protection_was_barely_exposed(
        loaded_state_path, local_server_command, local_server_env):
    """The FAIL path, against the pre-redesign T1b (storm-svc-1: working
    0.0969, protection 0.0176 -- its cone points away from the claimant
    corridor and storm-svc-1's protection leg, satna<->jhansi, sits outside
    it). Repointed at the discarded fixture by Task 15: the shipped T1b now
    PASSES this check by construction, which is the whole point of the
    rebuild, so keeping the negative case means keeping the old geometry."""
    scenario = load_scenario(DISCARDED_DIR / "T1b-2026-08-30.yaml")

    async def _run():
        async with _connected(
                loaded_state_path, local_server_command, local_server_env
        ) as client:
            with pytest.raises(PairInvalid, match="not both legs"):
                await assert_both_legs_exposed(
                    client, scenario, topology_path=TOPOLOGY_PATH)

    asyncio.run(_run())


# A hand-built scenario proving `assert_spend_is_real`'s PASS path against a
# real escape: T1b's own t1-issuance t3 cone (bearing away from the
# claimant corridor, so it never sweeps satna<->jabalpur), against the real,
# live, UNPROTECTED claimant `claimant-satna-jabalpur-fwd` (satna->jabalpur,
# 100 Gbps, `protection_path: []` -- confirmed live). Its own direct span
# (`oms_satna_jabalpur`) is what the SUT would normally reroute onto for a
# free escape, so avoiding whatever this cone's own risk group covers forces
# a REAL, spare-consuming reroute -- unlike storm-svc-1, this service has no
# protection to collide with, so `escape_objective`'s protection-collision
# gate can never be the reason it fails.
_SPEND_REAL_YAML = textwrap.dedent("""
    id: SPEND_REAL_PASS
    seed: 17
    state_file: eval/states/loaded-s17.json
    service_under_test: claimant-satna-jabalpur-fwd
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
        t3: {cone: {type: Polygon, coordinates: []}, width_km: 90, center: {lat: 25.442800716819043, lon: 81.32777666666667}}
      t1:
        t3: {cone: {type: Polygon, coordinates: []}, width_km: 90, center: {lat: 25.389623000292865, lon: 81.82382629023604}}
    realized: {}
    gold:
      survived: [claimant-satna-jabalpur-fwd]
      max_spares_wasted: 0
      decision_at_t0: wait
      label: wait
      rationale: assert_spend_is_real live-test fixture; not scored
    flip_variable: [test]
    metadata:
      cone_width_km: 90
      cone_motion_kmh: 20
      n_future_claimants: 0
      exposure_horizon_hours: 2
      spares_on_hand: 1
      claimant_services: []
""")


def _spend_real_scenario(tmp_path):
    path = tmp_path / "SPEND_REAL_PASS.yaml"
    path.write_text(_SPEND_REAL_YAML, encoding="utf-8")
    return load_scenario(path)


def test_spend_is_real_passes_on_a_real_unprotected_escape(
        tmp_path, loaded_state_path, local_server_command, local_server_env):
    scenario = _spend_real_scenario(tmp_path)

    async def _run():
        async with _connected(
                loaded_state_path, local_server_command, local_server_env
        ) as client:
            await assert_spend_is_real(
                client, scenario, topology_path=TOPOLOGY_PATH)

    asyncio.run(_run())


def test_spend_is_real_rejects_an_unreachable_residual_ceiling(
        tmp_path, loaded_state_path, local_server_command, local_server_env):
    """The same real escape as above, but a negative ceiling is unreachable
    by construction (`p_cut` is a probability, never negative) -- proving
    the residual check, not just the "no candidate" check, actually fires."""
    scenario = _spend_real_scenario(tmp_path)

    async def _run():
        async with _connected(
                loaded_state_path, local_server_command, local_server_env
        ) as client:
            with pytest.raises(PairInvalid, match="above the ceiling"):
                await assert_spend_is_real(
                    client, scenario, topology_path=TOPOLOGY_PATH,
                    residual_ceiling=-1.0)

    asyncio.run(_run())


@pytest.mark.parametrize("half", ("T1a-2026-08-30", "T1b-2026-08-30"))
def test_spend_is_real_rejects_storm_svc_1_which_had_no_escape(
        half, loaded_state_path, local_server_command, local_server_env):
    """Neither half of the PRE-REDESIGN T1 had a real escape for
    storm-svc-1 under the spend decider's own avoid (see the module comment
    above) -- a real, live-confirmed property of that geometry and the
    failure mode this test documents. Repointed at the discarded fixtures by
    Task 15: the rebuilt T1 has a real escape in BOTH halves (8 candidates,
    residual p_cut 0.0, no protection collision), which
    `test_episodes.py::test_t1_spend_is_real` now checks on the PASS side."""
    scenario = load_scenario(DISCARDED_DIR / f"{half}.yaml")

    async def _run():
        async with _connected(
                loaded_state_path, local_server_command, local_server_env
        ) as client:
            with pytest.raises(PairInvalid, match="no real escape"):
                await assert_spend_is_real(
                    client, scenario, topology_path=TOPOLOGY_PATH)

    asyncio.run(_run())


from storm_reoptimizer.eval.assertions import probe_flip_mismatches, probe_reading

RESTORABLE = {"status": "solution", "full_restore_candidates": 2,
              "min_spares_needed_by_site": {"jalgaon": 1, "khandwa": 1},
              "levers": ["optical_reroute"]}
FREE = {"status": "solution", "full_restore_candidates": 1,
        "min_spares_needed_by_site": {}, "levers": ["ip_reroute"]}
DEAD = {"status": "no_solution", "full_restore_candidates": 0,
        "min_spares_needed_by_site": None, "levers": []}
# Answers with scope fields (as returned from probe.answer_probe after Task 3)
RESTORABLE_WITH_SCOPE_A = {**RESTORABLE, "scope": "answered while avoiding every asset in rg_T1a_t1_t3; narrower avoid sets were not evaluated"}
RESTORABLE_WITH_SCOPE_B = {**RESTORABLE, "scope": "answered while avoiding every asset in rg_T1b_t1_t3; narrower avoid sets were not evaluated"}


def test_probe_reading_names_the_fact_each_kind_asks_about():
    assert probe_reading("restorable", RESTORABLE) == "restorable"
    assert probe_reading("restorable", DEAD) == "not_restorable"
    assert probe_reading("spares_needed", RESTORABLE) == "spare"
    assert probe_reading("spares_needed", FREE) == "free"
    assert probe_reading("spares_needed", DEAD) == "not_restorable"


def _twin_with_probe_flip(tmp_path, a_flip, b_flip, claimants=("c1",)):
    a = _half(tmp_path, "A", metadata=dict(claimant_services=list(claimants),
                                           probe_flip=a_flip))
    b = _half(tmp_path, "B", metadata=dict(claimant_services=list(claimants),
                                           probe_flip=b_flip))
    return a, b


def test_a_t2_shaped_pair_flips_on_restorability(tmp_path):
    a, b = _twin_with_probe_flip(
        tmp_path,
        {"kind": "restorable", "claimant": "c1", "expected": "restorable"},
        {"kind": "restorable", "claimant": "c1", "expected": "not_restorable"})
    assert probe_flip_mismatches(a, {"c1": RESTORABLE}, b, {"c1": DEAD}) == []


def test_a_half_whose_reading_disagrees_with_its_declaration_is_reported(tmp_path):
    a, b = _twin_with_probe_flip(
        tmp_path,
        {"kind": "restorable", "claimant": "c1", "expected": "restorable"},
        {"kind": "restorable", "claimant": "c1", "expected": "not_restorable"})
    problems = probe_flip_mismatches(a, {"c1": RESTORABLE}, b, {"c1": RESTORABLE})
    assert any("B" in p and "not_restorable" in p for p in problems)


def test_a_t3_shaped_pair_flips_on_spares_needed_across_two_claimants(tmp_path):
    a, b = _twin_with_probe_flip(
        tmp_path,
        {"kind": "spares_needed", "claimant": "ck", "expected": "spare"},
        {"kind": "spares_needed", "claimant": "cd", "expected": "free"},
        claimants=("ck", "cd"))
    answers = {"ck": RESTORABLE, "cd": FREE}
    assert probe_flip_mismatches(a, answers, b, answers) == []


def test_an_unnamed_claimant_must_answer_identically_in_both_halves(tmp_path):
    a, b = _twin_with_probe_flip(
        tmp_path,
        {"kind": "restorable", "claimant": "c1", "expected": "restorable"},
        {"kind": "restorable", "claimant": "c1", "expected": "not_restorable"},
        claimants=("c1", "c2"))
    problems = probe_flip_mismatches(
        a, {"c1": RESTORABLE, "c2": FREE}, b, {"c1": DEAD, "c2": RESTORABLE})
    assert any("c2" in p for p in problems)


def test_a_pair_with_no_declared_probe_flip_must_probe_identically(tmp_path):
    # T1's shape: the flip is in the observation, so the probe must not
    # carry one. Identical answers pass; a difference is an undeclared flip.
    a = _half(tmp_path, "A", metadata=dict(claimant_services=["c1"]))
    b = _half(tmp_path, "B", metadata=dict(claimant_services=["c1"]))
    assert probe_flip_mismatches(a, {"c1": RESTORABLE}, b, {"c1": RESTORABLE}) == []
    assert probe_flip_mismatches(a, {"c1": RESTORABLE}, b, {"c1": DEAD})


def test_scope_differences_are_ignored_for_t1_shaped_pairs(tmp_path):
    """Task 3 added `scope` field to probe answers. `scope` names the
    risk-group id (which embeds scenario.id), so it ALWAYS differs between
    halves by construction. The comparison should ignore scope and compare
    only the semantic content (status, full_restore_candidates, etc.)."""
    a = _half(tmp_path, "A", metadata=dict(claimant_services=["c1"]))
    b = _half(tmp_path, "B", metadata=dict(claimant_services=["c1"]))
    # Same semantic content but different scope (as it naturally differs by scenario)
    problems = probe_flip_mismatches(
        a, {"c1": RESTORABLE_WITH_SCOPE_A},
        b, {"c1": RESTORABLE_WITH_SCOPE_B})
    assert problems == [], f"expected no problems but got: {problems}"


def test_scope_differences_dont_hide_real_semantic_differences(tmp_path):
    """Even though scope is ignored, real semantic differences should still
    be caught."""
    a = _half(tmp_path, "A", metadata=dict(claimant_services=["c1"]))
    b = _half(tmp_path, "B", metadata=dict(claimant_services=["c1"]))
    # Same scope, but different semantic content
    problems = probe_flip_mismatches(
        a, {"c1": RESTORABLE_WITH_SCOPE_A},
        b, {"c1": {**DEAD, "scope": RESTORABLE_WITH_SCOPE_A["scope"]}})
    assert len(problems) > 0, "expected to catch the real semantic difference"
    assert "c1" in problems[0]


def test_the_two_halves_must_declare_different_expectations(tmp_path):
    a, b = _twin_with_probe_flip(
        tmp_path,
        {"kind": "restorable", "claimant": "c1", "expected": "restorable"},
        {"kind": "restorable", "claimant": "c1", "expected": "restorable"})
    problems = probe_flip_mismatches(a, {"c1": RESTORABLE}, b, {"c1": RESTORABLE})
    assert any("differ" in p for p in problems)
