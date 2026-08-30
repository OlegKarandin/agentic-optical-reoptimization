"""The twin-pair validity checks (eval design spec, "Twin-pair discipline")
and their own tests -- "a deliberately invalid pair (same gold both halves)
must be rejected by assertions.py"."""
import textwrap

import pytest

from storm_reoptimizer.eval.assertions import (
    PairInvalid, _check_no_free_escape, assert_flip_dominates,
    assert_gold_choices_differ, assert_issuance_prefix_shared,
    assert_no_global_policy_solves_the_suite, assert_shared_scalars_equal,
)
from storm_reoptimizer.eval.derived import FlipScalars
from storm_reoptimizer.eval.scenario_file import (
    SCENARIOS_DIR, load_scenario,
)

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
    reference_avoid: {{risk_groups: [rg_ref]}}
    forecast:
      t0:
        t3: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: {t0_lat}, lon: 81.0}}}}
      t1:
        t3: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: {t1_lat}, lon: 81.0}}}}
    realized:
      t3: [fiber_004]
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
""")


def _half(tmp_path, sid, *, label, t0_lat=25.0, t1_lat=25.0):
    path = tmp_path / f"{sid}.yaml"
    path.write_text(
        TWIN.format(id=sid, label=label, t0_lat=t0_lat, t1_lat=t1_lat),
        encoding="utf-8")
    return load_scenario(path)


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
    reference_avoid: {{risk_groups: [rg_ref]}}
    forecast:
      t0:
        t3: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: 25.0, lon: 81.0}}}}
      t1:
        t3: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: 25.6, lon: 81.0}}}}
    realized:
      t3: [fiber_004]
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
# -- "moves the service" alone is necessary but not sufficient, and two of the
# suite's three shipped conserve halves (T2b, T3b) were false positives under
# the old, broader check. `_T1_KW`/etc. below fix the label_rule/gold_label
# a T1a-shaped half needs; the new tests exercise T2/T3-shaped halves whose
# free escape does NOT flip the label.
_T1_KW = dict(label_rule="timing_at_decision_hour", gold_label="wait",
             reference_avoid={})


def test_a_free_candidate_that_moves_the_service_defeats_a_conserve_gold():
    """The T1a failure mode: a 0-pair candidate the agent can take to improve
    its own position while keeping the spare. Gold says "don't spend"; the
    label rule reads "acted"; both are satisfiable at once -- and for
    `timing_at_decision_hour`, ANY commit reads "act", so this is a genuine
    escape regardless of lever."""
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


def test_a_free_escape_that_does_not_flip_an_avoid_horizon_label_is_safe():
    """T2b-shaped false positive (Finding #5): the free candidate is offered
    under a neutral `reference_avoid={}`, so its `risk_groups` is empty and
    `avoid_horizon_at_decision_hour` still reads "narrow" -- unchanged from
    gold. The old, broader predicate (any zero-pair, service-moving
    candidate) would have flagged this; the label-aware one must not."""
    menu = {"status": "solution", "candidates": [
        {"lever": "ip_reroute", "reused_lightpaths": ["lp-prot-storm-svc-1-0"],
         "new_lightpaths": [], "restored_gbps": 300.0,
         "shortfall_gbps": 0.0, "cost_vector": {}}]}
    _check_no_free_escape(
        "T2b", menu, current={"lp-cand-storm-svc-1-0"},
        label_rule="avoid_horizon_at_decision_hour", gold_label="narrow",
        reference_avoid={}, wide_avoid_risk_group="rg_T2b_t1_t6")


def test_a_free_escape_that_would_flip_an_avoid_horizon_label_still_raises():
    """The mirror case: if the free candidate's `reference_avoid` DID name the
    wide risk group, committing it would read "wide" -- a real escape against
    a "narrow" gold, and the check must still catch it."""
    menu = {"status": "solution", "candidates": [
        {"lever": "ip_reroute", "reused_lightpaths": ["lp-prot-storm-svc-1-0"],
         "new_lightpaths": [], "restored_gbps": 300.0,
         "shortfall_gbps": 0.0, "cost_vector": {}}]}
    with pytest.raises(PairInvalid, match="GRADED LABEL"):
        _check_no_free_escape(
            "T2b", menu, current={"lp-cand-storm-svc-1-0"},
            label_rule="avoid_horizon_at_decision_hour", gold_label="narrow",
            reference_avoid={"risk_groups": ["rg_T2b_t1_t6"]},
            wide_avoid_risk_group="rg_T2b_t1_t6")


def test_a_free_escape_that_does_not_flip_a_chosen_lever_label_is_safe():
    """T3b-shaped false positive (Finding #5): the free candidate IS an
    ip_reroute, and `label_by_lever` already maps ip_reroute to gold's own
    label "B" -- committing it reads the SAME label, not a different one."""
    menu = {"status": "solution", "candidates": [
        {"lever": "ip_reroute", "reused_lightpaths": ["lp-prot-storm-svc-1-0"],
         "new_lightpaths": [], "restored_gbps": 300.0,
         "shortfall_gbps": 0.0, "cost_vector": {}}]}
    _check_no_free_escape(
        "T3b", menu, current={"lp-cand-storm-svc-1-0"},
        label_rule="chosen_lever_at_decision_hour", gold_label="B",
        reference_avoid={},
        label_by_lever={"optical_reroute": "A", "hybrid": "A",
                        "ip_reroute": "B"})


def test_a_free_escape_that_would_flip_a_chosen_lever_label_still_raises():
    """The mirror case: an optical_reroute free escape against a "B" gold
    would read "A" -- a real, catchable escape."""
    menu = {"status": "solution", "candidates": [
        {"lever": "optical_reroute", "reused_lightpaths": ["lp-elsewhere"],
         "new_lightpaths": [{"oms_sequence": ["oms_1"]}],
         "restored_gbps": 300.0, "shortfall_gbps": 0.0, "cost_vector": {}}]}
    # NOTE: pairs_needed reads len(new_lightpaths), so this candidate is not
    # actually zero-pair and would be skipped by the real predicate -- this
    # fixture exists only to exercise _label_if_committed's chosen_lever
    # branch directly, via the module-private helper, not through
    # _check_no_free_escape's pairs_needed gate.
    from storm_reoptimizer.eval.assertions import _label_if_committed
    label = _label_if_committed(
        label_rule="chosen_lever_at_decision_hour",
        candidate=menu["candidates"][0], avoid_used={},
        wide_avoid_risk_group=None,
        label_by_lever={"optical_reroute": "A", "hybrid": "A",
                        "ip_reroute": "B"})
    assert label == "A"
