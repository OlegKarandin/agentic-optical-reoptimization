"""Scalars derived from an episode's real forecast geometry (eval design
spec, "The one-variable check"; whole-branch review 2026-08-23, finding C2).

These are the numbers an episode's author did NOT declare. The confound that
survived three rounds of review of T1 -- `p_cut(storm-svc-1)` = 0.1349 in one
half and 0.8834 in the other, a bare threshold that answers the pair 2/2 --
was invisible precisely because nothing computed it. Everything here is pure:
the service-under-test coordinate is an argument, so the geometry is testable
without a server. The MCP-backed half is exercised in test_episodes.py and
test_rules.py."""
import textwrap

import pytest

from storm_reoptimizer.eval.cone import cut_probability, radial_offset_km
from storm_reoptimizer.eval.derived import (
    DERIVED_VARS, FLIP_VARS, DerivedGeometryError, derived_geometry_from_point,
    exposure_horizon_hour, flip_scalars_from_points, horizon_widths_km,
    sut_p_cut_at_exposure_horizon, within_issuance_cone_motion_kmh,
)
from storm_reoptimizer.eval.rules import (
    ENUMERATED_VARS, OBSERVABLE_VARS, best_rule, observables,
)
from storm_reoptimizer.eval.scenario_file import load_scenario

# Two horizons in the decision-hour issuance, so within-issuance motion is a
# real number rather than the degenerate 0.0 a single-horizon issuance gives.
EPISODE = textwrap.dedent("""
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
    damage_radius_km: 74
    reference_avoid: {{}}
    forecast:
      t0:
        t3: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: 25.0, lon: 81.0}}}}
      t1:
        t2: {{cone: {{type: Polygon, coordinates: []}}, width_km: 90, center: {{lat: 25.0, lon: 81.0}}}}
        t3: {{cone: {{type: Polygon, coordinates: []}}, width_km: {t3_width}, center: {{lat: {t3_lat}, lon: 81.0}}}}
    realized: {{}}
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
      exposure_horizon_hours: {horizon}
      spares_on_hand: 1
""")

# Somewhere on the toy topology, far enough off the t3 cone axis in one
# fixture and dead-centre in the other that p_cut genuinely separates.
SUT_POINT = (25.0, 81.0)


def _episode(tmp_path, sid, *, label="wait", t3_lat=25.0, t3_width=90,
             horizon=2):
    path = tmp_path / f"{sid}.yaml"
    path.write_text(EPISODE.format(id=sid, label=label, t3_lat=t3_lat,
                                   t3_width=t3_width, horizon=horizon),
                    encoding="utf-8")
    return load_scenario(path)


def test_exposure_horizon_counts_hour_LABEL_index_steps(tmp_path):
    # decision_hour t1 is index 1; exposure_horizon_hours=2 names hours[3].
    assert exposure_horizon_hour(_episode(tmp_path, "A")) == "t3"


def test_an_exposure_horizon_the_issuance_does_not_publish_is_an_error(tmp_path):
    # horizon=1 names t2, which the t1 issuance DOES publish...
    assert exposure_horizon_hour(_episode(tmp_path, "A", horizon=1)) == "t2"
    # ...but horizon=-1 names t0, which it cannot forecast.
    with pytest.raises(DerivedGeometryError, match="publishes no cone"):
        exposure_horizon_hour(_episode(tmp_path, "A", horizon=-1))


def test_an_exposure_horizon_off_the_end_of_the_timeline_is_an_error(tmp_path):
    with pytest.raises(DerivedGeometryError, match="names no hour"):
        exposure_horizon_hour(_episode(tmp_path, "A", horizon=9))


def test_p_cut_is_the_same_arithmetic_the_observation_shows_the_decider(tmp_path):
    """Not an independent reimplementation -- deliberately the same
    radial_offset_km -> cut_probability chain build_observation puts in front
    of the decider, because the point is that this is a number the agent can
    read and therefore a number a one-line rule can key on."""
    scenario = _episode(tmp_path, "A", t3_lat=25.9)
    offset, p_cut = sut_p_cut_at_exposure_horizon(scenario, SUT_POINT)
    expected_offset = radial_offset_km(25.9, 81.0, *SUT_POINT)
    assert offset == pytest.approx(expected_offset)
    assert p_cut == pytest.approx(cut_probability(expected_offset, 90.0, 74.0))
    # Unrounded, unlike the observation's 4-decimal display copy: the
    # equality check across halves must not paper over a real difference.
    assert p_cut != round(p_cut, 4) or p_cut in (0.0, 1.0)


def test_p_cut_is_near_one_on_the_cone_axis_and_falls_off_it(tmp_path):
    on_axis = sut_p_cut_at_exposure_horizon(
        _episode(tmp_path, "A", t3_lat=25.0), SUT_POINT)[1]
    off_axis = sut_p_cut_at_exposure_horizon(
        _episode(tmp_path, "B", t3_lat=26.0), SUT_POINT)[1]
    assert on_axis > off_axis
    assert 0.0 < off_axis < on_axis < 1.0


def test_within_issuance_motion_is_zero_when_the_horizons_share_a_centre(tmp_path):
    # This is T1b's shape: t2 and t3 both centred on the same point.
    assert within_issuance_cone_motion_kmh(
        _episode(tmp_path, "B", t3_lat=25.0)) == pytest.approx(0.0)


def test_within_issuance_motion_measures_the_issuances_OWN_horizons(tmp_path):
    """The declared `cone_motion_kmh` describes the t0 -> decision-hour
    REVISION of one horizon. This is a different quantity: how far the
    decision-hour issuance's cone moves between its own t2 and t3 cones. T1
    declares 63.0 in both halves while this one is 126.1 vs 0.0 -- which is
    the whole reason it needed deriving."""
    scenario = _episode(tmp_path, "A", t3_lat=26.0)
    expected = radial_offset_km(25.0, 81.0, 26.0, 81.0)   # one index step
    assert within_issuance_cone_motion_kmh(scenario) == pytest.approx(expected)
    assert scenario.metadata["cone_motion_kmh"] != pytest.approx(expected)


def test_horizon_widths_cover_every_cone_the_issuance_publishes(tmp_path):
    # The declared cone_width_km names ONE horizon; a pair whose halves
    # differ only by a width epsilon on ANOTHER horizon slips past it.
    assert horizon_widths_km(_episode(tmp_path, "A", t3_width=90.2)) == {
        "t2": 90.0, "t3": 90.2}


def test_derived_geometry_reports_the_inputs_alongside_the_numbers(tmp_path):
    derived = derived_geometry_from_point(
        _episode(tmp_path, "A", t3_lat=25.9), SUT_POINT)
    assert derived.scenario_id == "A"
    assert derived.sut_point == SUT_POINT
    assert derived.decision_hour == "t1"
    assert derived.exposure_horizon == "t3"
    assert set(derived.scalars()) == {
        "sut_p_cut_at_exposure_horizon", "within_issuance_cone_motion_kmh"}


def test_the_rule_enumeration_widens_when_derived_values_are_supplied(tmp_path):
    """The acceptance property for finding C2, in miniature: a pair whose
    halves are indistinguishable on every DECLARED scalar is nonetheless
    solved outright by a threshold on the derived p_cut -- and the check now
    sees it only because the derived value is passed in."""
    a = _episode(tmp_path, "Pa", label="wait", t3_lat=25.0)   # SUT dead-centre
    b = _episode(tmp_path, "Pb", label="act", t3_lat=26.4)    # SUT far off-axis
    pair = [a, b]
    # Every declared scalar is equal, so declared-only enumeration is at
    # chance: only the constant greedy rules can reach 0.5-per-half parity.
    assert best_rule(pair)[1] < 1.0

    derived = {e.id: derived_geometry_from_point(e, SUT_POINT).scalars()
               for e in pair}
    rule, score = best_rule(pair, derived)
    assert score == pytest.approx(1.0)
    assert rule.name.startswith("threshold:sut_p_cut_at_exposure_horizon")


def test_derived_values_override_a_declaration_of_the_same_name(tmp_path):
    """`observables` prefers the measured number to the claim about it --
    the declaration is what an author typed, and this whole module exists
    because authors' declarations were not being checked."""
    scenario = _episode(tmp_path, "A")
    assert observables(scenario)["cone_width_km"] == 90
    assert observables(scenario, {"A": {"cone_width_km": 12.0}})[
        "cone_width_km"] == 12.0


def test_the_enumeration_is_the_declared_list_plus_the_derived_list():
    assert DERIVED_VARS == ("sut_p_cut_at_exposure_horizon",)
    assert ENUMERATED_VARS == OBSERVABLE_VARS + DERIVED_VARS


# One earlier horizon (t2) plus the exposure horizon (t3), so the
# "before" aggregate is a real sum over a real horizon rather than 0.0.
_TWO_HORIZON_T1 = textwrap.dedent("""
      t1:
        t2: {cone: {type: Polygon, coordinates: [[[81.0, 25.0], [81.1, 25.0], [81.1, 25.1], [81.0, 25.0]]]}, width_km: 90, center: {lat: 25.0, lon: 81.0}}
        t3: {cone: {type: Polygon, coordinates: [[[81.0, 25.0], [81.1, 25.0], [81.1, 25.1], [81.0, 25.0]]]}, width_km: 90, center: {lat: 25.2, lon: 81.0}}
""").rstrip("\n")


def _two_horizon_scenario(example_scenario_yaml, write_scenario):
    old = "\n".join(example_scenario_yaml.splitlines()[18:20])   # the single-horizon t1 block
    # The dedented replacement above lands at 0/2-space indent; re-indent by
    # 2 to match `old`'s own 2/4-space nesting under `forecast:` -- a plain
    # `.strip("\n")` swap (as first drafted) collapses "  t1:" to "t1:" and
    # silently unparents it from `forecast:`, confirmed by round-tripping
    # this exact substitution through yaml.safe_load before trusting it.
    new = textwrap.indent(_TWO_HORIZON_T1.strip("\n"), "  ")
    return load_scenario(write_scenario(
        example_scenario_yaml.replace(old, new)))


def test_the_claimant_aggregate_excludes_the_service_under_test(
        example_scenario_yaml, write_scenario):
    scenario = _two_horizon_scenario(example_scenario_yaml, write_scenario)
    points = {"storm-svc-1": (25.2, 81.0), "svc-b": (25.2, 81.0)}
    demands = {"storm-svc-1": 300.0, "svc-b": 100.0}

    flip = flip_scalars_from_points(scenario, points=points,
                                    demands_gbps=demands)

    # Both services sit on the SAME point, so the SUT's own exposure and the
    # claimant's are the same p_cut. If the SUT leaked into the claimant sum
    # the aggregate would be 4x svc-b's contribution, not 1x.
    sut = flip.sut_ecar_by_horizon[flip.exposure_horizon]
    assert flip.claimant_ecar_at_exposure_horizon == pytest.approx(sut / 3.0)


def test_the_before_aggregate_sums_only_strictly_earlier_horizons(
        example_scenario_yaml, write_scenario):
    scenario = _two_horizon_scenario(example_scenario_yaml, write_scenario)
    points = {"storm-svc-1": (25.2, 81.0), "svc-b": (25.0, 81.0)}
    demands = {"storm-svc-1": 300.0, "svc-b": 100.0}

    flip = flip_scalars_from_points(scenario, points=points,
                                    demands_gbps=demands)

    assert flip.exposure_horizon == "t3"
    assert flip.earlier_horizons == ("t2",)
    # svc-b sits exactly on the t2 cone centre, so its p_cut there is the
    # highest it gets anywhere in this issuance.
    assert (flip.claimant_ecar_before_exposure_horizon
            > flip.claimant_ecar_at_exposure_horizon)
    assert flip.claimant_ecar_peak_over_horizons == pytest.approx(
        flip.claimant_ecar_before_exposure_horizon)


def test_a_single_horizon_issuance_has_no_before_aggregate(
        example_scenario_yaml, write_scenario):
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    flip = flip_scalars_from_points(
        scenario, points={"storm-svc-1": (25.2, 81.0), "svc-b": (25.0, 81.0)},
        demands_gbps={"storm-svc-1": 300.0, "svc-b": 100.0})

    assert flip.earlier_horizons == ()
    assert flip.claimant_ecar_before_exposure_horizon == 0.0


def test_the_flip_values_view_carries_exactly_the_swept_scalars(
        example_scenario_yaml, write_scenario):
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    flip = flip_scalars_from_points(
        scenario, points={"storm-svc-1": (25.2, 81.0)},
        demands_gbps={"storm-svc-1": 300.0})

    assert set(flip.values()) == set(FLIP_VARS)


def test_a_missing_sut_point_raises_rather_than_silently_zeroing(
        example_scenario_yaml, write_scenario):
    """Mirrors `derived_geometry`'s existing behaviour for the identical
    failure (no working-path coordinates for service_under_test): a missing
    SUT point must raise, not silently zero `sut_ecar_by_horizon` -- exactly
    the field W1.5's `assert_flip_dominates` reads as the SUT's own side of
    the comparison. A silent zero there would let that check draw a wrong
    conclusion with no error signal at all."""
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    with pytest.raises(DerivedGeometryError, match="storm-svc-1"):
        flip_scalars_from_points(
            scenario, points={"svc-b": (25.0, 81.0)},
            demands_gbps={"storm-svc-1": 300.0, "svc-b": 100.0})
