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

from storm_reoptimizer.eval.cone import (
    expected_capacity_at_risk_gbps, nearest_span_offset_km, p_cut_region,
    p_cut_service, radial_offset_km,
)
from storm_reoptimizer.eval.derived import (
    DERIVED_VARS, FLIP_VARS, DerivedGeometryError, decision_issuance,
    derived_geometry_from_spans, exposure_horizon_hour, flip_scalars_from_spans,
    horizon_widths_km, sut_p_cut_at_exposure_horizon,
    within_issuance_cone_motion_kmh,
)
from storm_reoptimizer.eval.observation import _restorable_groups
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
    depot_site: satna
    spare_inventory: {{satna: 1}}
    damage_radius_km: 74
    track_revision_km_per_hour_ahead: 30
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
      claimant_services: []
""")

# Somewhere on the toy topology, far enough off the t3 cone axis in one
# fixture and dead-centre in the other that p_cut genuinely separates.
def _span(lat, lon):
    """A degenerate (zero-length) span at one point. p_cut_region's own
    correctness argument (test_exposure_model.py) is that the region model
    degenerates EXACTLY to the closed-form point model as a span shrinks to
    zero length -- so this stands in for the old representative-point tests
    without changing what any of them assert."""
    return (lat, lon), (lat, lon)


SUT_SPANS = (_span(25.0, 81.0),)


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
    nearest_span_offset_km -> p_cut_region chain build_observation puts in
    front of the decider, because the point is that this is a number the
    agent can read and therefore a number a one-line rule can key on."""
    scenario = _episode(tmp_path, "A", t3_lat=25.9)
    offset, p_cut = sut_p_cut_at_exposure_horizon(scenario, SUT_SPANS)
    expected_offset = nearest_span_offset_km(SUT_SPANS, 25.9, 81.0)
    assert offset == pytest.approx(expected_offset)
    assert p_cut == pytest.approx(
        p_cut_region(SUT_SPANS, 25.9, 81.0, 90.0, 74.0))
    # Unrounded, unlike the observation's 4-decimal display copy: the
    # equality check across halves must not paper over a real difference.
    assert p_cut != round(p_cut, 4) or p_cut in (0.0, 1.0)


def test_p_cut_is_near_one_on_the_cone_axis_and_falls_off_it(tmp_path):
    on_axis = sut_p_cut_at_exposure_horizon(
        _episode(tmp_path, "A", t3_lat=25.0), SUT_SPANS)[1]
    off_axis = sut_p_cut_at_exposure_horizon(
        _episode(tmp_path, "B", t3_lat=26.0), SUT_SPANS)[1]
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
    derived = derived_geometry_from_spans(
        _episode(tmp_path, "A", t3_lat=25.9), SUT_SPANS)
    assert derived.scenario_id == "A"
    assert derived.sut_spans == SUT_SPANS
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

    derived = {e.id: derived_geometry_from_spans(e, SUT_SPANS).scalars()
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
    # Locate the t1 issuance's own single-horizon block by CONTENT, not a
    # hardcoded line-index slice (Finding #10, 2026-08-26 re-review: the
    # original `splitlines()[18:20]` would silently break if an unrelated
    # edit to `example_scenario_yaml` shifted its line count -- contained
    # blast radius, since a wrong slice makes `earlier_horizons` come out
    # empty and a downstream assertion like `earlier_horizons == ("t2",)`
    # then fails loudly, but a content anchor survives the edit instead of
    # merely failing safely after it). "  t1:" (exactly two-space indent) is
    # the forecast issuance header for hour t1 -- unique in this fixture; the
    # per-hour issuance headers under `forecast:` are the only lines at that
    # indent, and every OTHER "t1" token in the fixture (decision_hour: t1,
    # the t0 issuance's own "t1:" horizon key) sits at a different indent or
    # column. Take that header line plus its immediately-following horizon
    # line -- the whole single-horizon block this fixture declares for t1.
    lines = example_scenario_yaml.splitlines()
    start = next(i for i, line in enumerate(lines) if line == "  t1:")
    old = "\n".join(lines[start:start + 2])
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
    spans = {"storm-svc-1": (_span(25.2, 81.0),), "svc-b": (_span(25.2, 81.0),)}
    demands = {"storm-svc-1": 300.0, "svc-b": 100.0}

    flip = flip_scalars_from_spans(scenario, spans=spans,
                                   demands_gbps=demands)

    # Both services sit on the SAME point, so the SUT's own exposure and the
    # claimant's are the same p_cut. If the SUT leaked into the claimant sum
    # the aggregate would be 4x svc-b's contribution, not 1x.
    sut = flip.sut_ecar_by_horizon[flip.exposure_horizon]
    assert flip.claimant_ecar_at_exposure_horizon == pytest.approx(sut / 3.0)


def test_the_before_aggregate_sums_only_strictly_earlier_horizons(
        example_scenario_yaml, write_scenario):
    scenario = _two_horizon_scenario(example_scenario_yaml, write_scenario)
    spans = {"storm-svc-1": (_span(25.2, 81.0),), "svc-b": (_span(25.0, 81.0),)}
    demands = {"storm-svc-1": 300.0, "svc-b": 100.0}

    flip = flip_scalars_from_spans(scenario, spans=spans,
                                   demands_gbps=demands)

    assert flip.exposure_horizon == "t3"
    assert flip.earlier_horizons == ("t2",)
    # svc-b sits exactly on the t2 cone centre, so its p_cut there is the
    # highest it gets anywhere in this issuance.
    assert (flip.claimant_ecar_before_exposure_horizon
            > flip.claimant_ecar_at_exposure_horizon)
    assert flip.claimant_ecar_peak_over_horizons == pytest.approx(
        flip.claimant_ecar_before_exposure_horizon)


def test_the_min_aggregate_picks_the_other_horizon_than_the_peak(
        example_scenario_yaml, write_scenario):
    """`claimant_ecar_min_over_horizons` reads the SMALLEST horizon, which on
    this fixture is the exposure horizon rather than the earlier one the peak
    reads. That is the whole point of the variant (added 2026-08-26): which
    horizon it reads is decided by the VALUES, not fixed in advance, so it can
    follow a pair's discriminating horizon around without being told where it
    is -- which is how it solved the shipped suite 6/6 when the other three
    variants could not."""
    scenario = _two_horizon_scenario(example_scenario_yaml, write_scenario)
    spans = {"storm-svc-1": (_span(25.2, 81.0),), "svc-b": (_span(25.0, 81.0),)}
    demands = {"storm-svc-1": 300.0, "svc-b": 100.0}

    flip = flip_scalars_from_spans(scenario, spans=spans,
                                   demands_gbps=demands)

    assert (flip.claimant_ecar_min_over_horizons
            < flip.claimant_ecar_peak_over_horizons)
    # The peak is the t2 (earlier) horizon here, so the min must be the t3
    # (exposure) one -- a different horizon, from the same two numbers.
    assert flip.claimant_ecar_min_over_horizons == pytest.approx(
        flip.claimant_ecar_at_exposure_horizon)
    assert flip.claimant_ecar_peak_over_horizons == pytest.approx(
        flip.claimant_ecar_before_exposure_horizon)


def test_the_min_aggregate_reads_the_EARLIER_horizon_when_that_is_smaller(
        example_scenario_yaml, write_scenario):
    """The mirror of the test above, and the direction the SHIPPED suite
    actually turns on: T2's flip lives in its NEAR (earlier) horizon, which
    is the smaller one, so `min` reads it and `..._at_exposure_horizon` --
    which reads T2's byte-identical far horizon -- cannot see the flip at
    all.

    Same fixture, same two horizons; only the claimant's position moves, from
    the `t2` centre to the `t3` centre. That alone swaps which horizon each
    of `min`/`peak` reads, with no change to the code path -- which is the
    property that makes this variable able to follow a pair's discriminating
    horizon around, and the reason it solved the suite."""
    scenario = _two_horizon_scenario(example_scenario_yaml, write_scenario)
    # svc-b now sits on the t3 (exposure) centre rather than the t2 one, so
    # the EXPOSURE horizon is its high-water mark and the earlier horizon is
    # the small one.
    spans = {"storm-svc-1": (_span(25.0, 81.0),), "svc-b": (_span(25.2, 81.0),)}
    demands = {"storm-svc-1": 300.0, "svc-b": 100.0}

    flip = flip_scalars_from_spans(scenario, spans=spans,
                                   demands_gbps=demands)

    assert flip.earlier_horizons == ("t2",)
    assert (flip.claimant_ecar_before_exposure_horizon
            < flip.claimant_ecar_at_exposure_horizon)
    # min reads the EARLIER horizon here; peak reads the exposure one --
    # exactly the opposite assignment to the previous test.
    assert flip.claimant_ecar_min_over_horizons == pytest.approx(
        flip.claimant_ecar_before_exposure_horizon)
    assert flip.claimant_ecar_peak_over_horizons == pytest.approx(
        flip.claimant_ecar_at_exposure_horizon)


def test_the_min_and_peak_aggregates_coincide_on_a_single_horizon(
        example_scenario_yaml, write_scenario):
    """One horizon means one value, so min == max == that value -- and it is
    the exposure horizon's, since the issuance publishes nothing else. (This
    is T1's shape after Task 4 deleted its dominating `t2` nowcast.)"""
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    flip = flip_scalars_from_spans(
        scenario, spans={"storm-svc-1": (_span(25.2, 81.0),),
                         "svc-b": (_span(25.0, 81.0),)},
        demands_gbps={"storm-svc-1": 300.0, "svc-b": 100.0})

    assert flip.earlier_horizons == ()
    assert flip.claimant_ecar_min_over_horizons > 0.0
    # Deliberately NOT chained through pytest.approx: `a == approx(b) ==
    # approx(c)` would compare two approx objects in its second link, which is
    # not the assertion intended here.
    assert flip.claimant_ecar_min_over_horizons == pytest.approx(
        flip.claimant_ecar_peak_over_horizons)
    assert flip.claimant_ecar_min_over_horizons == pytest.approx(
        flip.claimant_ecar_at_exposure_horizon)


def test_a_single_horizon_issuance_has_no_before_aggregate(
        example_scenario_yaml, write_scenario):
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    flip = flip_scalars_from_spans(
        scenario, spans={"storm-svc-1": (_span(25.2, 81.0),),
                         "svc-b": (_span(25.0, 81.0),)},
        demands_gbps={"storm-svc-1": 300.0, "svc-b": 100.0})

    assert flip.earlier_horizons == ()
    assert flip.claimant_ecar_before_exposure_horizon == 0.0


def test_the_flip_values_view_carries_exactly_the_swept_scalars(
        example_scenario_yaml, write_scenario):
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    flip = flip_scalars_from_spans(
        scenario, spans={"storm-svc-1": (_span(25.2, 81.0),)},
        demands_gbps={"storm-svc-1": 300.0})

    assert set(flip.values()) == set(FLIP_VARS)
    # Named explicitly, not just "whatever FLIP_VARS says": enumerating three
    # of these four (now five) and believing the enumeration complete is
    # exactly the mistake that let a solving policy sit in the shipped suite
    # (see derived.py's module docstring).
    assert set(FLIP_VARS) == {"claimant_ecar_at_exposure_horizon",
                              "claimant_ecar_before_exposure_horizon",
                              "claimant_ecar_peak_over_horizons",
                              "claimant_ecar_min_over_horizons",
                              "largest_restorable_group_ecar_gbps"}


def test_a_missing_sut_point_raises_rather_than_silently_zeroing(
        example_scenario_yaml, write_scenario):
    """Mirrors `derived_geometry`'s existing behaviour for the identical
    failure (no working-path coordinates for service_under_test): a missing
    SUT entry must raise, not silently zero `sut_ecar_by_horizon` -- exactly
    the field W1.5's `assert_flip_dominates` reads as the SUT's own side of
    the comparison. A silent zero there would let that check draw a wrong
    conclusion with no error signal at all."""
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    with pytest.raises(DerivedGeometryError, match="storm-svc-1"):
        flip_scalars_from_spans(
            scenario, spans={"svc-b": (_span(25.0, 81.0),)},
            demands_gbps={"storm-svc-1": 300.0, "svc-b": 100.0})


def test_the_new_group_total_is_swept_by_the_whole_suite_check():
    # rules.py enumerated only author-DECLARED metadata, which is why T1's
    # original confound survived three review rounds -- nobody had declared
    # the number that solved the pair. We are adding a new number to the
    # observation; if it is not in the enumerated rule space we repeat that
    # mistake exactly.
    #
    # It belongs to the WHOLE-SUITE check, not the per-pair one: it is a
    # claimant-side aggregate, it IS the flip, and rules.py's per-pair
    # scoring would rate any differing variable 1.0 by construction. See the
    # deviation note in Task 11's brief, and rules.py's two-check doctrine.
    assert "largest_restorable_group_ecar_gbps" in FLIP_VARS
    assert "largest_restorable_group_ecar_gbps" not in ENUMERATED_VARS


def test_the_group_total_is_computed_from_the_same_grouping_the_agent_sees(
        example_scenario_yaml, write_scenario):
    """One function, two readers: if the check and the payload could
    disagree, the check is validating a number nobody was shown."""
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    # svc-b and svc-c co-terminate at satna<->raipur -- the depot site is
    # satna (example_scenario_yaml's own depot_site) -- and jointly restored
    # by one lightpath, so their ECARs add. svc-d carries by far the largest
    # demand but terminates kolkata<->mumbai, nowhere near the depot: if the
    # new field were a naive sum over every non-SUT service (T2a's D2
    # mistake) it would be dominated by svc-d's 900 Gbps; the honest figure
    # excludes it entirely.
    endpoint_sites = {"svc-b": ("satna", "raipur"),
                      "svc-c": ("satna", "raipur"),
                      "svc-d": ("kolkata", "mumbai")}
    spans = {"storm-svc-1": (_span(25.0, 81.0),),
             "svc-b": (_span(25.2, 81.0),),
             "svc-c": (_span(25.2, 81.0),),
             "svc-d": (_span(25.2, 81.0),)}
    demands = {"storm-svc-1": 300.0, "svc-b": 100.0, "svc-c": 50.0,
              "svc-d": 900.0}

    flip = flip_scalars_from_spans(scenario, spans=spans, demands_gbps=demands,
                                   endpoint_sites=endpoint_sites)

    cone = decision_issuance(scenario).horizons["t3"]
    exposure = {
        svc_id: {"t3": {"expected_capacity_at_risk_gbps":
                       expected_capacity_at_risk_gbps(
                           p_cut_region(svc_spans, cone.center["lat"],
                                       cone.center["lon"], cone.width_km,
                                       scenario.damage_radius_km),
                           demands[svc_id])}}
        for svc_id, svc_spans in spans.items()}
    groups = _restorable_groups(exposure, endpoint_sites,
                                depot_site=scenario.depot_site,
                                service_under_test="storm-svc-1")

    assert flip.largest_restorable_group_ecar_gbps == pytest.approx(
        max(g["ecar_gbps"] for g in groups["t3"]), abs=1e-6)
    assert flip.largest_restorable_group_ecar_gbps < 900.0
    assert flip.largest_restorable_group_ecar_gbps > 0.0


# Task 4 (protection-aware p_cut): a protection leg with no cuttable span
# anywhere near the cone drives the JOINT (both-legs-cut) probability toward
# zero, even though the working leg alone sits dead-centre on it -- the whole
# architectural point of reading `p_cut_service` instead of `p_cut_region`
# for a protected service. `far_protection` stands in for "a protection leg
# this storm cannot touch."
FAR_PROTECTION = (_span(40.0, 81.0),)


def test_sut_p_cut_at_exposure_horizon_uses_the_joint_probability_when_protected(
        tmp_path):
    scenario = _episode(tmp_path, "A", t3_lat=25.0)   # SUT dead-centre
    working_only = sut_p_cut_at_exposure_horizon(scenario, SUT_SPANS)[1]
    joint = sut_p_cut_at_exposure_horizon(
        scenario, SUT_SPANS, FAR_PROTECTION)[1]
    assert working_only > 0.9
    assert joint == pytest.approx(
        p_cut_service(SUT_SPANS, FAR_PROTECTION, 25.0, 81.0, 90.0, 74.0))
    assert joint < working_only


def test_derived_geometry_from_spans_threads_sut_protection_spans(tmp_path):
    scenario = _episode(tmp_path, "A", t3_lat=25.0)
    derived = derived_geometry_from_spans(scenario, SUT_SPANS, FAR_PROTECTION)
    assert derived.sut_p_cut_at_exposure_horizon == pytest.approx(
        sut_p_cut_at_exposure_horizon(scenario, SUT_SPANS, FAR_PROTECTION)[1])


def test_flip_scalars_from_spans_threads_protection_spans_for_the_claimant(
        example_scenario_yaml, write_scenario):
    scenario = load_scenario(write_scenario(example_scenario_yaml))
    spans = {"storm-svc-1": (_span(25.2, 81.0),), "svc-b": (_span(25.2, 81.0),)}
    demands = {"storm-svc-1": 300.0, "svc-b": 100.0}

    unprotected = flip_scalars_from_spans(scenario, spans=spans,
                                          demands_gbps=demands)
    protected = flip_scalars_from_spans(
        scenario, spans=spans, demands_gbps=demands,
        protection_spans={"svc-b": FAR_PROTECTION})

    # svc-b is the sole claimant (storm-svc-1 is excluded as the SUT), so
    # giving it a protection leg the storm cannot touch collapses its own
    # contribution to the claimant aggregate.
    assert (protected.claimant_ecar_at_exposure_horizon
            < unprotected.claimant_ecar_at_exposure_horizon)
    assert protected.claimant_ecar_at_exposure_horizon == pytest.approx(
        0.0, abs=1e-3)


import asyncio
from contextlib import asynccontextmanager

from storm_reoptimizer.eval.derived import (
    derived_scalars_for_suite, scenarios_by_state_file,
)


@pytest.fixture
def scenario(write_scenario, example_scenario_yaml):
    return load_scenario(write_scenario(example_scenario_yaml))


def test_scenarios_are_grouped_by_state_file_in_first_seen_order(scenario):
    import dataclasses
    a = dataclasses.replace(scenario, id="A", state_file="eval/states/x.json")
    b = dataclasses.replace(scenario, id="B", state_file="eval/states/y.json")
    c = dataclasses.replace(scenario, id="C", state_file="eval/states/x.json")
    groups = scenarios_by_state_file([a, b, c])
    assert list(groups) == ["eval/states/x.json", "eval/states/y.json"]
    assert [s.id for s in groups["eval/states/x.json"]] == ["A", "C"]


def test_the_suite_helper_connects_once_per_state_file(scenario, monkeypatch):
    import dataclasses
    from storm_reoptimizer.eval import derived
    a = dataclasses.replace(scenario, id="A", state_file="eval/states/x.json")
    b = dataclasses.replace(scenario, id="B", state_file="eval/states/y.json")
    opened = []

    def connect_for(state_file):
        @asynccontextmanager
        async def _connect():
            opened.append(state_file)
            yield object()
        return _connect

    async def fake_scalars(client, scenarios, *, topology_path):
        return {s.id: {"sut_p_cut_at_exposure_horizon": 0.5} for s in scenarios}
    monkeypatch.setattr(derived, "derived_scalars_for", fake_scalars)
    out = asyncio.run(derived_scalars_for_suite(connect_for, [a, b], topology_path="t"))
    assert opened == ["eval/states/x.json", "eval/states/y.json"]
    assert set(out) == {"A", "B"}
