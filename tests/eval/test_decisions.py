"""The three decision interfaces (eval design spec, Decisions 1-3)."""
import pytest

from storm_reoptimizer.eval.decisions import (
    CONSTRAINT_JSON_SCHEMA, COST_TERMS, ConstraintDecision, DecisionError,
    OBJECTIVE_JSON_SCHEMA, ObjectiveDecision, TIMING_JSON_SCHEMA,
    TimingDecision, candidate_index, rank_by_priority,
)


def test_timing_decision_round_trips():
    d = TimingDecision.from_dict({"action": "wait", "reasoning": "cone is wide"})
    assert d.action == "wait"
    assert d.to_dict() == {"action": "wait", "reasoning": "cone is wide",
                           "contested_claim": None, "claim_priority": []}


def test_timing_rejects_an_unknown_action():
    with pytest.raises(DecisionError, match="action"):
        TimingDecision.from_dict({"action": "panic", "reasoning": "x"})


def test_reasoning_is_mandatory_on_all_three_decisions():
    with pytest.raises(DecisionError, match="reasoning"):
        TimingDecision.from_dict({"action": "act"})
    with pytest.raises(DecisionError, match="reasoning"):
        ObjectiveDecision.from_dict({"choice": "candidate_0", "priority": None})


def test_constraint_decision_derives_the_protection_posture():
    """The posture is DERIVED from `avoid`, not stated (spec 6.4): it used
    to be four fields a caller could pass however it liked -- protected=True/
    srlg/srlg by the server signature's own default -- and the whole-branch
    review (2026-08-23, finding I5) confirmed that ZERO of the 7+ real call
    sites on this branch wanted that; every one of them passed
    protected=False/basis="physical"/level="link" explicitly. Stating four
    fields a model could only get wrong bought nothing, so they left the
    dataclass entirely."""
    d = ConstraintDecision.from_dict(
        {"avoid": {"risk_groups": ["rg_t3"]}, "reasoning": "clear of t+3"})
    assert d.avoid == {"risk_groups": ["rg_t3"]}
    assert (d.protected, d.best_effort, d.basis, d.level) == (
        False, False, "risk_group", "risk_group")
    # The bare constructor and from_dict must agree -- they are two of the
    # three paths into this object (ScriptedDecider's own fallback is the
    # third) and a divergence between them would be invisible.
    bare = ConstraintDecision(avoid={}, reasoning="x")
    assert (bare.protected, bare.best_effort, bare.basis, bare.level) == (
        False, False, "physical", "link")


def test_risk_group_is_a_legal_constraint_level():
    """T2a/T3a/T3b's gold decisions only validate under basis="risk_group"/
    level="risk_group". The payload no longer carries `basis`/`level` at
    all -- from_dict derives them from `avoid` alone."""
    payload = {"avoid": {"risk_groups": ["rg_T2a_t1_t6"]},
               "reasoning": "avoid the full t+6 cone"}
    d = ConstraintDecision.from_dict(payload)
    assert (d.basis, d.level) == ("risk_group", "risk_group")
    assert ConstraintDecision.from_dict(d.to_dict()) == d


def test_constraint_decision_rejects_an_unknown_avoid_key():
    with pytest.raises(DecisionError, match="avoid"):
        ConstraintDecision.from_dict(
            {"avoid": {"riskgroups": ["rg_t3"]}, "reasoning": "typo"})


def test_objective_priority_must_name_only_real_cost_terms():
    """`priority` is no longer read from a payload (spec 6.5) -- what
    remains is rank_by_priority itself, which the baseline still calls
    directly with a real ordering over COST_TERMS."""
    order = rank_by_priority(
        [{"cost_vector": {"transponders": 4.0, "dropped_traffic": 1.0}},
         {"cost_vector": {"transponders": 2.0, "dropped_traffic": 5.0}}],
        ("transponders", "dropped_traffic"))
    assert order == [1, 0]


def test_declining_to_state_a_priority_is_legitimate():
    d = ObjectiveDecision.from_dict(
        {"choice": "candidate_2", "priority": None,
         "reasoning": "not a strict ordering"})
    assert d.priority is None


def test_candidate_index_parses_the_choice_label():
    assert candidate_index("candidate_0") == 0
    assert candidate_index("candidate_12") == 12
    assert candidate_index("infeasible") is None
    assert candidate_index("hold") is None
    with pytest.raises(DecisionError):
        candidate_index("candidate_x")


def test_rank_by_priority_is_lexicographic_and_treats_margin_as_a_benefit():
    cands = [
        {"cost_vector": {"transponders": 4.0, "total_margin": 1.0}},
        {"cost_vector": {"transponders": 2.0, "total_margin": 0.0}},
        {"cost_vector": {"transponders": 2.0, "total_margin": 9.0}},
    ]
    # Fewer transponders first; among ties, MORE margin first.
    assert rank_by_priority(cands, ("transponders", "total_margin")) == [2, 1, 0]


def test_cost_terms_are_the_seven_the_server_reports():
    assert COST_TERMS == (
        "spectrum_used", "transponders", "max_util", "dropped_traffic",
        "added_latency", "total_margin", "services_at_risk")


def test_a_decision_carries_the_rival_claim_it_weighed_or_null():
    d = TimingDecision.from_dict({
        "action": "wait", "reasoning": "the cone sharpens next hour",
        "contested_claim": {"service_id": "d0462",
                            "expected_capacity_at_risk_gbps": 88.3}})
    assert d.contested_claim == {"service_id": "d0462",
                                 "expected_capacity_at_risk_gbps": 88.3}


def test_a_decision_with_no_rival_claim_says_so_explicitly():
    d = TimingDecision.from_dict({
        "action": "act", "reasoning": "nothing else is exposed",
        "contested_claim": None})
    assert d.contested_claim is None
    assert d.to_dict()["contested_claim"] is None
    assert d.to_dict()["claim_priority"] == []


def test_claim_priority_round_trips_and_validates():
    d = TimingDecision.from_dict({"action": "wait", "reasoning": "hold",
                                  "contested_claim": None,
                                  "claim_priority": ["c", "s"]})
    assert d.claim_priority == ("c", "s")
    assert d.to_dict()["claim_priority"] == ["c", "s"]
    with pytest.raises(DecisionError):
        TimingDecision.from_dict(
            {"action": "wait", "reasoning": "x", "claim_priority": "c"})


def test_an_absent_contested_claim_is_the_same_as_null():
    # baseline.py and every gold decision construct these positionally with no
    # claim; they must keep working untouched. `contested_claim` stays on
    # TimingDecision only -- it left ObjectiveDecision and ConstraintDecision
    # entirely (spec 6.4/6.5).
    assert TimingDecision.from_dict(
        {"action": "act", "reasoning": "x"}).contested_claim is None
    assert TimingDecision("act", "x").contested_claim is None


@pytest.mark.parametrize("bad, match", [
    ({"service_id": "d0462"}, "expected_capacity_at_risk_gbps"),
    ({"expected_capacity_at_risk_gbps": 1.0}, "service_id"),
    ({"service_id": "", "expected_capacity_at_risk_gbps": 1.0},
     "service_id"),
    ({"service_id": "d0462", "expected_capacity_at_risk_gbps": "big"},
     "expected_capacity_at_risk_gbps"),
    ("d0462", "contested_claim"),
])
def test_a_malformed_contested_claim_is_rejected(bad, match):
    with pytest.raises(DecisionError, match=match):
        TimingDecision.from_dict(
            {"action": "act", "reasoning": "x", "contested_claim": bad})


def test_timing_decision_requires_the_field():
    # `contested_claim` stays on TimingDecision only -- it left the
    # constraint and objective wire schemas entirely (spec 6.4/6.5).
    assert "contested_claim" in TIMING_JSON_SCHEMA["required"]
    assert "contested_claim" in TIMING_JSON_SCHEMA["properties"]


def test_reasoning_is_the_first_property_in_every_tool_schema():
    """Strict tool use emits properties in schema order. Two failures in the
    2026-09-09 run (T3b t0, T2b t1) emitted the enum BEFORE the reasoning
    that was supposed to justify it, and the text then contradicted it."""
    for schema in (TIMING_JSON_SCHEMA, CONSTRAINT_JSON_SCHEMA,
                   OBJECTIVE_JSON_SCHEMA):
        assert list(schema["properties"])[0] == "reasoning"


@pytest.mark.parametrize("bad", [
    "I chose to wait</reasoning><parameter name=\"contested_claim\">",
    "the claimant is bigger<parameter name=\"action\">",
])
def test_tool_call_markup_inside_reasoning_is_rejected(bad):
    """T1a t4 and T2b t3 both emitted a literal tool-call fragment inside the
    reasoning string and the validator accepted it."""
    with pytest.raises(DecisionError, match="tool-call markup"):
        TimingDecision.from_dict({"reasoning": bad, "action": "wait"})


def test_reasoning_still_accepts_ordinary_angle_brackets():
    d = TimingDecision.from_dict(
        {"reasoning": "claimant 149.0 > SUT 35.1", "action": "wait"})
    assert d.action == "wait"


def test_the_constraint_posture_is_derived_from_the_avoid_set():
    physical = ConstraintDecision(avoid={"assets": ["fiber_a_b_0"]},
                                  reasoning="named spans only")
    assert (physical.basis, physical.level) == ("physical", "link")
    grouped = ConstraintDecision(avoid={"risk_groups": ["rg_x"]},
                                 reasoning="the whole group")
    assert (grouped.basis, grouped.level) == ("risk_group", "risk_group")
    mixed = ConstraintDecision(
        avoid={"assets": ["fiber_a_b_0"], "risk_groups": ["rg_x"]},
        reasoning="both granularities")
    # A mixed avoid is LEGAL: the server's own _forbidden_assets unions
    # avoid["assets"] with the named groups' members and never looks at
    # basis/level, so the risk_group basis cannot suppress a named asset.
    assert (mixed.basis, mixed.level) == ("risk_group", "risk_group")
    assert mixed.route_service_args("svc")["avoid"] == mixed.avoid
    empty = ConstraintDecision(avoid={}, reasoning="unconstrained")
    assert (empty.basis, empty.level) == ("physical", "link")
    assert empty.protected is False and empty.best_effort is False


def test_an_empty_risk_group_list_still_derives_the_physical_basis():
    """`{"risk_groups": []}` names no group, so it must not flip the basis --
    assertions.PLAUSIBLE_ALTERNATIVES replays exactly this avoid set."""
    d = ConstraintDecision(avoid={"risk_groups": []}, reasoning="empty list")
    assert (d.basis, d.level) == ("physical", "link")


def test_the_constraint_schema_carries_only_reasoning_and_avoid():
    assert set(CONSTRAINT_JSON_SCHEMA["properties"]) == {"reasoning", "avoid"}
    assert set(CONSTRAINT_JSON_SCHEMA["required"]) == {"reasoning", "avoid"}
    assert "srlgs" not in CONSTRAINT_JSON_SCHEMA["properties"]["avoid"]["properties"]


def test_the_objective_schema_carries_only_reasoning_and_choice():
    assert set(OBJECTIVE_JSON_SCHEMA["properties"]) == {"reasoning", "choice"}
    assert set(OBJECTIVE_JSON_SCHEMA["required"]) == {"reasoning", "choice"}


def test_an_objective_payload_never_carries_a_priority_any_more():
    d = ObjectiveDecision.from_dict(
        {"reasoning": "cheapest real escape", "choice": "candidate_1"})
    assert d.priority is None
    assert d.to_dict() == {"choice": "candidate_1", "priority": None,
                           "reasoning": "cheapest real escape"}


def test_the_baseline_still_states_a_priority_positionally():
    """rank_by_priority and COST_TERMS stay for the baseline (spec 6.5); only
    the WIRE schema drops the field."""
    d = ObjectiveDecision("candidate_0", COST_TERMS[:2], "fixed ordering")
    assert d.priority == COST_TERMS[:2]
