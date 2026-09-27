"""spare_value.py, pure: `DecisionFacts` built by hand from spec
2026-09-27-t2-correlated-claims-and-honest-scope-design.md §3.3's T2a/T2b
joint-cut tables, checked against that document's own worked expected-value
numbers (217.5/149.4 for T2b, 217.5/433.8 for T2a). No server is involved --
`decision_facts` (the async gatherer) is exercised only by the live smoke
test in tests/eval/test_episodes.py."""
import pytest

from storm_reoptimizer.eval.spare_value import (
    DecisionFacts, best_hold_ranking, expected_spare_value,
)

SUT = "t2-svc-jalgaon-nagpur"
INDORE = "t1-svc-jalgaon-indore"
KHANDWA = "t2-claimant-jalgaon-khandwa"
DFWD = "t1-claimant-jalgaon-dhulia-fwd"
DREV = "t1-claimant-jalgaon-dhulia-rev"
D0346 = "d0346"
D0422 = "d0422"

DEMANDS = {SUT: 300.0, INDORE: 300.0, KHANDWA: 200.0,
          DFWD: 100.0, DREV: 100.0, D0346: 100.0, D0422: 100.0}

SHOWN = frozenset(DEMANDS)


def _facts(rows, *, khandwa_restorable: bool) -> DecisionFacts:
    restorable = {s: True for s in SHOWN}
    restorable[KHANDWA] = khandwa_restorable
    depot_spares = {s: (1 if restorable[s] else 0) for s in SHOWN}
    return DecisionFacts(
        sut=SUT, depot_site="jalgaon", spares_on_hand=1,
        horizon="t3", rg_id="rg_T2_t1_t3",
        rows=rows, demands=dict(DEMANDS), restorable=restorable,
        depot_spares=depot_spares, unrestored=5, restored_after_cut=2,
        shown_through_cut=SHOWN)


# spec §3.3's T2b table.
T2B_ROWS = (
    (frozenset({KHANDWA}), 0.820),
    (frozenset({SUT, INDORE, KHANDWA, DFWD, DREV, D0346, D0422}), 0.145),
    (frozenset({INDORE, KHANDWA}), 0.021),
    (frozenset(), 0.015),
)

# spec §3.3's T2a table.
T2A_ROWS = (
    (frozenset(), 0.426),
    (frozenset({KHANDWA}), 0.279),
    (frozenset({INDORE, KHANDWA}), 0.151),
    (frozenset({SUT, INDORE, KHANDWA, DFWD, DREV, D0346, D0422}), 0.145),
)


def test_t2b_expected_spare_value_matches_spec():
    facts = _facts(T2B_ROWS, khandwa_restorable=False)
    assert expected_spare_value(facts) == {
        "spend": pytest.approx(217.5), "hold": pytest.approx(149.4)}


def test_t2b_best_hold_ranking_leads_with_indore_then_sut():
    facts = _facts(T2B_ROWS, khandwa_restorable=False)
    assert best_hold_ranking(facts)[:2] == (INDORE, SUT)
    # khandwa is not restorable in T2b, so it never enters the ranking at all.
    assert KHANDWA not in best_hold_ranking(facts)


def test_t2a_expected_spare_value_matches_spec():
    facts = _facts(T2A_ROWS, khandwa_restorable=True)
    assert expected_spare_value(facts) == {
        "spend": pytest.approx(217.5), "hold": pytest.approx(433.8)}


def test_t2a_best_hold_ranking_leads_with_indore_then_sut_then_khandwa():
    facts = _facts(T2A_ROWS, khandwa_restorable=True)
    ranking = best_hold_ranking(facts)
    assert ranking[:2] == (INDORE, SUT)
    assert KHANDWA in ranking
    assert ranking.index(KHANDWA) > ranking.index(SUT)


def test_expected_spare_value_rejects_anything_but_one_spare_on_hand():
    facts = _facts(T2B_ROWS, khandwa_restorable=False)
    two_spares = DecisionFacts(
        sut=facts.sut, depot_site=facts.depot_site, spares_on_hand=2,
        horizon=facts.horizon, rg_id=facts.rg_id, rows=facts.rows,
        demands=facts.demands, restorable=facts.restorable,
        depot_spares=facts.depot_spares, unrestored=facts.unrestored,
        restored_after_cut=facts.restored_after_cut,
        shown_through_cut=facts.shown_through_cut)
    with pytest.raises(ValueError, match="single-spare"):
        expected_spare_value(two_spares)


def test_a_free_at_depot_service_ranked_first_is_skipped_for_the_spare():
    """If the top-ranked restorable service in a `down` row needs no real
    spare to restore (`depot_spares[s] == 0` -- it is free at the depot),
    the harness's own cheapest-first walk restores it either way and it
    never claims the held spare; `expected_spare_value`'s `hold` must move
    on to the next eligible, actually-down service in `best_hold_ranking`,
    not credit the free service (or nobody) with the spare's value.

    A minimal, self-contained table (not T2a/T2b's own rows): `indore` ties
    the SUT for rank #1 (demand x delay = 900) and is free at the depot;
    `khandwa` (600) ranks third. The one non-empty row cuts indore and
    khandwa together, but NOT the SUT, so a correct walk must skip past
    both indore (free) AND -- because the SUT is simply not down in this
    row -- past the SUT too, landing on khandwa."""
    demands = {SUT: 300.0, INDORE: 300.0, KHANDWA: 200.0}
    shown = frozenset(demands)
    restorable = {s: True for s in shown}
    depot_spares = {SUT: 1, INDORE: 0, KHANDWA: 1}  # indore is free
    rows = ((frozenset({INDORE, KHANDWA}), 0.5), (frozenset(), 0.5))
    facts = DecisionFacts(
        sut=SUT, depot_site="jalgaon", spares_on_hand=1,
        horizon="t3", rg_id="rg_T2_t1_t3",
        rows=rows, demands=demands, restorable=restorable,
        depot_spares=depot_spares, unrestored=5, restored_after_cut=2,
        shown_through_cut=shown)
    ranking = best_hold_ranking(facts)
    assert ranking[0] == INDORE  # tied with the SUT, wins the id tiebreak
    values = expected_spare_value(facts)
    # spend: the SUT is never in `down` here.
    assert values["spend"] == pytest.approx(0.0)
    # hold: indore (rank 1) is free -- skip; the SUT (rank 2) is not down
    # in this row -- skip; khandwa (rank 3) is down, restorable, and needs
    # a real spare -- credited.
    assert values["hold"] == pytest.approx(0.5 * 3 * 200.0)
