"""The harness-owned spare ledger (eval design spec, build order item 4).
The unit is transponder PAIRS -- one per new lightpath."""
import pytest

from storm_reoptimizer.eval.ledger import (
    InsufficientSpares, SpareLedger, pairs_needed,
)


def _candidate(n_new, *, lever="optical_reroute", total_lightpaths=40):
    return {
        "lever": lever,
        "reused_lightpaths": ["lp-a"] if lever != "optical_reroute" else [],
        "new_lightpaths": [
            {"oms_sequence": ["oms_1"], "lam": i, "mode_id": "300G@4.8dB",
             "gsnr_db": 12.0, "bitrate_gbps": 300.0} for i in range(n_new)],
        "restored_gbps": 300.0, "shortfall_gbps": 0.0,
        # Whole-network count, exactly as score_candidate reports it. The
        # ledger must NOT read this field.
        "cost_vector": {"transponders": 2.0 * total_lightpaths},
    }


def test_cost_is_one_pair_per_new_lightpath_not_the_network_transponder_count():
    assert pairs_needed(_candidate(2, total_lightpaths=40)) == 2
    assert pairs_needed(_candidate(0, lever="ip_reroute")) == 0


def test_an_ip_reroute_costs_no_spare():
    ledger = SpareLedger(on_hand=1)
    ledger.debit(_candidate(0, lever="ip_reroute"), hour="t1", service_id="s")
    assert ledger.on_hand == 1
    assert ledger.spent == 0


def test_an_affordable_candidate_decrements_by_its_own_count():
    ledger = SpareLedger(on_hand=2)
    assert ledger.can_afford(_candidate(2))
    ledger.debit(_candidate(2), hour="t1", service_id="storm-svc-1")
    assert ledger.on_hand == 0
    assert ledger.spent == 2
    assert ledger.debits == [
        {"hour": "t1", "service_id": "storm-svc-1", "pairs": 2}]


def test_an_unaffordable_candidate_is_rejected_and_nothing_is_spent():
    ledger = SpareLedger(on_hand=1)
    candidate = _candidate(2)
    assert not ledger.can_afford(candidate)
    with pytest.raises(InsufficientSpares) as exc:
        ledger.debit(candidate, hour="t1", service_id="storm-svc-1")
    assert exc.value.needed == 2
    assert exc.value.on_hand == 1
    assert ledger.on_hand == 1
    assert ledger.spent == 0
    assert ledger.debits == []


def test_the_rejection_is_typed_for_the_retry_loop():
    ledger = SpareLedger(on_hand=1)
    assert ledger.rejection(_candidate(2)) == {
        "type": "insufficient_spares", "needed": 2, "on_hand": 1}
