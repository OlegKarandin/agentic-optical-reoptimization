"""The node-local spare depot (exposure-and-depot design, §4.1).

A bare integer had a hole a hybrid candidate walks straight through: groom
satna -> X on an existing lightpath, light a new X -> allahabad, and pay
nothing from satna's depot. That is not a loophole to forbid -- a transponder
at X really is what it costs -- so the ledger models it."""
import pytest

from storm_reoptimizer.eval.ledger import (
    InsufficientSpares, SpareLedger, spares_needed,
)

OMS_NODES = {
    "oms_satna_rewa": ["satna", "rewa"],
    "oms_rewa_allahabad": ["rewa", "allahabad"],
    "oms_satna_jabalpur": ["satna", "jabalpur"],
    "oms_jabalpur_raipur": ["jabalpur", "raipur"],
}


def _lp(*oms):
    return {"oms_sequence": list(oms)}


def test_an_ip_reroute_costs_nothing_anywhere():
    # The whole point of the lever label: `ip_reroute` means "this recovery
    # costs no transponders".
    assert spares_needed({"lever": "ip_reroute", "new_lightpaths": []},
                         OMS_NODES) == {}


def test_a_new_lightpath_charges_one_transponder_at_each_endpoint_site():
    candidate = {"lever": "optical_reroute",
                 "new_lightpaths": [_lp("oms_satna_rewa", "oms_rewa_allahabad")]}
    assert spares_needed(candidate, OMS_NODES) == {"satna": 1, "allahabad": 1}


def test_the_endpoints_are_the_nodes_a_leg_touches_once():
    # Interior junctions appear twice in the leg endpoints; the two that
    # appear ONCE are the lightpath's ends. Derived this way rather than from
    # leg ORDER, which is not guaranteed head-to-tail.
    candidate = {"lever": "optical_reroute",
                 "new_lightpaths": [_lp("oms_jabalpur_raipur",
                                        "oms_satna_jabalpur")]}
    assert spares_needed(candidate, OMS_NODES) == {"raipur": 1, "satna": 1}


def test_two_new_lightpaths_at_the_same_site_charge_it_twice():
    candidate = {"lever": "optical_reroute", "new_lightpaths": [
        _lp("oms_satna_rewa"), _lp("oms_satna_jabalpur")]}
    assert spares_needed(candidate, OMS_NODES) == {"satna": 2, "rewa": 1,
                                                   "jabalpur": 1}


def test_a_candidate_that_never_touches_the_depot_pays_nothing_there():
    # The case that motivated the dict over a scalar.
    candidate = {"lever": "hybrid",
                 "new_lightpaths": [_lp("oms_rewa_allahabad")]}
    ledger = SpareLedger(inventory={"satna": 1}, depot_site="satna",
                         oms_nodes=OMS_NODES)
    assert spares_needed(candidate, OMS_NODES) == {"rewa": 1, "allahabad": 1}
    assert ledger.can_afford(candidate)


def test_only_the_depot_site_binds():
    ledger = SpareLedger(inventory={"satna": 1}, depot_site="satna",
                         oms_nodes=OMS_NODES, default_spares_per_site=8)
    one = {"lever": "optical_reroute",
           "new_lightpaths": [_lp("oms_satna_rewa", "oms_rewa_allahabad")]}
    assert ledger.can_afford(one)
    assert ledger.debit(one, hour="t1", service_id="storm-svc-1") == {
        "satna": 1, "allahabad": 1}
    assert ledger.on_hand == 0
    assert ledger.spent == 1
    # allahabad started at the generous default and is nowhere near binding.
    assert ledger.inventory["allahabad"] == 7
    assert not ledger.can_afford(one)


def test_the_rejection_names_the_binding_site():
    ledger = SpareLedger(inventory={"satna": 0}, depot_site="satna",
                         oms_nodes=OMS_NODES)
    candidate = {"lever": "optical_reroute",
                 "new_lightpaths": [_lp("oms_satna_rewa", "oms_rewa_allahabad")]}
    rejection = ledger.rejection(candidate)
    assert rejection["type"] == "insufficient_spares"
    assert rejection["site"] == "satna"
    assert rejection["needed"] == {"satna": 1, "allahabad": 1}
    with pytest.raises(InsufficientSpares) as exc:
        ledger.debit(candidate, hour="t1", service_id="storm-svc-1")
    assert exc.value.site == "satna"
    assert ledger.inventory == {"satna": 0}      # nothing changed


def test_debit_records_origin():
    ledger = SpareLedger(inventory={"satna": 1}, depot_site="satna",
                         oms_nodes={"oms_a": ["satna", "rewa"]})
    cand = {"new_lightpaths": [{"oms_sequence": ["oms_a"]}], "reused_lightpaths": []}
    ledger.debit(cand, hour="t3", service_id="c", origin="harness")
    assert ledger.debits == [{"hour": "t3", "service_id": "c",
                              "spares": {"satna": 1, "rewa": 1}, "origin": "harness"}]


def test_debit_origin_defaults_to_decider():
    ledger = SpareLedger(inventory={"satna": 1}, depot_site="satna",
                         oms_nodes={"oms_a": ["satna", "rewa"]})
    cand = {"new_lightpaths": [{"oms_sequence": ["oms_a"]}], "reused_lightpaths": []}
    ledger.debit(cand, hour="t1", service_id="s")
    assert ledger.debits[0]["origin"] == "decider"


def test_the_word_pair_has_left_the_module():
    # A pair spans two sites and cannot be scoped to one. `pairs_needed` is
    # renamed, not aliased -- an alias would let the old, site-blind reading
    # survive in a call site nobody re-read.
    import storm_reoptimizer.eval.ledger as ledger_module
    assert not hasattr(ledger_module, "pairs_needed")
