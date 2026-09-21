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


def test_a_mate_pair_over_the_same_endpoint_pair_charges_once_per_site():
    ledger = SpareLedger(inventory={"satna": 1}, depot_site="satna",
                         oms_nodes=OMS_NODES)
    fwd = {"lever": "optical_reroute",
          "new_lightpaths": [_lp("oms_satna_rewa")]}
    assert ledger.can_afford(fwd)
    assert ledger.debit(fwd, hour="t1", service_id="fwd") == {
        "satna": 1, "rewa": 1}
    assert ledger.on_hand == 0
    # The reverse direction over the SAME pair, still under test alone
    # (lit_runs=() here) would cost another pair -- but ledger.debit already
    # appended fwd's own run to ledger.lit_runs, so the SAME site charges
    # the mate for free.
    bwd_oms_nodes = {**OMS_NODES, "oms_rewa_satna": ["rewa", "satna"]}
    bwd = {"lever": "optical_reroute",
          "new_lightpaths": [{"oms_sequence": ["oms_rewa_satna"]}]}
    assert spares_needed(bwd, bwd_oms_nodes, lit_runs=ledger.lit_runs) == {}


def test_a_mate_pair_over_a_different_intermediate_route_still_charges_once():
    # The route need not mirror -- only the two terminating SITES matter
    # (spec 3.3). The lit run went satna->rewa directly; the "mate" charged
    # against it goes rewa->raipur->jabalpur->satna, a totally different
    # OMS path (fresh ids, not OMS_NODES', to keep this test's own chain
    # unambiguous) that still terminates at satna and rewa, in reverse.
    mate_oms_nodes = {"oms_rewa_raipur": ["rewa", "raipur"],
                      "oms_raipur_jabalpur": ["raipur", "jabalpur"],
                      "oms_jabalpur_satna": ["jabalpur", "satna"]}
    lit_runs = [("satna", "rewa")]
    mate = {"lever": "optical_reroute", "new_lightpaths": [{
        "oms_sequence": ["oms_rewa_raipur", "oms_raipur_jabalpur",
                         "oms_jabalpur_satna"]}]}
    assert spares_needed(mate, mate_oms_nodes, lit_runs=lit_runs) == {}


def test_two_codirectional_runs_to_the_same_far_end_charge_twice():
    candidate = {"lever": "optical_reroute", "new_lightpaths": [
        _lp("oms_satna_rewa"), {"oms_sequence": ["oms_satna_rewa"]}]}
    # Two DISTINCT single-leg runs over the same OMS id both terminate at
    # satna/rewa in the SAME direction -- co-directional, not a mate pair.
    assert spares_needed(candidate, OMS_NODES) == {"satna": 2, "rewa": 2}


def test_two_fwd_one_bwd_charges_the_max_not_the_sum():
    oms_nodes = {**OMS_NODES, "oms_rewa_satna": ["rewa", "satna"]}
    candidate = {"lever": "optical_reroute", "new_lightpaths": [
        _lp("oms_satna_rewa"), {"oms_sequence": ["oms_satna_rewa"]},
        {"oms_sequence": ["oms_rewa_satna"]}]}
    assert spares_needed(candidate, oms_nodes) == {"satna": 2, "rewa": 2}


def test_lit_runs_defaults_to_empty_and_reproduces_todays_arithmetic():
    # Every existing test above this one in the file calls spares_needed
    # with no lit_runs at all -- this just names the invariant directly.
    candidate = {"lever": "optical_reroute",
                 "new_lightpaths": [_lp("oms_satna_rewa", "oms_rewa_allahabad")]}
    assert (spares_needed(candidate, OMS_NODES)
           == spares_needed(candidate, OMS_NODES, lit_runs=()))


def test_debit_appends_to_lit_runs_and_a_second_mate_then_affords():
    ledger = SpareLedger(inventory={"satna": 1}, depot_site="satna",
                         oms_nodes=OMS_NODES)
    assert ledger.lit_runs == []
    fwd = {"lever": "optical_reroute",
          "new_lightpaths": [_lp("oms_satna_rewa")]}
    ledger.debit(fwd, hour="t1", service_id="fwd")
    assert ledger.lit_runs == [("satna", "rewa")]

    bwd_ledger_oms = {**OMS_NODES, "oms_rewa_satna": ["rewa", "satna"]}
    ledger.oms_nodes = bwd_ledger_oms
    bwd = {"lever": "optical_reroute",
          "new_lightpaths": [{"oms_sequence": ["oms_rewa_satna"]}]}
    # can_afford read False before the mate was lit (no spares left at
    # satna); it reads True now that the reverse run is free.
    assert not ledger.can_afford(fwd)          # a THIRD fwd run still can't
    assert ledger.can_afford(bwd)


def test_seeded_state_lightpaths_do_not_pair():
    # lit_runs is grown ONLY by debit() -- a fresh ledger's lit_runs is
    # empty regardless of what the seeded server state already has lit, so
    # a candidate's reverse mate there is charged full price (spec 3.4).
    ledger = SpareLedger(inventory={"satna": 1}, depot_site="satna",
                         oms_nodes=OMS_NODES)
    bwd_oms_nodes = {**OMS_NODES, "oms_rewa_satna": ["rewa", "satna"]}
    ledger.oms_nodes = bwd_oms_nodes
    bwd = {"lever": "optical_reroute",
          "new_lightpaths": [{"oms_sequence": ["oms_rewa_satna"]}]}
    assert ledger.can_afford(bwd)              # first run at this depot: fine
    assert spares_needed(bwd, bwd_oms_nodes) == {"satna": 1, "rewa": 1}
