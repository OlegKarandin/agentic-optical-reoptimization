"""Turning a horizon's risk group into rows an agent can name (spec 5.1)."""
from storm_reoptimizer.eval.risk_assets import (
    fiber_span_index, path_edges, risk_group_rows,
)

OMS = [
    {"id": "oms_sr", "src_node_id": "satna", "dst_node_id": "rewa",
     "elements": ["fiber_satna_rewa_0", "amp_1"]},
    {"id": "oms_sj", "src_node_id": "satna", "dst_node_id": "jhansi",
     "elements": ["fiber_satna_jhansi_0", "fiber_satna_jhansi_1"]},
    {"id": "oms_gap", "src_node_id": "satna", "dst_node_id": "nowhere",
     "elements": ["fiber_satna_nowhere_0"]},
]
COORDS = {"satna": (24.6, 80.8), "rewa": (24.5, 81.3),
          "jhansi": (25.4, 78.6)}


def test_the_index_maps_every_fiber_to_its_own_span():
    index = fiber_span_index(OMS, COORDS)
    assert index["fiber_satna_rewa_0"] == (
        ("satna", "rewa"), ((24.6, 80.8), (24.5, 81.3)))
    assert index["fiber_satna_jhansi_0"] == index["fiber_satna_jhansi_1"]
    # Non-fiber elements are not assets a storm cuts.
    assert "amp_1" not in index
    # A node the local topology lacks yields no span rather than a silent
    # half-resolved one.
    assert "fiber_satna_nowhere_0" not in index


def test_path_edges_are_unordered_consecutive_pairs():
    assert path_edges(["satna", "rewa", "allahabad"]) == {
        frozenset({"satna", "rewa"}), frozenset({"rewa", "allahabad"})}
    assert path_edges([]) == set()
    assert path_edges(["satna"]) == set()


def test_a_row_says_whether_the_span_is_on_the_service_it_can_act_on():
    index = fiber_span_index(OMS, COORDS)
    rows = risk_group_rows(
        ["fiber_satna_rewa_0", "fiber_satna_jhansi_0"], index,
        center_lat=24.6, center_lon=80.8, width_km=15.0,
        damage_radius_km=74.0,
        working_edges=path_edges(["satna", "rewa"]),
        protection_edges=path_edges(["satna", "jhansi"]))
    by_id = {r["asset_id"]: r for r in rows}
    assert by_id["fiber_satna_rewa_0"]["on"] == "working"
    assert by_id["fiber_satna_jhansi_0"]["on"] == "protection"
    assert all(0.0 <= r["p_cut"] <= 1.0 for r in rows)
    assert all(r["p_cut"] == round(r["p_cut"], 3) for r in rows)


def test_a_span_on_neither_of_the_services_paths_reads_none():
    index = fiber_span_index(OMS, COORDS)
    rows = risk_group_rows(
        ["fiber_satna_jhansi_0"], index, center_lat=24.6, center_lon=80.8,
        width_km=15.0, damage_radius_km=74.0,
        working_edges=path_edges(["satna", "rewa"]), protection_edges=set())
    assert rows[0]["on"] == "none"


def test_an_unknown_asset_id_is_skipped_rather_than_guessed():
    index = fiber_span_index(OMS, COORDS)
    rows = risk_group_rows(
        ["fiber_nope_0"], index, center_lat=24.6, center_lon=80.8,
        width_km=15.0, damage_radius_km=74.0, working_edges=set(),
        protection_edges=set())
    assert rows == []


def test_rows_are_ordered_most_likely_to_be_cut_first():
    index = fiber_span_index(OMS, COORDS)
    rows = risk_group_rows(
        ["fiber_satna_jhansi_0", "fiber_satna_rewa_0"], index,
        center_lat=24.6, center_lon=80.8, width_km=15.0,
        damage_radius_km=74.0, working_edges=set(), protection_edges=set())
    assert [r["p_cut"] for r in rows] == sorted(
        (r["p_cut"] for r in rows), reverse=True)
