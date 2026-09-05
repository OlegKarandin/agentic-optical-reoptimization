"""tools/find_sut.py's `prefilter` is pure GIS + a graph walk over the local
topology JSON -- no server, no solver -- so it gets a real, fast, offline
test. `evaluate` is live (shells out to build_eval_state.py against the
sibling conda env, then starts a real MCP server) and is exercised manually
instead; see the Task 13 report for the real run's output and the reasoning
for not adding an automated test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import find_sut  # noqa: E402

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)


def test_prefilter_finds_the_diverse_sites_including_tirupati():
    """The brief's own Step 1 claim: 16 rows total, tirupati among them with
    3 aerial and 1 buried neighbour, every row's track distance non-negative
    -- confirmed directly against toy_india_topology.json before writing this
    test (see task-13-report.md)."""
    rows = find_sut.prefilter(TOPOLOGY_PATH)

    assert len(rows) == 16
    by_site = {row["site"]: row for row in rows}
    assert "tirupati" in by_site
    assert by_site["tirupati"]["n_aerial"] == 3
    assert by_site["tirupati"]["n_buried"] == 1
    assert all(row["track_distance_km"] >= 0 for row in rows)


def test_prefilter_rows_are_sorted_by_track_distance():
    rows = find_sut.prefilter(TOPOLOGY_PATH)
    distances = [row["track_distance_km"] for row in rows]
    assert distances == sorted(distances)


def test_prefilter_respects_min_aerial():
    """Raising min_aerial strictly narrows the candidate set -- every row
    that survives at the higher bar must have survived at the lower one."""
    rows_default = {row["site"] for row in find_sut.prefilter(TOPOLOGY_PATH)}
    rows_strict = {
        row["site"]
        for row in find_sut.prefilter(TOPOLOGY_PATH, min_aerial=3)}
    assert rows_strict <= rows_default
    assert all(row["n_aerial"] >= 3
              for row in find_sut.prefilter(TOPOLOGY_PATH, min_aerial=3))
