"""tools/find_sut.py's `prefilter` is pure GIS + a graph walk over the local
topology JSON -- no server, no solver -- so it gets a real, fast, offline
test. `evaluate` is live (shells out to build_eval_state.py against the
sibling conda env, then starts a real MCP server) and is exercised manually
instead; see the Task 13 report for the real run's output and the reasoning
for not adding an automated test."""
import sys
from pathlib import Path

import networkx as nx

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


def _synthetic_graph() -> nx.Graph:
    """A small, controlled graph exercising `_escapes_aerial_pair` directly,
    independent of the real toy topology:

      S -- a1 -- b1        (b1's ONLY alternate route back to S, once the
                             direct S<->b1 edge is removed, is via a1)
      S -- a2
      S -- b2 -- x -- S     (b2's alternate route back to S goes via x,
                             touching neither a1 nor a2)
      S -- b3               (b3 has NO other edge at all -- an isolated
                             buried stub once S<->b3 is removed)
    """
    g = nx.Graph()
    g.add_edge("S", "a1")
    g.add_edge("S", "a2")
    g.add_edge("S", "b1")
    g.add_edge("a1", "b1")
    g.add_edge("S", "b2")
    g.add_edge("b2", "x")
    g.add_edge("x", "S")
    g.add_edge("S", "b3")
    return g


def test_escapes_aerial_pair_false_when_the_only_alternate_route_uses_a_leg():
    """b1's next-best route back to S (S<->b1 removed) is S-a1-b1 reversed,
    i.e. it must pass through a1 -- the buried leg's onward connectivity
    depends on one of the two aerial legs under test, so this is NOT a
    genuine independent escape."""
    graph = _synthetic_graph()
    assert find_sut._escapes_aerial_pair(graph, "S", "b1", "a1", "a2") is False


def test_escapes_aerial_pair_true_when_the_alternate_route_avoids_both_legs():
    """b2's next-best route back to S (S<->b2 removed) is b2-x-S, which
    touches neither a1 nor a2 -- a genuine independent escape."""
    graph = _synthetic_graph()
    assert find_sut._escapes_aerial_pair(graph, "S", "b2", "a1", "a2") is True


def test_escapes_aerial_pair_none_when_no_alternate_route_exists():
    """b3 has no edge other than the direct one to S -- once that's removed
    it's isolated, so there is no alternate path to judge at all."""
    graph = _synthetic_graph()
    assert find_sut._escapes_aerial_pair(graph, "S", "b3", "a1", "a2") is None
