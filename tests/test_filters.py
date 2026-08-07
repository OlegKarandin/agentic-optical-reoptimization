"""Event-type -> vulnerability-predicate table (CLAUDE.md's
"(event_type -> filter) table"). Only storm->aerial is real today; flood's
buried-asset filter is step 7's job (needs a "low-lying/river-crossing"
attribute this toy topology doesn't have yet -- see the step-3 design
spec)."""
import pytest
from shapely.geometry import LineString

from storm_reoptimizer.events.filters import get_filter
from storm_reoptimizer.geo_mapper import Edge


def test_storm_filter_keeps_aerial_and_rejects_buried():
    filter_fn = get_filter("storm")
    aerial = Edge(src="a", dst="b", mount_type="aerial",
                  geometry=LineString([(0, 0), (1, 1)]))
    buried = Edge(src="c", dst="d", mount_type="buried",
                  geometry=LineString([(0, 0), (1, 1)]))
    assert filter_fn(aerial) is True
    assert filter_fn(buried) is False


def test_get_filter_raises_for_an_unmapped_event_type():
    with pytest.raises(KeyError, match="flood"):
        get_filter("flood")
