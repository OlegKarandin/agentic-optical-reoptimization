import asyncio

import pytest

from storm_reoptimizer.eval.heat_margin import (
    MARGIN_QUANTILES, MarginCurve, MarginCurveError, curve_grid,
    measure_margin_curve, p_link_down, quantile_key,
)
from storm_reoptimizer.eval.thermal import NfMixture

CURVE = MarginCurve(nf_delta_db=(0.0, 1.0, 2.0), sut_margin_db=(0.4, -0.6, -1.6))


def test_keys_and_quantiles_are_the_spec_set():
    assert MARGIN_QUANTILES == (0.05, 0.25, 0.50, 0.75, 0.95)
    assert [quantile_key(q) for q in MARGIN_QUANTILES] == ["p05", "p25", "p50", "p75", "p95"]


def test_margin_at_interpolates_and_clamps_negative_nf_to_zero():
    assert CURVE.margin_at(0.5) == pytest.approx(-0.1)
    assert CURVE.margin_at(-0.3) == pytest.approx(0.4)      # Review Focus 3
    with pytest.raises(MarginCurveError):
        CURVE.margin_at(2.5)


@pytest.mark.parametrize("bad", [
    dict(nf_delta_db=(0.1, 1.0), sut_margin_db=(0.4, 0.0)),     # must start at 0
    dict(nf_delta_db=(0.0, 0.0), sut_margin_db=(0.4, 0.0)),     # strictly increasing
    dict(nf_delta_db=(0.0, 1.0), sut_margin_db=(0.4, 0.4)),     # strictly decreasing
])
def test_curve_validates(bad):
    with pytest.raises(MarginCurveError):
        MarginCurve(**bad)


def test_p_link_down_edges():
    point = lambda c, f, p: NfMixture(p, c, 0.0, f, 0.0)  # noqa: E731
    mix = point(0.6, 1.5, 0.10)
    assert p_link_down(CURVE, mix, 0.0) == 1.0               # 0.6 > 0.4 always down
    assert p_link_down(CURVE, mix, 1.0) == pytest.approx(0.10)
    assert p_link_down(CURVE, mix, 1.8) == 0.0               # never within the curve
    assert p_link_down(CURVE, mix, -1.0) == 1.0              # upshift above: margin(0)<target


def test_curve_grid_spans_the_tail_and_contains_every_realized_point():
    mix = NfMixture(0.1, 0.03, 0.003, 0.36, 0.02)
    grid = curve_grid([mix], [0.0, 0.123456789], points=5)
    assert grid[0] == 0.0 and grid[-1] == pytest.approx(mix.tail())
    assert 0.123456789 in grid and list(grid) == sorted(set(grid))


def test_measure_margin_curve_branches_and_restores():
    calls, state = [], {"applied": 0.0}

    async def call(name, arguments=None, **kw):
        calls.append(name)
        if name == "snapshot_create":
            return {"id": "base"}
        if name == "inject_degradation":
            state["applied"] += arguments["nf_delta"]
        if name == "snapshot_restore":
            state["applied"] = 0.0
        if name == "get_lightpaths":
            a = state["applied"]
            return [{"id": "lp", "qot": {"margin_db": 0.4 - a / 2}},
                    {"id": "by", "qot": {"margin_db": 3.0 - a / 4}}]
        return {}
    curve = asyncio.run(measure_margin_curve(
        call, amplifiers=("amp_x_y_0", "amp_y_x_0"), lightpath_id="lp",
        bystander_ids=("by",), grid=(0.0, 1.0)))
    assert curve.sut_margin_db == (0.4, -0.6)     # two amps x 1.0 each -> 2.0 applied
    assert curve.bystander_margin_db == {"by": (3.0, 2.5)}
    assert calls.count("snapshot_restore") == 1     # no branch at ΔNF = 0
