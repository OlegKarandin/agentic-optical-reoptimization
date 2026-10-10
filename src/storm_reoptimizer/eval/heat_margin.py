# src/storm_reoptimizer/eval/heat_margin.py
"""Heat spec §3.4 with plan deviation 5: ONE curve per episode, SUT margin
(at the provisioned mode) against an absolute ΔNF applied to every heated
amplifier, measured on server branches before any decision. The
observation's margin quantiles and the oracle's down-probabilities both
read THIS curve, so they can never disagree. Mode headroom is additive
(Task 1, `test_downshift_adds_exactly_the_threshold_gap...`): margin at
mode j = curve(δ) + θ_provisioned − θ_j."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Sequence

import numpy as np

from .thermal import NfMixture

MARGIN_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
CURVE_POINTS = 13
CURVE_TOLERANCE_DB = 1e-3   # == derived.DERIVED_TOLERANCE


class MarginCurveError(ValueError):
    """A curve that is malformed or read outside its measured range."""


def quantile_key(q: float) -> str:
    return f"p{round(q * 100):02d}"


@dataclass(frozen=True)
class MarginCurve:
    nf_delta_db: tuple[float, ...]
    sut_margin_db: tuple[float, ...]
    bystander_margin_db: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.nf_delta_db)
        if (n < 2 or len(self.sut_margin_db) != n
                or any(len(v) != n for v in self.bystander_margin_db.values())):
            raise MarginCurveError("curve arrays must align and hold >= 2 points")
        if self.nf_delta_db[0] != 0.0:
            raise MarginCurveError("curve must start at ΔNF = 0")
        if any(b <= a for a, b in zip(self.nf_delta_db, self.nf_delta_db[1:])):
            raise MarginCurveError("ΔNF grid must be strictly increasing")
        if any(b >= a for a, b in zip(self.sut_margin_db, self.sut_margin_db[1:])):
            raise MarginCurveError("SUT margin must fall strictly as ΔNF rises")

    def margin_at(self, nf_delta_db: float) -> float:
        """Negative ΔNF (a forecast tail below setpoint) clamps to 0: NF never
        improves on nominal (plan deviation 9)."""
        d = max(0.0, nf_delta_db)
        if d > self.nf_delta_db[-1] + 1e-9:
            raise MarginCurveError(
                f"ΔNF {d} beyond the measured curve (max {self.nf_delta_db[-1]})")
        return float(np.interp(d, self.nf_delta_db, self.sut_margin_db))

    def to_dict(self) -> dict:
        return {"nf_delta_db": list(self.nf_delta_db),
                "sut_margin_db": list(self.sut_margin_db),
                "bystander_margin_db": {k: list(v) for k, v in self.bystander_margin_db.items()}}

    @classmethod
    def from_dict(cls, raw: dict) -> "MarginCurve":
        return cls(tuple(raw["nf_delta_db"]), tuple(raw["sut_margin_db"]),
                   {k: tuple(v) for k, v in raw.get("bystander_margin_db", {}).items()})


def p_link_down(curve: MarginCurve, mixture: NfMixture, headroom_db: float) -> float:
    """P[SUT margin at a mode with `headroom_db` over the provisioned one < 0]
    = P[curve(ΔNF) < −headroom]. 0.0 when the curve never falls that far
    inside its grid: the grid reaches every component's 0.999 point, so the
    dropped mass is <= 0.001 (<= 0.8 Gbps at L = 800, far under the 25
    Gbps·h graded floor)."""
    target = -headroom_db
    if curve.sut_margin_db[0] < target:
        return 1.0
    if curve.sut_margin_db[-1] >= target:
        return 0.0
    delta_star = float(np.interp(target, curve.sut_margin_db[::-1], curve.nf_delta_db[::-1]))
    return mixture.p_exceed(delta_star)


def curve_grid(mixtures: Iterable[NfMixture], realized: Iterable[float],
               points: int = CURVE_POINTS) -> tuple[float, ...]:
    realized = list(realized)
    top = max([m.tail() for m in mixtures] + realized)
    if top <= 0.0:
        raise MarginCurveError("every forecast and realized ΔNF is zero; nothing to measure")
    grid = {round(top * i / (points - 1), 9) for i in range(points)}
    grid |= {round(r, 9) for r in realized}
    return tuple(sorted(grid))


async def measure_margin_curve(
    call: Callable[..., Awaitable], *, amplifiers: Sequence[str],
    lightpath_id: str, bystander_ids: Sequence[str], grid: Sequence[float],
) -> MarginCurve:
    """One branch per grid point: inject ΔNF on every heated amp, read
    margins, restore. Must run before any heat or mode change is applied."""
    base = (await call("snapshot_create"))["id"]
    sut: list[float] = []
    bys: dict[str, list[float]] = {b: [] for b in bystander_ids}
    for delta in grid:
        if delta:
            for amp in amplifiers:
                await call("inject_degradation", {"asset_id": amp, "nf_delta": delta})
        rows = {lp["id"]: lp for lp in await call("get_lightpaths", expect_list=True)}
        sut.append(rows[lightpath_id]["qot"]["margin_db"])
        for b in bystander_ids:
            bys[b].append(rows[b]["qot"]["margin_db"])
        if delta:
            await call("snapshot_restore", {"snapshot_id": base})
    return MarginCurve(tuple(grid), tuple(sut), {k: tuple(v) for k, v in bys.items()})
