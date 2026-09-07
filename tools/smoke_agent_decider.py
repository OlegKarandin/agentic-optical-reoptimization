# tools/smoke_agent_decider.py
"""One real API call, to prove the request shape ClaudeDecider builds is one
the API actually accepts. NOT a test -- it costs money and needs credentials,
so it lives outside tests/ and is never collected by pytest. Run it once
after any change to the request shape.

Run: python tools/smoke_agent_decider.py"""
from __future__ import annotations

import asyncio

from storm_reoptimizer.eval.agent import ClaudeDecider
from storm_reoptimizer.eval.observation import Observation
from storm_reoptimizer.eval.scenario_file import ConeAtHorizon, Issuance


def main() -> None:
    obs = Observation(
        scenario_id="SMOKE", service_under_test="storm-svc-1", hour="t1",
        hour_index=1, hours_remaining=2,
        issuance=Issuance(issued_at="t1", horizons={
            "t3": ConeAtHorizon(
                cone={"type": "Polygon", "coordinates": []}, width_km=320.0,
                center={"lat": 24.855553, "lon": 81.327777})}),
        exposure={"storm-svc-1": {"t3": {
            "hours_ahead": 2, "offset_km": 0.0, "width_km": 320.0,
            "p_cut": 0.341, "demand_gbps": 300.0}}},
        services=({"id": "storm-svc-1", "demand_gbps": 300.0,
                   "src_router": "router_satna",
                   "dst_router": "router_allahabad",
                   "working_path": [], "protection_path": []},),
        spares_on_hand=1, lead_time_hours=1,
        risk_group_ids={"t3": "rg_SMOKE_t1_t3"})

    decision = asyncio.run(ClaudeDecider().timing(obs))
    print(f"action={decision.action}\nreasoning={decision.reasoning}")


if __name__ == "__main__":
    main()
