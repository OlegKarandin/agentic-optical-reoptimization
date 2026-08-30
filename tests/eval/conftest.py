"""Shared eval-harness test fixtures. The example scenario lives here rather
than in a test module so several test modules can use it without
cross-module imports (tests/eval/ has no __init__.py by design)."""
import textwrap

import pytest

EXAMPLE_SCENARIO_YAML = textwrap.dedent("""
    id: EXAMPLE_A
    pair: EXAMPLE
    seed: 17
    state_file: eval/states/loaded-s17.json
    service_under_test: storm-svc-1
    track: hudhud
    hours: [t0, t1, t2, t3]
    decision_hour: t1
    lead_time_hours: 1
    spares_on_hand: 1
    depot_site: satna
    spare_inventory: {satna: 1}
    damage_radius_km: 74
    reference_avoid:
      risk_groups: [rg_ref]
    forecast:
      t0:
        t1: {cone: {type: Polygon, coordinates: [[[81.0, 25.0], [81.1, 25.0], [81.1, 25.1], [81.0, 25.0]]]}, width_km: 190, center: {lat: 25.0, lon: 81.0}}
        t3: {cone: {type: Polygon, coordinates: [[[81.0, 25.0], [81.1, 25.0], [81.1, 25.1], [81.0, 25.0]]]}, width_km: 90, center: {lat: 25.0, lon: 81.0}}
      t1:
        t3: {cone: {type: Polygon, coordinates: [[[81.0, 25.0], [81.1, 25.0], [81.1, 25.1], [81.0, 25.0]]]}, width_km: 90, center: {lat: 25.2, lon: 81.0}}
    realized:
      t3: [fiber_004, fiber_005]
    gold:
      survived: [storm-svc-1, svc-b]
      max_spares_wasted: 1
      decision_at_t0: wait
      label: wait
      rationale: |
        P(cut at t3 | issued_t0) ~ 0.25; 0.25*300G < 1.0*250G.
    flip_variable: [svc-b, cone, centre]
    metadata:
      cone_width_km: 90
      cone_motion_kmh: 20
      n_future_claimants: 1
      exposure_horizon_hours: 2
      spares_on_hand: 1
""")


@pytest.fixture
def example_scenario_yaml() -> str:
    return EXAMPLE_SCENARIO_YAML


@pytest.fixture
def write_scenario(tmp_path):
    """Write scenario YAML to a temp file and return its Path."""
    def _write(text: str, name: str = "EXAMPLE_A.yaml"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path
    return _write
