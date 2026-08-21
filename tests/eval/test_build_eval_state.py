"""The loaded operating network must be reproducible: same seed, same
target_mean_util, same service count and mean utilization within tolerance
(eval design spec, "Testing"). This is a real two-stage build against the
143-node toy topology under the sibling conda env -- slow, and cached on disk
by the loaded_state_path fixture so the rest of the suite pays for it once."""
import json
import subprocess
from pathlib import Path

TOPOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "storm_reoptimizer" / "data" / "toy_india_topology.json"
)
BUILDER = Path(__file__).parent.parent.parent / "tools" / "build_eval_state.py"


def test_loaded_state_has_the_service_under_test_and_a_real_background_load(
    loaded_state_path,
):
    doc = json.loads(loaded_state_path.read_text(encoding="utf-8"))
    service_ids = {s["id"] for s in doc["services"]}
    assert "storm-svc-1" in service_ids
    # Background load: enough services for spare contention, and enough
    # lit lightpaths for an ip_reroute candidate to exist at all.
    assert len(service_ids) >= 10
    assert len(doc["lightpaths"]) >= 10
    assert doc["meta"]["seed"] == 17
    assert 0.4 <= doc["meta"]["achieved_mean_util"] <= 0.8


def test_build_is_reproducible_at_the_same_seed(
    tmp_path, loaded_state_path, local_server_command, local_server_env,
):
    first = json.loads(loaded_state_path.read_text(encoding="utf-8"))
    out = tmp_path / "rebuild-s17.json"
    proc = subprocess.run(
        [local_server_command[0], str(BUILDER),
         "--topology", str(TOPOLOGY_PATH), "--out", str(out), "--seed", "17"],
        env=local_server_env, capture_output=True, text=True, timeout=1800,
        check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    second = json.loads(out.read_text(encoding="utf-8"))

    assert [s["id"] for s in second["services"]] == [s["id"] for s in first["services"]]
    assert abs(second["meta"]["achieved_mean_util"]
               - first["meta"]["achieved_mean_util"]) < 0.02
