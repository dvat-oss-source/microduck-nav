import json
from pathlib import Path

import numpy as np
import pytest

from tools.steering_metrics import command_metrics, fixed_sample, path_metrics, trace_metrics


def test_command_metrics_integrates_pure_yaw_and_counts_sign_changes():
    commands = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.1, 0.0, 0.0, 1.0],
        [0.2, 0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0, -1.0],
    ])
    result = command_metrics(commands, np.array([0.0, 0.4]))

    assert result["wz_sign_flips"] == 1
    assert result["pure_yaw_s"] == pytest.approx(0.3)
    assert result["pure_yaw_fraction"] == pytest.approx(0.75)
    assert result["commanded_abs_yaw_rad"] == pytest.approx(0.3)


def test_path_metrics_uses_unwrapped_actual_heading_changes():
    yaw = np.deg2rad([170.0, 179.0, -179.0, -170.0])
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    result = path_metrics(xy, yaw, geodesic=2.0, success=True)

    assert result["net_yaw_rad"] == pytest.approx(np.deg2rad(20.0))
    assert result["total_abs_yaw_rad"] == pytest.approx(np.deg2rad(20.0))
    assert result["total_to_net_yaw_ratio"] == pytest.approx(1.0)
    assert result["spl_term"] == pytest.approx(
        2.0 / max(2.0, result["path_length_ma30_m"]))


def test_fixed_sample_contains_failures_nav125_and_longest_success(tmp_path):
    episodes = {
        "nav_018": {"task_type": "nav", "success": False, "duration": 2.0},
        "nav_063": {"task_type": "nav", "success": False, "duration": 3.0},
        "nav_125": {"task_type": "nav", "success": True, "duration": 50.0},
        "nav_129": {"task_type": "nav", "success": True, "duration": 90.0},
        "nav_001": {"task_type": "nav", "success": True, "duration": 10.0},
    }
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"episodes": episodes, "seeds_sha256": "seed-hash"}))

    sample = fixed_sample(report, sample_size=4, rng_seed=20261109)

    assert sample["gt_failure_ids"] == ["nav_018", "nav_063"]
    assert sample["longest_success_id"] == "nav_129"
    assert sample["mandatory_ids"] == ["nav_018", "nav_063", "nav_125", "nav_129"]
    assert sample["episode_ids"] == sample["mandatory_ids"]
    assert sample["source_seeds_sha256"] == "seed-hash"


def test_frozen_nav125_matches_prompt3_baseline():
    root = Path(__file__).resolve().parents[1]
    trace = root / "reports/traces/frozen_v2_atlantis_walk_gt/nav_125.npz"
    report = json.loads((root / "reports/bench_frozen_v2_atlantis_walk_gt_20261110_020423.json")
                        .read_text())

    result = trace_metrics(trace, report["episodes"]["nav_125"])

    assert result["wz_sign_flips"] == 258
    assert result["wz_sign_flips_per_minute"] == pytest.approx(117.0, abs=0.5)
    assert result["pure_yaw_s"] == pytest.approx(38.1, abs=0.1)
    assert np.degrees(result["total_abs_yaw_rad"]) == pytest.approx(6306.0, abs=5.0)
    assert np.degrees(result["net_yaw_rad"]) == pytest.approx(115.0, abs=1.0)
    assert result["total_to_net_yaw_ratio"] == pytest.approx(54.8, abs=1.0)
    assert result["sway_rms_m"] == pytest.approx(0.0161, abs=0.0002)
    assert result["path_ratio_ma30_geodesic"] == pytest.approx(1.32, abs=0.01)
