import json

import numpy as np

from tools.check_trace_parity import compare_trace


def _write(path, *, command=0.25, plan_time=0.01, qpos_value=1.0):
    metadata = {
        "plan_time_s": plan_time,
        "plan_times_s": [plan_time],
        "plan_log": [{"complete": True, "plan_time_s": plan_time, "reason": None}],
        "commands": [[0.0, command, 0.0, 0.0]],
        "estimates": [[0.0, 1.0, 2.0, 3.0]],
        "transitions": [{"t": 0.0, "state": "follow"}],
        "wall_s": 0.2,
    }
    task = {"ep": "nav_000", "seed": 123}
    np.savez_compressed(path, qpos=np.array([[qpos_value, 0.0]], dtype=np.float64),
                        t=np.array([0.0], dtype=np.float64),
                        metadata=json.dumps(metadata), task=json.dumps(task))


def test_timing_differences_are_excluded_but_trace_is_exact(tmp_path):
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    _write(left, plan_time=0.01)
    _write(right, plan_time=0.09)

    result = compare_trace(left, right)

    assert result["passed"]
    assert result["checks"] == {
        "qpos_bit_identical": True,
        "t_bit_identical": True,
        "task_equal": True,
        "metadata_equal_excluding_timing": True,
    }


def test_command_difference_fails_metadata_parity(tmp_path):
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    _write(left)
    _write(right, command=0.25000000000000006)

    result = compare_trace(left, right)

    assert not result["passed"]
    assert not result["checks"]["metadata_equal_excluding_timing"]


def test_qpos_difference_fails_byte_identity(tmp_path):
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    _write(left)
    _write(right, qpos_value=np.nextafter(1.0, 2.0))

    result = compare_trace(left, right)

    assert not result["passed"]
    assert not result["checks"]["qpos_bit_identical"]
