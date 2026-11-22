"""A4: final goal repair (nav/follower.py _plan_repair / _translate)."""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nav.follower import RPPFollower, load_envelope  # noqa: E402

GOAL = (1.0, 0.0, 0.0)


def _f(key="atlantis_walk"):
    f = RPPFollower(load_envelope(key))
    f.set_path([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), GOAL])
    assert f.rm is not None
    f.hist.append((f.t, 0.0, 0.0, 0.0))
    return f


def _check_at(f, dx_body, dy_body, yaw_err):
    """Robot pose such that the goal is at body offset (dx, dy) and goal yaw - heading = yaw_err."""
    th = GOAL[2] - yaw_err
    c, s = math.cos(th), math.sin(th)
    x = GOAL[0] - (c * dx_body - s * dy_body)
    y = GOAL[1] - (s * dx_body + c * dy_body)
    f.hist[-1] = (f.t, x, y, th)
    return f._check(x, y, th, math.hypot(dx_body, dy_body), yaw_err)


def test_constants_derived_from_measurements():
    rm = _f().rm
    assert rm["tq_src"] == "primitives_atlantis_walk.json" and rm["tq"] == pytest.approx(0.6)
    assert rm["coarse"] == {1: 1.5, -1: -1.5}           # smallest measured turning circle
    assert rm["fine"] == {1: 0.5, -1: -1.0}            # unchanged CommandModel levels
    assert 0.010 < rm["d0"] < 0.025
    assert rm["n_max"] >= 3
    for c in rm["U"]:
        assert abs(c["cmd"][2]) < 1e-12 and c["step"] >= 2 * c["sigma"] - 1e-12


def test_large_yaw_fixed_first_with_coarse_level():
    f = _f()
    cmd, status = _check_at(f, 0.0, 0.03, math.radians(120))
    assert f.state == "rotate_fix" and status == "rotate"
    assert cmd[0] == 0 and cmd[1] == 0 and cmd[2] == pytest.approx(1.5)


def test_lateral_position_repair_without_turning():
    f = _f()
    cmd, status = _check_at(f, 0.0, 0.040 + 0.03, 0.0)    # goal 70 mm to the left, yaw fine
    assert f.state == "translate"
    assert abs(cmd[2]) < 1e-12 and cmd[1] > 0               # exact safe lateral-left command, no yaw
    assert any(np.allclose(cmd, s) for s in f.env.safe)


def test_turn_displacement_budget_translates_first():
    f = _f()
    yaw = math.radians(12)                                   # small (fine) yaw error
    d = f.rm["tp"] - 0.5 * f._D(yaw, f.rm["fine"][1])        # does not leave room for the turn's shift
    _check_at(f, 0.0, -d, yaw)
    assert f.state == "translate"
    f2 = _f()
    _check_at(f2, 0.0, -0.002, yaw)                          # fits: fine turn only
    assert f2.state == "rotate_fix" and f2._coarse_level == pytest.approx(0.5)


def test_reversal_guard_requires_progress():
    f = _f()
    _check_at(f, 0.0, 0.07, 0.0)
    assert f.state == "translate"
    f._set_state("settle")
    # overshoot to the other side with no net progress: opposite-sense lateral not allowed
    _, status = _check_at(f, 0.0, -0.07, 0.0)
    assert f.state != "translate" or float(f._tr["cand"]["u"][1]) > -0.5


def test_repair_is_bounded_and_reports_best_pose():
    f = _f()
    n = f.max_fixes                                      # original bound (A4 attempt 3)
    for k in range(n):
        f.stuck = False
        f._check(*(1.0, 0.07, 0.0), 0.07 + 0.001 * k, 0.0)
        f._set_state("settle")
    _, status = f._check(1.0, 0.07, 0.0, 0.07, 0.0)
    assert status == "stuck" and f.info["reason"] == "final fixes exhausted"
    assert f.info["best"][0] == pytest.approx(0.07 / f.rm["tp"])


def test_translate_stops_on_projection_and_never_turns():
    f = _f()
    _check_at(f, 0.0, 0.07, 0.0)
    cand = f._tr["cand"]
    for _ in range(3):
        f.t += 0.0
        cmd, _ = f._translate(1.0, -0.07, 0.0)
        assert abs(cmd[2]) < 1e-12
    cmd, _ = f._translate(1.0, 0.0, 0.0)                     # error gone along u
    assert f.state == "settle" and np.allclose(cmd, 0)


def test_fixture_envelope_keeps_legacy_repair():
    f = RPPFollower(None)
    f.set_path([(0.0, 0.0, 0.0), GOAL])
    assert f.rm is None


def test_fine_turn_first_when_translation_cannot_make_room():
    f = _f()
    yaw = math.radians(20)                                   # below theta_c, above ty
    assert yaw < f.rm["theta_c"][1]
    room = f.rm["tp"] - f._D(yaw, f.rm["fine"][1])
    # error straight ahead, just above the room the turn leaves; the forward/back quanta
    # (>= 65 mm) overshoot it and lateral steps do not reduce it: no translation helps
    d = room + 0.005
    _check_at(f, d, 0.0, yaw)
    assert f.state == "rotate_fix" and f._coarse_level == pytest.approx(0.5)


def test_translate_holds_planned_quanta():
    f = _f()
    _check_at(f, 0.0, 0.07, 0.0)
    n = f._tr["ticks"]
    assert n >= round(f.rm["tq"] / f.dt)
    for _ in range(n - 1):
        cmd, _ = f._translate(1.0, -0.07, 0.0)          # _out advances f.t
        assert f.state == "translate" and abs(cmd[1]) > 0


def test_non_moving_translation_is_not_reused():
    f = _f()
    _check_at(f, 0.0, 0.07, 0.0)
    first = tuple(f._tr["cand"]["cmd"].tolist())
    f._set_state("settle")
    _check_at(f, 0.0, 0.07, 0.0)                          # no progress at the next settled check
    assert first in f._banned
    assert f.state != "translate" or tuple(f._tr["cand"]["cmd"].tolist()) != first
