"""Reference geometry regressions; no simulator reports or gate files are written."""
import math

import numpy as np
import pytest

from nav.follower import DEFAULT_ENVELOPE, RPPFollower, wrap
from nav.kinsim import DEFAULT_FOL, KinSim


ENV = dict(DEFAULT_ENVELOPE, fol=DEFAULT_FOL)


@pytest.mark.parametrize("spacing", [0.01, 0.02, 0.035])
def test_xy_quantization_does_not_become_alternating_yaw(spacing):
    x = np.arange(0, 3, spacing)
    path = np.c_[x, 0.002 * (-1.0) ** np.arange(len(x)), np.zeros(len(x))]
    f = RPPFollower(ENV)
    f.set_path(path)
    commands = []
    for px in np.arange(0.2, 1.5, 0.016):
        cmd, status = f.compute((px, 0.01, 0))
        assert status == "tracking" and cmd[0] > 0
        commands.append(cmd)
    # Millimetre XY rounding must not demand alternating saturated turns.
    wz = np.asarray(commands)[:, 2]
    assert np.all(wz < 0)
    assert np.max(np.abs(wz)) < 0.5


@pytest.mark.parametrize("spacing", [0.01, 0.027])
def test_arc_heading_and_curvature_agree_across_sampling_and_wrap(spacing):
    theta = np.arange(0, 1.2, spacing)
    path = np.c_[np.sin(theta), 1 - np.cos(theta),
                 (theta + math.pi - 0.3 + math.pi) % (2 * math.pi) - math.pi]
    f = RPPFollower(ENV)
    f.set_path(path)
    for s in np.linspace(0.1, 1.0, 70):
        h, k = f._heading_reference(s)
        assert abs(wrap(h - (math.pi - 0.3 + s))) < 2e-4
        assert k == pytest.approx(1, abs=2e-4)
        delta = 1e-6
        dh = wrap(f._heading_reference(s + delta)[0] - f._heading_reference(s - delta)[0])
        assert k == pytest.approx(dh / (2 * delta), abs=1e-6)


def test_stopped_turn_is_a_heading_break_not_incoming_curvature():
    f = RPPFollower(ENV)
    f.set_path([(0, 0, 0), (0.5, 0, 0), (1, 0, 0),
                (1, 0, math.pi / 2), (1, 0.5, math.pi / 2), (1, 1, math.pi / 2)])
    for s in (0.2, 0.8, 0.999):
        assert f._heading_reference(s) == pytest.approx((0, 0))
    for s in (1.001, 1.2, 1.8):
        assert f._heading_reference(s) == pytest.approx((math.pi / 2, 0))


@pytest.mark.parametrize("x", [[0, 0.5, 1.5], [0, 0.5, 1, 1.5]])
def test_requested_endpoint_yaw_does_not_curve_a_straight_approach(x):
    f = RPPFollower(ENV)
    path = np.c_[x, np.zeros((len(x), 2))]
    path[0, 2], path[-1, 2] = -1, 1
    f.set_path(path)
    assert f.goal[2] == 1
    for s in np.linspace(0, 1.49, 50):
        assert f._heading_reference(s) == pytest.approx((0, 0))


def test_nonuniform_monotone_heading_has_no_overshoot():
    f = RPPFollower(ENV)
    f.set_path([(0, 0, 0), (0.2, 0, 0), (0.201, 0, 0.1),
                (0.4, 0, 0.1), (0.7, 0, 0.1)])
    for s in np.linspace(0.2, 0.201, 101):
        h, k = f._heading_reference(s)
        assert -1e-12 <= h <= 0.1 + 1e-12
        assert k >= -1e-9


def test_two_dimensional_paths_retain_geometric_reference():
    f = RPPFollower(ENV)
    f.set_path([(0, 0), (0.5, 0.2), (1, 0.4)])
    assert f.heading_coeff is None
    assert f._heading_reference(0.3)[0] == pytest.approx(math.atan2(0.2, 0.5))


def test_closed_loop_lateral_recovery_keeps_forward_progress():
    f = RPPFollower(ENV)
    f.set_path(np.c_[np.linspace(0, 4, 201), np.zeros((201, 2))])
    sim = KinSim(lateral_sigma=0)
    sim.reset((0, 0.08, 0))
    commands = []
    for tick in range(120):
        observed = sim.pose.copy()
        observed[2] += 0.025 * math.sin(2 * math.pi * tick * 0.1 / 0.4)
        cmd, status = f.compute(observed)
        assert status == "tracking"
        commands.append(cmd)
        for _ in range(5):
            sim.step(cmd)
    assert sim.pose[0] > 1.2
    assert abs(sim.pose[1]) < 0.01
    assert np.all(np.asarray(commands)[:, 0] > 0)
