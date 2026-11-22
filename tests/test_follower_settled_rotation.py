"""Settled rotation must produce lasting yaw before retaining a weak level."""
import math
import numpy as np
import pytest
from nav.follower import DEFAULT_ENVELOPE, RPPFollower

def _settled_rotation_fixture(stronger=True):
    levels = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1] if stronger else [-0.5, -0.25, 0, 0.25, 0.5]
    rows, safe = [], []
    for vx in (0, 0.2, 0.3):
        for wz in levels:
            command = [vx, 0, wz]
            # Low safe levels can twist the trunk without a reliable yaw step.
            achieved_w = 0.01 * np.sign(wz) if vx == 0 and abs(wz) == 0.25 else wz
            safe.append(command)
            rows.append({'command': command, 'falls': 0,
                         'mean_velocity': [vx, 0, achieved_w],
                         'end_mean': [4 * vx, 0, 4 * achieved_w], 'end_std': [0.001] * 3})
    env = dict(DEFAULT_ENVELOPE, safe_cmds=safe, vx=[0, 0.3], wz=[min(levels), max(levels)],
               measurement={'seconds_per_command': 4, 'rows': rows},
               fol={'vx': {'gain': 1, 'tau': 0.1, 'delay': 0.1},
                    'wz': {'gain': 1, 'tau': 0.1, 'delay': 0}})
    f = RPPFollower(env)
    f.set_path([(0, 0, 0)])
    return f, {tuple(c) for c in safe}


def _run_transient_rotation(f, sign, settled_heading):
    initial = -sign * 0.3
    for _ in range(20):
        command, status = f.compute((0, 0, initial))
        if command[2] != 0:
            break
    else:
        pytest.fail('final correction never started')
    assert status == 'rotate' and command[2] == sign * 0.5
    # The commanded level appears to reach the target while twisting, so the
    # online transient-progress monitor has no reason to raise its floor.
    command, _ = f.compute((0, 0, 0))
    assert f.state == 'settle' and np.array_equal(command, [0, 0, 0])
    for _ in range(30):
        command, status = f.compute((0, 0, settled_heading))
        if f.state != 'settle':
            return command, status
    pytest.fail('settled correction outcome was never checked')


@pytest.mark.parametrize('sign', [-1, 1])
def test_settled_posture_relaxation_escalates_above_actual_command(sign):
    f, safe = _settled_rotation_fixture()
    command, status = _run_transient_rotation(f, sign, -sign * 0.3)
    assert status == 'rotate' and command[2] == sign * 1.0
    assert tuple(command) in safe
    assert f.min_ticks[sign] == 2
    assert f._rotation_floor[sign] == sign * 1.0
    assert len(f.escalation_events) == 1
    event = f.escalation_events[0]
    assert event['reason'] == 'settled_rotation_relaxed'
    assert event['from_wz'] == sign * 0.5  # skip lower but safe +/-0.25 rows
    # A second failed burst at the strongest measured row stays bounded.
    for _ in range(4):
        f.compute((0, 0, 0))
        if f.state == 'settle':
            break
    assert f.state == 'settle'
    for _ in range(30):
        command, status = f.compute((0, 0, -sign * 0.3))
        if f.state != 'settle':
            break
    assert status == 'rotate' and command[2] == sign * 1.0
    assert tuple(command) in safe and len(f.escalation_events) == 1
    f.set_path([(0, 0, 0)])
    assert f._rotation_floor == {1: 0, -1: 0}
    assert not f.escalation_events


@pytest.mark.parametrize('sign', [-1, 1])
def test_successful_settled_rotation_does_not_escalate(sign):
    f, _ = _settled_rotation_fixture()
    command, status = _run_transient_rotation(f, sign, -sign * 0.1)
    assert status == 'done' and np.array_equal(command, [0, 0, 0])
    assert f.min_ticks[sign] == 1
    assert not f.escalation_events


@pytest.mark.parametrize('sign', [-1, 1])
def test_settled_rotation_cannot_invent_a_stronger_safe_level(sign):
    f, safe = _settled_rotation_fixture(stronger=False)
    command, status = _run_transient_rotation(f, sign, -sign * 0.3)
    assert status == 'rotate' and command[2] == sign * 0.5
    assert tuple(command) in safe
    assert f.min_ticks[sign] == 2
    assert not f.escalation_events


