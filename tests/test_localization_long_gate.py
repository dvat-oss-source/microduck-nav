"""Long localization gate orchestration; no physics/rendering in unit tests."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools import localization_gate as gate


class FakeSim:
    def reset(self, seed, start_pose):
        self.t = 0.
        self.pose = np.array(start_pose, float)
        return self.truth()

    def truth(self):
        return SimpleNamespace(t=self.t, pose2=self.pose.copy(), fallen=False)

    def step(self, command):
        self.t = round(self.t + .02, 8)
        self.pose = np.array(command, float)
        return self.truth(), None


class FakeLocalizer:
    seen = {'odom': [], 'odom+fiducial': []}
    drift = 0.

    def __init__(self, sim, mode, **kwargs):
        self.sim, self.mode = sim, mode
        self._ticks, self._cam_period = 0, 10
        self.stats = dict(det=0, used=0, gated=0, upd_range=0)

    def reset(self, pose):
        self.pose = np.array(pose, float)

    def update(self, sim, gt_for_eval_only=None):
        self.pose = sim.pose.copy()
        if self.mode != 'gt':
            assert gt_for_eval_only is None
            self.seen[self.mode].append((sim.t, tuple(sim.pose)))
            self._ticks += 1
            if self.mode == 'odom+fiducial':
                self.pose[0] += self.drift * sim.t
                if self._ticks % self._cam_period == 0:
                    self.stats['det'] += 2
                    self.stats['used'] += 1
                    self.stats['gated'] += 1
                    self.stats['upd_range'] += 1
        return self.pose.copy()


class FakeFollower:
    stuck = False

    def __init__(self, *args, **kwargs):
        pass

    def set_path(self, poses):
        self.goal = np.array(poses[-1], float)
        self.calls = 0

    def compute(self, pose, velocity):
        self.calls += 1
        status = 'stuck' if self.stuck else ('done' if self.calls > 1 else 'tracking')
        return self.goal.copy(), status


@pytest.fixture
def route(monkeypatch):
    import nav.follower
    import nav.planner
    FakeLocalizer.seen = {'odom': [], 'odom+fiducial': []}
    FakeLocalizer.drift = 0.
    FakeFollower.stuck = False
    monkeypatch.setattr(gate, 'Localizer', FakeLocalizer)
    monkeypatch.setattr(nav.follower, 'RPPFollower', FakeFollower)
    monkeypatch.setattr(nav.planner, 'plan', lambda gm, prims, start, goal, **kw:
                        SimpleNamespace(complete=True, poses=[start, goal], reason='', expansions=1))
    points = [{'room': name, 'pose': [float(i), 0., 0.]}
              for i, name in enumerate(gate.ROOM_ORDER)]
    gm = SimpleNamespace(room_of=lambda x, y: gate.ROOM_ORDER[int(round(x))])

    def run(**kwargs):
        return gate.run_long_route(FakeSim(), SimpleNamespace(n_renders=0), gm, None, None,
                                   points, seed=4, **kwargs)
    return run


def test_long_route_all_rooms_duration_and_identical_observations(route):
    result = route()
    assert result['passed'] and result['route_complete']
    assert result['duration_s'] >= 60.
    assert set(result['reached_rooms']) == set(gate.ROOM_ORDER)
    assert FakeLocalizer.seen['odom'] == FakeLocalizer.seen['odom+fiducial']
    assert len(FakeLocalizer.seen['odom']) == result['steps']
    assert result['error_vs_time'][-1]['t_s'] == result['duration_s']
    assert len(result['trajectory_sha256']) == 64
    stats = result['fiducial_stats']
    assert stats['acceptance_fraction'] == .5
    assert stats['accepted_tick_fraction'] == 1.


def test_long_route_exposes_drift_in_terminal_and_time_series(route):
    FakeLocalizer.drift = .002
    result = route()
    assert result['route_complete'] and not result['passed']
    assert result['odom+fiducial']['final_xy_m'] >= .12
    assert result['error_vs_time'][-1]['errors']['odom+fiducial'][0] >= .12
    assert any(t['errors']['odom+fiducial'][0] > .05 for t in result['per_room_terminal'])


def test_incomplete_route_cannot_pass_even_with_accurate_localization(route):
    FakeFollower.stuck = True
    result = route()
    assert not result['route_complete'] and not result['passed']
    assert result['reached_rooms'] == [] and result['duration_s'] == 0.
    assert 'stuck' in result['failure']


def test_long_route_rejects_short_horizon(route):
    with pytest.raises(ValueError, match='>=60'):
        route(min_seconds=59.)


def test_no_detections_do_not_imply_acceptance():
    stats = gate.fix_acceptance(dict(det=0, used=0, gated=0, upd_range=0), 5, 0)
    assert stats['acceptance_fraction'] is None
    assert stats['accepted_tick_fraction'] == 0.
