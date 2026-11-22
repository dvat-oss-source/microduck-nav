from types import SimpleNamespace
import numpy as np
from nav.fsm import PLAN_MAX_EXPANSIONS, NavFSM


class Sim:
    data = object()
    initial_policy_key = 'velocity_hard'
    t = 0
    trace = [None]
    skill_done = False
    def force_switch(self, key): self.key = key
    def switch_policy(self, key): self.key = key; return True

class Loc:
    def __init__(self):
        self.calls = 0
        frame = SimpleNamespace(imu_quat=np.array([1., 0, 0, 0]), q=np.zeros(14), foot_contacts=(True, False))
        self.last_sensor_frame = frame
        self.reader = SimpleNamespace(read=lambda data: frame)
        self.kin = SimpleNamespace(R_trunk_imu=np.eye(3), feet_in_trunk=lambda q: [(np.array([0, 0, -.1]), np.eye(3))] * 2)
    def update(self, sim, gt_for_eval_only=None):
        self.calls += 1
        return np.zeros(3)

class Follower:
    def __init__(self): self.calls = 0
    def set_path(self, poses): self.poses = poses
    def compute(self, pose, vel): self.calls += 1; return np.ones(3), 'tracking'


def test_follower_rate_and_localizer_each_tick(monkeypatch):
    monkeypatch.setattr('nav.fsm.plan', lambda *a, **k: SimpleNamespace(complete=True, poses=[(0,0,0),(1,0,0)], plan_time_s=.01))
    s, loc, follower = Sim(), Loc(), Follower()
    fsm = NavFSM(s, None, None, follower, loc, contracts={'recover': {'policy_key':'stand', 'max_time_s':10}})
    fsm.start((1, 0, 0)); gt = SimpleNamespace(fallen=False)
    fsm.tick(gt)
    commands = [fsm.tick(gt)[0] for _ in range(10)]
    assert fsm.state == 'follow' and loc.calls == 11 and follower.calls == 2
    assert all(np.array_equal(c, np.ones(3)) for c in commands)
    assert fsm.planner_times == [.01]
    assert fsm.plan_log[0]['plan_time_s'] == .01


def test_skill_refuses_contract_from_other_dynamics():
    sim = Sim(); sim.dynamics = 'training'
    fsm = NavFSM(sim, None, None, Follower(), Loc(), contracts={'protocol': {'dynamics':'infer'}})
    fsm.start(None, task='kick', target_xy=(0., 0.))
    assert fsm.state == 'fail' and fsm.reason == 'skill_dynamics_mismatch'


def test_skill_contract_provenance_and_support(monkeypatch):
    sim = Sim(); sim.dynamics = 'training'; sim._lock = {'kick': {'md5': 'abc'}, 'velocity_hard': {'md5': 'def'}}
    sim.model = SimpleNamespace(body=lambda name: SimpleNamespace(id=int(name[-1])),
        geom=lambda name: SimpleNamespace(id=int(name[5])), body_mass=np.full(3, .025),
        body_inertia=np.tile([1.6e-5]*3, (3, 1)), body_ipos=np.tile([0., 0., .04], (3, 1)),
        body_iquat=np.tile([1., 0., 0., 0.], (3, 1)), geom_quat=np.tile([1., 0., 0., 0.], (3, 1)),
        geom_margin=np.zeros(3), geom_gap=np.zeros(3),
        geom_type=np.full(3, 2), geom_contype=np.ones(3), geom_conaffinity=np.ones(3), geom_priority=np.zeros(3),
        geom_size=np.tile([.04, 0., 0.], (3, 1)), geom_pos=np.tile([0., 0., .04], (3, 1)),
        geom_friction=np.tile([1., .005, .0001], (3, 1)), geom_solref=np.tile([.02, 1.], (3, 1)),
        geom_solimp=np.tile([.9, .95, .001, .5, 2.], (3, 1)))
    sim.provenance = lambda: {'dynamics_version': 'v2'}
    monkeypatch.setattr('nav.fsm._pol.verified_entry', lambda key, lock, allow_rejected: lock[key])
    entry = {'supported': True, 'policy_key': 'kick', 'policy_md5': 'abc'}
    contract = {'protocol': {'dynamics': 'training', 'dynamics_version': 'v2',
                             'robot_morphology': 'robot_walk_backlash.xml',
                             'stand_policy': 'velocity_hard', 'stand_policy_md5': 'def',
                             'ball_model': 'apartment_sphere_40mm_25g_origin_floor_v1'},
                'kick': {'left': dict(entry), 'right': dict(entry)}, 'pick': {'supported': False}}
    fsm = NavFSM(sim, None, None, Follower(), Loc(), contracts=contract)
    assert fsm._skill_contract_error('kick') is None
    contract['protocol']['dynamics_version'] = 'old'
    assert fsm._skill_contract_error('kick') == 'skill_dynamics_version_mismatch'
    contract['protocol']['dynamics_version'] = 'v2'
    contract['protocol']['robot_morphology'] = 'different.xml'
    assert fsm._skill_contract_error('kick') == 'skill_morphology_mismatch'
    contract['protocol']['robot_morphology'] = 'robot_walk_backlash.xml'
    sim.model.geom_size[0, 0] = .035
    assert fsm._skill_contract_error('kick') == 'skill_ball_model_mismatch'
    sim.model.geom_size[0, 0] = .04
    contract['kick']['right']['policy_md5'] = 'old'
    assert fsm._skill_contract_error('kick') == 'skill_policy_mismatch'
    assert fsm._skill_contract_error('pick') == 'skill_unsupported_pick'
    contract['kick']['right']['policy_md5'] = 'abc'
    def rejected(*args, **kwargs):
        from nav.policies import PolicyLockError
        raise PolicyLockError('rejected')
    monkeypatch.setattr('nav.fsm._pol.verified_entry', rejected)
    assert fsm._skill_contract_error('kick') == 'skill_policy_invalid'


def test_unvalidated_recovery_fails_before_policy_switch():
    sim = Sim(); sim.dynamics = 'training'; sim.provenance = lambda: {'dynamics_version': 'v2'}
    loc = Loc(); loc.last_sensor_frame.imu_quat = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])
    contract = {'protocol': {'dynamics': 'training', 'dynamics_version': 'v2',
                             'robot_morphology': 'robot_walk_backlash.xml'},
                'recover': {'supported': False}}
    fsm = NavFSM(sim, None, None, Follower(), loc, contracts=contract)
    fsm.start((1, 0, 0)); fsm.tick(SimpleNamespace())
    assert fsm.reason == 'recovery_unsupported' and not hasattr(sim, 'key')


def test_sensor_safety_independent_of_ground_truth():
    from nav.fsm import sensor_safety
    l = Loc()
    frame = l.reader.read(None)
    tilt, height = sensor_safety(frame, l.kin)
    assert tilt == 0 and abs(height - .1) < 1e-12
    frame.foot_contacts = (False, False)
    assert sensor_safety(frame, l.kin)[1] is None
    frame.imu_quat = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])
    assert abs(sensor_safety(frame, l.kin)[0] - np.pi/2) < 1e-12


def test_planner_result_array_and_restore_initial_policy(monkeypatch):
    from nav.planner import PlanResult
    result = PlanResult(poses=np.array([[0., 0., 0.], [1., 0., 0.]]), primitive_ids=[],
                        cost=1., eps_achieved=1., complete=True, plan_time_s=.01)
    monkeypatch.setattr('nav.fsm.plan', lambda *a, **k: result)
    sim = Sim(); loc = Loc(); fsm = NavFSM(sim, None, None, Follower(), loc, contracts={})
    fsm.start((1, 0, 0)); fsm.tick(SimpleNamespace())
    assert fsm.state == 'follow'
    fsm.state = 'skill'; fsm._skill_t = 0; fsm._duration = 1
    sim.skill_done = True
    fsm.tick(SimpleNamespace())
    assert sim.key == 'velocity_hard' and fsm.state == 'settle'


def test_partial_plan_followed_then_replanned(monkeypatch):
    """D3: a timeout_partial plan with heuristic progress is followed; at its end the FSM replans."""
    from nav.planner import PlanResult
    calls = []
    def fake_plan(*a, **k):
        calls.append(k)
        complete = len(calls) > 1
        return PlanResult(poses=np.array([[0., 0., 0.], [.1, 0., 0.]]), primitive_ids=[], cost=1.,
                          eps_achieved=1., complete=complete, plan_time_s=.01,
                          reason="" if complete else "timeout_partial", end_state_pose=(.1, 0., 0.))
    monkeypatch.setattr('nav.fsm.plan', fake_plan)
    monkeypatch.setattr(NavFSM, '_partial_progress', lambda self, pose, r: 1.0)
    sim = Sim(); loc = Loc(); fol = Follower()
    fsm = NavFSM(sim, None, None, fol, loc, contracts={})
    fsm.start((1, 0, 0)); fsm.tick(SimpleNamespace())
    assert fsm.state == 'follow' and fsm.plan_log[-1]['reason'] == 'timeout_partial'
    assert calls[0]['max_expansions'] == PLAN_MAX_EXPANSIONS
    fol.compute = lambda *a: (np.zeros(3), 'done')
    fsm._tick = 0; fsm.tick(SimpleNamespace())
    assert fsm.state == 'plan'                      # partial segment ended -> replan, not done
    fsm.tick(SimpleNamespace())
    assert fsm.state == 'follow' and fsm.plan_log[-1]['complete']
    fsm._tick = 0; fsm.tick(SimpleNamespace())
    assert fsm.state == 'done'


def test_partial_plan_without_progress_fails(monkeypatch):
    from nav.planner import PlanResult
    r = PlanResult(poses=np.array([[0., 0., 0.], [.1, 0., 0.]]), primitive_ids=[], cost=1., eps_achieved=1.,
                   complete=False, plan_time_s=.01, reason="timeout_partial", end_state_pose=(.1, 0., 0.))
    monkeypatch.setattr('nav.fsm.plan', lambda *a, **k: r)
    monkeypatch.setattr(NavFSM, '_partial_progress', lambda self, pose, res: 0.0)
    fsm = NavFSM(Sim(), None, None, Follower(), Loc(), contracts={})
    fsm.start((1, 0, 0)); fsm.tick(SimpleNamespace())
    assert fsm.state == 'fail' and fsm.reason == 'plan_incomplete'


def test_kick_precision_respects_contract_and_restores_navigation_defaults():
    follower = Follower()
    follower.tol_pos, follower.tol_yaw = .05, np.deg2rad(10)
    follower.pos_margin, follower.yaw_margin = .002, np.deg2rad(2)
    follower.model = SimpleNamespace(stop_spread=.004)
    contract = {'kick': {'right': {'supported': True, 'target_trunk': [.095, -.037],
                                  'tol_half': [.025, .025]}}}
    fsm = NavFSM(Sim(), None, None, follower, Loc(), contracts=contract)
    assert fsm._configure_kick_precision('right')
    radius = np.linalg.norm(contract['kick']['right']['target_trunk'])
    assert follower.tol_pos + radius * follower.tol_yaw <= .025 + 1e-12
    assert follower.tol_pos > follower.pos_margin
    assert follower.tol_yaw > follower.yaw_margin
    assert (follower.tol_pos - follower.pos_margin) / np.sqrt(2) > follower.model.stop_spread
    fsm.start((1., 0., 0.), task='nav')
    assert follower.tol_pos == .05 and follower.tol_yaw == np.deg2rad(10)


def test_kick_precision_refuses_unsupported_degenerate_or_unreachable_limits():
    follower = Follower()
    follower.tol_pos, follower.tol_yaw = .05, np.deg2rad(10)
    follower.pos_margin, follower.yaw_margin = .002, np.deg2rad(2)
    follower.model = SimpleNamespace(stop_spread=.004)
    entry = {'supported': True, 'target_trunk': [.095, -.037], 'tol_half': [.025, .025]}
    fsm = NavFSM(Sim(), None, None, follower, Loc(), contracts={'kick': {'right': entry}})
    for change in ({'supported': False}, {'tol_half': [0., .025]}, {'tol_half': [.005, .005]}):
        original = dict(entry)
        entry.update(change)
        assert not fsm._configure_kick_precision('right')
        assert follower.tol_pos == .05 and follower.tol_yaw == np.deg2rad(10)
        entry.clear(); entry.update(original)
    follower.model.stop_spread = .02
    assert not fsm._configure_kick_precision('right')


def test_kick_stage_preserves_navigation_tolerance_and_refuses_infeasible_contract(monkeypatch):
    from nav.planner import PlanResult
    result = PlanResult(poses=np.array([[0., 0., 0.], [1., 0., 0.]]), primitive_ids=[],
                        cost=1., eps_achieved=1., complete=True, plan_time_s=.01)
    monkeypatch.setattr('nav.fsm.plan', lambda *args, **kwargs: result)
    monkeypatch.setattr(NavFSM, '_skill_contract_error', lambda *args, **kwargs: None)
    follower = Follower()
    follower.tol_pos, follower.tol_yaw = .05, np.deg2rad(10)
    follower.pos_margin, follower.yaw_margin = .002, np.deg2rad(2)
    follower.model = SimpleNamespace(stop_spread=.004)
    side = {'supported': True, 'target_trunk': [.095, -.037], 'tol_half': [.025, .025],
            'direction_bias_rad': 0.}
    contracts = {'kick': {'right': side, 'left': dict(side)}}
    fsm = NavFSM(Sim(), None, None, follower, Loc(), contracts=contracts)
    fsm.start(None, task='kick', target_xy=(1., -.1))
    fsm.tick(SimpleNamespace())
    assert fsm.state == 'follow' and follower.tol_pos == .025
    fsm.start(None, task='kick', target_xy=(1., -.1))
    assert follower.tol_pos == .05
    side['tol_half'] = [0., .025]
    fsm.tick(SimpleNamespace())
    assert fsm.state == 'fail' and fsm.reason == 'skill_precision_unachievable'
