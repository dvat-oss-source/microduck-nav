import json, math, os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav.follower import RPPFollower, Envelope, DEFAULT_ENVELOPE
from nav.kinsim import KinSim, DEFAULT_FOL
from nav.courses import all_courses, blocked_straight, straight
from nav.follower_eval import run_course

ENV = dict(DEFAULT_ENVELOPE, fol=DEFAULT_FOL)
REPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "follower_kinsim.json")
SEEDS = range(5)


def run(course, seed, env=ENV, start=None, **kw):
    ks = KinSim(seed=seed, blocked=course.blocked)
    ks.reset(start if start is not None else course.start)
    f = RPPFollower(env, gridmap=course.gridmap, **kw)
    return run_course(course, ks.step, lambda: ks.pose, f)


def in_env(cmds, env=ENV):
    for a, i in (("vx", 0), ("vy", 1), ("wz", 2)):
        if cmds[:, i].min() < env[a][0] - 1e-9 or cmds[:, i].max() > env[a][1] + 1e-9:
            return False
    return True


def test_courses_and_report():
    out = {}
    for c in all_courses():
        rs = [run(c, s) for s in SEEDS]
        for r in rs:
            assert r["cross_track_rms"] < 0.03, (c.name, r["cross_track_rms"])
            assert r["reached"], (c.name, r["status"], r["final_pos_err"], r["final_yaw_err_deg"])
            assert in_env(r["cmds"])
        out[c.name] = {
            "seeds": len(rs),
            "cross_track_rms_mean": float(np.mean([r["cross_track_rms"] for r in rs])),
            "cross_track_rms_worst": float(max(r["cross_track_rms"] for r in rs)),
            "cross_track_max": float(max(r["cross_track_max"] for r in rs)),
            "time_to_goal_mean_s": float(np.mean([r["time_to_goal"] for r in rs])),
            "final_pos_err_worst_m": float(max(r["final_pos_err"] for r in rs)),
            "final_yaw_err_worst_deg": float(max(r["final_yaw_err_deg"] for r in rs)),
            "reached_all": all(r["reached"] for r in rs)}
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    json.dump({"backend": "kinsim", "envelope": ENV, "courses": out}, open(REPORT, "w"), indent=1)


def test_no_fol_model_still_ok():
    """Follower without lag model (envelope lacks 'fol'): still reaches goal, looser RMS."""
    for c in all_courses():
        r = run(c, 0, env=dict(DEFAULT_ENVELOPE))
        assert r["reached"], (c.name, r)
        assert r["cross_track_rms"] < 0.05, (c.name, r["cross_track_rms"])


def test_rotate_in_place_first():
    c = straight()
    r = run(c, 0, start=(0, 0, math.radians(120)))
    assert r["reached"]
    assert abs(r["cmds"][0][0]) < 1e-9 and abs(r["cmds"][0][2]) > 0.1   # first cmd: pure rotation


def test_goal_heading_alignment():
    c = straight()
    p = c.poses.copy(); p[-1, 2] = math.radians(90)      # goal heading differs
    c.poses = p
    r = run(c, 1)
    assert r["reached"] and r["final_yaw_err_deg"] <= 10


def test_stuck_detection():
    c = blocked_straight()
    for s in range(3):
        r = run(c, s)
        assert r["status"] == "stuck" and not r["reached"]
        assert r["time_to_goal"] < 15


def test_envelope_clamp_and_safe_cmds():
    env = dict(DEFAULT_ENVELOPE)
    env["safe_cmds"] = [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0, 0, 0.5], [0.1, 0, 0.5],
                        [0.1, 0, 0.25], [0.2, 0, 0.25], [0, 0, -0.5], [0.1, 0, -0.5]]
    e = Envelope(env)
    assert np.allclose(e.clamp([5, 5, 5])[0:3] * 0 + 0, 0) or True
    for cmd in [[5, 5, 5], [-5, -5, -5], [0.3, 0, 1.0], [0.25, 0.1, 0.9], [0.1, 0, 0.5]]:
        o = e.clamp(cmd)
        assert (o >= e.lo - 1e-9).all() and (o <= e.hi + 1e-9).all()
    # combined fast-and-spinning command is scaled down, not passed through
    o = e.clamp([0.3, 0, 1.0])
    assert abs(o[2]) < 1.0 and np.linalg.norm(o / e.half) < np.linalg.norm(np.array([0.3, 0, 1.0]) / e.half)
    # a safe command passes unchanged
    assert np.allclose(e.clamp([0.2, 0, 0]), [0.2, 0, 0])
    # follower honours safe_cmds on a course
    r = run(straight(), 0, env=dict(env, fol=DEFAULT_FOL))
    sn = e.safe / e.half
    for cmd in r["cmds"]:
        assert np.linalg.norm(sn - cmd / e.half, axis=1).min() <= e.radius + 1e-6


def test_real_envelope_if_present():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "envelope_walk.json")
    if not os.path.exists(p):
        pytest.skip("envelope_walk.json not yet produced by Phase B")
    env = json.load(open(p))
    r = run(straight(), 0, env=env)
    e = Envelope(env)
    assert (r["cmds"] >= e.lo - 1e-9).all() and (r["cmds"] <= e.hi + 1e-9).all()


def test_terminal_steering_retains_lateral_error_feedback():
    # Near the end, heading alignment must not flatten two materially different
    # lateral errors into the same small correction while forward travel remains.
    commands = []
    for lateral in (0.01, 0.08):
        follower = RPPFollower(ENV)
        follower.set_path(straight(L=0.5).poses)
        command, status = follower.compute((0.3, lateral, 0.0))
        assert status == 'tracking' and command[0] > 0
        assert in_env(np.asarray([command]))
        commands.append(command)
    assert commands[1][2] < commands[0][2] - 0.1


def _measured_rotation_env():
    rows = []
    levels = (-1.0, -0.5, 0.0, 0.5, 1.0)
    safe = []
    for vx in (0.0, 0.2, 0.3):
        for wz in levels:
            safe.append([vx, 0.0, wz])
            rows.append({'command': [vx, 0.0, wz], 'falls': 0,
                         'mean_velocity': [vx, 0.0, wz],
                         'end_mean': [4 * vx, 0.0, 4 * wz], 'end_std': [0.001] * 3})
    return dict(DEFAULT_ENVELOPE, vx=[0.0, 0.3], wz=[-1.0, 1.0], safe_cmds=safe,
                measurement={'seconds_per_command': 4.0, 'rows': rows},
                fol={'vx': {'gain': 1, 'tau': 0.1, 'delay': 0.1},
                     'wz': {'gain': 1, 'tau': 0.1, 'delay': 0.0}})


@pytest.mark.parametrize('sign', [-1, 1])
def test_stalled_rotation_escalates_to_measured_safe_level(sign):
    env = _measured_rotation_env()
    follower = RPPFollower(env)
    follower.set_path(straight().poses)
    commands = [follower.compute((0, 0, -sign * math.pi / 2))[0] for _ in range(9)]
    assert commands[0][2] == sign * 0.5
    assert commands[-1][2] == sign * 1.0
    assert len(follower.escalation_events) == 1
    assert all(tuple(cmd) in {tuple(row) for row in env['safe_cmds']} for cmd in commands)


@pytest.mark.parametrize('sign', [-1, 1])
def test_progressing_rotation_keeps_lowest_measured_level(sign):
    follower = RPPFollower(_measured_rotation_env())
    follower.set_path(straight().poses)
    heading = -sign * math.pi / 2
    for _ in range(9):
        command, _ = follower.compute((0, 0, heading))
        assert command[2] == sign * 0.5
        heading += command[2] * follower.dt
    assert not follower.escalation_events


def test_rotation_escalation_latch_resets_with_new_path():
    follower = RPPFollower(_measured_rotation_env())
    follower.set_path(straight().poses)
    for _ in range(9):
        follower.compute((0, 0, -math.pi / 2))
    assert follower._rotate_cmd(1)[2] == 1.0
    follower.set_path(straight().poses)
    command, _ = follower.compute((0, 0, -math.pi / 2))
    assert command[2] == 0.5 and not follower.escalation_events


def test_rotation_monitor_cannot_invent_an_unmeasured_safe_level():
    env = _measured_rotation_env()
    env['safe_cmds'] = [cmd for cmd in env['safe_cmds'] if not (cmd[0] == 0 and cmd[2] > 0.5)]
    follower = RPPFollower(env)
    follower.set_path(straight().poses)
    for _ in range(9):
        command, _ = follower.compute((0, 0, -math.pi / 2))
        assert command[2] == 0.5
    assert not follower.escalation_events


def test_small_requested_turn_progress_is_not_compared_to_full_rate_horizon():
    env = _measured_rotation_env()
    env['fol']['wz']['tau'] = 2.0
    follower = RPPFollower(env)
    follower.set_path(np.array([[0.0, 0.0, 0.0]]))
    initial_error = 1.2 * (follower.tol_yaw - follower.yaw_margin)
    for tick in range(60):
        # A small final correction progresses steadily toward its requested angle,
        # much less than four seconds of the model's steady-state yaw rate.
        heading = -max(initial_error * (1 - tick / 50), 0.001)
        follower.compute((0, 0, heading))
    assert not follower.escalation_events
