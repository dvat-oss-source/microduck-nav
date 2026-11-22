"""Generic course runner: same gate for KinSim now, nav.sim.Sim later.

    run_course(course, step_fn, pose_fn, follower, hold=5, vel_fn=None)
step_fn(cmd3) advances ONE 50 Hz policy step. pose_fn() -> (x, y, theta) (ground truth).
vel_fn() -> optional body velocity (passed to the follower; None = follower uses its model).

Sim adapter:  step_fn, pose_fn = sim_backend(sim)   (sim must have been reset first)
"""
from __future__ import annotations
import math
import numpy as np


def cross_track(poses, p):
    """Distance from p (x,y) to the polyline of `poses`."""
    A, B = poses[:-1, :2], poses[1:, :2]
    AB = B - A
    t = np.clip(((p[:2] - A) * AB).sum(1) / np.maximum((AB ** 2).sum(1), 1e-12), 0, 1)
    return float(np.hypot(*(A + t[:, None] * AB - p[:2]).T).min())


def sim_backend(sim, initial_gt=None):
    state = {"gt": initial_gt}
    def step_fn(cmd):
        gt, _obs = sim.step(np.asarray(cmd, float)); state["gt"] = gt
    def pose_fn():
        return tuple(state["gt"].pose2)
    return step_fn, pose_fn


def run_course(course, step_fn, pose_fn, follower, hold=5, vel_fn=None, settle_steps=50):
    follower.set_path(course.poses)
    cmds, errs = [], []
    traj = []
    t = 0.0
    status = "tracking"
    while t < course.max_time:
        pose = pose_fn()
        cmd, status = follower.compute(pose, vel_fn() if vel_fn else None)
        cmds.append(np.array(cmd))
        errs.append(cross_track(course.poses, np.array(pose)))
        traj.append(tuple(pose))
        if status in ("done", "stuck"):
            break
        for _ in range(hold):
            step_fn(cmd)
        t += hold * 0.02
    for _ in range(settle_steps):    # let the lag settle with zero command
        step_fn(np.zeros(3))
    pose = np.array(pose_fn())
    g = course.poses[-1]
    dpos = float(math.hypot(*(pose[:2] - g[:2])))
    dyaw = float(abs((pose[2] - g[2] + math.pi) % (2 * math.pi) - math.pi))
    errs = np.array(errs)
    return {"course": course.name, "status": status, "time_to_goal": t,
            "cross_track_rms": float(np.sqrt(np.mean(errs ** 2))), "cross_track_max": float(errs.max()),
            "final_pos_err": dpos, "final_yaw_err_deg": math.degrees(dyaw),
            "reached": bool(status == "done" and dpos <= 0.05 and dyaw <= math.radians(10)),
            "cmds": np.array(cmds), "traj": np.array(traj)}
