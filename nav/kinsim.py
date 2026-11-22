"""Kinematic stand-in for the MuJoCo sim: omni integration at 50 Hz with per-axis
first-order lag + pure delay + gain + Gaussian lateral drift. API mirrors Sim enough
for follower_eval (step(cmd), pose)."""
from __future__ import annotations
import math
import numpy as np

DEFAULT_FOL = {"vx": {"tau": 0.3, "delay": 0.1, "gain": 0.9},
               "vy": {"tau": 0.3, "delay": 0.1, "gain": 0.9},
               "wz": {"tau": 0.3, "delay": 0.1, "gain": 0.9}}


class KinSim:
    dt = 0.02

    def __init__(self, fol=None, lateral_sigma=0.01, seed=0, blocked=None):
        fol = fol or DEFAULT_FOL
        self.tau = np.array([fol[a]["tau"] for a in ("vx", "vy", "wz")])
        self.delay = np.array([fol[a]["delay"] for a in ("vx", "vy", "wz")])
        self.gain = np.array([fol[a]["gain"] for a in ("vx", "vy", "wz")])
        self.sigma = lateral_sigma
        self.blocked = blocked      # fn(x, y) -> bool; motion into blocked cells is rejected
        self.rng = np.random.default_rng(seed)
        self.reset((0, 0, 0))

    def reset(self, pose):
        self.pose = np.array(pose, float)
        self.v = np.zeros(3)
        nd = int(round(self.delay.max() / self.dt)) + 2
        self.buf = [np.zeros(3) for _ in range(nd)]
        self.t = 0.0
        return self.pose.copy()

    def step(self, cmd):
        self.buf.append(np.asarray(cmd, float))
        self.buf = self.buf[-(int(round(self.delay.max() / self.dt)) + 2):]
        eff = np.array([self.buf[-1 - int(round(self.delay[a] / self.dt))][a] for a in range(3)])
        self.v += self.dt / self.tau * (self.gain * eff - self.v)
        vb = self.v.copy()
        vb[1] += self.rng.normal(0, self.sigma)       # lateral drift (body frame)
        x, y, th = self.pose
        c, s = math.cos(th), math.sin(th)
        nx = x + (c * vb[0] - s * vb[1]) * self.dt
        ny = y + (s * vb[0] + c * vb[1]) * self.dt
        if self.blocked is not None and self.blocked(nx, ny):
            self.v[:2] = 0.0
        else:
            self.pose[:2] = (nx, ny)
        self.pose[2] = th + vb[2] * self.dt
        self.t += self.dt
        return self.pose.copy()

    @property
    def vel_body(self):
        return self.v.copy()
