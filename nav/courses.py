"""Test courses (dense poses at 2 cm) reusable with kinsim and the real Sim."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

SPACING = 0.02


def densify(pts, spacing=SPACING):
    """Resample a polyline (N,2) at uniform arc spacing; theta = tangent."""
    pts = np.asarray(pts, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = int(round(s[-1] / spacing))
    si = np.linspace(0, s[-1], n + 1)
    x, y = np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])
    th = np.arctan2(np.gradient(y), np.gradient(x))
    return np.stack([x, y, th], 1)


class WallClearance:
    """Minimal GridMap-like object (world_to_cell + clearance) for analytic courses."""
    def __init__(self, dist_fn, extent=(-1, -2, 6, 2), res=0.02):
        self.res, self.x0, self.y0 = res, extent[0], extent[1]
        xs = np.arange(extent[0], extent[2], res); ys = np.arange(extent[1], extent[3], res)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self.clearance = dist_fn(X, Y).astype(np.float32)

    def world_to_cell(self, x, y):
        return (int(np.clip((x - self.x0) / self.res, 0, self.clearance.shape[0] - 1)),
                int(np.clip((y - self.y0) / self.res, 0, self.clearance.shape[1] - 1)))


@dataclass
class Course:
    name: str
    poses: np.ndarray
    blocked: object = None            # fn(x, y)->bool for kinsim (obstacle)
    gridmap: object = None            # optional clearance provider
    expect_goal: bool = True
    max_time: float = 40.0

    @property
    def start(self): return tuple(self.poses[0])
    @property
    def goal(self): return tuple(self.poses[-1])


def straight(L=2.0):
    return Course("straight", densify([[0, 0], [L, 0]]), max_time=25)


def s_bend(L=3.0, A=0.25):
    x = np.linspace(0, L, 600)
    return Course("s_bend", densify(np.stack([x, A * np.sin(2 * math.pi * x / L)], 1)), max_time=35)


def doorway(gap=0.6, wall_x=1.5, L=3.0):
    def dist(X, Y):
        dx = np.abs(X - wall_x) - 0.05
        # Outside the opening the nearest wall point is straight across in x;
        # inside it the nearest point is a jamb (|Y| = gap/2).
        dy = gap / 2 - np.abs(Y)
        wall = np.where(np.abs(Y) > gap / 2, np.maximum(dx, 0), np.hypot(np.maximum(dx, 0), dy))
        return np.minimum(wall, 1.0)
    gm = WallClearance(dist, extent=(-0.5, -1.5, L + 0.5, 1.5))
    return Course("doorway", densify([[0, 0], [L, 0]]), gridmap=gm, max_time=35)


def u_turn(r=0.5, L=1.5):
    a = np.linspace(-math.pi / 2, math.pi / 2, 200)
    arc = np.stack([L + r * np.cos(a), r + r * np.sin(a)], 1)
    pts = np.vstack([[[0, 0]], arc, [[0, 2 * r]]])
    return Course("u_turn", densify(pts), max_time=45)


def blocked_straight(L=2.0, wall_x=1.0):
    return Course("blocked", densify([[0, 0], [L, 0]]), blocked=lambda x, y: x > wall_x,
                  expect_goal=False, max_time=30)


def all_courses():
    return [straight(), s_bend(), doorway(), u_turn()]
