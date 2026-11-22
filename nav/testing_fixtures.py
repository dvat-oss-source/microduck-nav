"""Synthetic stand-ins for Phase C (GridMap) and Phase B (primitives JSON) used by planner tests.

Nothing here touches MuJoCo. Both objects follow nav/CONTRACTS.md so the planner
code paths that consume them are identical to the ones that will consume
``nav/data/map_*.npz`` and ``nav/data/primitives_*.json``.

SyntheticGridMap convention: occ[i, j] with i = row along +y, j = column along +x,
cell_to_world returns the cell *center*.  (The planner does not rely on this; it
probes cell_to_world/world_to_cell to discover the convention of whatever GridMap
it is given.)
"""
from __future__ import annotations

import math
import json

import numpy as np
from scipy import ndimage

N_HEADINGS = 16


class SyntheticGridMap:
    """GridMap-compatible map: res, origin, occ, cliff, lethal, cost, clearance, rooms."""

    def __init__(self, occ, cliff, res=0.02, origin=(0.0, 0.0), robot_radius=0.10,
                 cost_decay_m=0.12, rooms=None):
        self.res = float(res)
        self.origin = (float(origin[0]), float(origin[1]))
        self.occ = np.asarray(occ, bool)
        self.cliff = np.asarray(cliff, bool)
        self.robot_radius = float(robot_radius)
        hazard = self.occ | self.cliff
        # clearance: distance (m) from cell center to nearest hazard cell
        self.clearance = (ndimage.distance_transform_edt(~hazard) * self.res).astype(np.float32)
        self.lethal = self.clearance <= self.robot_radius
        cost = np.exp(-(self.clearance - self.robot_radius) / cost_decay_m).astype(np.float32)
        cost = np.clip(cost, 0.0, 1.0)
        cost[self.lethal] = np.inf
        self.cost = cost
        self.rooms = dict(rooms or {})

    @property
    def shape(self):
        return self.occ.shape

    def world_to_cell(self, x, y):
        return (int(math.floor((y - self.origin[1]) / self.res)),
                int(math.floor((x - self.origin[0]) / self.res)))

    def cell_to_world(self, i, j):
        return (self.origin[0] + (j + 0.5) * self.res, self.origin[1] + (i + 0.5) * self.res)

    def in_bounds(self, i, j):
        return 0 <= i < self.occ.shape[0] and 0 <= j < self.occ.shape[1]

    def is_lethal(self, x, y):
        i, j = self.world_to_cell(x, y)
        return (not self.in_bounds(i, j)) or bool(self.lethal[i, j])

    def cost_at(self, x, y):
        i, j = self.world_to_cell(x, y)
        if not self.in_bounds(i, j):
            return float("inf")
        return float(self.cost[i, j])

    def save(self, path):
        np.savez_compressed(path, res=self.res, origin=np.array(self.origin), occ=self.occ,
                            cliff=self.cliff, robot_radius=self.robot_radius,
                            rooms=json.dumps(self.rooms))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(d["occ"], d["cliff"], float(d["res"]), tuple(d["origin"]),
                   float(d["robot_radius"]), rooms=json.loads(str(d["rooms"])))


def make_house_map(res=0.02, robot_radius=0.10):
    """~7 x 6 m, three rooms, two doorways (0.6 m), a stair-hole cliff, a box obstacle.

    Room A: x 0..3.5 (whole height), contains the cliff (stair hole) x 1.0..2.0, y 4.0..5.0
    Room B: x 3.5..7, y 0..3, contains a box x 4.5..5.0, y 1.0..1.6
    Room C: x 3.5..7, y 3..6
    Doorway A<->B: wall x=3.5, y 1.2..1.8;  doorway B<->C: wall y=3.0, x 5.0..5.6
    """
    W, H = int(round(7.0 / res)), int(round(6.0 / res))
    occ = np.zeros((H, W), bool)
    cliff = np.zeros((H, W), bool)
    xs = (np.arange(W) + 0.5) * res
    ys = (np.arange(H) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    t = 0.05  # half wall thickness
    occ |= (X < 0.1) | (X > 6.9) | (Y < 0.1) | (Y > 5.9)
    occ |= (np.abs(X - 3.5) < t) & ~((Y > 1.2) & (Y < 1.8))
    occ |= (np.abs(Y - 3.0) < t) & (X > 3.5) & ~((X > 5.0) & (X < 5.6))
    occ |= (X > 4.5) & (X < 5.0) & (Y > 1.0) & (Y < 1.6)
    cliff |= (X > 1.0) & (X < 2.0) & (Y > 4.0) & (Y < 5.0)
    rooms = {"A": [0.1, 0.1, 3.45, 5.9], "B": [3.55, 0.1, 6.9, 2.95], "C": [3.55, 3.05, 6.9, 5.9]}
    return SyntheticGridMap(occ, cliff, res=res, origin=(0.0, 0.0), robot_radius=robot_radius,
                            rooms=rooms)


def _integrate(cmd_sequence, theta0, dt=0.02):
    """Unicycle (+ lateral) integration of [[vx, vy, wz, dur], ...] from (0,0,theta0)."""
    x = y = 0.0
    th = theta0
    pts = []
    t_acc = 0.0
    next_sample = 0.1
    for vx, vy, wz, dur in cmd_sequence:
        n = max(1, int(round(dur / dt)))
        h = dur / n
        for _ in range(n):
            # midpoint integration
            thm = th + 0.5 * wz * h
            x += (vx * math.cos(thm) - vy * math.sin(thm)) * h
            y += (vx * math.sin(thm) + vy * math.cos(thm)) * h
            th += wz * h
            t_acc += h
            if t_acc >= next_sample - 1e-9:
                pts.append([x, y])
                next_sample += 0.1
    return x, y, th - theta0, pts


def make_synthetic_primitives(n_headings=N_HEADINGS, sigma_pos=0.008, sigma_pos_per_m=0.03,
                              sigma_th=0.02):
    """Primitives JSON dict (contract schema), per heading:
    straight 1 s @ 0.2 m/s; gentle arcs L/R (0.2 m/s, 22.5 deg/s, 1 s); tight arcs L/R
    (0.15 m/s, 45 deg/s, 1 s); turn-in-place +-22.5 deg (0.5 s) and +-45 deg (1 s);
    backward step 1 s @ -0.08 m/s.  Small covariances.
    """
    bin_w = 2 * math.pi / n_headings
    base = [
        ("straight", [[0.2, 0.0, 0.0, 1.0]]),
        ("arc_gentle_L", [[0.2, 0.0, bin_w, 1.0]]),
        ("arc_gentle_R", [[0.2, 0.0, -bin_w, 1.0]]),
        ("arc_tight_L", [[0.15, 0.0, 2 * bin_w, 1.0]]),
        ("arc_tight_R", [[0.15, 0.0, -2 * bin_w, 1.0]]),
        ("turn_L22", [[0.0, 0.0, 2 * bin_w, 0.5]]),
        ("turn_R22", [[0.0, 0.0, -2 * bin_w, 0.5]]),
        ("turn_L45", [[0.0, 0.0, 2 * bin_w, 1.0]]),
        ("turn_R45", [[0.0, 0.0, -2 * bin_w, 1.0]]),
        ("back", [[-0.08, 0.0, 0.0, 1.0]]),
    ]
    prims = []
    for k in range(n_headings):
        th0 = k * bin_w
        for name, seq in base:
            dx, dy, dth, pts = _integrate(seq, th0)
            dist = math.hypot(dx, dy)
            sp_ = sigma_pos + sigma_pos_per_m * dist
            st = sigma_th + 0.02 * abs(dth)
            cov = np.diag([sp_ ** 2, sp_ ** 2, st ** 2])
            cov[0, 1] = cov[1, 0] = 0.2 * sp_ ** 2
            prims.append({
                "id": f"h{k:02d}_{name}",
                "start_heading_idx": k,
                "cmd_sequence": seq,
                "end_mean": [dx, dy, dth],
                "end_cov": cov.tolist(),
                "duration": float(sum(s[3] for s in seq)),
                "fall_rate": 0.0,
                "swept": pts,
            })
    return {"policy_md5": "synthetic", "n_headings": n_headings, "primitives": prims}


def sample_free_poses(gridmap, n, rng, min_margin=0.06):
    """Random poses whose cell is >= min_margin (m) away from any lethal cell.

    Uses only ``gridmap.lethal`` / ``res`` / ``cell_to_world`` so it works on real Phase C maps.
    """
    lethal = np.asarray(gridmap.lethal, bool)
    margin = ndimage.distance_transform_edt(~lethal) * float(gridmap.res)
    ii, jj = np.nonzero(margin >= min_margin)
    out = []
    for _ in range(n):
        k = rng.integers(len(ii))
        x, y = gridmap.cell_to_world(int(ii[k]), int(jj[k]))
        out.append((float(x), float(y), float(rng.uniform(-math.pi, math.pi))))
    return out
