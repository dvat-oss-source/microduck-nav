"""Heading-aware cliff checks for the isolated v5/A6 geometry candidate.

The calibrated polygon is the convex hull of collision-geom AABB corners
recorded in trunk coordinates during flat practice runs.  It includes the
collision body and feet across the sampled gait phases.  Cliff cells are tested
as closed world-space squares, so this retains the map's conservative
cell-square allowance without subtracting an unrelated circular robot radius.

No live nav module imports this file.  The coordinator's candidate mapping may
attach ``Footprint`` as ``gridmap.footprint`` after loading the versioned
calibration artifact.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def convex_hull(points: Iterable[Iterable[float]]) -> np.ndarray:
    """Return the CCW monotone-chain hull, dropping collinear interior points."""
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) < 3:
        raise ValueError("footprint needs at least three distinct points")

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    out = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(out) < 3 or not np.isfinite(out).all():
        raise ValueError("invalid footprint hull")
    return out


def _wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _minkowski_segment(poly: np.ndarray, axis: np.ndarray, radius: float) -> np.ndarray:
    """Convex hull of poly + [-radius,+radius] * axis."""
    if radius <= 0:
        return poly
    delta = np.asarray(axis, dtype=float) * float(radius)
    return convex_hull(np.vstack((poly + delta, poly - delta)))


def _minkowski_square(poly: np.ndarray, radius: float) -> np.ndarray:
    """Conservative axis-aligned square pad used for angular-arc sagitta."""
    if radius <= 0:
        return poly
    corners = np.asarray([[-radius, -radius], [-radius, radius],
                          [radius, -radius], [radius, radius]], dtype=float)
    return convex_hull((p + c for p in poly for c in corners))


def _sat_poly_square(poly: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> bool:
    """Closed-set convex polygon / axis-aligned square intersection."""
    square = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    axes = [(1.0, 0.0), (0.0, 1.0)]
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        edge = b - a
        if float(edge @ edge) > 1e-24:
            axes.append((-float(edge[1]), float(edge[0])))
    sq = np.asarray(square, dtype=float)
    for axis in axes:
        n = np.asarray(axis, dtype=float)
        pproj = poly @ n
        sproj = sq @ n
        if float(pproj.max()) < float(sproj.min()) or float(sproj.max()) < float(pproj.min()):
            return False
    return True


class Footprint:
    """Exact polygon-to-cliff-cell-square checker.

    ``pose`` and each row of ``poses`` use trunk-frame ``(x, y, theta)``.
    ``heading_delta`` is the travel tangent relative to trunk yaw.  The full
    p99.5 cross-track excursion is applied as a two-sided segment along the
    tangent's local normal; the value is never divided, clipped, or capped.
    """

    def __init__(self, gridmap, calibration):
        if isinstance(calibration, (str, Path)):
            path = Path(calibration)
            self.calibration_path = str(path)
            self.calibration = json.loads(path.read_text())
            self._verify_calibration(path, self.calibration)
        else:
            self.calibration_path = None
            self.calibration = dict(calibration)
        self.gridmap = gridmap
        self.hull = np.asarray(self.calibration["body_frame_hull_m"], dtype=float)
        if self.hull.ndim != 2 or self.hull.shape[1] != 2 or len(self.hull) < 3:
            raise ValueError("calibration body_frame_hull_m must be Nx2")
        self.hull = convex_hull(self.hull)
        self.cross_track_q_m = float(self.calibration["cross_track_excursion_p995_m"])
        self.turn_swept_radius_m = float(self.calibration["turn_swept_radius_m"])
        if (not np.isfinite([self.cross_track_q_m, self.turn_swept_radius_m]).all()
                or self.cross_track_q_m <= 0 or self.turn_swept_radius_m <= 0):
            raise ValueError("invalid measured footprint margins")
        self.res = float(gridmap.res)
        self.origin = np.asarray(gridmap.origin, dtype=float)
        self.cliff = np.asarray(gridmap.cliff, dtype=bool)
        if self.cliff.ndim != 2 or self.res <= 0:
            raise ValueError("gridmap needs a 2-D cliff mask and positive res")
        if hasattr(gridmap, "cliff_clearance"):
            self.cliff_clearance = np.asarray(gridmap.cliff_clearance, dtype=float)
        else:
            from scipy.ndimage import distance_transform_edt
            self.cliff_clearance = (np.maximum(0.0,
                distance_transform_edt(~self.cliff) * self.res - math.sqrt(2.0) * self.res)
                if np.any(self.cliff) else np.full(self.cliff.shape, np.inf))

    @staticmethod
    def _verify_calibration(path, calibration):
        """Fail closed on stale/unproven measurement files."""
        if calibration.get("schema") != "microduck-a6-footprint-v1" or calibration.get("status") != "ready":
            raise ValueError("footprint calibration is not a ready v1 measurement")
        if calibration.get("cross_track_quantile") != 0.995:
            raise ValueError("footprint calibration must retain the full p99.5 quantile")
        if calibration.get("practice_policy") != "atlantis_walk":
            raise ValueError("footprint calibration policy mismatch")
        provenance = calibration.get("provenance", {})
        files = provenance.get("sha256", {})
        if not files:
            raise ValueError("footprint calibration has no input hashes")
        root = path.resolve().parents[5]
        for relative, expected in files.items():
            artifact = (root / relative).resolve()
            if (not artifact.is_relative_to(root) or not artifact.is_file()
                    or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected):
                raise ValueError(f"footprint calibration provenance mismatch: {relative}")

    def _pose_polygon(self, pose, heading_delta=0.0):
        x, y, theta = map(float, pose)
        a = np.asarray([[math.cos(theta), -math.sin(theta)],
                        [math.sin(theta), math.cos(theta)]])
        poly = self.hull @ a.T + np.asarray([x, y])
        normal_theta = theta + float(heading_delta) + math.pi / 2.0
        normal = np.asarray([math.cos(normal_theta), math.sin(normal_theta)])
        return _minkowski_segment(poly, normal, self.cross_track_q_m)

    def _polygon_safe(self, poly):
        # Cheap broad phase: the old conservative cliff-clearance field is used
        # only to prove a whole polygon is far from every cliff. Near a cliff,
        # the exact square intersection below decides safety.
        center = (poly.min(axis=0) + poly.max(axis=0)) * 0.5
        circ = float(np.linalg.norm(poly - center, axis=1).max())
        ox0, oy0 = map(float, self.origin[:2])
        j = int(math.floor((center[0] - ox0) / self.res))
        i = int(math.floor((center[1] - oy0) / self.res))
        if 0 <= i < self.cliff.shape[0] and 0 <= j < self.cliff.shape[1]:
            if self.cliff_clearance[i, j] > circ:
                return True
        # Include cells touching the polygon bounds, including exact boundary
        # contact. The squares are closed, matching raster collision semantics.
        xmin, ymin = poly.min(axis=0)
        xmax, ymax = poly.max(axis=0)
        ox, oy = map(float, self.origin[:2])
        j0 = max(0, int(math.floor((xmin - ox) / self.res)) - 1)
        i0 = max(0, int(math.floor((ymin - oy) / self.res)) - 1)
        j1 = min(self.cliff.shape[1] - 1, int(math.floor((xmax - ox) / self.res)))
        i1 = min(self.cliff.shape[0] - 1, int(math.floor((ymax - oy) / self.res)))
        if j0 > j1 or i0 > i1:
            return False
        for i, j in np.argwhere(self.cliff[i0:i1 + 1, j0:j1 + 1]):
            i, j = int(i + i0), int(j + j0)
            x0, y0 = ox + j * self.res, oy + i * self.res
            if _sat_poly_square(poly, x0, y0, x0 + self.res, y0 + self.res):
                return False
        return True

    def pose_safe(self, pose, heading_delta=0.0) -> bool:
        """Check a single actual pose against all cliff-cell squares."""
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("pose must be finite (x,y,theta)")
        return self._polygon_safe(self._pose_polygon(pose, heading_delta))

    def _turn_disk_safe(self, pose):
        """Preserve the independently measured full in-place turn envelope."""
        x, y = float(pose[0]), float(pose[1])
        radius = self.turn_swept_radius_m + self.cross_track_q_m
        ox, oy = map(float, self.origin[:2])
        j0 = max(0, int(math.floor((x - radius - ox) / self.res)) - 1)
        i0 = max(0, int(math.floor((y - radius - oy) / self.res)) - 1)
        j1 = min(self.cliff.shape[1] - 1, int(math.floor((x + radius - ox) / self.res)))
        i1 = min(self.cliff.shape[0] - 1, int(math.floor((y + radius - oy) / self.res)))
        if j0 > j1 or i0 > i1:
            return False
        for i, j in np.argwhere(self.cliff[i0:i1 + 1, j0:j1 + 1]):
            i, j = int(i + i0), int(j + j0)
            x0, y0 = ox + j * self.res, oy + i * self.res
            dx = max(x0 - x, 0.0, x - (x0 + self.res))
            dy = max(y0 - y, 0.0, y - (y0 + self.res))
            if dx * dx + dy * dy <= radius * radius:
                return False
        return True

    def swept_safe(self, poses, heading_deltas=None) -> bool:
        """Check a continuous polygon sweep represented by an Nx3 pose path.

        Translation and shortest-angle yaw are interpolated. Each adjacent
        pair is checked as the convex hull of its endpoint polygons, with a
        square pad at least as large as the maximum rotational-arc sagitta.
        Sampling bounds translation to res/2 and sagitta to res/4. Any segment
        with yaw change also checks the existing full measured turn disk at
        every interpolated center, preserving the prior turn envelope.
        """
        path = np.asarray(poses, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 1 or not np.isfinite(path).all():
            raise ValueError("poses must be a finite Nx3 array")
        if heading_deltas is None:
            deltas = np.zeros(len(path), dtype=float)
            for k in range(len(path) - 1):
                dxy = path[k + 1, :2] - path[k, :2]
                if np.linalg.norm(dxy) > 1e-12:
                    tangent = math.atan2(float(dxy[1]), float(dxy[0]))
                    deltas[k] = _wrap(tangent - float(path[k, 2]))
            if len(path) > 1:
                deltas[-1] = deltas[-2]
        else:
            deltas = np.asarray(heading_deltas, dtype=float)
            if deltas.shape != (len(path),) or not np.isfinite(deltas).all():
                raise ValueError("heading_deltas must contain one finite angle per pose")
        if len(path) == 1:
            return self.pose_safe(path[0], deltas[0])

        max_radius = float(np.max(np.linalg.norm(self.hull, axis=1))) + self.cross_track_q_m
        sagitta_limit = self.res / 4.0
        max_yaw_step = (math.pi if max_radius <= sagitta_limit else
                        2.0 * math.acos(max(-1.0, min(1.0, 1.0 - sagitta_limit / max_radius))))
        max_yaw_step = max(1e-4, min(max_yaw_step, 0.05))

        for k in range(len(path) - 1):
            a, b = path[k], path[k + 1]
            dyaw = _wrap(float(b[2] - a[2]))
            dist = float(np.linalg.norm(b[:2] - a[:2]))
            ddelta = _wrap(float(deltas[k + 1] - deltas[k]))
            dnormal = _wrap((float(b[2]) + float(deltas[k + 1]))
                            - (float(a[2]) + float(deltas[k])))
            n = max(1, int(math.ceil(dist / (self.res / 2.0))),
                    int(math.ceil(max(abs(dyaw), abs(dnormal)) / max_yaw_step)))
            prev_poly = None
            for step in range(n + 1):
                u = step / n
                pose = np.asarray([a[0] + u * (b[0] - a[0]),
                                   a[1] + u * (b[1] - a[1]),
                                   float(a[2]) + u * dyaw])
                delta = deltas[k] + u * ddelta
                cur_poly = self._pose_polygon(pose, delta)
                if not self._polygon_safe(cur_poly):
                    return False
                if prev_poly is not None:
                    swept = convex_hull(np.vstack((prev_poly, cur_poly)))
                    local_angle = max(abs(dyaw), abs(dnormal)) / n
                    sagitta = max_radius * (1.0 - math.cos(local_angle / 2.0))
                    swept = _minkowski_square(swept, sagitta)
                    if not self._polygon_safe(swept):
                        return False
                if dist <= 1e-12 and abs(dyaw) > 1e-10 and not self._turn_disk_safe(pose):
                    return False
                prev_poly = cur_poly
        return True
