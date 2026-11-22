"""Phase C: 2D planning map (occupancy / cliff / inflated cost) built from MuJoCo geometry.

See nav/CONTRACTS.md. Key conventions
-------------------------------------
* Grid arrays are indexed [i, j] with i = row along +y, j = column along +x
  (cell (i, j) centre = origin + ((j+.5)*res, (i+.5)*res)); `origin` is the lower-left corner.
* occ      : static, collidable geoms overlapping the robot height band [step_over_height, robot_height].
             Rasterised CONSERVATIVELY (any cell square touching a geom footprint is occupied).
* cliff    : no walkable floor under the cell (ray down finds no surface with z >= -step_over_height),
             or outside the floor. Evaluated with rays at cell centre and (near floor edges) at 4 sub-points,
             so a cell is cliff if ANY sample misses. Cells already `occ` are not flagged cliff.
* lethal   : (occ | cliff) inflated by `r_inflate = robot_radius + 3*lateral_sigma` (clearance <= r_inflate).
* cost     : float32; inf where lethal; else 0.99*exp(-k*(d - r_inflate)) (nav2-style), d = clearance.
* rooms    : name -> (xmin, ymin, xmax, ymax) bbox (interior, wall centre-lines) ; room_id raster too.
* doorways : list of dicts {name, rooms:(a,b), axis, center:(x,y), width, p0:(x,y), p1:(x,y)}.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

NAV_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = NAV_ROOT / "data"
DEFAULT_SCENE = (Path.home() / "microduck/upstream/microduck_rl/src/mjlab_microduck/robot/microduck/"
                 "scene_apartment.xml")
MAP_VERSION = 5
ROBOT_ROOT_BODY = "trunk_base"
GROUP_MASKED = 5  # geom group used (on a private model copy) to hide non-obstacle geoms from rays


# --------------------------------------------------------------------------- model helpers
def _robot_body_ids(m, root_name=ROBOT_ROOT_BODY):
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_name)
    ids = set()
    for b in range(m.nbody):
        r = b
        while r != 0 and r != root:
            r = m.body_parentid[r]
        if r == root:
            ids.add(b)
    return root, ids


def _free_body_ids(m):
    """Bodies that are movable: have a free joint themselves or in an ancestor."""
    out = set()
    for b in range(1, m.nbody):
        r = b
        while r != 0:
            js = m.body_jntadr[r]
            if m.body_jntnum[r] > 0 and m.jnt_type[js] == mujoco.mjtJoint.mjJNT_FREE:
                out.add(b)
                break
            r = m.body_parentid[r]
    return out


def classify_geoms(m):
    """Returns (robot_geoms, movable_geoms, visual_geoms, static_obstacle_candidates) as lists of ids."""
    root, rb = _robot_body_ids(m)
    fb = _free_body_ids(m) - rb
    robot, movable, visual, static = [], [], [], []
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if b in rb:
            robot.append(g)
        elif b in fb:
            movable.append(g)
        elif m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            visual.append(g)
        else:
            static.append(g)
    return robot, movable, visual, static


def masked_ray_model(scene_xml):
    """Private model copy where everything that is not a static collidable geom is moved to group 5.
    Use with geomgroup mask GEOMGROUP_MASK. Returns (model, data)."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    set_stand(m, d)
    robot, movable, visual, _ = classify_geoms(m)
    for g in robot + movable + visual:
        m.geom_group[g] = GROUP_MASKED
    mujoco.mj_forward(m, d)
    return m, d


GEOMGROUP_MASK = np.array([1, 1, 1, 1, 1, 0], dtype=np.uint8)


def set_stand(m, d):
    try:
        k = m.key("STAND")
        d.qpos[:] = k.qpos
        if m.nu:
            d.ctrl[:] = k.ctrl
    except Exception:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)


# --------------------------------------------------------------------------- geometry -> point clouds
def _sphere_dirs(nu=16, nv=32):
    th = np.linspace(0, np.pi, nu)[:, None]
    ph = np.linspace(0, 2 * np.pi, nv, endpoint=False)[None, :]
    return np.stack([np.sin(th) * np.cos(ph) + 0 * th, np.sin(th) * np.sin(ph) + 0 * th,
                     np.cos(th) + 0 * ph], -1).reshape(-1, 3)


def geom_world_points(m, d, g, nseg=96):
    """Vertices of a convex polytope (exact for box/mesh, inscribed approximation for curved) in world frame.
    Returns None for plane/hfield."""
    t = m.geom_type[g]
    s = m.geom_size[g]
    R = d.geom_xmat[g].reshape(3, 3)
    p = d.geom_xpos[g]
    GT = mujoco.mjtGeom
    if t == GT.mjGEOM_BOX:
        loc = np.array([[sx * s[0], sy * s[1], sz * s[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    elif t == GT.mjGEOM_CYLINDER:
        a = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
        ring = np.stack([s[0] * np.cos(a), s[0] * np.sin(a)], 1)
        loc = np.concatenate([np.c_[ring, np.full(nseg, -s[1])], np.c_[ring, np.full(nseg, s[1])]])
    elif t == GT.mjGEOM_SPHERE:
        loc = _sphere_dirs() * s[0]
    elif t == GT.mjGEOM_ELLIPSOID:
        loc = _sphere_dirs() * s[:3]
    elif t == GT.mjGEOM_CAPSULE:
        dirs = _sphere_dirs()
        top = dirs[dirs[:, 2] >= -1e-9] * s[0] + [0, 0, s[1]]
        bot = dirs[dirs[:, 2] <= 1e-9] * s[0] - [0, 0, s[1]]
        loc = np.concatenate([top, bot])
    elif t == GT.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        a0, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        loc = m.mesh_vert[a0:a0 + n].astype(float)
    else:
        return None
    return loc @ R.T + p


def slab_footprint(pts, zlo, zhi):
    """2D convex footprint (Nx2 hull vertices) of the convex hull of `pts` intersected with zlo<=z<=zhi.
    Returns None if empty."""
    if pts[:, 2].max() < zlo or pts[:, 2].min() > zhi:
        return None
    cand = [pts[(pts[:, 2] >= zlo) & (pts[:, 2] <= zhi)]]
    if len(pts) >= 4 and np.ptp(pts, axis=0).min() > 1e-9:
        try:
            hull = ConvexHull(pts)
            edges = set()
            for tri in hull.simplices:
                for a, b in ((0, 1), (1, 2), (0, 2)):
                    e = tuple(sorted((tri[a], tri[b])))
                    edges.add(e)
            E = np.array(sorted(edges))
            A, B = pts[E[:, 0]], pts[E[:, 1]]
            for z in (zlo, zhi):
                da, db = A[:, 2] - z, B[:, 2] - z
                m_ = (da * db) < 0
                if m_.any():
                    tt = da[m_] / (da[m_] - db[m_])
                    cand.append(A[m_] + tt[:, None] * (B[m_] - A[m_]))
        except Exception:
            # degenerate (flat) hull: fall back to clipping by projecting everything inside the slab
            cand.append(np.c_[pts[:, :2], np.clip(pts[:, 2], zlo, zhi)])
    else:
        cand.append(np.c_[pts[:, :2], np.clip(pts[:, 2], zlo, zhi)])
    c = np.concatenate(cand)
    return c[:, :2] if len(c) else None


def raster_convex(poly_pts, x0, y0, res, H, W, grid):
    """Mark in `grid` (bool HxW) all cells whose square touches the convex hull of poly_pts (Nx2)."""
    P = np.unique(np.round(poly_pts, 9), axis=0)
    xmin, ymin = P.min(0) - 1e-9
    xmax, ymax = P.max(0) + 1e-9
    j0 = max(int(np.floor((xmin - x0) / res)), 0)
    j1 = min(int(np.floor((xmax - x0) / res)), W - 1)
    i0 = max(int(np.floor((ymin - y0) / res)), 0)
    i1 = min(int(np.floor((ymax - y0) / res)), H - 1)
    if j1 < j0 or i1 < i0:
        return
    jj, ii = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1))
    cx = x0 + (jj + 0.5) * res
    cy = y0 + (ii + 0.5) * res
    h = 0.5 * res + 1e-9
    inside = np.ones_like(cx, dtype=bool)
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    if len(P) >= 3:
        try:
            hl = ConvexHull(P)
            V = P[hl.vertices]
            for k in range(len(V)):
                e = V[(k + 1) % len(V)] - V[k]
                n = np.array([-e[1], e[0]])
                nn = np.linalg.norm(n)
                if nn > 1e-12:
                    axes.append(n / nn)
        except Exception:
            V = P
            axes.append(np.array([1.0, 0.0]))
    else:
        V = P
        if len(P) == 2:
            e = P[1] - P[0]
            n = np.array([-e[1], e[0]])
            if np.linalg.norm(n) > 1e-12:
                axes.append(n / np.linalg.norm(n))
    for n in axes:
        pr = V @ n
        c = cx * n[0] + cy * n[1]
        rad = h * (abs(n[0]) + abs(n[1]))
        inside &= (c + rad >= pr.min() - 1e-9) & (c - rad <= pr.max() + 1e-9)
    sub = grid[i0:i1 + 1, j0:j1 + 1]
    sub |= inside


# --------------------------------------------------------------------------- robot dims
def robot_dimensions(m, d):
    """(height, radius, zmin, centre_xy) from robot COLLISION geoms (contype/conaffinity != 0) at STAND."""
    robot, _, _, _ = classify_geoms(m)
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ROBOT_ROOT_BODY)
    c = d.xpos[root][:2].copy()
    pts = []
    for g in robot:
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        p = geom_world_points(m, d, g)
        if p is not None:
            pts.append(p)
    pts = np.concatenate(pts)
    radius = float(np.linalg.norm(pts[:, :2] - c, axis=1).max())
    return float(pts[:, 2].max()), radius, float(pts[:, 2].min()), c


# --------------------------------------------------------------------------- GridMap
@dataclass
class GridMap:
    res: float
    origin: tuple
    occ: np.ndarray
    cliff: np.ndarray
    lethal: np.ndarray
    cost: np.ndarray
    clearance: np.ndarray
    rooms: dict = field(default_factory=dict)       # name -> (xmin, ymin, xmax, ymax)
    doorways: list = field(default_factory=list)
    room_id: np.ndarray | None = None                # int8 raster, -1 none; index into room_names
    room_names: list = field(default_factory=list)
    params: dict = field(default_factory=dict)       # robot_radius, r_inflate, robot_height, ...
    footprint: object | None = field(default=None, repr=False, compare=False)
    footprint_fingerprint: str = ""

    @property
    def shape(self):
        return self.occ.shape

    @property
    def bounds(self):
        H, W = self.occ.shape
        return (self.origin[0], self.origin[1], self.origin[0] + W * self.res, self.origin[1] + H * self.res)

    def world_to_cell(self, x, y):
        j = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(int)
        i = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(int)
        return (int(i), int(j)) if i.ndim == 0 else (i, j)

    def cell_to_world(self, i, j):
        return (self.origin[0] + (np.asarray(j) + 0.5) * self.res,
                self.origin[1] + (np.asarray(i) + 0.5) * self.res)

    def in_bounds(self, x, y):
        i, j = self.world_to_cell(x, y)
        H, W = self.occ.shape
        return (np.asarray(i) >= 0) & (np.asarray(i) < H) & (np.asarray(j) >= 0) & (np.asarray(j) < W)

    def _lookup(self, arr, x, y, oob):
        i, j = self.world_to_cell(x, y)
        H, W = self.occ.shape
        ii, jj = np.asarray(i), np.asarray(j)
        ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
        v = arr[np.clip(ii, 0, H - 1), np.clip(jj, 0, W - 1)]
        return np.where(ok, v, oob) if ii.ndim else (v if ok else oob)

    def is_lethal(self, x, y):
        """Cheap point test; cliff footprint safety requires is_pose_lethal(x,y,theta)."""
        r = self._lookup(self.lethal, x, y, True)
        return bool(r) if np.ndim(r) == 0 else r

    def is_pose_lethal(self, x, y, theta, heading_delta=0.0):
        """Heading-aware cliff footprint plus unchanged inflated obstacle point test."""
        point_bad = self.is_lethal(x, y)
        if np.ndim(point_bad):
            out = np.asarray(point_bad, bool).copy()
            for k in range(out.size):
                if not out.flat[k]:
                    out.flat[k] = not self.footprint.pose_safe(
                        (np.asarray(x).flat[k], np.asarray(y).flat[k], np.asarray(theta).flat[k]),
                        heading_delta=float(np.asarray(heading_delta).flat[k])
                        if np.ndim(heading_delta) else float(heading_delta))
            return out
        return bool(point_bad or not self.footprint.pose_safe((x, y, theta), heading_delta=heading_delta))

    def cost_at(self, x, y):
        r = self._lookup(self.cost, x, y, np.float32(np.inf))
        return float(r) if np.ndim(r) == 0 else r

    def clearance_at(self, x, y):
        r = self._lookup(self.clearance, x, y, np.float32(0.0))
        return float(r) if np.ndim(r) == 0 else r

    def room_of(self, x, y):
        if self.room_id is None:
            return None
        k = int(self._lookup(self.room_id, x, y, -1))
        return self.room_names[k] if k >= 0 else None

    def sample_free(self, room, n=5, min_clearance=None, rng=None, margin=0.05):
        """n well-spread non-lethal points in `room` (farthest-point sampling over cells with clearance >=
        min_clearance, default r_inflate + 0.05). rng=None -> deterministic (start from highest clearance)."""
        x0, y0, x1, y1 = self.rooms[room]
        if min_clearance is None:
            min_clearance = self.params["r_inflate"] + 0.05
        H, W = self.occ.shape
        ii, jj = np.mgrid[0:H, 0:W]
        X, Y = self.cell_to_world(ii, jj)
        ok = (~self.lethal) & (self.clearance >= min_clearance) & (X > x0 + margin) & (X < x1 - margin) & \
             (Y > y0 + margin) & (Y < y1 - margin)
        pts = np.c_[X[ok], Y[ok]]
        if len(pts) == 0:
            return []
        if rng is None:
            sel = [int(np.argmax(self.clearance[ok]))]
        else:
            sel = [int(rng.integers(len(pts)))]
        dmin = np.linalg.norm(pts - pts[sel[0]], axis=1)
        while len(sel) < min(n, len(pts)):
            k = int(np.argmax(dmin))
            sel.append(k)
            dmin = np.minimum(dmin, np.linalg.norm(pts - pts[k], axis=1))
        return [(float(pts[k, 0]), float(pts[k, 1])) for k in sel]

    # -- io
    def save(self, path):
        np.savez_compressed(
            path, res=self.res, origin=np.array(self.origin), occ=self.occ, cliff=self.cliff, lethal=self.lethal,
            cost=self.cost, clearance=self.clearance,
            room_id=self.room_id if self.room_id is not None else np.zeros(0, np.int8),
            meta=json.dumps({"rooms": {k: list(v) for k, v in self.rooms.items()}, "doorways": self.doorways,
                             "room_names": self.room_names, "params": self.params}))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        for dw in meta["doorways"]:
            for k in ("rooms", "center", "p0", "p1"):
                dw[k] = tuple(dw[k])
        rid = z["room_id"]
        return cls(res=float(z["res"]), origin=tuple(float(v) for v in z["origin"]), occ=z["occ"], cliff=z["cliff"],
                   lethal=z["lethal"], cost=z["cost"], clearance=z["clearance"],
                   rooms={k: tuple(v) for k, v in meta["rooms"].items()}, doorways=meta["doorways"],
                   room_id=rid if rid.size else None, room_names=meta["room_names"], params=meta["params"])


# --------------------------------------------------------------------------- build
def _scene_files(scene_xml):
    """scene xml plus recursively <include>d xml files and mesh/asset files they sit beside (hashed by content)."""
    seen, stack = [], [Path(scene_xml).resolve()]
    while stack:
        f = stack.pop()
        if f in seen or not f.exists():
            continue
        seen.append(f)
        for inc in re.findall(r'<include\s+file="([^"]+)"', f.read_text()):
            stack.append((f.parent / inc).resolve())
    return seen


def scene_hash(scene_xml, params):
    h = hashlib.sha256()
    for f in sorted(_scene_files(scene_xml)):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    asset_dir = Path(scene_xml).resolve().parent / "assets"
    if asset_dir.exists():
        for f in sorted(asset_dir.iterdir()):
            if f.suffix.lower() in (".stl", ".obj"):
                h.update(f.name.encode())
                h.update(f.read_bytes())
    h.update(json.dumps(params, sort_keys=True).encode())
    h.update(str(MAP_VERSION).encode())
    return h.hexdigest()[:16]


def _floor_ok(m, d, x, y, z0, step_over):
    """Vectorised-ish floor test: ray down from z0 at (x,y); floor present iff the first hit surface z >= -step_over."""
    gid = np.zeros(1, np.int32)
    out = np.zeros(len(x), bool)
    vec = np.array([0.0, 0.0, -1.0])
    pnt = np.zeros(3)
    for k in range(len(x)):
        pnt[0], pnt[1], pnt[2] = x[k], y[k], z0
        dist = mujoco.mj_ray(m, d, pnt, vec, GEOMGROUP_MASK, 1, -1, gid)
        out[k] = dist >= 0 and (z0 - dist) >= -step_over - 1e-9
    return out


def _find_rooms(m):
    """Rooms from the per-room floor overlay geoms (visual 'floor_<room>' boxes): bbox = pos +- size.
    Corridor: x-range between the kitchen/living east edge and the office/bedroom/bath west edge,
    y-range = union of the floor_cor_* overlays."""
    def bbox(name):
        g = m.geom(name)
        return (float(g.pos[0] - g.size[0]), float(g.pos[1] - g.size[1]),
                float(g.pos[0] + g.size[0]), float(g.pos[1] + g.size[1]))
    rooms = {}
    names = {"kitchen": "floor_kitchen", "living": "floor_living", "bedroom": "floor_bedroom",
             "office": "floor_office", "bathroom": "floor_bath"}
    for r, gname in names.items():
        rooms[r] = bbox(gname)
    cn, cs = bbox("floor_cor_n"), bbox("floor_cor_s")
    xw = max(rooms["kitchen"][2], rooms["living"][2])
    xe = min(rooms["office"][0], rooms["bedroom"][0], rooms["bathroom"][0])
    rooms["corridor"] = (xw, min(cn[1], cs[1]), xe, max(cn[3], cs[3]))
    return rooms


def _find_doorways(rooms, occ_raw, x0, y0, res, tol=0.2):
    names = list(rooms)
    dws = []
    H, W = occ_raw.shape

    def occ_at(x, y):
        j, i = int((x - x0) // res), int((y - y0) // res)
        return occ_raw[i, j] if (0 <= i < H and 0 <= j < W) else True

    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = rooms[names[a]], rooms[names[b]]
            for axis in ("x", "y"):  # 'x': wall line is x = const (doorway runs along y)
                if axis == "x":
                    gap = max(B[0] - A[2], A[0] - B[2])
                    lo, hi = max(A[1], B[1]), min(A[3], B[3])
                    line = 0.5 * (min(A[2], B[2]) + max(A[0], B[0])) if gap >= 0 else None
                else:
                    gap = max(B[1] - A[3], A[1] - B[3])
                    lo, hi = max(A[0], B[0]), min(A[2], B[2])
                    line = 0.5 * (min(A[3], B[3]) + max(A[1], B[1])) if gap >= 0 else None
                if gap < -1e-9 or gap > tol or hi - lo < 0.2:
                    continue
                ts = np.arange(lo + res / 2, hi, res)
                free = np.array([not occ_at(*((line, t) if axis == "x" else (t, line))) for t in ts])
                # runs of free cells strictly inside (bounded by walls on both ends)
                k = 0
                while k < len(ts):
                    if free[k]:
                        k2 = k
                        while k2 + 1 < len(ts) and free[k2 + 1]:
                            k2 += 1
                        if k > 0 and k2 < len(ts) - 1:
                            t0, t1 = ts[k] - res / 2, ts[k2] + res / 2
                            p0 = (float(line), float(t0)) if axis == "x" else (float(t0), float(line))
                            p1 = (float(line), float(t1)) if axis == "x" else (float(t1), float(line))
                            dws.append({"name": f"{names[a]}-{names[b]}#{len(dws)}", "rooms": (names[a], names[b]),
                                        "axis": "along_y" if axis == "x" else "along_x",
                                        "center": (float(0.5 * (p0[0] + p1[0])), float(0.5 * (p0[1] + p1[1]))),
                                        "width": float(t1 - t0), "p0": p0, "p1": p1})
                        k = k2 + 1
                    else:
                        k += 1
    return dws


def build_map(scene_xml=None, robot_radius=None, lateral_sigma=0.0, step_over_height=0.02, res=0.02,
              decay_k=5.0, margin=0.2, cache=True, cache_dir=None, force=False):
    scene_xml = str(Path(scene_xml) if scene_xml else DEFAULT_SCENE)
    params = dict(robot_radius=robot_radius, lateral_sigma=float(lateral_sigma),
                  step_over_height=float(step_over_height), res=float(res), decay_k=float(decay_k),
                  margin=float(margin))
    cache_dir = Path(cache_dir) if cache_dir else DATA_DIR
    cpath = cache_dir / f"map_{scene_hash(scene_xml, params)}.npz"
    if cache and cpath.exists() and not force:
        gm = GridMap.load(cpath)
        gm.footprint, gm.footprint_fingerprint = _attach_footprint(gm)
        return gm

    m = mujoco.MjModel.from_xml_path(scene_xml)
    d = mujoco.MjData(m)
    set_stand(m, d)
    height, r_fp, zmin_robot, _ = robot_dimensions(m, d)
    if robot_radius is None:
        robot_radius = r_fp
    r_inflate = robot_radius + 3.0 * lateral_sigma

    robot, movable, visual, static = classify_geoms(m)

    # world extent from static geoms (collidable), plus margin
    ext = []
    for g in static:
        p = geom_world_points(m, d, g)
        if p is not None:
            ext.append(p[:, :2])
    ext = np.concatenate(ext)
    x0 = np.floor((ext[:, 0].min() - margin) / res) * res
    y0 = np.floor((ext[:, 1].min() - margin) / res) * res
    W = int(np.ceil((ext[:, 0].max() + margin - x0) / res))
    H = int(np.ceil((ext[:, 1].max() + margin - y0) / res))

    # ---- obstacles: static geoms overlapping the robot band
    occ = np.zeros((H, W), bool)
    n_obst = 0
    for g in static:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue  # plane = floor
        pts = geom_world_points(m, d, g)
        if pts is None:
            continue
        # Support slabs ending at the lower boundary are floor, including when
        # the walker has no measured step-over ability (height exactly zero).
        if pts[:, 2].max() <= step_over_height + 1e-9:
            continue
        fp = slab_footprint(pts, step_over_height, height)
        if fp is None:
            continue
        raster_convex(fp, x0, y0, res, H, W, occ)
        n_obst += 1

    # ---- floor / cliff via rays on a private model copy (robot, movables, visual-only geoms hidden)
    mr, dr = masked_ray_model(scene_xml)
    z0 = step_over_height + 0.005
    ci = np.arange(H)
    cj = np.arange(W)
    CX = x0 + (cj + 0.5) * res
    CY = y0 + (ci + 0.5) * res
    GX, GY = np.meshgrid(CX, CY)
    center_ok = _floor_ok(mr, dr, GX.ravel(), GY.ravel(), z0, step_over_height).reshape(H, W)
    # refine: cells whose 3x3 neighbourhood has mixed floor status get 4 extra corner-ish samples
    ok_f = center_ok.astype(np.uint8)
    mixed = (ndimage.maximum_filter(ok_f, 3, mode="nearest") != ndimage.minimum_filter(ok_f, 3, mode="nearest"))
    # also treat grid border as mixed (the void beyond is cliff)
    mixed[0, :] = mixed[-1, :] = mixed[:, 0] = mixed[:, -1] = True
    mi, mj = np.nonzero(mixed)
    allok = center_ok.copy()
    q = 0.45 * res
    for dx, dy in ((-q, -q), (q, -q), (-q, q), (q, q), (0, -q), (0, q), (-q, 0), (q, 0)):
        allok[mi, mj] &= _floor_ok(mr, dr, CX[mj] + dx, CY[mi] + dy, z0, step_over_height)
    cliff = (~allok) & (~occ)

    bad = occ | cliff
    # Cost shaping still sees the full cliff margin, but hard expansion checks use
    # unchanged inflated occupancy plus the raw cliff mask. Exact oriented cliff
    # footprint checks are performed lazily on provisional planner routes.
    clearance = (ndimage.distance_transform_edt(~bad) * res).astype(np.float32)
    occ_clearance = (ndimage.distance_transform_edt(~occ) * res).astype(np.float32)
    lethal = (occ_clearance <= r_inflate + 1e-9) | cliff
    cost = (0.99 * np.exp(-decay_k * np.maximum(clearance - r_inflate, 0.0))).astype(np.float32)
    cost[cost < 1e-3] = 0.0
    cost[lethal] = np.inf

    rooms = _find_rooms(m)
    names = list(rooms)
    rid = -np.ones((H, W), np.int8)
    for k, nm in enumerate(names):
        xa, ya, xb, yb = rooms[nm]
        rid[(GX >= xa) & (GX < xb) & (GY >= ya) & (GY < yb)] = k
    doors = _find_doorways(rooms, occ, x0, y0, res)

    gm = GridMap(res=res, origin=(float(x0), float(y0)), occ=occ, cliff=cliff, lethal=lethal, cost=cost,
                 clearance=clearance, rooms=rooms, doorways=doors, room_id=rid, room_names=names,
                 params=dict(params, robot_radius=float(robot_radius), r_inflate=float(r_inflate),
                             robot_height=height, robot_footprint_radius=float(r_fp), robot_zmin=zmin_robot,
                             n_obstacle_geoms=n_obst, scene_xml=scene_xml))
    gm.footprint, gm.footprint_fingerprint = _attach_footprint(gm)
    if cache:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        gm.save(cpath)
    return gm


def _attach_footprint(gridmap):
    """Load the hash-pinned A6 checker injected by the candidate validation runner."""
    import os
    import sys
    from nav.a6_footprint import Footprint

    calibration = os.environ.get("MICRODUCK_A6_CALIBRATION_PATH")
    if not calibration:
        # Promoted default: the accepted, hash-pinned v5/A6 measurement (fail closed on any change).
        calibration = str(Path(__file__).resolve().parents[1]
                          / "reports/dev/v6/P2/c3_recal/calibration_c3_v1.json")
        if (not Path(calibration).is_file() or hashlib.sha256(Path(calibration).read_bytes()).hexdigest()
                != "b7d79c033fcd285908b3f2309bbb8500bf1d68531ef7c04e965e04f651371031"):
            raise RuntimeError("A6 footprint calibration missing or hash mismatch; refusing cliff-map use")
    path = Path(calibration).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    fp = Footprint(gridmap, path)
    module_path = Path(sys.modules[Footprint.__module__].__file__).resolve()
    digest = hashlib.sha256(path.read_bytes() + b"\0" + module_path.read_bytes()).hexdigest()
    return fp, digest


def overlay_discs(gridmap, obstacles):
    """Private planning overlay for surveyed free objects, with full inflation.

    The cached static map and its parameters are never modified. Disc surfaces
    include a cell-square allowance, matching the static map's conservative
    rasterization. All objects use the existing measured r_inflate.
    """
    from dataclasses import replace
    if not obstacles:
        return gridmap
    ii, jj = np.indices(gridmap.occ.shape)
    xx, yy = gridmap.cell_to_world(ii,jj)
    distance = np.full(gridmap.occ.shape,np.inf)
    for obj in obstacles:
        xy, radius = np.asarray(obj['xy'],float), float(obj['radius'])
        if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(radius) or radius <= 0:
            raise ValueError('invalid free-object disc')
        distance = np.minimum(distance,np.maximum(0.,np.hypot(xx-xy[0],yy-xy[1])-radius-gridmap.res/np.sqrt(2)))
    inflate = float(gridmap.params['r_inflate'])
    lethal = gridmap.lethal | (distance <= inflate+1e-9)
    clearance = np.minimum(gridmap.clearance,distance).astype(np.float32)
    extra_cost = .99*np.exp(-float(gridmap.params.get('decay_k',5.))*np.maximum(distance-inflate,0.))
    extra_cost[extra_cost < 1e-3] = 0.
    cost = np.maximum(gridmap.cost,extra_cost).astype(np.float32)
    cost[lethal] = np.inf
    return replace(gridmap,occ=gridmap.occ | (distance <= 0.),lethal=lethal,cost=cost,
                   clearance=clearance,params=dict(gridmap.params,free_object_discs=list(obstacles)))
