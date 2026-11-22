"""Independent ground-truth checker (Phase H).

Imports ONLY mujoco, numpy, json, math, dataclasses, typing (enforced by tests/test_checker.py).
Everything is recomputed from MuJoCo state: the trace's qpos is loaded into a freshly built model
and mj_forward'd.  The stair hole is derived here from apartment geometry (NOT from mapping.py):
it is every cell inside the building footprint that is not covered by a static collidable geom
whose top surface is at floor level (z >= -0.01).

Trace: dict {"qpos": (T,nq), "t": (T,), "model_path": str}  (model_path may instead be passed to check()).
Task specs (dict, "type" selects):
  nav : goal=[x,y,theta], pos_tol=0.05, yaw_tol_deg=10, final 1 s continuously in tolerance
  kick: ball_body, direction (rad, world) , min_speed (m/s), angle_tol_deg
  pick: target_body, min_lift (m, default 0.02 since recorded pick-policy onset), max_drift
        (m, default 0.01: trunk xy drift since onset), upright_tol_deg (default 30)
Skill windows use sim-recorded policy_events (or legacy policy_timeline), never FSM indices.
Fall = tilt > 60 degrees or trunk height above lowest foot-geom centre < 0.02 m.
SPL path_length uses centred MA30 trunk xy with endpoint padding; path_length_raw is retained.
Verdict: JSON-serializable dict with "success", "reasons" (codes), and measurements.
"""
import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

TRUNK = "trunk_base"
FALL_TILT_DEG = 60.0
FALL_Z = 0.02  # trunk height above the lowest foot collision-geom centre
PATH_SMOOTHING_STEPS = 30
FLOOR_TOP_Z = -0.01     # a static box whose top is above this counts as walkable/solid cover
FLOOR_CONTACT_Z = 0.012  # static geoms whose top is below this are "floor", not obstacles
HOLE_RES = 0.01

_MODELS: Dict[str, Any] = {}


def load_model(path: str):
    if path not in _MODELS:
        _MODELS[path] = (mujoco.MjModel.from_binary_path(path) if path.endswith(".mjb")
                         else mujoco.MjModel.from_xml_path(path))
    return _MODELS[path]


@dataclass
class Hole:
    x0: float
    y0: float
    res: float
    mask: np.ndarray  # (H, W) bool, True = hole cell

    def contains(self, x: float, y: float) -> bool:
        j = int(math.floor((x - self.x0) / self.res))
        i = int(math.floor((y - self.y0) / self.res))
        h, w = self.mask.shape
        return 0 <= i < h and 0 <= j < w and bool(self.mask[i, j])

    def bbox(self) -> Optional[List[float]]:
        ii, jj = np.nonzero(self.mask)
        if len(ii) == 0:
            return None
        return [self.x0 + jj.min() * self.res, self.y0 + ii.min() * self.res,
                self.x0 + (jj.max() + 1) * self.res, self.y0 + (ii.max() + 1) * self.res]


def _static_geom_ids(model) -> List[int]:
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0]


def derive_hole(model, res: float = HOLE_RES) -> Hole:
    """Hole = cells inside the footprint of the outer walls not covered by any solid floor-level box."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    boxes = []
    for g in _static_geom_ids(model):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        R = data.geom_xmat[g].reshape(3, 3)
        c = data.geom_xpos[g]
        s = model.geom_size[g]
        top = c[2] + np.abs(R[2]) @ s
        if top < FLOOR_TOP_Z:
            continue
        ext = np.abs(R[:2, :]) @ s  # AABB half extents in xy
        boxes.append((c[0], c[1], ext[0], ext[1], R, s))
    if not boxes:
        return Hole(0.0, 0.0, res, np.zeros((1, 1), bool))
    # footprint: bbox of thin floor slabs whose top surface is at z ~ 0 (stairs are lower, walls taller)
    slabs = [b for b in boxes if b[5][2] < 0.02] or boxes
    x0 = min(b[0] - b[2] for b in slabs)
    x1 = max(b[0] + b[2] for b in slabs)
    y0 = min(b[1] - b[3] for b in slabs)
    y1 = max(b[1] + b[3] for b in slabs)
    xs = x0 + (np.arange(int(math.ceil((x1 - x0) / res))) + 0.5) * res
    ys = y0 + (np.arange(int(math.ceil((y1 - y0) / res))) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    covered = np.zeros(X.shape, bool)
    for cx, cy, ex, ey, R, s in boxes:
        dx, dy = X - cx, Y - cy
        lx = R[0, 0] * dx + R[1, 0] * dy
        ly = R[0, 1] * dx + R[1, 1] * dy
        covered |= (np.abs(lx) <= s[0]) & (np.abs(ly) <= s[1])
    return Hole(x0, y0, res, ~covered)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _quat_tilt_yaw(q) -> Tuple[float, float]:
    w, x, y, z = q
    r22 = 1 - 2 * (x * x + y * y)
    tilt = math.acos(max(-1.0, min(1.0, r22)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tilt, yaw


def _subtree_bodies(model, root: int) -> set:
    out = {root}
    for b in range(model.nbody):
        p = b
        while p != 0:
            if p == root:
                out.add(b)
                break
            p = model.body_parentid[p]
    return out


def _free_qadr(model, body: int) -> int:
    j = model.body_jntadr[body]
    if j < 0 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("body %s has no free joint" % model.body(body).name)
    return int(model.jnt_qposadr[j])


def _static_geometry_matches(model, reference):
    actual = {model.geom(g).name: g for g in _static_geom_ids(model)
              if model.geom_contype[g] or model.geom_conaffinity[g]}
    if set(actual) != {entry["name"] for entry in reference}:
        return False
    for entry in reference:
        g = actual[entry["name"]]
        for key in ("type", "pos", "size", "quat", "contype", "conaffinity"):
            if not np.allclose(np.asarray(getattr(model, "geom_" + key)[g]), entry[key], rtol=0, atol=1e-8):
                return False
    return True


def _smooth_xy(xy, window=PATH_SMOOTHING_STEPS):
    """Centred MA30, with endpoint padding so short traces retain their shape."""
    if len(xy) < 2:
        return xy.copy()
    left, right = window // 2, window - 1 - window // 2
    padded = np.pad(xy, ((left, right), (0, 0)), mode="edge")
    return np.column_stack([np.convolve(padded[:, j], np.ones(window) / window,
                                        mode="valid") for j in range(2)])


def _geom_over_hole(model, data, g, hole):
    # MuJoCo's local AABB includes mesh vertices and primitive extents. Transform
    # all extents conservatively: overlap includes cells between corner samples.
    bounds = model.geom_aabb[g]
    R = data.geom_xmat[g].reshape(3, 3)
    centre = data.geom_xpos[g] + R @ bounds[:3]
    extent = np.abs(R) @ bounds[3:]
    if centre[2] - extent[2] > 0.01:
        return False  # swinging feet above floor + 1 cm are not incursions
    low, high = centre[:2] - extent[:2], centre[:2] + extent[:2]
    j0, i0 = np.floor((low - [hole.x0, hole.y0]) / hole.res).astype(int)
    j1, i1 = np.floor((high - [hole.x0, hole.y0]) / hole.res).astype(int)
    h, w = hole.mask.shape
    return bool(np.any(hole.mask[max(0, i0):min(h, i1 + 1),
                                 max(0, j0):min(w, j1 + 1)])) if i1 >= 0 and j1 >= 0 and i0 < h and j0 < w else False


def _skill_onset(trace, kind, t):
    """Use sim-recorded policy switches, never the FSM's skill_start_index."""
    timeline = trace.get("policy_events", trace.get("policy_timeline"))
    if timeline is None:
        return None
    for entry in timeline:
        policy = entry.get("policy", "") if isinstance(entry, dict) else entry[1]
        if kind not in str(policy):
            continue
        when = entry.get("t", 0.0) if isinstance(entry, dict) else entry[0]
        # The sim recorder's step is exact; legacy switch times can accumulate
        # floating-point MuJoCo clock error. Neither comes from the FSM index.
        onset = (int(entry["step"]) if isinstance(entry, dict) and "step" in entry
                 else int(np.searchsorted(t, float(when) - 1e-9, side="left")))
        return onset if 0 <= onset < len(t) else None
    return None


def check(trace: Dict[str, Any], task: Dict[str, Any], model_path: Optional[str] = None,
          fall_tilt_deg: float = FALL_TILT_DEG, fall_z: float = FALL_Z) -> Dict[str, Any]:
    model = load_model(model_path or trace["model_path"])
    reference = trace.get("static_geometry_reference")
    if reference is not None and not _static_geometry_matches(model, reference):
        return {"success": False, "reasons": ["bad_trace"], "task_type": task.get("type"),
                "fallen": False, "hole_incursion": False, "safety_unknown": True,
                "error": "checker geometry differs from independent frozen scene"}
    data = mujoco.MjData(model)
    qpos = np.asarray(trace.get("qpos", []), float)
    t = np.asarray(trace.get("t", []), float)
    T = len(qpos) if qpos.ndim else 0
    if (qpos.ndim != 2 or T == 0 or qpos.shape[1] != model.nq or t.ndim != 1 or len(t) != T
            or not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(t))
            or (T > 1 and np.any(np.diff(t) <= 0))):
        return {"success": False, "reasons": ["bad_trace"], "task_type": task.get("type"),
                "fallen": False, "hole_incursion": False, "safety_unknown": True}
    trunk = model.body(TRUNK).id
    qa = _free_qadr(model, trunk)
    robot = _subtree_bodies(model, trunk)
    hole = derive_hole(model)
    static = set(_static_geom_ids(model))
    robot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in robot
                   and (model.geom_contype[g] or model.geom_conaffinity[g])]
    feet = [model.geom(f"{side}_foot_collision").id for side in ("left", "right")]
    relative_height = np.zeros(T)
    floorish = set()
    for g in static:
        top = model.geom_pos[g][2] + (model.geom_size[g][2] if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else 0)
        if top < FLOOR_CONTACT_Z:
            floorish.add(g)

    tilt = np.zeros(T); xy = np.zeros((T, 2)); yaw = np.zeros(T); z = np.zeros(T)
    hole_steps = []; coll_steps = 0; coll_geoms = set()
    for k in range(T):
        data.qpos[:] = qpos[k]
        mujoco.mj_forward(model, data)
        xy[k] = data.qpos[qa:qa + 2]; z[k] = data.qpos[qa + 2]
        tilt[k], yaw[k] = _quat_tilt_yaw(data.qpos[qa + 3:qa + 7])
        relative_height[k] = z[k] - min(data.geom_xpos[g][2] for g in feet)
        # Trunk centre always counts; collision extents count near/below the floor.
        if hole.contains(*xy[k]) or any(_geom_over_hole(model, data, g, hole) for g in robot_geoms):
            hole_steps.append(k)
        hit = False
        for c in range(data.ncon):
            g1, g2 = int(data.contact[c].geom1), int(data.contact[c].geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if model.geom_bodyid[a] in robot and b in static and b not in floorish:
                    hit = True
                    coll_geoms.add(model.geom(b).name or str(b))
        coll_steps += hit

    tilt_deg = np.degrees(tilt)
    fell_idx = np.nonzero((tilt_deg > fall_tilt_deg) | (relative_height < fall_z))[0]
    fallen = len(fell_idx) > 0
    path_raw = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if T > 1 else 0.0
    smooth_xy = _smooth_xy(xy)
    path_len = float(np.sum(np.linalg.norm(np.diff(smooth_xy, axis=0), axis=1))) if T > 1 else 0.0
    reasons: List[str] = []
    v: Dict[str, Any] = {
        "task_type": task.get("type"), "n_steps": T, "duration": float(t[-1] - t[0]),
        "fallen": bool(fallen), "fall_step": int(fell_idx[0]) if fallen else None,
        "max_tilt_deg": float(tilt_deg.max()), "min_trunk_z": float(z.min()),
        "hole_incursion": bool(hole_steps), "hole_incursion_steps": len(hole_steps),
        "hole_bbox": hole.bbox(), "path_length": path_len, "path_length_raw": path_raw,
        "path_smoothing_steps": PATH_SMOOTHING_STEPS, "min_trunk_height_above_feet": float(relative_height.min()),
        "safety_unknown": False,
        "collision_steps": int(coll_steps), "collision_geoms": sorted(coll_geoms),
        "final_pose": [float(xy[-1, 0]), float(xy[-1, 1]), float(yaw[-1])],
    }
    if fallen:
        reasons.append("fall")
    if hole_steps:
        reasons.append("stair_hole")

    ty = task.get("type")
    ok = False
    if ty == "nav":
        gx, gy, gth = task["goal"]
        ptol = task.get("pos_tol", 0.05); ytol = math.radians(task.get("yaw_tol_deg", 10.0))
        perr = np.hypot(xy[:, 0] - gx, xy[:, 1] - gy)
        yerr = np.abs([_wrap(a - gth) for a in yaw])
        inside = (perr <= ptol) & (yerr <= ytol)
        v["pos_err"] = float(perr[-1]); v["yaw_err_deg"] = float(math.degrees(yerr[-1]))
        v["time_to_goal"] = None
        if inside[-1]:
            k = T - 1
            while k > 0 and inside[k - 1]:
                k -= 1
            v["time_to_goal"] = float(t[k] - t[0])
        elif perr[-1] > ptol:
            reasons.append("pos_tol")
        if not inside[-1] and perr[-1] <= ptol:
            reasons.append("yaw_tol")
        settle = t >= t[-1] - 1.0 - 1e-9
        dwell_ok = bool(t[-1] - t[0] >= 1.0 - 1e-9 and np.all(inside[settle]))
        if inside[-1] and not dwell_ok:
            reasons.append("goal_not_settled")
        v["goal_settled"] = dwell_ok
        ok = bool(inside[-1] and dwell_ok)
    elif ty == "kick":
        b = model.body(task["ball_body"]).id
        ba = _free_qadr(model, b)
        bxy = qpos[:, ba:ba + 2]
        onset = _skill_onset(trace, "kick", t)
        v["skill_onset_index"] = onset
        if onset is None:
            reasons.append("skill_not_started")
            onset = T - 1
        pre_move = float(np.max(np.linalg.norm(bxy[:onset + 1] - bxy[0], axis=1)))
        v["ball_pre_kick_displacement"] = pre_move
        if pre_move > 0.02:
            reasons.append("ball_moved_before_kick")
        dt = np.diff(t[onset:])
        vel = np.diff(bxy[onset:], axis=0) / dt[:, None] if len(dt) else np.zeros((0, 2))
        speed = np.linalg.norm(vel, axis=1) if len(vel) else np.zeros(0)
        v["ball_displacement"] = float(np.linalg.norm(bxy[-1] - bxy[0]))
        if len(speed) == 0 or speed.max() <= 0:
            v["ball_peak_speed"] = 0.0; v["angle_err_deg"] = None
            reasons.append("ball_not_moved")
        else:
            kmax = int(np.argmax(speed))
            v["ball_peak_speed"] = float(speed[kmax])
            ang = math.atan2(vel[kmax][1], vel[kmax][0])
            err = abs(_wrap(ang - task["direction"]))
            v["angle_err_deg"] = float(math.degrees(err))
            if speed[kmax] < task.get("min_speed", 0.3):
                reasons.append("too_slow")
            if math.degrees(err) > task.get("angle_tol_deg", 30.0):
                reasons.append("wrong_direction")
        ok = not any(r in reasons for r in ("ball_not_moved", "too_slow", "wrong_direction", "ball_moved_before_kick", "skill_not_started"))
    elif ty == "pick":
        b = model.body(task["target_body"]).id
        ba = _free_qadr(model, b)
        # The skill contract measures lift/drift from skill onset, after the
        # navigation approach. Safety above still examines the entire trace.
        onset = _skill_onset(trace, "pick", t)
        v["skill_onset_index"] = onset
        if onset is None:
            reasons.append("skill_not_started")
            onset = T - 1
        lift = float(qpos[-1, ba + 2] - qpos[onset, ba + 2])
        drift = float(np.linalg.norm(xy[-1] - xy[onset]))
        upright = float(tilt_deg[-1])
        v.update(lift=lift, drift=drift, final_tilt_deg=upright)
        if lift < task.get("min_lift", 0.02):
            reasons.append("not_lifted")
        if drift >= task.get("max_drift", 0.01):
            reasons.append("drift")
        if upright > task.get("upright_tol_deg", 30.0):
            reasons.append("not_upright")
        ok = not any(r in reasons for r in ("not_lifted", "drift", "not_upright", "skill_not_started"))
    else:
        reasons.append("unknown_task")
        v["safety_unknown"] = True
    success = bool(ok and not fallen and not hole_steps)
    v["success"] = success
    v["reasons"] = reasons if not success else []
    if success and reasons:
        v["reasons"] = []
    return json.loads(json.dumps(v))
