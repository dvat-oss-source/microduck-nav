"""Phase F -- localization for the Microduck nav stack (simulation only).

    loc = Localizer(sim, mode="odom+fiducial")      # modes: "gt" | "odom" | "odom+fiducial" | "learned+fiducial"
    loc.reset(gt.pose2)                              # start pose (world frame); call right after sim.reset
    loc.set_command(vx, vy, wz)                      # learned+fiducial only: the command last passed to sim.step
    est = loc.update(sim, gt_for_eval_only=gt)       # call EVERY 50 Hz policy step -> np.array([x, y, theta])

Modes
-----
F0 "gt"            returns the ground-truth pose2 (gt_for_eval_only, or sim.ground_truth()).
F1 "odom"          contact / leg odometry. Uses ONLY what the robot can sense (see SensorFrame):
                     * 14 actuated-joint positions (+ velocities, carried but not needed by the estimator),
                     * IMU: `orientation` framequat, `angular-velocity` gyro, `imu_accel` accelerometer
                       (read from data.sensordata by sensor name; sensors.xml noise std is applied by us because
                       MuJoCo only stores it as metadata).
                   Binary foot contact is simulated from foot collision contacts (a virtual tactile sensor). As in Pollen's Rust odometry (replicated in
                   upstream scripts/infer_policy.py PyOdometry) the stance anchor is selected from contacting feet using forward
                   kinematics of the sole anchor points (upstream scripts/odom_anchor_sets.json: v15 / alpha4 /
                   alpha16) on a scratch MjData with measured joint angles and IMU orientation, the lowest point is the
                   anchor (fixed in the world), the trunk moves so the anchor stays put. When both feet lose contact translation is held. No pip `pyodometry` package
                   exists (pip: "No matching distribution"), so PyOdometry is re-implemented here (same algorithm).
                   Heading source: "imu_quat" (default, the IMU fused orientation) or "gyro" (integrated gyro z).
                   `yaw_bias_dps_std` optionally degrades the IMU yaw with a per-episode constant bias
                   (stress test; the simulated orientation sensor is otherwise ground truth + 0.001 quat noise).
F2 "odom+fiducial" F1 as the EKF prediction + bearing/range updates from coloured fiducial markers detected in
                   head_camera renders at `camera_hz` (default 5 Hz). Markers are added at runtime to a COPY of the
                   apartment scene via mujoco.MjSpec (upstream XML untouched; visual-only geoms, contype 0), see
                   build_fiducial_spec(). Each marker = two stacked 8x8 cm patches (top colour / bottom colour from a
                   6-hue palette), the ordered colour pair is the ID. Detector: HSV hue segmentation -> connected
                   components -> vertically stacked pair -> ID, bearing from the pair centre, range from apparent
                   height. Camera pose from FK on the measured neck/head joints. Marker map is "surveyed" (known).

F3 "learned+fiducial" (candidate, Stage 1a). Same heading path and fiducial EKF update as F2. The translation
                   prediction and its covariance come from a learned proprioceptive odometry net (ONNX, CPU,
                   one thread) instead of the per-tick PyOdometry increment. Every 5 ticks (0.1 s), once a window
                   of `window` feature frames is full, the net predicts the body-frame displacement over the last 5
                   ticks plus per-axis log-variance; the displacement is rotated by the heading at the start of that
                   horizon, and its variance (rotated) becomes process noise. Until the window is full the PyOdometry
                   increment is used, exactly as in F2. The coordinator must call `set_command(vx, vy, wz)` with the
                   command last passed to `sim.step` before every `update()` (see integration/README.md).
                   `learned_yaw="imu"` (default): heading stays on the IMU path, the net is used for translation only.
                   `learned_yaw="net"`: the net's dyaw also corrects the heading.

All updates are intended at the 50 Hz policy rate (the odometry anchor-switching logic is tick based).
"""
from __future__ import annotations

import colorsys
import hashlib
import importlib.util
import json
import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

# ----------------------------------------------------------------------------------------------- paths
def _root() -> Path:
    return Path(os.environ.get("MICRODUCK_ROOT", str(Path.home() / "microduck"))).expanduser()


ANCHOR_JSON = _root() / "upstream/microduck_rl/scripts/odom_anchor_sets.json"
CONTROL_HZ = 50.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64) / np.linalg.norm(q))
    return m.reshape(3, 3)


def load_anchor_sets(name="alpha4"):
    with open(ANCHOR_JSON) as f:
        sets = json.load(f)["sets"]
    s = sets[name]
    return [np.asarray(s["left"], float), np.asarray(s["right"], float)]


# ----------------------------------------------------------------------------------------------- sensors
@dataclass
class SensorFrame:
    """What the robot itself can sense at one control tick (no ground truth; virtual tactile contacts)."""
    t: float
    q: np.ndarray            # (14,) actuated joint positions
    qd: np.ndarray           # (14,) actuated joint velocities
    imu_quat: np.ndarray     # (4,) wxyz, `orientation` framequat of the imu site (noise applied)
    imu_gyro: np.ndarray     # (3,) rad/s, `angular-velocity` sensor (noise applied), imu frame
    imu_accel: np.ndarray    # (3,) m/s^2, `imu_accel` (specific force), imu frame
    foot_contacts: tuple[bool, bool] | None = None  # simulated binary tactile switches


class SensorReader:
    """Reads SensorFrame from a Sim's MjData by sensor name, applying the noise declared in sensors.xml."""

    def __init__(self, model, joint_qpos_indices, joint_qvel_indices, seed=0, noise=True):
        self.m = model
        self.qi = np.asarray(joint_qpos_indices)
        self.vi = np.asarray(joint_qvel_indices)
        self.rng = np.random.default_rng(seed)
        self.noise = noise

        def adr(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if i < 0:
                raise RuntimeError(f"sensor {name!r} missing from model")
            return int(model.sensor_adr[i]), int(model.sensor_dim[i]), float(model.sensor_noise[i])

        self.s_quat = adr("orientation")
        self.s_gyro = adr("angular-velocity")
        self.s_acc = adr("imu_accel")
        self.feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision") for side in ("left", "right")]
        # Actual geom names vary by model; resolve from the foot body when needed.
        for i, side in enumerate(("left", "right")):
            if self.feet[i] < 0:
                candidates = [g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(side + "_foot") and model.geom_contype[g]]
                if candidates: self.feet[i] = candidates[0]
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.robot_geoms = np.array([model.body_rootid[model.geom_bodyid[g]] == trunk for g in range(model.ngeom)])

    def _get(self, d, spec):
        a, n, sig = spec
        v = d.sensordata[a:a + n].copy()
        if self.noise and sig > 0:
            v = v + self.rng.normal(0.0, sig, n)
        return v

    def read(self, data) -> SensorFrame:
        q = self._get(data, self.s_quat)
        q = q / np.linalg.norm(q)
        contacts = [False, False]
        for c in data.contact[:data.ncon]:
            for i, fg in enumerate(self.feet):
                if fg >= 0 and ((c.geom1 == fg and not self.robot_geoms[c.geom2]) or
                                (c.geom2 == fg and not self.robot_geoms[c.geom1])):
                    contacts[i] = True
        return SensorFrame(t=float(data.time), q=data.qpos[self.qi].copy(), qd=data.qvel[self.vi].copy(),
                           imu_quat=q, imu_gyro=self._get(data, self.s_gyro), imu_accel=self._get(data, self.s_acc), foot_contacts=tuple(contacts))


# ----------------------------------------------------------------------------------------------- kinematics
class RobotKinematics:
    """FK on a scratch MjData: trunk at identity, measured joint angles -> foot sites / camera in the trunk frame."""

    def __init__(self, model, joint_qpos_indices):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qi = np.asarray(joint_qpos_indices)
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.free_adr = int(model.jnt_qposadr[model.body_jntadr[self.trunk]])
        self.foot_sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in ("left", "right")]
        self.imu_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        if min(self.foot_sites) < 0 or self.cam < 0 or self.imu_site < 0:
            raise RuntimeError("model lacks left_foot/right_foot/imu sites or head_camera")
        self.d.qpos[:] = model.qpos0
        a = self.free_adr
        self.d.qpos[a:a + 7] = [0, 0, 0, 1, 0, 0, 0]
        # imu orientation relative to trunk (constant calibration, identity in this model)
        self.R_trunk_imu = quat_to_mat(model.site_quat[self.imu_site])

    def _fk(self, q):
        self.d.qpos[self.qi] = q
        mujoco.mj_kinematics(self.m, self.d)

    def feet_in_trunk(self, q):
        self._fk(q)
        d = self.d
        return [(d.site_xpos[s].copy(), d.site_xmat[s].reshape(3, 3).copy()) for s in self.foot_sites]

    def camera_in_trunk(self, q):
        """(position (3,), rotation (3,3)) of head_camera in the trunk frame (MuJoCo cam axes: x right, y up, -z fwd)."""
        self._fk(q)
        mujoco.mj_camlight(self.m, self.d)          # cam_xpos/xmat are computed here, not in mj_kinematics
        return self.d.cam_xpos[self.cam].copy(), self.d.cam_xmat[self.cam].reshape(3, 3).copy()


# ----------------------------------------------------------------------------------------------- odometry
class PyOdometry:
    """Replica of microduck/odometry `Odometry::update` (== upstream scripts/infer_policy.py PyOdometry).
    Anchor the lowest sole point to the ground, orient the trunk by the IMU, move the anchor when another sole
    point drops below it and holds the claim for SWITCH_CONFIRM_TICKS."""
    SWITCH_MARGIN = -0.010
    SWITCH_CONFIRM_TICKS = 2

    def __init__(self, corners):
        self.corners = corners
        self.reset()

    def reset(self):
        self.anchor_foot = 0
        self.anchor_local = np.zeros(3)
        self.anchor_xy = np.zeros(2)
        self.position = np.zeros(3)
        self.pending = None
        self.pending_ticks = 0
        self.needs_init = True

    def update(self, feet, R, contacts=None):
        if contacts is not None and not any(contacts):
            return
        if self.needs_init:
            if contacts is not None:
                self.anchor_foot = next(i for i, active in enumerate(contacts) if active)
            self.anchor_xy = (R @ feet[self.anchor_foot][0])[:2].copy()
            self.needs_init = False
        if contacts is not None and not contacts[self.anchor_foot]:
            foot = next(i for i, active in enumerate(contacts) if active)
            pos, rot = feet[foot]
            points = self.position + (self.corners[foot] @ rot.T + pos) @ R.T
            idx = int(np.argmin(points[:, 2]))
            self.anchor_foot, self.anchor_local = foot, self.corners[foot][idx].copy()
            self.anchor_xy = points[idx, :2].copy()
        self._reproject(R, feet)
        cand = self._lowest_corner(R, feet, contacts)
        if cand is None:
            self.pending, self.pending_ticks = None, 0
        else:
            if self.pending is not None and self.pending[0] == cand[0]:
                self.pending_ticks += 1
            else:
                self.pending, self.pending_ticks = cand, 1
            if self.pending_ticks >= self.SWITCH_CONFIRM_TICKS:
                foot, local, world_xy = self.pending
                self.pending, self.pending_ticks = None, 0
                self.anchor_foot, self.anchor_local, self.anchor_xy = foot, local, world_xy
                self._reproject(R, feet)

    def _reproject(self, R, feet):
        pos, rot = feet[self.anchor_foot]
        c = R @ (rot @ self.anchor_local + pos)
        self.position = np.array([self.anchor_xy[0] - c[0], self.anchor_xy[1] - c[1], -c[2]])

    def _lowest_corner(self, R, feet, contacts=None):
        lowest, best = -self.SWITCH_MARGIN, None
        for foot, (pos, rot) in enumerate(feet):
            if contacts is not None and not contacts[foot]:
                continue
            world = self.position + (self.corners[foot] @ rot.T + pos) @ R.T
            i = int(np.argmin(world[:, 2]))
            if world[i, 2] < lowest:
                lowest = world[i, 2]
                best = (foot, self.corners[foot][i].copy(), world[i, :2].copy())
        return best


# ----------------------------------------------------------------------------------------------- fiducials
# 6-hue palette (rgb). Each marker = ordered (top, bottom) pair, so 30 distinct IDs.
FID_PALETTE = {
    "R": (0.95, 0.05, 0.05), "Y": (0.98, 0.88, 0.00), "G": (0.00, 0.80, 0.10),
    "C": (0.00, 0.85, 0.90), "B": (0.10, 0.15, 0.95), "M": (0.90, 0.00, 0.85),
}
FID_PATCH = 0.08          # patch side [m]; marker is 0.08 wide x 0.16 tall
FID_Z = 0.20              # marker centre height [m] (head camera is ~0.23 m up)
# (wall geom name, axis the face normal points along, sign, tangent coordinate along the wall)
# Face positions are computed from the wall geoms in apartment.xml (half thickness 0.06). Candidates were checked
# for occlusion by furniture with mj_ray (see tests/test_localization.py::test_marker_layout_unoccluded).
FID_LAYOUT = [
    # office (x .5..4, y -1..1)
    ("wF", "y", -1, 1.2), ("wF", "y", -1, 2.2), ("wF", "y", -1, 3.2),
    ("wG", "y", +1, 1.4), ("wG", "y", +1, 2.4), ("wG", "y", +1, 3.4),
    ("wC_m2", "x", +1, 0.8), ("wC_m1", "x", +1, -0.5),
    # corridor (x -1..0.5)
    ("wB_m", "x", +1, 0.9), ("wB_m", "x", +1, -0.3), ("wC_m1", "x", -1, -1.2),
    ("wall_north", "y", -1, -0.2), ("wall_south", "y", +1, -0.3),
    # bathroom (x .5..4, y -3..-1)
    ("wG", "y", -1, 1.5), ("wG", "y", -1, 3.0), ("wall_south", "y", +1, 2.0), ("wall_south", "y", +1, 3.2),
    # bedroom (y 1..3)
    ("wF", "y", +1, 1.5), ("wF", "y", +1, 3.0), ("wall_north", "y", -1, 2.0),
    # living (x -4..-1, y -3..0.5)
    ("wA_e", "y", -1, -1.6), ("wall_south", "y", +1, -2.2), ("wall_west", "x", +1, -1.0), ("wB_s", "x", -1, -2.4),
    # kitchen
    ("wall_north", "y", -1, -2.5), ("wall_west", "x", +1, 1.5), ("wA_w", "y", +1, -3.5),
]
_COLORS = list(FID_PALETTE)
FID_CODES = [(a, b) for a in _COLORS for b in _COLORS if a != b]        # 30 codes
FID_Z_HALF_SPAN = 2 * FID_PATCH / 2


@dataclass
class Fiducial:
    id: int
    top: str
    bot: str
    center: np.ndarray       # world xyz of the marker centre
    normal: np.ndarray       # outward (room-facing) unit normal


def fiducial_map(model_or_spec_geoms: dict) -> list[Fiducial]:
    """Build the marker map from a {geom_name: (pos, size)} dict of the wall geoms."""
    out = []
    for i, (gname, axis, sign, tcoord) in enumerate(FID_LAYOUT):
        pos, size = model_or_spec_geoms[gname]
        ax = 0 if axis == "x" else 1
        tx = 1 - ax
        face = pos[ax] + sign * (size[ax] + 0.004)
        c = np.zeros(3)
        c[ax], c[tx], c[2] = face, tcoord, FID_Z
        n = np.zeros(3)
        n[ax] = sign
        out.append(Fiducial(i, FID_CODES[i][0], FID_CODES[i][1], c, n))
    return out


def build_fiducial_spec(scene_xml: str):
    """Load the scene via MjSpec, add the marker geoms (visual only), return (spec, markers).
    Upstream XML is not modified. The spec compiles to a model with identical nq/nv/body layout."""
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    walls = {}
    for gname in {g for g, *_ in FID_LAYOUT}:
        g = spec.geom(gname)
        walls[gname] = (np.array(g.pos), np.array(g.size))
    markers = fiducial_map(walls)
    for cname, rgb in FID_PALETTE.items():
        mat = spec.add_material(name=f"fid_{cname}")
        mat.rgba = [*rgb, 1.0]
        mat.emission = 0.6
        mat.specular = 0.0
        mat.shininess = 0.0
        mat.reflectance = 0.0
    for mk in markers:
        n = mk.normal
        # box thin along the normal axis
        if abs(n[0]) > 0.5:
            size = [0.002, FID_PATCH / 2, FID_PATCH / 2]
        else:
            size = [FID_PATCH / 2, 0.002, FID_PATCH / 2]
        for part, dz, col in (("t", +FID_PATCH / 2, mk.top), ("b", -FID_PATCH / 2, mk.bot)):
            pos = mk.center + np.array([0, 0, dz])
            spec.worldbody.add_geom(name=f"fid{mk.id}_{part}", type=mujoco.mjtGeom.mjGEOM_BOX, size=size,
                                    pos=list(pos), material=f"fid_{col}", contype=0, conaffinity=0)
    return spec, markers


def _rgb_to_hsv(img):
    x = img.astype(np.float32) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    df = mx - mn
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.zeros_like(mx)
    nz = df > 1e-6
    rc = nz & (mx == r)
    gc = nz & (mx == g) & ~rc
    bc = nz & ~rc & ~gc
    h[rc] = ((g - b)[rc] / df[rc]) % 6
    h[gc] = (b - r)[gc] / df[gc] + 2
    h[bc] = (r - g)[bc] / df[bc] + 4
    s = np.where(mx > 1e-6, df / np.maximum(mx, 1e-6), 0)
    return h * 60.0, s, mx


@dataclass
class Detection:
    id: int
    u: float            # pair centre, px
    v: float
    h_px: float         # pair vertical extent
    w_px: float
    top: str = ""
    bot: str = ""
    # filled when the camera model is applied
    bearing: float = float("nan")      # rad, horizontal bearing of the camera->marker ray, relative to body heading
    range_h: float = float("nan")      # m, horizontal camera->marker distance
    sigma_b: float = 0.03
    sigma_r: float = 0.3


class FiducialDetector:
    """Colour-pair marker detector for the head_camera image."""
    HUE_TOL = 18.0
    MIN_S, MIN_V = 0.45, 0.25
    MIN_AREA = 6

    def __init__(self):
        self.hues = {c: colorsys.rgb_to_hsv(*rgb)[0] * 360.0 for c, rgb in FID_PALETTE.items()}
        self.code_to_id = {code: i for i, code in enumerate(FID_CODES)}

    def detect(self, img) -> list[Detection]:
        from scipy import ndimage
        H, W = img.shape[:2]
        h, s, v = _rgb_to_hsv(img)
        ok = (s > self.MIN_S) & (v > self.MIN_V)
        blobs = []
        for c, hc in self.hues.items():
            dh = np.abs((h - hc + 180.0) % 360.0 - 180.0)
            mask = ok & (dh < self.HUE_TOL)
            if mask.sum() < self.MIN_AREA:
                continue
            lab, n = ndimage.label(mask)
            if n == 0:
                continue
            for k, sl in enumerate(ndimage.find_objects(lab), start=1):
                ys, xs = sl
                area = int((lab[sl] == k).sum())
                bh, bw = ys.stop - ys.start, xs.stop - xs.start
                if area < self.MIN_AREA or area < 0.55 * bh * bw:      # solid rectangle only
                    continue
                if bh > 3 * bw or bw > 3 * bh:                          # patches are ~square
                    continue
                cy, cx = ndimage.center_of_mass(lab[sl] == k)
                blobs.append(dict(c=c, x0=xs.start, x1=xs.stop, y0=ys.start, y1=ys.stop, area=area,
                                  cx=float(cx + xs.start + 0.5), cy=float(cy + ys.start + 0.5)))
        dets = []
        for a in blobs:                  # a = top blob
            for b in blobs:              # b = bottom blob
                if a is b or a["c"] == b["c"]:
                    continue
                wa, wb = a["x1"] - a["x0"], b["x1"] - b["x0"]
                ha, hb = a["y1"] - a["y0"], b["y1"] - b["y0"]
                if not (0.6 < wa / wb < 1.67 and 0.6 < ha / hb < 1.67):
                    continue
                if abs((a["x0"] + a["x1"]) - (b["x0"] + b["x1"])) / 2 > 0.4 * max(wa, wb):
                    continue
                gap = b["y0"] - a["y1"]
                if gap < -0.3 * ha or gap > 0.35 * ha:
                    continue
                code = (a["c"], b["c"])
                if code not in self.code_to_id:
                    continue
                x0, x1 = min(a["x0"], b["x0"]), max(a["x1"], b["x1"])
                y0, y1 = a["y0"], b["y1"]
                if x0 <= 0 or y0 <= 0 or x1 >= W or y1 >= H:            # clipped by the image border
                    continue
                # Center separation estimates projected marker height without
                # contamination from the width of a tilted bounding box.
                dets.append(Detection(id=self.code_to_id[code], u=(a["cx"] + b["cx"]) / 2.0,
                                      v=(a["cy"] + b["cy"]) / 2.0,
                                      h_px=2.0 * abs(b["cy"] - a["cy"]), w_px=float(x1 - x0),
                                      top=a["c"], bot=b["c"]))
        # one detection per id: keep the largest
        best = {}
        for d in dets:
            if d.id not in best or d.h_px > best[d.id].h_px:
                best[d.id] = d
        return list(best.values())


FIDUCIAL_RENDER_VERSION = 2   # v2: sim->render qpos copied per joint NAME (v1 copied a raw qpos prefix, which
                              # mis-assigned joints when the sim model had extra joints, e.g. training backlash joints)


class FiducialCamera:
    """Render model with markers (a copy of the scene), head_camera renderer, detector. Shareable between Localizers.
    The render model's qpos layout may differ from the sim model's (training dynamics interleave backlash joints), so
    qpos is copied joint-by-joint by name after `bind(sim_model)`. Fails closed if unbound or if any render-model joint
    is missing from the sim model."""

    def __init__(self, scene_xml: str, width=320, height=240):
        self.spec, self.markers = build_fiducial_spec(scene_xml)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.W, self.H = width, height
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        self.fovy = float(self.model.cam_fovy[self.cam_id])
        self.f = (height / 2.0) / math.tan(math.radians(self.fovy) / 2.0)   # px, square pixels
        # The head_camera is mounted rolled by a multiple of 90 deg in the model (image "up" is trunk +-x / +-y).
        # Calibration constant (from FK at the default pose): rotate the image by rot_k*90 deg CCW so world-up is
        # image-up; `rot_theta` is the matching virtual-camera rotation used by the camera model.
        kin = RobotKinematics(self.model, list(range(7, 21)))
        _, Rc0 = kin.camera_in_trunk(self.model.qpos0[7:21])
        up_c = Rc0.T @ np.array([0.0, 0.0, 1.0])
        a = math.degrees(math.atan2(up_c[1], up_c[0]))
        self.rot_k = int(round((90.0 - a) / 90.0)) % 4
        self.rot_theta = math.radians(90.0 * self.rot_k)
        if self.rot_k % 2:
            self.W, self.H = height, width            # virtual (rotated) image size
        self.rw, self.rh = width, height              # raw render size
        self.renderer = None
        self.detector = FiducialDetector()
        self._cache = (None, None, None)
        self.n_renders = 0
        self._bound_model = None
        self._qmap = None   # (render_idx, sim_idx) int arrays

    def bind(self, sim_model):
        """Build the name-based qpos map from `sim_model` into the render model. Idempotent per sim model."""
        if self._bound_model is sim_model:
            return
        width = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4,
                 mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}
        def keys(m):
            # named joints by name; unnamed joints (scene objects' free joints) by "body:<owning body name>"
            out = {}
            for j in range(m.njnt):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                if not name:
                    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j]))
                    name = f"body:{body}" if body else None
                if name is None or name in out:
                    raise ValueError(f"FiducialCamera: cannot key joint {j} uniquely ({name!r})")
                out[name] = j
            return out
        rk, sk = keys(self.model), keys(sim_model)
        r_idx, s_idx, missing = [], [], []
        for name, j in rk.items():
            sj = sk.get(name, -1)
            if sj < 0:
                missing.append(name)
                continue
            w = width[int(self.model.jnt_type[j])]
            if int(sim_model.jnt_type[sj]) != int(self.model.jnt_type[j]):
                raise ValueError(f"FiducialCamera: joint {name!r} type differs between render and sim models")
            ra, sa = int(self.model.jnt_qposadr[j]), int(sim_model.jnt_qposadr[sj])
            r_idx += range(ra, ra + w); s_idx += range(sa, sa + w)
        if missing:
            raise ValueError(f"FiducialCamera: render-model joints missing from sim model: {missing}")
        self._qmap = (np.asarray(r_idx, int), np.asarray(s_idx, int))
        self._bound_model = sim_model
        self._cache = (None, None, None)

    def _set_qpos(self, sim_data):
        if self._qmap is None:
            raise RuntimeError("FiducialCamera.bind(sim_model) must be called before rendering")
        r, s = self._qmap
        self.data.qpos[r] = sim_data.qpos[s]
        mujoco.mj_forward(self.model, self.data)

    def render(self, sim_data):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.rh, self.rw)
        self._set_qpos(sim_data)
        self.renderer.update_scene(self.data, camera="head_camera")
        self.n_renders += 1
        return np.ascontiguousarray(np.rot90(self.renderer.render(), self.rot_k))

    def detect(self, sim_data) -> list[Detection]:
        """Render + detect, cached per sim time so several Localizers share one render."""
        t = float(sim_data.time)
        if self._cache[0] == t and self._cache[1] is sim_data and np.array_equal(self._cache_qpos, sim_data.qpos):
            return self._cache[2]
        dets = self.detector.detect(self.render(sim_data))
        self._cache = (t, sim_data, dets)
        self._cache_qpos = sim_data.qpos.copy()
        return dets

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # --- ground-truth visibility (evaluation only) -------------------------------------------------
    def gt_visible(self, sim_data, max_range=3.5, min_incidence_cos=0.35):
        """IDs of markers truly in view (centre in image, unoccluded, facing the camera). Evaluation only."""
        self._set_qpos(sim_data)
        m, d = self.model, self.data
        cp = d.cam_xpos[self.cam_id].copy()
        Rc = d.cam_xmat[self.cam_id].reshape(3, 3)
        trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        out = {}
        for mk in self.markers:
            v = mk.center - cp
            dist = float(np.linalg.norm(v))
            if dist > max_range or dist < 0.15:
                continue
            u = v / dist
            if float(-u @ mk.normal) < min_incidence_cos:
                continue
            pc = rz(self.rot_theta) @ (Rc.T @ v)      # virtual (upright) camera frame
            if pc[2] >= -1e-6:
                continue
            px = self.W / 2 + self.f * pc[0] / -pc[2]
            py = self.H / 2 - self.f * pc[1] / -pc[2]
            hpx = self.f * 0.16 / (-pc[2])
            full = (hpx / 2 + 1 < py < self.H - hpx / 2 - 1) and (hpx / 4 + 1 < px < self.W - hpx / 4 - 1)
            if not (0 < px < self.W and 0 < py < self.H):
                continue
            # occlusion: ray towards the marker must reach it (skip robot bodies)
            gid = np.zeros(1, dtype=np.int32)
            start = cp.copy()
            clear, left = True, dist
            for _ in range(6):
                dd = mujoco.mj_ray(m, d, start, u, None, 1, -1, gid)
                if dd < 0 or dd > left - 0.012:
                    break
                g = int(gid[0])
                if g >= 0 and m.geom_bodyid[g] != 0 and m.body_rootid[m.geom_bodyid[g]] == trunk:
                    start = start + u * (dd + 1e-3)
                    left -= dd + 1e-3
                    continue
                clear = False
                break
            if clear:
                out[mk.id] = dict(dist=dist, full=bool(full), h_px=float(hpx), u=float(px), v=float(py))
        return out


# ----------------------------------------------------------------------------------------------- Localizer
@dataclass
class LocalizerParams:
    anchor_set: str = "alpha4"
    heading_source: str = "imu_quat"        # "imu_quat" | "gyro"
    sensor_noise: bool = True               # apply sensors.xml noise
    yaw_bias_dps_std: float = 0.0           # degraded IMU yaw: per-episode constant bias ~ N(0, std) [deg/s]
    camera_hz: float = 5.0
    # EKF
    q_pos: float = 0.0004                    # position process noise [m^2 per m travelled]
    q_pos_floor: float = 1e-8               # [m^2 per tick]
    q_head: float = 1.5e-5                  # heading random walk [rad^2 / s]
    q_head_trav: float = 0.0                # heading noise per rad rotated [rad^2/rad]
    p0_pos: float = 0.01                    # initial position std [m]
    p0_head: float = 0.02                   # initial heading std [rad]
    sigma_bearing: float = math.radians(5.0)
    sigma_range_rel: float = 0.15
    sigma_range_px: float = 1.5             # px of apparent-height noise
    gate_chi2: float = 9.21                 # 2-dof 99%
    use_range: bool = True
    min_hpx_range: float = 12.0             # use range only when the marker is big enough
    max_hpx_use: float = 400.0


# ----------------------------------------------------------------------------------------------- learned odometry (F3)
LEARNED_MODE = "learned+fiducial"
LEARNED_HORIZON_TICKS = 5                     # net horizon: 0.1 s at CONTROL_HZ
LEARNED_LOGVAR_RANGE = (-18.0, 8.0)           # clip of the predicted log-variance (numerical guard only)
LEARNED_OUTPUT = ["dx", "dy", "dyaw", "logvar_dx", "logvar_dy", "logvar_dyaw"]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LearnedOdometry:
    """ONNX learned odometry net + feature ring buffer (F3 mode only).

    model_dir holds odom_net.onnx, odom_net_meta.json and features.py (the interface in
    reports/dev/v5/A3/MODEL_INTERFACE.md). Fails closed with RuntimeError if a file is missing, if the ONNX
    sha256 differs from `expected_sha256` (required, no default), or if the meta file disagrees with the ONNX.
    """

    def __init__(self, model_dir, expected_sha256: str | None):
        import onnxruntime as ort          # lazy: the existing modes never import onnxruntime
        d = Path(model_dir).expanduser()
        onnx_path, meta_path, feat_path = d / "odom_net.onnx", d / "odom_net_meta.json", d / "features.py"
        for p in (onnx_path, meta_path, feat_path):
            if not p.is_file():
                raise RuntimeError(f"learned odometry: missing model file {p}")
        if not expected_sha256:
            raise RuntimeError("learned odometry: learned_sha256 is required (fail closed)")
        digest = _sha256_file(onnx_path)
        if digest != str(expected_sha256).lower():
            raise RuntimeError(f"learned odometry: odom_net.onnx sha256 {digest} != expected {expected_sha256}")
        meta = json.loads(meta_path.read_text())
        if meta.get("onnx_sha256") != digest:
            raise RuntimeError(f"learned odometry: meta onnx_sha256 {meta.get('onnx_sha256')} != file {digest}")
        if "features_py_sha256" in meta and _sha256_file(feat_path) != meta["features_py_sha256"]:
            raise RuntimeError("learned odometry: features.py sha256 differs from meta")
        if int(meta.get("hz", CONTROL_HZ)) != int(CONTROL_HZ):
            raise RuntimeError(f"learned odometry: model hz {meta.get('hz')} != {CONTROL_HZ}")
        if round(float(meta.get("horizon_s", 0.1)) * CONTROL_HZ) != LEARNED_HORIZON_TICKS:
            raise RuntimeError(f"learned odometry: model horizon_s {meta.get('horizon_s')} != 5 ticks")
        if list(meta.get("output", [])) != LEARNED_OUTPUT:
            raise RuntimeError(f"learned odometry: unexpected output order {meta.get('output')}")
        self.window = int(meta["window"])
        if self.window < LEARNED_HORIZON_TICKS + 1:
            raise RuntimeError(f"learned odometry: window {self.window} too short")
        self.n_features = len(meta["features"])
        self.mean = np.asarray(meta["norm_mean"], np.float32)
        self.std = np.asarray(meta["norm_std"], np.float32)
        if self.mean.shape != (self.n_features,) or self.std.shape != (self.n_features,):
            raise RuntimeError("learned odometry: norm_mean/norm_std length != features")
        spec = importlib.util.spec_from_file_location(f"learned_features_{digest[:16]}", feat_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.features_fn = mod.features_from_frame
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        if [i.name for i in self.session.get_inputs()] != ["x"] or [o.name for o in self.session.get_outputs()] != ["y"]:
            raise RuntimeError("learned odometry: ONNX must have input 'x' and output 'y'")
        self.sha256 = digest
        self.clear()

    def clear(self):
        self.feat = deque(maxlen=self.window)
        self.theta = deque(maxlen=LEARNED_HORIZON_TICKS + 1)     # heading at the start of the last horizon is theta[0]
        self.dxy = deque(maxlen=LEARNED_HORIZON_TICKS)           # odometry increments of the last horizon
        self.dth = deque(maxlen=LEARNED_HORIZON_TICKS)

    @property
    def full(self) -> bool:
        return len(self.feat) >= self.window

    def push(self, frame, cmd, theta):
        f = np.asarray(self.features_fn(frame, cmd), np.float64).reshape(-1)
        if f.shape != (self.n_features,) or not np.all(np.isfinite(f)):
            raise RuntimeError(f"learned odometry: feature vector shape {f.shape} != ({self.n_features},) or non-finite")
        self.feat.append(f)
        self.theta.append(float(theta))

    def push_increment(self, dxy, dth):
        self.dxy.append(np.asarray(dxy, float).copy())
        self.dth.append(float(dth))

    def predict(self):
        """-> (dx, dy, dyaw) body frame over the last horizon, and clipped log-variances (3,)."""
        x = (np.stack(self.feat).astype(np.float32) - self.mean) / self.std
        y = np.asarray(self.session.run(["y"], {"x": x[None].astype(np.float32)})[0], np.float64).reshape(-1)
        if y.shape != (6,) or not np.all(np.isfinite(y)):
            raise RuntimeError(f"learned odometry: bad net output {y}")
        return y[:3], np.clip(y[3:], *LEARNED_LOGVAR_RANGE)


class Localizer:
    MODES = ("gt", "odom", "odom+fiducial", LEARNED_MODE)

    def __init__(self, sim, mode="odom", params: LocalizerParams | None = None, seed: int = 0, camera: FiducialCamera | None = None,
                 marker_scene: str | None = None, learned_model_dir: str | None = None, learned_sha256: str | None = None,
                 learned_yaw: str = "imu", **kw):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.sim = sim
        self.mode = mode
        self.p = params or LocalizerParams(**kw)
        if params is not None and kw:
            raise ValueError("pass either params or keyword overrides")
        self.seed = seed
        m = sim.model
        self.reader = SensorReader(m, sim.joint_qpos_indices, sim.joint_qvel_indices, seed=seed + 1000,
                                   noise=self.p.sensor_noise)
        self.kin = RobotKinematics(m, sim.joint_qpos_indices)
        self.odom = PyOdometry(load_anchor_sets(self.p.anchor_set))
        self.rng = np.random.default_rng(seed + 2000)
        self.camera = camera
        # F3 (learned+fiducial) only. Other modes never reach this code and keep active_look False.
        self.active_look = mode == LEARNED_MODE
        self.cmd = (0.0, 0.0, 0.0)
        self.learned_stats = dict(net=0, fallback=0)
        self.last_net = None                  # (dx, dy, dyaw, logvar[3]) of the last net step (diagnostics)
        self._learned = None
        self._learned_yaw = learned_yaw
        if mode == LEARNED_MODE:
            if learned_model_dir is None:
                raise ValueError("mode 'learned+fiducial' requires learned_model_dir")
            if learned_yaw not in ("imu", "net"):
                raise ValueError("learned_yaw must be 'imu' or 'net'")
            self._learned = LearnedOdometry(learned_model_dir, learned_sha256)
        if mode in ("odom+fiducial", LEARNED_MODE) and camera is None:
            self.camera = FiducialCamera(marker_scene or sim.scene_xml)
        if self.camera is not None:
            self.camera.bind(m)
        self.n_cam_ticks = 0
        self.stats = dict(det=0, used=0, gated=0, upd_range=0)
        self.last_dets: list[Detection] = []
        self.pose = np.zeros(3)
        self._ticks = 0
        self._cam_period = max(1, int(round(CONTROL_HZ / self.p.camera_hz)))
        self.reset(np.zeros(3))

    # ---------------------------------------------------------------- command (F3)
    def set_command(self, vx, vy, wz):
        """Body-frame command last passed to sim.step (m/s, m/s, rad/s). The episode runner must call this before
        every update(). It is a no-op unless mode == 'learned+fiducial'."""
        if self._learned is None:
            return
        self.cmd = (float(vx), float(vy), float(wz))

    # ---------------------------------------------------------------- reset
    def reset(self, pose2, sensors: SensorFrame | None = None):
        """Set the world pose of the trunk (x, y, theta) and re-initialise odometry from current sensors."""
        pose2 = np.asarray(pose2, float).copy()
        self.pose0 = pose2
        self.pose = pose2.copy()
        self._ticks = 0
        self.odom.reset()
        self.stats = dict(det=0, used=0, gated=0, upd_range=0)
        self.bias = math.radians(self.rng.normal(0.0, self.p.yaw_bias_dps_std)) if self.p.yaw_bias_dps_std > 0 else 0.0
        self._gyro_yaw = None
        self._prev_world_xy = None
        self._prev_yaw_h = None
        self._t0 = None
        self.P = np.diag([self.p.p0_pos ** 2, self.p.p0_pos ** 2, self.p.p0_head ** 2])
        if self._learned is not None:
            self._learned.clear()
            self.learned_stats = dict(net=0, fallback=0)
            self.last_net = None
        self.last_sensor_frame = sensors if sensors is not None else self.reader.read(self.sim.data)
        if self.mode != "gt":
            fr = self.last_sensor_frame
            xy, theta = self._process_odom(fr, first=True)
            self._prev_world_xy, self._prev_theta = xy.copy(), theta
        return self.pose.copy()

    # ---------------------------------------------------------------- core odometry step
    def _heading(self, fr: SensorFrame, R_imu):
        """Heading used by the estimator (IMU yaw, optionally gyro-integrated / bias-degraded)."""
        psi_q = yaw_of(R_imu)
        if self._t0 is None:
            self._t0 = fr.t
            self._gyro_yaw = psi_q
            self._last_t = fr.t
        if self.p.heading_source == "gyro":
            dt = fr.t - self._last_t
            wz_world = float((R_imu @ fr.imu_gyro)[2])      # R_imu here = trunk-in-world; imu frame == trunk frame
            self._gyro_yaw += wz_world * dt
            psi = self._gyro_yaw
        else:
            psi = psi_q
        self._last_t = fr.t
        psi = psi + self.bias * (fr.t - self._t0)
        return psi_q, psi

    def _process_odom(self, fr: SensorFrame, first=False):
        R_imu = quat_to_mat(fr.imu_quat)                    # IMU-world <- imu frame
        R_tr = R_imu @ self.kin.R_trunk_imu.T               # trunk-in-world (calibration identity here)
        psi_q, psi = self._heading(fr, R_tr)
        R_used = rz(psi - psi_q) @ R_tr
        feet = self.kin.feet_in_trunk(fr.q)
        self.odom.update(feet, R_used, fr.foot_contacts)
        if first:
            self._off = self.pose0[2] - psi
            self._Roff = rz(self._off)
            self._pos_odom0 = self.odom.position[:2].copy()
        xy_odom = self._Roff[:2, :2] @ (self.odom.position[:2] - self._pos_odom0) + self.pose0[:2]
        theta = wrap(psi + self._off)
        self._R_tr, self._psi = R_tr, psi
        return xy_odom, theta

    # ---------------------------------------------------------------- update
    def update(self, sim_obs=None, gt_for_eval_only=None):
        """sim_obs: the Sim (default: the bound sim), or a SensorFrame (no camera then). Returns pose2 estimate."""
        if isinstance(sim_obs, SensorFrame):
            fr, data = sim_obs, None
        else:
            sim = sim_obs if sim_obs is not None else self.sim
            data = sim.data
            fr = self.reader.read(data)
        self.last_sensor_frame = fr
        if self.mode == "gt":
            gt = gt_for_eval_only if gt_for_eval_only is not None else self.sim.ground_truth()
            self.pose = np.asarray(gt.pose2, float).copy()
            return self.pose.copy()
        xy_odom, theta_odom = self._process_odom(fr)
        self._ticks += 1
        if self.mode == "odom":
            self.pose = np.array([xy_odom[0], xy_odom[1], theta_odom])
            return self.pose.copy()
        # ---- EKF predict (F1/F2: odometry increments; F3: learned net, PyOdometry fallback until the window is full)
        if self._learned is not None:
            self._learned_predict(fr, xy_odom, theta_odom)
        else:
            self._odom_predict(xy_odom, theta_odom)
        # ---- fiducial update at camera rate
        if data is not None and self._ticks % self._cam_period == 0:
            self.n_cam_ticks += 1
            dets = self.camera.detect(data)
            self.last_dets = self._apply_camera_model(dets, fr)
            for det in self.last_dets:
                self._ekf_update(det, fr)
        return self.pose.copy()

    # ---------------------------------------------------------------- EKF predict helpers
    def _odom_increment(self, xy_odom, theta_odom):
        """Move pose by the odometry increment (same operations as the original inline predict). Returns (dxy, dth)."""
        if self._prev_world_xy is None:
            self._prev_world_xy, self._prev_theta = xy_odom.copy(), theta_odom
            self.pose = np.array([*xy_odom, theta_odom])
        dxy = xy_odom - self._prev_world_xy
        dth = wrap(theta_odom - self._prev_theta)
        self._prev_world_xy, self._prev_theta = xy_odom.copy(), theta_odom
        self.pose[:2] += dxy
        self.pose[2] = wrap(self.pose[2] + dth)
        return dxy, dth

    def _odom_cov(self, dxy, dth):
        """Per-tick odometry process noise (F1/F2, and the F3 fallback)."""
        dist = float(np.hypot(*dxy))
        Q = np.diag([self.p.q_pos * dist + self.p.q_pos_floor, self.p.q_pos * dist + self.p.q_pos_floor,
                     self.p.q_head / CONTROL_HZ + self.p.q_head_trav * abs(dth)])
        self.P = self.P + Q

    def _odom_predict(self, xy_odom, theta_odom):
        dxy, dth = self._odom_increment(xy_odom, theta_odom)
        self._odom_cov(dxy, dth)

    def _learned_predict(self, fr, xy_odom, theta_odom):
        """F3 predict. Each tick: push the frame features (with the last command) and the odometry increment.
        Before the window is full: the PyOdometry increment and its noise (identical to F2).
        Every LEARNED_HORIZON_TICKS ticks after that: the net's body-frame displacement over the last horizon
        replaces the odometry translation of those ticks (the odometry increments are removed and the net
        displacement, rotated by the horizon-start heading, is added). Its rotated variance is the process noise."""
        L, H = self._learned, LEARNED_HORIZON_TICKS
        L.push(fr, self.cmd, theta_odom)
        dxy, dth = self._odom_increment(xy_odom, theta_odom)
        L.push_increment(dxy, dth)
        if not L.full:
            self.learned_stats["fallback"] += 1
            self._odom_cov(dxy, dth)
            return
        if self._ticks % H != 0:
            return                                    # between net steps the odometry increment drives the pose
        (dx, dy, dyaw), lv = L.predict()
        self.learned_stats["net"] += 1
        self.last_net = (dx, dy, dyaw, lv.copy())
        theta0 = L.theta[0]                           # heading at the start of the horizon (IMU path)
        c, s = math.cos(theta0), math.sin(theta0)
        R0 = np.array([[c, -s], [s, c]])
        acc_xy = np.sum(np.asarray(L.dxy), axis=0)     # odometry translation already applied over this horizon
        acc_th = float(sum(L.dth))
        self.pose[:2] += R0 @ np.array([dx, dy]) - acc_xy
        if self._learned_yaw == "net":
            self.pose[2] = wrap(self.pose[2] + dyaw - acc_th)
        floor = H * self.p.q_pos_floor
        Qxy = R0 @ np.diag(np.exp(lv[:2])) @ R0.T + floor * np.eye(2)
        if self._learned_yaw == "net":
            qyaw = float(np.exp(lv[2])) + H * self.p.q_head / CONTROL_HZ
        else:
            qyaw = H * self.p.q_head / CONTROL_HZ + self.p.q_head_trav * abs(acc_th)
        Q = np.zeros((3, 3))
        Q[:2, :2] = Qxy
        Q[2, 2] = qyaw
        self.P = self.P + 0.5 * (Q + Q.T)

    # ---------------------------------------------------------------- camera model
    def _apply_camera_model(self, dets, fr):
        cam_pos, Rc = self.kin.camera_in_trunk(fr.q)
        R_tr = self._R_tr
        Ryf = rz(-yaw_of(R_tr)) @ R_tr                   # trunk frame -> gravity-aligned, yaw-free frame
        self._cam_off_yf = (Ryf @ cam_pos)               # camera offset from trunk origin, yaw-free
        f = self.camera.f
        W, H = self.camera.W, self.camera.H
        out = []
        for d in dets:
            ray_c = np.array([(d.u - W / 2) / f, -(d.v - H / 2) / f, -1.0])
            ray_c /= np.linalg.norm(ray_c)
            R_virtual = Ryf @ Rc @ rz(-self.camera.rot_theta)
            r = R_virtual @ ray_c
            bearing = math.atan2(r[1], r[0])
            # Solve the pinhole projection of the surveyed vertical marker's top and
            # bottom. This includes camera pitch and off-axis perspective. The
            # measured image height is a pixel measurement, not an oracle range.
            vertical_c = R_virtual.T @ np.array([0.0, 0.0, 1.0])
            a = f * (2 * FID_PATCH) * abs(vertical_c[1] * (-ray_c[2]) + ray_c[1] * vertical_c[2])
            q = max(ray_c[2] ** 2, 1e-12)
            b = d.h_px * (FID_PATCH * vertical_c[2]) ** 2
            distance = (a + math.sqrt(a * a + 4 * d.h_px * q * b)) / (2 * d.h_px * q)
            d2 = Detection(**{**d.__dict__})
            d2.bearing = bearing
            d2.range_h = distance * math.hypot(r[0], r[1])
            d2.sigma_b = math.sqrt(self.p.sigma_bearing ** 2 + (1.0 / f) ** 2)
            d2.sigma_r = d2.range_h * math.sqrt(self.p.sigma_range_rel ** 2 + (self.p.sigma_range_px / d.h_px) ** 2)
            out.append(d2)
        return out

    def _h(self, x, m_xy, cam_off_yf, use_range):
        c = x[:2] + rz(x[2])[:2, :2] @ cam_off_yf[:2]
        v = m_xy - c
        z = [wrap(math.atan2(v[1], v[0]) - x[2])]
        if use_range:
            z.append(math.hypot(v[0], v[1]))
        return np.array(z)

    def _ekf_update(self, det: Detection, fr):
        mk = self.camera.markers[det.id]
        m_xy = mk.center[:2]
        self.stats["det"] += 1
        use_range = self.p.use_range and det.h_px >= self.p.min_hpx_range
        x = self.pose.copy()
        z = np.array([det.bearing, det.range_h]) if use_range else np.array([det.bearing])
        h0 = self._h(x, m_xy, self._cam_off_yf, use_range)
        innov = z - h0
        innov[0] = wrap(innov[0])
        J = np.zeros((len(z), 3))
        eps = 1e-5
        for k in range(3):
            xp = x.copy()
            xp[k] += eps
            hp = self._h(xp, m_xy, self._cam_off_yf, use_range)
            dh = hp - h0
            dh[0] = wrap(dh[0])
            J[:, k] = dh / eps
        Rm = np.diag([det.sigma_b ** 2] + ([det.sigma_r ** 2] if use_range else []))
        S = J @ self.P @ J.T + Rm
        chi2 = float(innov @ np.linalg.solve(S, innov))
        gate = self.p.gate_chi2 if use_range else 6.63
        if chi2 > gate:
            self.stats["gated"] += 1
            return
        K = self.P @ J.T @ np.linalg.inv(S)
        dx = K @ innov
        self.pose = x + dx
        self.pose[2] = wrap(self.pose[2])
        I3 = np.eye(3)
        self.P = (I3 - K @ J) @ self.P @ (I3 - K @ J).T + K @ Rm @ K.T      # Joseph form
        self.stats["used"] += 1
        self.stats["upd_range"] += int(use_range)
        # keep the odometry increment chain consistent: the filter pose is the output, odometry feeds increments only

    def close(self):
        if self.camera is not None:
            self.camera.close()
