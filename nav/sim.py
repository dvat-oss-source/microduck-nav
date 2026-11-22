"""Headless, deterministic, step-able Microduck sim (Phase A).

Mirrors the CPU deployment rehearsal in
upstream/microduck_rl/scripts/infer_policy.py (BAM M6 XL330 actuators driven by
bam.mujoco.MujocoController, 61D obs with projected gravity, unified 13D command
block, 50 Hz policy = 4 x 5 ms MuJoCo substeps with bam_ctrl.update() before
each substep). infer_policy.py is NOT imported here (it pulls termios/viewer);
its constants are copied and locked by tests/test_sim.py.

Dependencies: mujoco, bam, onnxruntime, numpy (no macOS-only imports).

The default ``dynamics="infer"`` preserves that deployment loop exactly.
``dynamics="training"`` opts into nominal cb8561f6 backlash morphology,
collision/solver configuration, current limiting, and observation/action
delays. It retains seeded voltage/sag variation but does not reproduce the
complete training domain randomization. See reports/sim_fidelity.md.

Usage:
    sim = Sim("apartment", "walk")
    gt = sim.reset(seed=0, start_pose=(1.5, -0.3, 0.0))
    for _ in range(500):
        gt, obs = sim.step(np.array([0.2, 0.0, 0.0]))
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import policies as _pol

# ---------------------------------------------------------------------------
# Constants copied from infer_policy.py (asserted equal in tests/test_sim.py)
# ---------------------------------------------------------------------------
BAM_MOTOR_NAME = "xl330"
BAM_MODEL = "m6"
BAM_KP_FW = 200.0
BAM_VIN_RANGE = (6.5, 8.2)
BAM_VIN_DROP_GAIN_RANGE = (0.0, 0.2)
BAM_VIN_MIN = 6.0
BAM_MAX_CURRENT = None
BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
BALL_RADIUS = 0.035
DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)
SPAWN_Z = 0.125            # infer_policy.py initial trunk height (non-roller)
TIMESTEP = 0.005
DECIMATION = 4
CONTROL_DT = TIMESTEP * DECIMATION   # 0.02 s, 50 Hz

# Sitting start (microduck_standup_env_cfg: SIT_Z, SITTING_JOINT_OVERRIDES)
SIT_Z = 0.060
SITTING_JOINT_OVERRIDES = {1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0,
                           10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0}

# Fall criteria
FALL_TILT_RAD = math.radians(60.0)
FALL_HEIGHT_M = 0.020      # trunk z minus lowest foot-geom z; standing ~0.110, sitstand sit ~0.031

# Hand-off protocol (switch_policy)
HANDOFF_TILT_RAD = math.radians(5.0)
HANDOFF_JVEL_RAD_S = 0.25  # max |dq| over the 14 servos (settled stance measures ~0.02-0.07)
HANDOFF_SETTLE_STEPS = 5   # both conditions must hold for 5 consecutive steps (0.1 s)
HANDOFF_TIMEOUT_S = 3.0

# --- dynamics="training" only (never used by the infer path) ------------------------------------
# Bump whenever training-mode physics changes: envelopes / primitives / follower gates / skill
# contracts / traces record sim.provenance()["dynamics_version"] and must be re-measured on change.
TRAINING_DYNAMICS_VERSION = "cb8561f6_nominal_v2"
# Terrain contact softening of the training env (play cfg of Mjlab-Velocity2-Rough-Backlash):
# upstream/rl_cb8561f6/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:151-169
# (_soften_terrain_contacts, installed as cfg.scene.spec_fn at :778 when rough=True) sets these on
# every geom of the static "terrain" body; nav applies them to every static worldbody geom.
# Only solref/solimp change there (friction untouched). Robot feet have priority 1, so foot-floor
# contacts keep the foot's parameters; the softening acts on non-foot robot / floor contacts.
TERRAIN_SOLREF = (0.04, 1.0)
TERRAIN_SOLIMP = (0.85, 0.95, 0.001, 0.5, 2.0)

SCENE_DIR_REL = "upstream/microduck_rl/src/mjlab_microduck/robot/microduck"
SCENE_ALIASES = {
    "apartment": "scene_apartment.xml",
    "flat": "scene.xml",
    "ball": "scene_ball.xml",
}


def scene_path(scene_xml: str) -> Path:
    name = SCENE_ALIASES.get(scene_xml, scene_xml)
    p = Path(name).expanduser()
    if p.is_absolute() and p.exists():
        return p
    cand = _pol.microduck_root() / SCENE_DIR_REL / name
    if cand.exists():
        return cand
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(f"scene {scene_xml!r} not found (tried {cand})")


def yaw_from_quat(q) -> float:
    qw, qx, qy, qz = q
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class GroundTruth:
    t: float
    pose2: np.ndarray           # (x, y, theta) world, theta = trunk yaw
    pos3: np.ndarray            # trunk_base freejoint position (world)
    quat: np.ndarray            # trunk_base orientation (w, x, y, z)
    lin_vel_body: np.ndarray    # trunk linear velocity in trunk frame
    ang_vel_body: np.ndarray    # trunk angular velocity in trunk frame
    tilt_rad: float             # angle between trunk z axis and world z
    qpos: np.ndarray
    qvel: np.ndarray
    foot_contacts: dict = field(default_factory=dict)   # {"left": bool, "right": bool}
    fallen: bool = False
    # --- additions beyond CONTRACTS.md (non-breaking) ---
    height: float = 0.0         # trunk z minus lowest foot-geom center z
    policy: str = ""            # active policy key


def _load_bam_model(vin: float):
    from bam.model import load_model
    bm = load_model(motor_name=BAM_MOTOR_NAME, model=BAM_MODEL)
    bm.actuator.kp = BAM_KP_FW
    bm.actuator.vin = vin
    bm.actuator.max_current = BAM_MAX_CURRENT
    return bm


def _add_ball(spec: mujoco.MjSpec):
    """Same body as robot/microduck/ball.xml (70 mm / 15 g hollow ball)."""
    b = spec.worldbody.add_body(name="ball", pos=[0.3, 0.0, 0.035])
    b.add_freejoint(name="ball_free")
    b.mass = 0.015
    b.inertia = [1.225e-5, 1.225e-5, 1.225e-5]
    b.explicitinertial = True
    b.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.035, 0, 0],
               rgba=[1, 0.55, 0, 1], friction=[0.5, 0.005, 0.0001])


def training_frictionloss(ctl, model, data) -> np.ndarray:
    """BAM friction budget exactly as the mjlab training actuator computes it (dynamics="training").

    Port of bam.mjlab.BamActuator.compute / _compute_friction_budget
    (.venv-ref/lib/python3.13/site-packages/bam/mjlab.py:425-481 budget, :580-676 compute), which differs
    from bam.mujoco.MujocoController.update (used unchanged by the infer path) in three ways:
      1. the DOF-friction constraint force is gathered by DOF index (efc_id of a
         mjCNSTR_FRICTION_DOF row is a DOF id; MujocoController compares it to joint ids);
      2. external torque omits qfrc_constraint: under mujoco_warp 3.8.1 with the sparse Jacobian
         mjlab uses, d.qfrc_constraint is zeroed for every world whose solver converged before the
         last iteration (mujoco_warp/_src/solver.py:1955-1985 + :2187-2195: zero_() then the
         scatter kernel returns early for ctx.done worlds), so in a batched
         training run BAM reads 0 there (measured: 0 in 71% of world-substeps with only 4 worlds,
         every world but the slowest one);
      3. the m6 quadratic term has no sign gate (drive / backdrive masks only).
    Returns dof_frictionloss for ctl.dof_indexes.
    """
    b = ctl.model
    di = ctl.dof_indexes
    sel = data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
    qfrc_fric = np.zeros(model.nv)
    np.add.at(qfrc_fric, data.efc_id[sel], data.efc_force[sel])
    ext = -data.qfrc_bias[di] - qfrc_fric[di]
    mot = data.qfrc_actuator[di]
    dq = data.qvel[di]
    sc = np.exp(-np.power(np.abs(dq) / b.dtheta_stribeck.value, b.alpha.value))
    fl = b.friction_base.value + sc * b.friction_stribeck.value
    fl = fl + np.abs(ext * b.load_friction_external.value - mot * b.load_friction_motor.value)
    fl = fl + sc * np.abs(ext * b.load_friction_external_stribeck.value - mot * b.load_friction_motor_stribeck.value)
    drive = (np.abs(mot) > np.abs(ext)).astype(float)
    fl = fl + sc * (drive * b.load_friction_external_quad.value * ext ** 2
                    + (1.0 - drive) * b.load_friction_motor_quad.value * mot ** 2)
    return fl


def build_model(xml_path: Path, vin: float, add_ball: bool = False, dynamics: str = "infer"):
    """Port of infer_policy.load_mujoco_with_bam (+ optional ball body).

    Position actuators -> torque motors with voltage-bounded forcerange, joint
    damping/frictionloss zeroed, stiff friction constraint; armature set by
    MujocoController.
    """
    from bam.mujoco import MujocoController  # noqa: F401  (import check)

    bam_model = _load_bam_model(vin)
    kt, R = bam_model.kt.value, bam_model.R.value
    force_limit = bam_model.actuator.vin * kt / R
    if dynamics == "training":
        # cb8561f6 training morphology, retaining the requested scene geometry.
        import xml.etree.ElementTree as ET
        root = ET.parse(xml_path).getroot()
        robot_dir = _pol.microduck_root() / "upstream/rl_cb8561f6/src/mjlab_microduck/robot/microduck"
        for inc in root.findall("include"):
            name = inc.get("file")
            inc.set("file", str(robot_dir / "robot_walk_backlash.xml") if name.startswith("robot")
                    else str(xml_path.parent / name))
        for key in root.findall("keyframe"):
            root.remove(key)  # deployment keyframes have 14 fewer qpos entries
        ET.SubElement(root, "compiler", meshdir=str(robot_dir / "assets"))
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
        # FULL_COLLISION in robot constants, including mjlab's unnamed-geom behaviour.
        # Scene geoms are preserved; robot descendants are selected by parent chain.
        robot_geoms = []
        for g in spec.geoms:
            b = g.parent
            while b is not None and getattr(b, "name", "world") != "world":
                if b.name == "trunk_base":
                    robot_geoms.append(g)
                    break
                b = getattr(b, "parent", None)
        subset = {g.name for g in robot_geoms if g.name.endswith("_collision")}
        for g in robot_geoms:
            if g.name in subset:
                g.contype = g.conaffinity = 1
                g.condim = 3 if g.name in ("left_foot_collision", "right_foot_collision") else 1
                g.priority = int(g.name in ("left_foot_collision", "right_foot_collision"))
                if g.priority:
                    g.friction[0] = 1.0
        for name in {g.name for g in robot_geoms} - subset:
            g = spec.geom(name)
            g.contype = g.conaffinity = 0
        bam_model.actuator.max_current = 1.75
        force_limit = BAM_VIN_RANGE[1] * kt / R
        for g in spec.worldbody.geoms:          # _soften_terrain_contacts (see TERRAIN_SOLREF)
            g.solref = list(TERRAIN_SOLREF)
            g.solimp = list(TERRAIN_SOLIMP)
    else:
        spec = mujoco.MjSpec.from_file(str(xml_path))
    if add_ball and not any(b.name == "ball" for b in spec.bodies):
        _add_ball(spec)
    names = []
    for act in spec.actuators:
        tgt = act.target
        tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
        if tgt_name.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for joint in spec.joints:
            if joint.name == tgt_name:
                joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0
                joint.solref_friction = BAM_STIFF_SOLREF_FRICTION
                joint.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
                break
    model = spec.compile()
    model.opt.timestep = TIMESTEP
    if dynamics == "training":
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 30
        model.opt.tolerance = 1e-8
        model.opt.ls_iterations = model.opt.ccd_iterations = 50
    return model, bam_model, names, kt, R


class Sim:
    """One deterministic MuJoCo + BAM + ONNX-policy instance.

    Public attributes (read-only for other modules): model, data, t, policy_key,
    lock_entries_used, params (per-episode randomization), trace (list of qpos).
    """

    def __init__(self, scene_xml: str = "apartment", policy_key: str = "walk",
                 lock_path=_pol.DEFAULT_LOCK_PATH, ball: bool = False, record: bool = True,
                 allow_rejected: bool = False, dynamics: str = "infer"):
        from bam.mujoco import MujocoController

        self.scene_xml = str(scene_path(scene_xml))
        self.lock_path = Path(lock_path)
        self._lock = _pol.load_lock(self.lock_path)
        self.initial_policy_key = policy_key
        self.record = record
        self.allow_rejected = allow_rejected
        if dynamics not in ("infer", "training"):
            raise ValueError("dynamics must be 'infer' or 'training'")
        self.dynamics = dynamics
        self._ball_requested = ball
        nominal_vin = 7.4
        self.model, self.bam_model, self._act_names, self._kt, self._R = build_model(
            Path(self.scene_xml), nominal_vin, add_ball=ball, dynamics=dynamics)
        self.data = mujoco.MjData(self.model)
        self.bam_ctrl = MujocoController(self.bam_model, self._act_names, self.model, self.data,
                                         vin_drop_gain=None, vin_min=BAM_VIN_MIN)
        if dynamics == "training":
            # Every caller of bam_ctrl.update() (step, skills.make_fallen, ...) gets the mjlab
            # friction budget; the torque/vin-sag part of MujocoController.update is unchanged.
            ctl, m_, d_ = self.bam_ctrl, self.model, self.data
            _bam_update = ctl.update

            def _training_update():
                _bam_update()
                m_.dof_frictionloss[ctl.dof_indexes] = training_frictionloss(ctl, m_, d_)
            ctl.update = _training_update
        m = self.model
        # Pristine copies of the fields BAM rewrites every update (restored on reset).
        self._dof_frictionloss0 = m.dof_frictionloss.copy()
        self._dof_damping0 = m.dof_damping.copy()
        self._act_ids = np.array([m.actuator(n).id for n in self._act_names])

        self.n_joints = m.nu
        self.joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        self.joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        self.backlash_qpos_indices = []
        self.backlash_qvel_indices = []
        if dynamics == "training":
            for i in range(m.nu):
                name = m.joint(m.actuator_trnid[i, 0]).name
                j = m.joint(f"passive_{name}_backlash").id
                self.backlash_qpos_indices.append(int(m.jnt_qposadr[j]))
                self.backlash_qvel_indices.append(int(m.jnt_dofadr[j]))
        self.imu_ang_vel_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")])
        self.trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.fj_qadr = int(m.jnt_qposadr[fj])
        self.fj_vadr = int(m.jnt_dofadr[fj])
        bj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_qadr = int(m.jnt_qposadr[bj]) if bj >= 0 else None
        self.ball_vadr = int(m.jnt_dofadr[bj]) if bj >= 0 else None
        self.robot_body_ids = {b for b in range(m.nbody) if m.body_rootid[b] == self.trunk_id}
        self.robot_geoms = np.array([m.geom_bodyid[g] in self.robot_body_ids for g in range(m.ngeom)])
        self.foot_geoms = {
            side: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
            for side in ("left", "right")
        }
        self.default_pose = DEFAULT_POSE[: self.n_joints]
        # Floor slabs (world, colliding, top within 1 cm of z=0) + planes: allowed spawn contacts.
        mujoco.mj_kinematics(m, self.data)
        self.floor_geoms = set()
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != 0 or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                self.floor_geoms.add(g)
                continue
            lo, hi = self._geom_world_aabb(g)
            if abs(hi[2]) <= 0.011 and lo[2] > -0.05:
                self.floor_geoms.add(g)

        self._sessions: dict = {}
        self.lock_entries_used: dict = {}
        self.policy_key = None
        self._activate(policy_key)

        self.t = 0.0
        self.params: dict = {}
        self.trace: list = []
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.command = np.zeros(13, dtype=np.float32)
        self.policy_steps = 0           # steps since the active policy was started
        self.last_obs = None
        self._reset_done = False

    # ------------------------------------------------------------------ policies
    def _session(self, key: str):
        if key not in self._sessions:
            import onnxruntime as ort
            entry = _pol.verified_entry(key, self.lock_path, self._lock,     # md5 check
                                        allow_rejected=self.allow_rejected)
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(_pol.resolve(entry)), sess_options=so,
                                        providers=["CPUExecutionProvider"])
            dp = entry.get("default_joint_pos")
            if dp is not None and np.max(np.abs(np.asarray(dp) - DEFAULT_POSE)) > 1.5e-3:
                raise _pol.PolicyLockError(f"{key}: ONNX default_joint_pos differs from DEFAULT_POSE")
            self._sessions[key] = (sess, entry)
            self.lock_entries_used[key] = {"path": entry["path"], "md5": entry["md5"], "kind": entry["kind"]}
        return self._sessions[key]

    def _activate(self, key: str):
        sess, entry = self._session(key)
        self.policy_key = key
        self._sess = sess
        self._entry = entry
        self._in_name = entry["input_name"]
        self._out_name = entry["output_name"]
        self.action_scale = float(entry.get("action_scale", 1.0))
        self.command_mode = entry["command_mode"]
        self.policy_steps = 0

    @property
    def policy_entry(self) -> dict:
        return self._entry

    def provenance(self) -> dict:
        """What a run must record: scene + every lock entry used."""
        return {"scene_xml": self.scene_xml, "dynamics": self.dynamics,
                "dynamics_version": TRAINING_DYNAMICS_VERSION if self.dynamics == "training" else "infer_policy_v1",
                "lock_path": str(self.lock_path),
                "policies": dict(self.lock_entries_used), "params": dict(self.params)}

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int, start_pose=None, extra: dict | None = None) -> GroundTruth:
        """Reset to a standing (or sitting) start.

        seed -> np.random.default_rng(seed) draws, in this order:
          vin ~ U(6.5, 8.2) [V], vin_drop_gain ("sag") ~ U(0, 0.2) [V/Nm],
          yaw ~ U(-pi, pi) (drawn always, used only if start_pose is None).
        start_pose: (x, y, theta) or None -> (DEFAULT_SPAWN[scene] xy, seeded random yaw).
        extra keys (all optional):
          vin, vin_drop_gain: override the draws (parity tests use 7.4 / 0.1)
          init: "stand" (default, DEFAULT_POSE, z = floor + 0.125) | "sit" (standup start)
          ball_xy: (x, y) ball position (Sim(ball=True) or a scene with a ball)
          ball_for: "kick_right" | "kick_left" place ball in front of that foot (infer_policy offsets)
          policy: policy key to start with (no hand-off)
          check_spawn: bool (default True) raise if robot spawns in collision / off the floor
        """
        extra = dict(extra or {})
        rng = np.random.default_rng(seed)
        self._rng = rng
        vin = float(rng.uniform(*BAM_VIN_RANGE))
        gain = float(rng.uniform(*BAM_VIN_DROP_GAIN_RANGE))
        yaw_draw = float(rng.uniform(-math.pi, math.pi))
        vin = float(extra.get("vin", vin))
        gain = float(extra.get("vin_drop_gain", gain))
        if start_pose is None:
            x, y = DEFAULT_SPAWN.get(Path(self.scene_xml).name, (0.0, 0.0))
            theta = yaw_draw
        else:
            x, y, theta = (float(v) for v in start_pose)
        init = extra.get("init", "stand")
        self._activate(extra.get("policy", self.initial_policy_key))

        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        m.dof_frictionloss[:] = self._dof_frictionloss0
        m.dof_damping[:] = self._dof_damping0
        # Battery: BAM vin + MuJoCo forcerange exactly as load_mujoco_with_bam builds them for this vin.
        self.bam_model.actuator.vin = vin
        fl = (BAM_VIN_RANGE[1] if self.dynamics == "training" else vin) * self._kt / self._R
        m.actuator_forcerange[self._act_ids, 0] = -fl
        m.actuator_forcerange[self._act_ids, 1] = fl
        self.bam_ctrl.vin_drop_gain = gain if gain > 0 else None
        self.bam_ctrl.last_ts = d.time

        floor_z = self.floor_height(x, y) if extra.get("check_spawn", True) else 0.0
        a = self.fj_qadr
        z0 = floor_z + (SIT_Z if init == "sit" else SPAWN_Z)
        d.qpos[a:a + 3] = [x, y, z0]
        d.qpos[a + 3:a + 7] = [math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)]
        for i, qi in enumerate(self.joint_qpos_indices):
            d.qpos[qi] = self.default_pose[i]
        if init == "sit":
            for idx, ang in SITTING_JOINT_OVERRIDES.items():
                d.qpos[self.joint_qpos_indices[idx]] = ang
            targets = np.array([d.qpos[qi] for qi in self.joint_qpos_indices], dtype=np.float32)
        else:
            targets = self.default_pose
        self.bam_ctrl.reset(d.qpos)
        self.bam_ctrl.q_target[:] = targets
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.command = np.zeros(13, dtype=np.float32)
        if self.ball_qadr is not None:
            if "ball_xy" in extra:
                bx, by = extra["ball_xy"]
            else:
                bx, by = self._ball_in_front(extra.get("ball_for", "kick_right"))
            self.place_ball(bx, by)
        mujoco.mj_forward(m, d)
        if extra.get("check_spawn", True):
            self._check_spawn()

        self.t = float(d.time)
        self.params = {"seed": int(seed), "vin": vin, "vin_drop_gain": gain,
                       "start_pose": [x, y, theta], "init": init, "policy": self.policy_key,
                       "dynamics": self.dynamics}
        self._action_history = []
        self._obs_history = []
        self._obs_cache_t = None
        self._obs_lags = [0, 0]
        self._obs_phase = rng.integers(0, 64, 2) if self.dynamics == "training" else [0, 0]
        self._obs_count = 0
        self.trace = [d.qpos.copy()] if self.record else []
        self.last_obs = None
        self._reset_done = True
        return self.ground_truth()

    def floor_height(self, x: float, y: float) -> float:
        """Top z of the highest colliding non-robot geom whose world AABB contains (x, y).

        Raises ValueError if nothing is under (x, y) or the top is not the main
        floor (|z| > 2 cm: stairs hole, furniture, free objects).
        """
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        best, best_g = None, -1
        for g in range(m.ngeom):
            if self.robot_geoms[g] or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                top = float(d.geom_xpos[g][2])
            else:
                lo, hi = self._geom_world_aabb(g)
                if not (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]):
                    continue
                top = float(hi[2])
            if best is None or top > best:
                best, best_g = top, g
        if best is None:
            raise ValueError(f"no floor under ({x:.2f}, {y:.2f})")
        if abs(best) > 0.02:
            raise ValueError(f"spawn ({x:.2f},{y:.2f}) is not on the main floor: "
                             f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, best_g)} (top z={best:.3f})")
        return best

    def _geom_world_aabb(self, g: int):
        d = self.data
        a = self.model.geom_aabb[g]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * a[3:] + a[:3]
        wc = corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        return wc.min(0), wc.max(0)

    def _check_spawn(self):
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            r1, r2 = self.robot_geoms[c.geom1], self.robot_geoms[c.geom2]
            if r1 != r2 and c.dist < -1e-3:
                other = c.geom2 if r1 else c.geom1
                if other in self.floor_geoms:
                    continue
                raise ValueError("robot spawns in collision with "
                                 f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other)}")

    # ------------------------------------------------------------------ ball
    def _ball_in_front(self, behavior: str):
        a = self.fj_qadr
        x, y = float(self.data.qpos[a]), float(self.data.qpos[a + 1])
        yaw = yaw_from_quat(self.data.qpos[a + 3:a + 7])
        off_y = -BALL_OFFSET_ABS_Y if behavior == "kick_right" else BALL_OFFSET_ABS_Y
        return (x + math.cos(yaw) * BALL_OFFSET_X - math.sin(yaw) * off_y,
                y + math.sin(yaw) * BALL_OFFSET_X + math.cos(yaw) * off_y)

    def place_ball(self, x: float, y: float, for_kick: str | None = None):
        """Teleport the ball (infer_policy._place_ball). for_kick ignores x, y."""
        if self.ball_qadr is None:
            raise RuntimeError("no ball in this model: construct Sim(..., ball=True)")
        if for_kick:
            x, y = self._ball_in_front(for_kick)
        self.data.qpos[self.ball_qadr:self.ball_qadr + 7] = [x, y, BALL_RADIUS, 1, 0, 0, 0]
        self.data.qvel[self.ball_vadr:self.ball_vadr + 6] = 0.0

    def ball_pos(self):
        if self.ball_qadr is None:
            return None
        return self.data.qpos[self.ball_qadr:self.ball_qadr + 3].copy()

    # ------------------------------------------------------------------ obs
    @staticmethod
    def _quat_rotate_inverse(quat, vec):
        w = quat[0]
        xyz = quat[1:4]
        t = np.cross(xyz, vec) * 2
        return vec - w * t + np.cross(xyz, t)

    def observe(self) -> np.ndarray:
        """61D obs for the current data/last_action/command (what the policy sees).

        As in infer_policy, gyro (sensordata) and trunk xquat come from the last
        mj_step's forward pass, i.e. one 5 ms substep behind qpos.
        """
        d = self.data
        ang = d.sensordata[self.imu_ang_vel_adr:self.imu_ang_vel_adr + 3].copy().astype(np.float32)
        quat = d.xquat[self.trunk_id].copy().astype(np.float32)
        grav = self._quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        jp = d.qpos[self.joint_qpos_indices].copy().astype(np.float32) - self.default_pose
        jv = d.qvel[self.joint_qvel_indices].copy().astype(np.float32)
        if self.dynamics == "training":
            jp += d.qpos[self.backlash_qpos_indices].astype(np.float32)
            jv += d.qvel[self.backlash_qvel_indices].astype(np.float32)
            raw = np.concatenate([ang, grav, jv])
            if self._obs_cache_t != d.time:
                self._obs_history.append(raw.copy())
                self._obs_history = self._obs_history[-2:]
                for i in range(2):
                    if (self._obs_count + self._obs_phase[i]) % 64 == 0:
                        self._obs_lags[i] = int(self._rng.integers(0, 2))
                self._obs_count += 1
                self._obs_cache_t = d.time
            ang = self._obs_history[-1 - min(self._obs_lags[0], len(self._obs_history)-1)][:3]
            grav = self._obs_history[-1 - min(self._obs_lags[1], len(self._obs_history)-1)][3:6]
            jv = self._obs_history[0][6:]
        return np.concatenate([ang, grav, jp, jv, self.last_action, self.command]).astype(np.float32)

    def _fill_command(self, command):
        c = np.zeros(13, dtype=np.float32)
        mode = self.command_mode
        if mode == "twist":
            cmd = np.asarray(command, dtype=np.float32).ravel()
            if cmd.size == 13:
                c[:] = cmd
            else:
                c[0:3] = cmd[0:3]
        elif mode == "posture_flag":
            c[0] = float(np.asarray(command).ravel()[0])
        elif mode == "phase":
            ph = self.phase
            c[0] = np.cos(2 * np.pi * ph)
            c[1] = np.sin(2 * np.pi * ph)
            c[2] = 0.0
        # "zero": all zeros
        self.command = c

    @property
    def phase(self) -> float:
        """Phase-encoded policies: infer_policy advances phase by dt/period BEFORE each inference."""
        if self.command_mode != "phase":
            return float("nan")
        return self.policy_steps * CONTROL_DT / float(self._entry.get("period_s", 4.0))

    @property
    def policy_elapsed(self) -> float:
        return self.policy_steps * CONTROL_DT

    @property
    def skill_done(self) -> bool:
        """Episodic policies: True once infer_policy would have handed control back.

        pick: next phase >= end_phase (0.7); kicks/roulade: elapsed >= duration_s.
        Perpetual policies: always False. The caller decides what to switch to.
        """
        e = self._entry
        if self.command_mode == "phase":
            nxt = (self.policy_steps + 1) * CONTROL_DT / float(e.get("period_s", 4.0))
            return nxt >= float(e.get("end_phase", 0.7))
        if "duration_s" in e and e["kind"] != "walk":
            return self.policy_elapsed >= float(e["duration_s"])
        return False

    # ------------------------------------------------------------------ step
    def step(self, command) -> tuple[GroundTruth, np.ndarray]:
        """One 50 Hz policy step. Returns (ground truth AFTER the step, obs fed to the policy)."""
        if not self._reset_done:
            raise RuntimeError("call reset() first")
        self.policy_steps += 1
        self._fill_command(command)
        obs = self.observe()
        action = self._sess.run([self._out_name], {self._in_name: obs.reshape(1, -1)})[0]
        action = action.squeeze(0).astype(np.float32)
        self.last_action = action.copy()
        targets = self.default_pose + action * self.action_scale
        self.bam_ctrl.q_target[:] = targets
        m, d = self.model, self.data
        for _ in range(DECIMATION):
            if self.dynamics == "training":
                self._action_history.append(targets.copy())
                self._action_history = self._action_history[-7:]
                lag = min(int(self._rng.integers(3, 7)), len(self._action_history) - 1)
                # Training reads the encoder after the passive backlash hinge,
                # but motor back-EMF/friction use motor-side speed.
                self.bam_ctrl.q_target[:] = self._action_history[-1-lag] - d.qpos[self.backlash_qpos_indices]
            self.bam_ctrl.update()
            mujoco.mj_step(m, d)
        self.bam_ctrl.q_target[:] = targets
        if self.dynamics == "training":
            mujoco.mj_forward(m, d)
        self.t = float(d.time)
        self.last_obs = obs
        if self.record:
            self.trace.append(d.qpos.copy())
        return self.ground_truth(), obs

    # ------------------------------------------------------------------ hand-off
    def switch_policy(self, key: str, timeout_s: float = HANDOFF_TIMEOUT_S) -> bool:
        """Hand-off protocol.

        1. Keep the CURRENT policy running with its idle command (zero twist;
           posture flag 0; phase/zero policies keep their own command) until
           tilt < 5 deg AND max |joint vel| < 0.5 rad/s for 5 consecutive steps.
        2. last_action := (current motor targets - DEFAULT_POSE) / action_scale of
           the new policy, so the first obs it sees is consistent with the
           targets actually being held (no command jump).
        3. Activate `key` (md5-verified), policy_steps = 0.
        Returns False (and stays on the current policy) if step 1 times out.
        """
        if key == self.policy_key:
            return True
        self._session(key)                      # md5-verify before moving
        n_max = int(round(timeout_s / CONTROL_DT))
        ok_run = 0
        for _ in range(n_max + 1):
            if self.is_settled():
                ok_run += 1
                if ok_run >= HANDOFF_SETTLE_STEPS:
                    break
            else:
                ok_run = 0
            self.step(np.zeros(3))
        else:
            return False
        self.force_switch(key)
        return True

    def is_settled(self) -> bool:
        gt_tilt = self._tilt()
        jv = np.abs(self.data.qvel[self.joint_qvel_indices]).max()
        return gt_tilt < HANDOFF_TILT_RAD and jv < HANDOFF_JVEL_RAD_S

    def force_switch(self, key: str):
        """Swap policy immediately (no settling), keeping targets continuous."""
        _, entry = self._session(key)
        scale = float(entry.get("action_scale", 1.0))
        self.last_action = ((np.asarray(self.bam_ctrl.q_target, dtype=np.float32) - self.default_pose)
                            / np.float32(scale)).astype(np.float32)
        self._activate(key)

    # ------------------------------------------------------------------ ground truth
    def _tilt(self) -> float:
        q = self.data.qpos[self.fj_qadr + 3:self.fj_qadr + 7]
        w, x, y, z = q / np.linalg.norm(q)
        r22 = 1.0 - 2.0 * (x * x + y * y)
        return float(math.acos(max(-1.0, min(1.0, r22))))

    def foot_contacts(self) -> dict:
        d = self.data
        out = {"left": False, "right": False}
        fl, fr = self.foot_geoms["left"], self.foot_geoms["right"]
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            for side, fg in (("left", fl), ("right", fr)):
                if (g1 == fg and not self.robot_geoms[g2]) or (g2 == fg and not self.robot_geoms[g1]):
                    out[side] = True
        return out

    def ground_truth(self) -> GroundTruth:
        m, d = self.model, self.data
        a, v = self.fj_qadr, self.fj_vadr
        pos = d.qpos[a:a + 3].copy()
        quat = d.qpos[a + 3:a + 7].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat / np.linalg.norm(quat))
        R = R.reshape(3, 3)
        lin_body = R.T @ d.qvel[v:v + 3]
        ang_body = d.qvel[v + 3:v + 6].copy()          # freejoint angular velocity is body-frame
        tilt = self._tilt()
        foot_z = min(d.geom_xpos[g][2] for g in self.foot_geoms.values())
        height = float(pos[2] - foot_z)
        fallen = bool(tilt > FALL_TILT_RAD or height < FALL_HEIGHT_M)
        return GroundTruth(
            t=float(d.time),
            pose2=np.array([pos[0], pos[1], yaw_from_quat(quat)]),
            pos3=pos, quat=quat, lin_vel_body=lin_body, ang_vel_body=ang_body,
            tilt_rad=tilt, qpos=d.qpos.copy(), qvel=d.qvel.copy(),
            foot_contacts=self.foot_contacts(), fallen=fallen, height=height,
            policy=self.policy_key or "",
        )


# Default spawn (x, y) per scene when reset(start_pose=None): yaw is randomized.
# Apartment: office centre-west, >= 0.4 m clearance to walls / furniture (see STATUS.md).
DEFAULT_SPAWN = {
    "scene.xml": (0.0, 0.0),
    "scene_ball.xml": (0.0, 0.0),
    "scene_apartment.xml": (2.0, 0.0),
}


def measure_rtf(sim: Sim, n_steps: int = 500, command=(0.2, 0.0, 0.0), seed: int = 0) -> dict:
    sim.reset(seed)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        sim.step(np.asarray(command))
    wall = time.perf_counter() - t0
    return {"steps": n_steps, "wall_s": wall, "sim_s": n_steps * CONTROL_DT,
            "rtf": n_steps * CONTROL_DT / wall, "steps_per_s": n_steps / wall}
