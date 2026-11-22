"""TEST-ONLY minimal MuJoCo kick rollout (does NOT use nav.sim).

Mirrors upstream scripts/infer_policy.py: scene_ball.xml + BAM M6 actuators,
50 Hz policy (4 x 5 ms substeps), obs = [ang_vel, proj_gravity, q-default,
qdot, last_action, zeros(13)], target = DEFAULT_POSE + action.  The ball is
teleported in front of the kicking foot exactly like infer_policy._place_ball.

Used by tests/test_mirror.py and tests/run_mirror_check.py.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

UP = Path.home() / "microduck" / "upstream"
RL = UP / "microduck_rl"
POL = UP / "policies"
SCENE = RL / "src/mjlab_microduck/robot/microduck/scene_ball.xml"
INFER = RL / "scripts" / "infer_policy.py"

_ip = None


def _infer_policy():
    """Import upstream infer_policy.py as a module (read-only; for its BAM loader + constants)."""
    global _ip
    if _ip is None:
        spec = importlib.util.spec_from_file_location("upstream_infer_policy", INFER)
        _ip = importlib.util.module_from_spec(spec)
        sys.modules["upstream_infer_policy"] = _ip
        spec.loader.exec_module(_ip)
    return _ip


def make_env(vin: float = 7.4, vin_drop_gain: float = 0.1):
    ip = _infer_policy()
    bam_model = ip.load_bam_model(ip.BAM_KP_FW, vin, 0.0)
    model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(
        str(SCENE), bam_model, 0.005, vin_drop_gain if vin_drop_gain > 0 else None, ip.BAM_VIN_MIN)
    return model, data, bam_ctrl


def rollout(policy_fn, foot: str, seed: int, ball_offset=(0.0, 0.0), steps: int = 100,
            noise_ball: float = 0.0):
    """Run `policy_fn(obs61)->act14` for `steps` control steps from a standing start.

    foot selects the nominal ball spawn side ("left" -> +y, "right" -> -y).
    ball_offset is added to the nominal spawn (trunk yaw frame, metres).
    Returns dict(obs (T,61), act (T,14), ball_xy (T+1,2) in start trunk frame,
    fell bool, tilt_max, ball_peak_speed).
    """
    ip = _infer_policy()
    rng = np.random.default_rng(seed)
    vin = float(rng.uniform(6.5, 8.2))
    model, data, bam_ctrl = make_env(vin=vin)
    default = ip.DEFAULT_POSE
    fj = model_name2id(model, "joint", "trunk_base_freejoint")
    qadr = int(model.jnt_qposadr[fj])
    vadr = int(model.jnt_dofadr[fj])
    jq = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    jv = [int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    bj = model_name2id(model, "joint", "ball_free")
    bq = int(model.jnt_qposadr[bj])
    bv = int(model.jnt_dofadr[bj])
    import mujoco
    bid_trunk = model_name2id(model, "body", "trunk_base")
    imu = model_name2id(model, "sensor", "imu_ang_vel")
    imu_adr = int(model.sensor_adr[imu])

    data.qpos[qadr:qadr + 3] = [0.0, 0.0, 0.125]
    data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
    for i, q in enumerate(jq):
        data.qpos[q] = default[i]
    bam_ctrl.reset(data.qpos)
    bam_ctrl.q_target[:] = default
    # ball placement (yaw = 0 at start)
    bx0, by0 = (ip.BALL_OFFSET_X, -ip.BALL_OFFSET_ABS_Y if foot == "right" else ip.BALL_OFFSET_ABS_Y)
    bx = bx0 + ball_offset[0] + (rng.uniform(-noise_ball, noise_ball) if noise_ball else 0.0)
    by = by0 + ball_offset[1] + (rng.uniform(-noise_ball, noise_ball) if noise_ball else 0.0)
    data.qpos[bq:bq + 7] = [bx, by, ip.BALL_RADIUS, 1, 0, 0, 0]
    data.qvel[bv:bv + 6] = 0.0
    mujoco.mj_forward(model, data)

    last_action = np.zeros(14, dtype=np.float32)
    O, A, B = [], [], [data.qpos[bq:bq + 2].copy()]
    fell, tilt_max, bpeak = False, 0.0, 0.0
    for _ in range(steps):
        quat = data.xquat[bid_trunk].astype(np.float32)
        grav = ip.PolicyInference.quat_rotate_inverse(None, quat, np.array([0, 0, -1], np.float32))
        obs = np.concatenate([
            data.sensordata[imu_adr:imu_adr + 3].astype(np.float32),
            grav.astype(np.float32),
            (data.qpos[jq] - default).astype(np.float32),
            data.qvel[jv].astype(np.float32),
            last_action,
            np.zeros(13, np.float32),
        ]).astype(np.float32)
        act = np.asarray(policy_fn(obs), dtype=np.float32).reshape(14)
        last_action = act.copy()
        O.append(obs)
        A.append(act)
        bam_ctrl.q_target[:] = default + act
        for _k in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)
        B.append(data.qpos[bq:bq + 2].copy())
        bpeak = max(bpeak, float(np.linalg.norm(data.qvel[bv:bv + 2])))
        tilt = math.acos(float(np.clip(-grav[2], -1, 1)))
        tilt_max = max(tilt_max, tilt)
        if tilt > math.radians(60) or data.qpos[qadr + 2] < 0.06:
            fell = True
    return dict(obs=np.array(O), act=np.array(A), ball_xy=np.array(B), fell=fell,
                tilt_max=tilt_max, ball_peak_speed=bpeak, vin=vin)


def model_name2id(model, kind, name):
    import mujoco
    k = {"joint": mujoco.mjtObj.mjOBJ_JOINT, "body": mujoco.mjtObj.mjOBJ_BODY,
         "sensor": mujoco.mjtObj.mjOBJ_SENSOR}[kind]
    i = mujoco.mj_name2id(model, k, name)
    assert i >= 0, name
    return i
