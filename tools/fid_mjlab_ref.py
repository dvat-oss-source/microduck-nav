"""OFFLINE reference (.venv-ref, Warp on CPU): alpha_walking in Pollen's real mjlab training env
(Mjlab-Velocity2-Rough-Backlash-MicroDuck, code rl_cb8561f6), flat tiles, no pushes, head/body
commands 0, final curricula, per-env command SCHEDULES (piecewise constant).

Usage (from ~/microduck):
  OMP_NUM_THREADS=1 .venv-ref/bin/python nav/tools/fid_mjlab_ref.py <spec.json> <mode> <out.json>
mode: "train_dr"  = every training DR / obs corruption active (as in training, minus pushes)
      "nominal"   = DR off: no obs noise, no IMU misalignment, no encoder bias, foot mu 1.0,
                    no mass/CoM/armature/friction DR; vin 7.4 V, sag 0.1 V/Nm (infer_policy defaults)
                    for every env (forcerange stays at training's 8.2*kt/R). Actuator delay (3-6
                    substeps) and obs delays stay (they are part of the training env, not DR).
spec.json: {"seconds": float, "envs": [{"name":..., "schedule": [[t0, vx, vy, wz], ...]}, ...]}
Timing convention (same as nav/tools/fid_nav.py): the action computed at control step k
(t = 0.02 k) sees the command active at t.
Logs per env at 50 Hz: x, y, yaw, body vx, vy, wz (root link), alive.
"""
import json
import math
import sys
import time

sys.path.insert(0, "upstream/rl_cb8561f6/src")
import numpy as np
import torch

torch.set_num_threads(1)
import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

sys.path.insert(0, ".")
from duckmac.onnx_mlp import NumpyOnnxMLP

spec = json.load(open(sys.argv[1]))
mode = sys.argv[2]
out = sys.argv[3]
policy_path = spec.get("policy", "upstream/policies/alpha_walking.onnx")
seconds = float(spec["seconds"])
envs = spec["envs"]
N = len(envs)
n_steps = int(round(seconds / 0.02))


def cmd_at(sched, t):
    c = sched[0][1:]
    for row in sched:
        if t + 1e-9 >= row[0]:
            c = row[1:]
    return c


CMD = np.array([[cmd_at(e["schedule"], k * 0.02) for e in envs] for k in range(n_steps + 1)], dtype=np.float32)

pol = NumpyOnnxMLP(policy_path)
cfg = load_env_cfg("Mjlab-Velocity2-Rough-Backlash-MicroDuck")
cfg.scene.num_envs = N
g = cfg.scene.terrain.terrain_generator
g.num_rows = g.num_cols = 5
g.curriculum = False
g.seed = 0
g.sub_terrains = {"flat": g.sub_terrains["flat"]}
cfg.events.pop("push_robot")
if mode == "nominal":
    for k in ("foot_friction", "encoder_bias", "randomize_com", "randomize_head_com",
              "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature"):
        cfg.events.pop(k)
    # Curriculum terms otherwise try to mutate removed DR events at reset.
    for k, term in list(cfg.curriculum.items()):
        if term.params.get("event_name") in ("randomize_com", "randomize_head_com",
                                              "randomize_mass_inertia", "randomize_joint_friction",
                                              "randomize_armature", "foot_friction", "encoder_bias"):
            cfg.curriculum.pop(k)
    rb = cfg.events["reset_base"].params["pose_range"]
    rb.update({"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.125, 0.125), "yaw": (0.0, 0.0)})
    actor = cfg.observations["actor"]
    actor.enable_corruption = False
    for t in ("base_ang_vel", "projected_gravity"):
        actor.terms[t].params["max_angle_deg"] = 0.0
elif mode != "train_dr":
    raise SystemExit(f"unknown mode {mode}")

torch.manual_seed(int(spec.get("seed", 0)))
env = ManagerBasedRlEnv(cfg, device="cpu")
env.common_step_counter = 10**9
robot = env.scene["robot"]
if mode == "nominal":
    for act in robot.actuators:
        if hasattr(act, "vin_tensor") and act.vin_tensor is not None:
            act.vin_tensor[:] = 7.4
            act.vin_drop_gain[:] = 0.1
            print("set vin 7.4 / sag 0.1 on", type(act).__name__, flush=True)

tw = env.command_manager.get_term("twist")
cur = {"k": 0}


def rs(ids):
    tw.vel_command_b[ids] = torch.from_numpy(CMD[cur["k"]])[ids]
    tw.is_standing_env[ids] = False
    tw.is_forward_env[ids] = False
    tw.time_left[ids] = 1e9


tw._resample_command = rs
for n in ("head_pose", "body_pose"):
    env.command_manager.get_term(n)._resample_command = lambda ids: None

obs, _ = env.reset()
tw.time_left[:] = 1e9
alive = np.ones(N, bool)
fall_t = np.full(N, np.nan)
log = np.zeros((n_steps + 1, N, 6), dtype=np.float32)


def snap(k):
    p = robot.data.root_link_pos_w.numpy()
    q = robot.data.root_link_quat_w.numpy()
    v = robot.data.root_link_lin_vel_b.numpy()
    w = robot.data.root_link_ang_vel_b.numpy()
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    log[k, :, 0:2] = p[:, :2]
    log[k, :, 2] = yaw
    log[k, :, 3:5] = v[:, :2]
    log[k, :, 5] = w[:, 2]


snap(0)
t0 = time.time()
for k in range(n_steps):
    with torch.no_grad():
        a = torch.from_numpy(pol(obs["actor"].numpy()).astype(np.float32))
    cur["k"] = k + 1
    tw.vel_command_b[:] = torch.from_numpy(CMD[k + 1])
    obs, _, term, trunc, _ = env.step(a)
    tm = term.numpy()
    newly = tm & alive
    fall_t[newly] = (k + 1) * 0.02
    alive &= ~tm
    snap(k + 1)
    log[k + 1, ~alive] = np.nan
    if k % 50 == 0:
        print(k, round(time.time() - t0, 1), "falls", int((~alive).sum()), flush=True)

res = dict(mode=mode, policy=policy_path, seconds=seconds, num_envs=N, wall_s=time.time() - t0,
           names=[e["name"] for e in envs], fall_t=[None if np.isnan(t) else float(t) for t in fall_t],
           log_cols=["x", "y", "yaw", "vx_b", "vy_b", "wz_b"], log_hz=50,
           log=np.where(np.isnan(log), None, np.round(log, 5)).tolist())
json.dump(res, open(out, "w"))
print("DONE", time.time() - t0, flush=True)
