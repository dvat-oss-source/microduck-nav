"""Stand-up / fall-recovery diagnosis (Phase G).  One factor at a time:

  dynamics x robot model : infer | training+walk model (nav.sim default) | training+allcollisions (skill_sim)
  init protocol          : nav   = skills.make_fallen (drop from 0.15 m, motors hold HOME for 1.5 s, then hand-off)
                           train = StandUp env set_random_ground_state (face-down/face-up(+-90 deg log-roll)/
                                   sit-keyframe quats, z U(0.05,0.09), joints HOME, qvel 0, policy from t=0,
                                   last_action 0) -- tasks/mdp.py:4157, standup_env_cfg EPISODE_LENGTH_S 6
  policy                 : atlantis_standup_flat, standup_rough (partial 6750), velstand (Pollen v8, VelStand-Rough
                           -Backlash, trained with fallen resets; zero twist)

Success = tilt < 15 deg and trunk-foot height >= 0.09 m held 0.5 s; reported within 6 s (training episode) and 10 s.

  $PY -m tools.standup_check --seeds 5 --out reports/skills_diag_standup.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import mujoco
import numpy as np

from nav import skills
from nav.sim import SITTING_JOINT_OVERRIDES, SIT_Z

STATES = ["front", "back", "side_left", "side_right", "sit"]
S = 2 ** -0.5


def _qmul(a, b):
    return skills._quat_mul(a, b)


def train_init(sim, state, seed):
    """Reproduce set_random_ground_state for one env (flat terrain)."""
    rng = np.random.default_rng(5000 + seed)
    init = "sit" if state == "sit" else "stand"
    sim.reset(seed, start_pose=(0.0, 0.0, 0.0), extra={"policy": "walk", "init": init, "check_spawn": False})
    yaw = rng.uniform(-math.pi, math.pi)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    if state == "front":
        q = np.array([S * cy, -S * sy, S * cy, S * sy])
        z = rng.uniform(0.05, 0.09)
    elif state in ("back", "side_left", "side_right"):
        q = np.array([S * cy, S * sy, -S * cy, S * sy])
        # face_up_roll_max = 90 deg about body z; "back" = |roll| small, sides = +-90 (edges of the range)
        th = {"back": rng.uniform(-0.25, 0.25), "side_left": math.radians(80) + rng.uniform(0, 0.17),
              "side_right": -math.radians(80) - rng.uniform(0, 0.17)}[state]
        q = _qmul(q, np.array([math.cos(th / 2), 0, 0, math.sin(th / 2)]))
        z = rng.uniform(0.05, 0.09)
    else:  # sit: +-10 deg pitch/roll, joints SIT overrides + N(0, 0.12)
        p, r = rng.uniform(-1, 1, 2) * math.radians(10)
        q = skills._quat_axis_angle("z", yaw)
        q = _qmul(_qmul(q, skills._quat_axis_angle("y", p)), skills._quat_axis_angle("x", r))
        z = rng.uniform(0.05, 0.09)
    d = sim.data
    a = sim.fj_qadr
    d.qpos[a + 2] = z
    d.qpos[a + 3:a + 7] = q / np.linalg.norm(q)
    d.qvel[:] = 0.0
    if state == "sit":
        for i, qi in enumerate(sim.joint_qpos_indices):
            d.qpos[qi] += rng.normal(0, 0.12)
    mujoco.mj_forward(sim.model, d)
    # Policy starts at once with zero action history; motor targets = current joints (as the env's
    # reset_action_history + default-pose action offset would hold HOME, last_action = 0).
    sim.bam_ctrl.reset(d.qpos)
    sim.bam_ctrl.q_target[:] = sim.default_pose
    sim.last_action[:] = 0.0
    return sim.ground_truth()


def nav_init(sim, state, seed):
    if state == "sit":
        sim.reset(seed, start_pose=(0.0, 0.0, 0.0), extra={"policy": "walk", "init": "sit"})
        return sim.ground_truth()
    return skills.make_fallen(sim, state, seed)


def trial(sim, key, state, seed, protocol, max_steps=500):
    gt0 = (train_init if protocol == "train" else nav_init)(sim, state, seed)
    sim.force_switch(key)
    sim.last_action[:] = 0.0 if protocol == "train" else sim.last_action
    hold, t_up = 0, None
    min_tilt = math.pi
    for k in range(max_steps):
        gt, _ = sim.step(np.zeros(3))
        min_tilt = min(min_tilt, gt.tilt_rad)
        if gt.tilt_rad < skills.UPRIGHT_TILT and gt.height >= skills.STAND_HEIGHT_OK:
            hold += 1
            if hold >= 25:
                t_up = (k + 1 - 24) * skills.CONTROL_DT
                break
        else:
            hold = 0
    return dict(state=state, seed=seed, success_10s=t_up is not None,
                success_6s=t_up is not None and t_up <= 6.0, t_up_s=t_up,
                start_z=float(gt0.pos3[2]), start_tilt_deg=math.degrees(gt0.tilt_rad),
                min_tilt_deg=math.degrees(min_tilt), end_tilt_deg=math.degrees(gt.tilt_rad),
                end_height=float(gt.height))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--policies", default="atlantis_standup_flat,standup_rough,velstand")
    ap.add_argument("--configs", default="infer:walk,training:walk,training:allcollisions")
    ap.add_argument("--protocols", default="nav,train")
    ap.add_argument("--out", default="reports/skills_diag_standup.json")
    a = ap.parse_args()
    res = {"seeds": a.seeds, "states": STATES, "runs": []}
    for cfg in a.configs.split(","):
        dyn, robot = cfg.split(":")
        sim = skills.skill_sim("flat", "walk", dynamics=dyn, robot=robot)
        for proto in a.protocols.split(","):
            for key in a.policies.split(","):
                t0 = time.time()
                tr = [trial(sim, key, st, s, proto) for st in STATES for s in range(a.seeds)]
                per = {st: [int(sum(t["success_6s"] for t in tr if t["state"] == st)),
                            int(sum(t["success_10s"] for t in tr if t["state"] == st))] for st in STATES}
                fallen = [t for t in tr if t["state"] != "sit"]
                row = dict(dynamics=dyn, robot=robot, protocol=proto, policy=key, per_state_6s_10s=per,
                           fallen_success_6s=int(sum(t["success_6s"] for t in fallen)),
                           fallen_success_10s=int(sum(t["success_10s"] for t in fallen)), n_fallen=len(fallen),
                           mean_start_z_fallen=float(np.mean([t["start_z"] for t in fallen])), trials=tr)
                res["runs"].append(row)
                print(f"{dyn:8s} {robot:13s} {proto:5s} {key:22s} fallen {row['fallen_success_6s']:2d}/"
                      f"{row['fallen_success_10s']:2d} of {len(fallen)}  z0 {row['mean_start_z_fallen']:+.3f}  "
                      + " ".join(f"{st}:{v[0]}/{v[1]}" for st, v in per.items()) + f"  ({time.time()-t0:.0f}s)",
                      flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
