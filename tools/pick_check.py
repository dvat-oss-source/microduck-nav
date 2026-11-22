"""Ground-pick diagnosis (Phase G).

The upstream GroundPick task (tasks/microduck_ground_pick_env_cfg.py) has NO object: it rewards the mouth tip
going as close as possible to the floor WITHOUT touching it (mouth_ground_proximity target 0 + head_impact_penalty),
mouth x-axis pointing down, then a clean return to stand, while a 10-40 g "payload" weight is applied as an external
force at mouth_tip during the rise (phase >= HOLD_END 0.425).  The jaw is passive, there is no grasp.  This tool
scores the trained task on its own terms (the eval_ground_pick.py protocol: stand 2 s on the walker, pick one cycle
phase 0->0.7 at period 4 s, stand 3 s) plus:
  * payload: the training payload (U(10,40) g at mouth_tip from phase 0.425) applied -> still upright / drift?
  * object: a 70 mm / 15 g ball (only free object in nav.sim) at the measured mouth point -> does it leave the
    floor (z > r + 5 mm), and how far is it pushed?

  $PY -m tools.pick_check --seeds 10 --out reports/skills_diag_pick.json
"""
from __future__ import annotations

import argparse
import json
import math

import mujoco
import numpy as np

from nav import skills
from nav.sim import BALL_RADIUS

HOLD_END = 0.425


def trial(sim, key, seed, payload=False, ball_trunk=None, t_pre=100, t_post=150):
    rng = np.random.default_rng(9000 + seed)
    m, d = sim.model, sim.data
    sim.reset(seed, start_pose=(0.0, 0.0, 0.0), extra={"policy": "walk", "ball_xy": (5.0, 5.0)})
    mouth = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
    jaw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    floor = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and (m.geom_contype[g] or m.geom_conaffinity[g])}
    feet = set(sim.foot_geoms.values())
    start_xy = sim.ground_truth().pose2[:2].copy()
    for _ in range(t_pre):
        sim.step(np.zeros(3))
    ok = sim.switch_policy(key)
    gt = sim.ground_truth()
    if ball_trunk is not None:
        sim.place_ball(*skills.trunk_to_world(gt.pose2, ball_trunk))
    kg = rng.uniform(0.01, 0.04) if payload else 0.0
    min_h, down, head_floor, fell, ball_zmax, n = 1e9, -1.0, 0, bool(gt.fallen), 0.0, 0
    up_run, up_best, robot_ball = 0, 0, 0
    b0 = sim.ball_pos()[:2].copy() if ball_trunk is not None else None
    bg = m.geom('ball_geom').id
    while not sim.skill_done and n < 300:
        d.xfrc_applied[:] = 0.0
        if kg > 0 and sim.phase >= HOLD_END:
            F = np.array([0.0, 0.0, -kg * 9.81])
            r = d.site_xpos[mouth] - d.xipos[jaw]
            d.xfrc_applied[jaw, :3] = F
            d.xfrc_applied[jaw, 3:] = np.cross(r, F)
        gt, _ = sim.step(np.zeros(3))
        n += 1
        fell |= bool(gt.fallen)
        h = float(d.site_xpos[mouth][2])
        if h < min_h:
            min_h = h
            down = float(-d.site_xmat[mouth].reshape(3, 3)[2, 0])
        for c in d.contact[:d.ncon]:
            if (c.geom1 in floor) != (c.geom2 in floor):
                rg = c.geom2 if c.geom1 in floor else c.geom1
                if sim.robot_geoms[rg] and rg not in feet:
                    head_floor += 1
                    break
        if ball_trunk is not None:
            bz = float(sim.ball_pos()[2])
            ball_zmax = max(ball_zmax, bz)
            up_run = up_run + 1 if bz > BALL_RADIUS + 0.005 else 0
            up_best = max(up_best, up_run)
            robot_ball += any(bg in (c.geom1, c.geom2) and sim.robot_geoms[c.geom2 if c.geom1 == bg else c.geom1]
                              for c in d.contact[:d.ncon])
    # payload stays on during the post-pick stand (object carried), as the payload hook does to the end of the
    # cycle in training; switch back to the walker
    sim.force_switch("walk")
    for _ in range(t_post):
        d.xfrc_applied[:] = 0.0
        if kg > 0:
            F = np.array([0.0, 0.0, -kg * 9.81])
            d.xfrc_applied[jaw, :3] = F
            d.xfrc_applied[jaw, 3:] = np.cross(d.site_xpos[mouth] - d.xipos[jaw], F)
        gt, _ = sim.step(np.zeros(3))
        fell |= bool(gt.fallen)
    d.xfrc_applied[:] = 0.0
    drift = float(np.hypot(*(gt.pose2[:2] - start_xy)))
    up = bool((not fell) and gt.tilt_rad < skills.UPRIGHT_TILT)
    out = dict(seed=seed, handoff_ok=bool(ok), payload_kg=kg, fell=fell, end_upright=up, drift_mm=1000 * drift,
               min_mouth_h_mm=1000 * min_h, mouth_down_cos=down, head_floor_contact_steps=head_floor,
               # training objective met: mouth reached the floor band without touching, oriented down, back up
               task_ok=bool(up and head_floor == 0 and min_h < 0.035 and down > 0.9))
    if ball_trunk is not None:
        b = sim.ball_pos()
        # lifted = ball held >= 5 mm off the floor for >= 0.2 s (10 steps); a transient bounce does not count
        out.update(ball_peak_lift_mm=1000 * (ball_zmax - BALL_RADIUS), ball_lifted=bool(up_best >= 10),
                   ball_airborne_max_steps=int(up_best), robot_ball_contact_steps=int(robot_ball),
                   ball_push_m=float(np.hypot(*(b[:2] - b0))), ball_end_z_mm=1000 * float(b[2]))
    return out


def summarize(tr):
    s = dict(n=len(tr), task_ok=int(sum(t["task_ok"] for t in tr)), falls=int(sum(t["fell"] for t in tr)),
             min_mouth_h_mm_median=float(np.median([t["min_mouth_h_mm"] for t in tr])),
             head_floor_contact_trials=int(sum(t["head_floor_contact_steps"] > 0 for t in tr)),
             drift_mm_median=float(np.median([t["drift_mm"] for t in tr])))
    if "ball_lifted" in tr[0]:
        s["ball_lifted"] = int(sum(t["ball_lifted"] for t in tr))
        s["ball_peak_lift_mm_max"] = float(max(t["ball_peak_lift_mm"] for t in tr))
        s["ball_push_m_median"] = float(np.median([t["ball_push_m"] for t in tr]))
        s["robot_ball_contact_trials"] = int(sum(t["robot_ball_contact_steps"] > 0 for t in tr))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--policies", default="pick,atlantis_pick_flat")
    ap.add_argument("--configs", default="infer:walk,training:walk,training:allcollisions")
    ap.add_argument("--out", default="reports/skills_diag_pick.json")
    a = ap.parse_args()
    res = {"runs": []}
    for cfg in a.configs.split(","):
        dyn, robot = cfg.split(":")
        sim = skills.skill_sim("flat", "walk", dynamics=dyn, robot=robot, ball=True)
        for key in a.policies.split(","):
            base = [trial(sim, key, s) for s in range(a.seeds)]
            mx = float(np.mean([t["min_mouth_h_mm"] for t in base]))
            pay = [trial(sim, key, s, payload=True) for s in range(a.seeds)]
            # ball centred under the mean lowest mouth point (trunk frame, from the contract measurement)
            mxy = skills.load_contracts()["pick"]["mouth_trunk_xy"]
            obj = [trial(sim, key, s, ball_trunk=(mxy[0], mxy[1])) for s in range(a.seeds)]
            row = dict(dynamics=dyn, robot=robot, policy=key, base=summarize(base), payload=summarize(pay),
                       ball=summarize(obj), trials=dict(base=base, payload=pay, ball=obj))
            res["runs"].append(row)
            print(dyn, robot, key, "\n  base", row["base"], "\n  payload", row["payload"], "\n  ball", row["ball"], flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
