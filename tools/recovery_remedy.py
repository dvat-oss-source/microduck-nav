"""Ball-free paired recovery protocol audit, isolated from frozen benchmark.

The training-support arm approximates pinned reset distributions in local Sim;
it is deliberately not labelled an independent mjlab reproduction.
"""
from __future__ import annotations
import argparse
import json
import math
import hashlib
from pathlib import Path
import mujoco
import numpy as np
from nav import skills
from nav.sim import TRAINING_DYNAMICS_VERSION
from nav.provenance import scene_sha256
from tools.standup_check import train_init, nav_init


def initialize(sim, state, seed, protocol):
    if protocol == "settled_walk":
        return nav_init(sim, state, seed)
    train_init(sim, state, seed)
    # Prior diagnostic truncated face-up logroll to near-flat or near-side.
    # Pinned set_random_ground_state samples its full uniform +/-90deg range.
    if state == "back":
        rng = np.random.default_rng(17000 + seed)
        yaw, theta = rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi / 2, math.pi / 2)
        cy, sy, s = math.cos(yaw / 2), math.sin(yaw / 2), 2 ** -.5
        q = np.array([s * cy, s * sy, -s * cy, s * sy])
        sim.data.qpos[sim.fj_qadr + 3:sim.fj_qadr + 7] = skills._quat_mul(q, np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)]))
        mujoco.mj_forward(sim.model, sim.data)
    return sim.ground_truth()


def trial(sim, state, seed, protocol, policy):
    start = initialize(sim, state, seed, protocol)
    sim.force_switch(policy)
    if protocol == "training_support":
        sim.last_action[:] = 0
    trace, hold, recovered = [], 0, False
    for _ in range(500):
        gt, _ = sim.step(np.zeros(3))
        upright = gt.tilt_rad < skills.UPRIGHT_TILT and gt.height >= skills.STAND_HEIGHT_OK
        hold = hold + 1 if upright else 0
        trace.append(dict(t=sim.t, z=float(gt.pos3[2]), height=float(gt.height),
                          tilt_deg=math.degrees(gt.tilt_rad), upright=bool(upright)))
        if hold >= 25:
            recovered = True
            break
    # Exercise the actual selected walking policy after recovery only.
    handoff_fell = None
    if recovered:
        sim.force_switch("atlantis_walk")
        handoff_fell = False
        for _ in range(100):
            gt, _ = sim.step(np.zeros(3))
            handoff_fell |= bool(gt.fallen)
    return dict(state=state, seed=seed, params=sim.params, start_z=float(start.pos3[2]),
                start_tilt_deg=math.degrees(start.tilt_rad), success_10s=recovered,
                success_6s=recovered and len(trace) <= 300,
                walker_handoff_fell=handoff_fell, trace=trace)


def main():
    if TRAINING_DYNAMICS_VERSION != "cb8561f6_nominal_v2":
        raise RuntimeError("diagnostic frozen dynamics changed")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="reports/pick_recovery_remedy_recovery.json")
    ap.add_argument("--policies", default="atlantis_standup_flat,standup_rough,velstand")
    a = ap.parse_args()
    runs = []
    for robot in ("walk", "allcollisions"):
        sim = skills.skill_sim("flat", "atlantis_walk", dynamics="training", robot=robot, ball=False)
        for protocol in ("training_support", "settled_walk"):
            states = ("front", "back", "sit") if protocol == "training_support" else ("front", "back", "side_left", "side_right")
            for policy in a.policies.split(","):
                ts = [trial(sim, st, s, protocol, policy) for st in states for s in range(a.seeds)]
                morphology = skills.NAV_ROOT.parent / "upstream/rl_cb8561f6/src/mjlab_microduck/robot/microduck" / ("robot_walk_backlash.xml" if robot == "walk" else "robot_allcollisions_backlash.xml")
                row = dict(robot=robot, protocol=protocol, policy=policy, locks=sim.lock_entries_used.copy(),
                           base_training_scene_closure_sha256=scene_sha256(sim.scene_xml, "training"),
                           morphology_override_xml=str(morphology),
                           morphology_override_xml_sha256=hashlib.sha256(morphology.read_bytes()).hexdigest(), trials=ts)
                runs.append(row)
                print(robot, protocol, policy, sum(t["success_10s"] for t in ts), "/", len(ts), flush=True)
    Path(a.out).write_text(json.dumps(dict(development_only=True, supported=False,
        dynamics_version="cb8561f6_nominal_v2", ball_present=False,
        allow_rejected=False,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__), skills.NAV_ROOT / "tools/standup_check.py",
                                 skills.NAV_ROOT / "nav/sim.py", skills.NAV_ROOT / "nav/skills.py",
                                 skills.NAV_ROOT / "nav/policies.py", skills.NAV_ROOT / "policies.lock.json")},
        caveat="local Sim reset distribution audit, not independent mjlab replay", runs=runs), indent=2))


if __name__ == "__main__":
    main()
