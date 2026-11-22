"""Development-only real-object lift audit; never changes skill contracts.

Use apartment ball geometry and measure its geom's world z, not body origin,
robot mouth height, or payload force. No attachment or external forces.
"""
from __future__ import annotations
import argparse
import json
import copy
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from nav import skills
from nav.sim import Sim, scene_path, TRAINING_DYNAMICS_VERSION
from nav.taskscene import place_object
from nav.provenance import scene_sha256


def trial(sim, seed, target, policy="pick", walker="atlantis_walk"):
    sim.reset(seed, start_pose=(0., 0., 0.), extra={"policy": walker, "ball_xy": (5., 5.)})
    for _ in range(100):
        sim.step(np.zeros(3))
    accepted = sim.switch_policy(policy)
    gt = sim.ground_truth()
    start_xy = gt.pose2[:2].copy()
    xy = skills.trunk_to_world(gt.pose2, skills.load_contracts()["pick"]["mouth_trunk_xy"])
    place_object(sim, target, [*xy, 0., 1., 0., 0., 0.])
    mujoco.mj_forward(sim.model, sim.data)
    bg = sim.model.geom(target + "_geom").id
    body = sim.model.body(target).id
    qa = int(sim.model.jnt_qposadr[int(sim.model.body_jntadr[body])])
    origin_z0 = float(sim.data.qpos[qa + 2])
    z0 = float(sim.data.geom_xpos[bg, 2])
    trace, fell, run, best = [], False, 0, 0
    while not sim.skill_done and len(trace) < 300:
        gt, _ = sim.step(np.zeros(3))
        dz = float(sim.data.geom_xpos[bg, 2]) - z0
        contact = any(bg in (c.geom1, c.geom2) and sim.robot_geoms[c.geom2 if c.geom1 == bg else c.geom1]
                      for c in sim.data.contact[:sim.data.ncon])
        run = run + 1 if dz >= .020 else 0
        best = max(best, run)
        fell |= bool(gt.fallen)
        trace.append(dict(t=sim.t, object_center_world=sim.data.geom_xpos[bg].tolist(),
                          lift_m=dz, body_origin_lift_m=float(sim.data.qpos[qa + 2]) - origin_z0,
                          robot_contact=bool(contact), fallen=bool(gt.fallen)))
    # Frozen episodes include a final one-second zero-command dwell. Object
    # retention is measured through this selected-walker handoff as well.
    sim.force_switch(walker)
    for _ in range(50):
        gt, _ = sim.step(np.zeros(3))
        dz = float(sim.data.geom_xpos[bg, 2]) - z0
        contact = any(bg in (c.geom1, c.geom2) and sim.robot_geoms[c.geom2 if c.geom1 == bg else c.geom1]
                      for c in sim.data.contact[:sim.data.ncon])
        run = run + 1 if dz >= .020 else 0
        best = max(best, run)
        fell |= bool(gt.fallen)
        trace.append(dict(t=sim.t, object_center_world=sim.data.geom_xpos[bg].tolist(),
                          lift_m=dz, body_origin_lift_m=float(sim.data.qpos[qa + 2]) - origin_z0,
                          robot_contact=bool(contact), fallen=bool(gt.fallen)))
    final_lift = trace[-1]["lift_m"]
    drift = float(np.linalg.norm(gt.pose2[:2] - start_xy))
    return dict(seed=seed, target=target, policy=policy, handoff_ok=bool(accepted), params=sim.params,
                peak_lift_m=max(t["lift_m"] for t in trace),
                lift_20mm=any(t["lift_m"] >= .020 for t in trace),
                final_lift_m=final_lift, drift_m=drift,
                final_body_origin_lift_m=trace[-1]["body_origin_lift_m"],
                frozen_criterion_local_verdict=bool(accepted
                                and trace[-1]["body_origin_lift_m"] >= .020 and drift < .01
                                and gt.tilt_rad <= np.deg2rad(30) and not fell),
                stricter_physical_lift_verdict=bool(accepted and final_lift >= .020
                                and trace[-1]["body_origin_lift_m"] >= .020 and drift < .01
                                and gt.tilt_rad <= np.deg2rad(30) and not fell),
                longest_20mm_dwell_s=best * .02,
                robot_contact_steps=sum(t["robot_contact"] for t in trace),
                fell=fell, trace=trace)


def main():
    if TRAINING_DYNAMICS_VERSION != "cb8561f6_nominal_v2":
        raise RuntimeError("diagnostic frozen dynamics changed")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--policies", default="pick,atlantis_pick_flat")
    ap.add_argument("--robot", choices=("walk", "allcollisions"), default="walk")
    ap.add_argument("--out", default="reports/pick_recovery_remedy_pick.json")
    a = ap.parse_args()
    # Copy physical target bodies verbatim from the actual apartment include.
    # Only material names are removed because this headless flat scene has no
    # apartment visual asset table; collision and inertial attributes are intact.
    source = scene_path("apartment").parent / "apartment.xml"
    root = ET.Element("mujoco", model="pick_target_development")
    robot_xml = skills.NAV_ROOT.parent / "upstream/rl_cb8561f6/src/mjlab_microduck/robot/microduck" / ("robot_walk_backlash.xml" if a.robot == "walk" else "robot_allcollisions_backlash.xml")
    ET.SubElement(root, "include", file=str(robot_xml))
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 .05")
    apartment = ET.parse(source)
    targets = ["obj_" + str(i) for i in range(4)]
    for target in targets:
        body = copy.deepcopy(apartment.find(".//body[@name='" + target + "']"))
        if body is None:
            raise RuntimeError("missing frozen target " + target)
        body.set("pos", f"{5 + targets.index(target)} 5 0")
        for geom in body.iter("geom"):
            geom.attrib.pop("material", None)
        world.append(body)
    scene = Path(a.out).with_suffix(".scene.xml")
    ET.ElementTree(root).write(scene)
    # Resolve closure before running any trials so provenance errors cannot
    # discard a completed trial batch.
    diagnostic_closure = scene_sha256(scene.resolve(), "training")
    sim = Sim(str(scene.resolve()), "atlantis_walk", dynamics="training")
    reference = Sim("apartment", "atlantis_walk", dynamics="training", record=False)
    checked_fields = ("type", "size", "friction", "solref", "solimp", "margin", "gap",
                      "condim", "contype", "conaffinity")
    for target in targets:
        sg, rg = sim.model.geom(target + "_geom"), reference.model.geom(target + "_geom")
        for field in checked_fields:
            if not np.array_equal(getattr(sg, field), getattr(rg, field)):
                raise RuntimeError(f"compiled target physics differs: {target} geom {field}")
        sb, rb = sim.model.body(target), reference.model.body(target)
        for field in ("mass", "inertia", "ipos", "iquat"):
            if not np.array_equal(getattr(sb, field), getattr(rb, field)):
                raise RuntimeError(f"compiled target physics differs: {target} body {field}")
    del reference
    trials = [trial(sim, s, target, policy=policy) for policy in a.policies.split(",")
              for target in targets for s in range(a.seeds)]
    result = dict(development_only=True, dynamics=sim.params["dynamics"],
                  dynamics_version="cb8561f6_nominal_v2", robot=robot_xml.name,
                  object="actual obj_0..obj_3 70mm boxes / 50g; origin floor",
                  source_xml_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  compiled_physics_matches_actual_apartment=True,
                  scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                  scene_closure_sha256=diagnostic_closure,
                  actual_apartment_closure_sha256=scene_sha256(scene_path("apartment"), "training"),
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), skills.NAV_ROOT / "nav/sim.py",
                                           skills.NAV_ROOT / "nav/skills.py", skills.NAV_ROOT / "nav/policies.py",
                                           skills.NAV_ROOT / "policies.lock.json")},
                  allow_rejected=False,
                  metric="frozen criterion: final body-origin delta>=.020m; stricter physical diagnostic additionally requires geom-center delta>=.020m",
                  dwell_requirement="none specified in frozen pick contract; reported separately",
                  policy_locks=sim.lock_entries_used, trials=trials,
                  lift_20mm_count=sum(t["lift_20mm"] for t in trials),
                  frozen_criterion_local_count=sum(t["frozen_criterion_local_verdict"] for t in trials),
                  stricter_physical_lift_count=sum(t["stricter_physical_lift_verdict"] for t in trials), supported=False)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k not in ("trials", "policy_locks")})


if __name__ == "__main__":
    main()
