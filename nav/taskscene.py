"""Explicit free-object placements shared by seed validation and episode resets."""
import numpy as np
import mujoco


def place_object(sim, body_name, pose):
    body = sim.model.body(body_name).id
    joint = int(sim.model.body_jntadr[body])
    if joint < 0 or sim.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"task object {body_name} has no free joint")
    qadr = int(sim.model.jnt_qposadr[joint])
    vadr = int(sim.model.jnt_dofadr[joint])
    pose = np.asarray(pose, float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("object pose must be finite xyz+wxyz")
    if body_name.startswith("ball"):
        # Sim.place_ball addresses its configured ball. Bind the named apartment ball;
        # its local geom offset means the frozen freejoint z must be restored afterwards.
        old = sim.ball_qadr, sim.ball_vadr
        try:
            sim.ball_qadr, sim.ball_vadr = qadr, vadr
            sim.place_ball(float(pose[0]), float(pose[1]))
        finally:
            sim.ball_qadr, sim.ball_vadr = old
    sim.data.qpos[qadr:qadr + 7] = pose
    sim.data.qvel[vadr:vadr + 6] = 0
    mujoco.mj_forward(sim.model, sim.data)


def apply_task_objects(sim, task):
    for name, pose in task.get("object_poses", {}).items():
        place_object(sim, name, pose)
    name = task.get("ball_body", task.get("target_body"))
    if name and "object_pose" in task:
        place_object(sim, name, task["object_pose"])
    if sim.record and sim.trace:
        sim.trace[-1] = sim.data.qpos.copy()


def body_pose(sim, name):
    body = sim.model.body(name).id
    joint = int(sim.model.body_jntadr[body])
    a = int(sim.model.jnt_qposadr[joint])
    return sim.data.qpos[a:a + 7].copy()
