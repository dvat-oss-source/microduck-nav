"""Training-Sim feasibility checks used before freezing benchmark tasks."""
import math
import numpy as np

from .taskscene import apply_task_objects, body_pose


class SeedValidator:
    def __init__(self, scene, policy, gridmap):
        from .sim import Sim
        self.sim = Sim(scene, policy, dynamics="training", record=False)
        self.gm = gridmap
        self.radius = float(gridmap.params["r_inflate"])
        self.objects = {}
        self.rejections = {"reset": 0, "object_clearance": 0, "object_stability": 0}
        for body in range(self.sim.model.nbody):
            if self.sim.model.body_rootid[body] == self.sim.trunk_id:
                continue
            j = int(self.sim.model.body_jntadr[body])
            if j >= 0 and self.sim.model.jnt_type[j] == 0:  # MuJoCo FREE
                name = self.sim.model.body(body).name
                self.objects[name] = body_pose(self.sim, name).tolist()

    def set_layout(self, poses):
        self.objects = {name: list(pose) for name, pose in poses.items()}

    def pose_ok(self, seed, pose, placements=None, exclude=None):
        objects = dict(self.objects)
        objects.update(placements or {})
        if self.gm.is_lethal(*pose[:2]):
            return False
        # Conservative object radius includes the apartment boxes' half diagonal.
        for name, obj in objects.items():
            if name != exclude and math.dist(pose[:2], obj[:2]) < self.radius + 0.05:
                self.rejections["object_clearance"] += 1
                return False
        try:
            self.sim.reset(int(seed), pose)
            apply_task_objects(self.sim, {"object_poses": objects})
            self.sim._check_spawn()
        except ValueError:
            self.rejections["reset"] += 1
            return False
        return True

    def stable(self, seed, start, placements, target=None):
        if not self.pose_ok(seed, start, placements):
            return False
        names = [target] if target else list(placements)
        initial = {name: body_pose(self.sim, name)[:3] for name in names}
        largest = 0.0
        for _ in range(50):
            self.sim.step(np.zeros(3))
            largest = max(largest, *(float(np.linalg.norm(body_pose(self.sim, name)[:3] - initial[name]))
                                     for name in names))
        if largest > 0.01:
            self.rejections["object_stability"] += 1
            return False
        return True


def planner_start_check(gridmap, primitives, pose):
    """A1: can the planner produce a start lattice node for this pose?  Uses the same snap (and
    bounded snap repair) as LatticePlanner.plan.  -> (ok, info)."""
    from .planner import get_planner
    pl = get_planner(gridmap, primitives)
    s0 = pl.snap(pose)
    if s0 < 0:
        return False, {"mode": "off_lattice"}
    if pl.node_ok[s0 // pl.nh]:
        return True, {"mode": "nearest"}
    s, info = pl.repair_start(pose)
    return s >= 0, info
