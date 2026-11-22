"""Connect the navigation stack to the independent benchmark checker.

No checker verdict is used to control an episode. The frozen task's initial
pose is the localization prior; subsequent non-GT estimates use sensors only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import traceback
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# F3 "learned+fiducial": sha256 of data/learned_odom/odom_net.onnx (= onnx_sha256 in odom_net_meta.json).
# None = fail closed: the learned mode refuses to construct until the coordinator pins the promoted model.
LEARNED_ODOM_SHA256 = "c98c145275b5e9fdae8df90fdbe896cf88b6b78e599c8b4b4878f1ae5e525076"
# Stage 1b: sha256 of data/anchor_graph/anchor_graph_live.json (its self-hash). None = fail closed for anchors.
ANCHOR_GRAPH_SHA256 = "b2461a7428cf6ad1eb69b372836c9f10b023f573d67687c17df44924d18113f7"
ANCHOR_RADIUS_M = 0.3
ANCHOR_SLOW_FRACTION = 0.5       # UNMEASURED slow-down factor of the default cruise; floored at measured vx_min speed
# Stage 1 ablations (selectable from tools/run_benchmark.py). "legacy" = live behaviour, no new code path.
ABLATIONS = {
    "legacy": dict(mode=None, look=False, anchors=False),
    "1a": dict(mode="learned+fiducial", look=False, anchors=False),
    "1a+1c": dict(mode="learned+fiducial", look=True, anchors=False),
    "all": dict(mode="learned+fiducial", look=True, anchors=True),
    "1b+1c": dict(mode="odom+fiducial", look=True, anchors=True),
}


def read_envelope(policy, data_dir=ROOT / "data"):
    from .policies import verified_entry
    entry = verified_entry(policy)
    data_dir = Path(data_dir)
    env = json.loads((data_dir / f"envelope_{policy}.json").read_text())
    if env.get("policy_md5") != entry["md5"]:
        raise RuntimeError(f"envelope policy md5 mismatch for {policy}")
    from .sim import TRAINING_DYNAMICS_VERSION
    want = TRAINING_DYNAMICS_VERSION if env.get("dynamics") == "training" else "infer_policy_v1"
    if env.get("dynamics_version") != want:
        raise RuntimeError(f"envelope_{policy}.json measured with dynamics_version="
                           f"{env.get('dynamics_version')!r}, current sim is {want!r}; rerun nav.measure")
    return env


def read_artifacts(policy, data_dir=ROOT / "data"):
    env = read_envelope(policy, data_dir)
    data_dir = Path(data_dir)
    primitives = json.loads((data_dir / f"primitives_{policy}.json").read_text())
    if primitives.get("dynamics_version") != env["dynamics_version"]:
        raise RuntimeError(f"primitives dynamics_version mismatch for {policy}")
    if primitives.get("policy_md5") != env["policy_md5"]:
        raise RuntimeError(f"primitives policy md5 mismatch for {policy}")
    return env, primitives


def check_ablation(ablation, mode):
    """Reject ablation/mode pairs that the ablation table does not define (fail closed)."""
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation {ablation!r}; choose from {sorted(ABLATIONS)}")
    spec = ABLATIONS[ablation]
    if spec["mode"] is None:
        if mode in ("learned+fiducial",):
            raise ValueError("learned+fiducial requires an explicit ablation (1a, 1a+1c or all)")
        return
    if mode != spec["mode"]:
        raise ValueError(f"ablation {ablation!r} requires mode {spec['mode']!r}, got {mode!r}")


@dataclass
class EpisodeRunner:
    policy: str = "walk"
    mode: str = "gt"
    scene: str = "apartment"
    dynamics: str = "training"
    max_time: float = 120.0
    comparison_policy: bool = False
    data_dir: str = str(ROOT / "data")
    follower_reports_dir: str | None = None
    ablation: str = "legacy"

    def __post_init__(self):
        check_ablation(self.ablation, self.mode)

    def __call__(self, task):
        self.partial_trace = None
        ablation = ABLATIONS[self.ablation]
        from .sim import Sim
        from .mapping import build_map
        from .localization import Localizer
        from .follower import RPPFollower
        from .fsm import NavFSM

        env, primitives = read_artifacts(self.policy, self.data_dir)
        # Explicit mode is saved in every trace. Legacy mode remains the default
        # until a measured fidelity gate selects a different configuration.
        artifact_mode = env.get("dynamics", "infer")
        if artifact_mode != self.dynamics or primitives.get("dynamics", "infer") != self.dynamics:
            raise RuntimeError("measurement/episode dynamics mismatch")
        sim = Sim(self.scene, self.policy, record=True, dynamics=self.dynamics)
        # The training robot adds backlash joints. The independent checker must
        # consume that exact model, rather than the deployment scene's nq/layout.
        model_dir = Path(self.data_dir) / "checker_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        from .provenance import scene_sha256
        closure_hash = scene_sha256(sim.scene_xml, self.dynamics)
        signature = hashlib.sha256((closure_hash.encode() +
                                    (Path(__file__).parent / "sim.py").read_bytes() +
                                    self.dynamics.encode())).hexdigest()[:20]
        checker_model = model_dir / f"{signature}.mjb"
        if not checker_model.exists():
            temp = model_dir / f"{signature}.{os.getpid()}.mjb"
            mujoco.mj_saveModel(sim.model, str(temp))
            os.replace(temp, checker_model)
        gt = sim.reset(task["seed"], task["start"])
        self.partial_trace = {"qpos": np.asarray(sim.trace), "t": np.array([0.]),
                              "model_path": str(checker_model), "scene_xml": sim.scene_xml,
                              "provenance": sim.provenance()}
        from .taskscene import apply_task_objects
        apply_task_objects(sim, task)
        gt = sim.ground_truth()
        controls, estimates, transitions, events = [], [], [], []
        timeline = []
        fsm = localizer = None
        error = error_tb = None
        ablation_record = dict(mode=self.mode, active_look=ablation["look"], anchors=ablation["anchors"],
                               learned_odom_sha256=LEARNED_ODOM_SHA256 if self.mode == "learned+fiducial" else None,
                               anchor_graph_sha256=None, convergence_threshold_sha256=None)
        base_step = sim.step

        def recorded_step(command):
            mirror = hasattr(sim._sess, "_mp")
            event = {"t": float(sim.t), "step": len(sim.trace) - 1,
                     "policy": sim.policy_key, "mirror": mirror}
            if not events or (event["policy"], mirror) != (events[-1]["policy"], events[-1]["mirror"]):
                events.append(event)
                timeline.append([event["t"], event["policy"]])
            controls.append([float(sim.t), *np.asarray(command, float).tolist()])
            return base_step(command)
        sim.step = recorded_step

        def collect():
            provenance = sim.provenance()
            provenance["scene_sha256"] = closure_hash
            mirrored = any(e["mirror"] for e in events)
            provenance["record_kick_mirror"] = mirrored
            trace = {"qpos": np.asarray(sim.trace), "t": np.arange(len(sim.trace)) * 0.02,
                     "model_path": str(checker_model), "scene_xml": sim.scene_xml,
                     "scene_sha256": closure_hash,
                     "static_geometry_reference": static_geometry_reference(sim.scene_xml, closure_hash),
                     "commands": controls, "estimates": estimates, "transitions": transitions,
                     "plan_time_s": max(fsm.planner_times, default=0.0) if fsm else 0.0,
                     "plan_times_s": list(fsm.planner_times) if fsm else [],
                     "plan_log": list(fsm.plan_log) if fsm else [], "policy": self.policy,
                     "rotation_monitor": {
                         "scope": "final follower path only; new paths reset monitor history",
                         "response_window_s": getattr(getattr(fsm, "follower", None), "rotation_window", None),
                         "escalation_events": list(getattr(getattr(fsm, "follower", None), "escalation_events", []))},
                     "mode": self.mode, "dynamics": self.dynamics,
                     "ablation": dict(name=self.ablation, **ablation_record),
                     "dynamics_version": provenance["dynamics_version"],
                     "path_smoothing_steps": 30, "policy_timeline": timeline,
                     "policy_events": events, "record_kick_mirror": mirrored,
                     "localization": {"fiducial_stats": dict(getattr(localizer, "stats", {})),
                                      "camera_renders": getattr(getattr(localizer, "camera", None), "n_renders", 0),
                                      "completion_events": list(getattr(fsm,"localization_events",[])),
                                      "final_quality": localizer.quality() if hasattr(localizer,'quality') else None},
                     "kick_approach": {"events":list(getattr(fsm,"kick_events",[])),
                         "runin_events":list(getattr(getattr(fsm,"_kick_approach",None),"events",[])),
                         "target_source":getattr(fsm,"target_source",None)},
                     "target_source": "frozen surveyed object placement; no visual object detector",
                     "provenance": provenance}
            if error:
                trace.update(error=error, traceback=error_tb)
            self.partial_trace = trace
            return trace

        try:
            from .mapsigma import planning_lateral_sigma
            sigma, sigma_prov = (planning_lateral_sigma(self.policy, env, self.follower_reports_dir)
                                if self.follower_reports_dir else planning_lateral_sigma(self.policy, env))
            if not sigma_prov["follower_gate_passed"]:
                raise RuntimeError("walker failed real-Sim follower gate")
            gm = build_map(sim.scene_xml, lateral_sigma=sigma,
                           step_over_height=env["step_over_height"])
            loc_kw = {}
            if self.mode == "learned+fiducial":
                loc_kw = dict(learned_model_dir=str(Path(self.data_dir) / "learned_odom"), learned_sha256=LEARNED_ODOM_SHA256)
            localizer = Localizer(sim, mode=self.mode, seed=task["seed"], **loc_kw)
            localizer.reset(task["start"])
            follower = RPPFollower(env, gridmap=gm)
            kind = task.get("type", "nav")
            target = None
            if kind == "nav":
                goal = task["goal"]
            else:
                body = sim.model.body(task["ball_body" if kind == "kick" else "target_body"]).id
                target = (sim.data.xpos[body, :2].copy() if self.mode == 'gt' else
                          np.asarray(task['object_pose'][:2],float))
                heading = task.get("direction", math.atan2(target[1] - gt.pose2[1], target[0] - gt.pose2[0]))
                goal = [float(target[0]), float(target[1]), heading]
            objects = []
            target_name = task.get('ball_body',task.get('target_body'))
            if kind == 'kick':
                for name, placement in task.get('object_poses',{}).items():
                    if name == target_name:
                        continue
                    obj_body = sim.model.body(name).id
                    radius = max(float(np.linalg.norm(sim.model.geom_size[g]))
                                 +float(np.linalg.norm(sim.model.geom_pos[g,:2]))
                                 for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == obj_body
                                 and (sim.model.geom_contype[g] or sim.model.geom_conaffinity[g]))
                    xy = (sim.data.xpos[obj_body,:2].copy() if self.mode == 'gt' else np.asarray(placement[:2],float))
                    objects.append(dict(name=name,xy=xy.tolist(),radius=radius))
            target_source = ('current simulator object observation (GT mode)' if self.mode == 'gt' else
                             'frozen surveyed target; displacement is unobservable without an object detector')
            observer = (lambda: sim.data.xpos[body,:2].copy()) if kind == 'kick' and self.mode == 'gt' else None
            look_heading = anchor_fn = None
            if ablation["look"]:
                from .fsm import ACTIVE_LOOK_JSON
                ablation_record["convergence_threshold_sha256"] = artifact_digest(ACTIVE_LOOK_JSON)
            if ablation["anchors"]:
                from .anchor_graph import anchor_fn_for, load_anchor_graph, look_provider
                graph_dir = Path(self.data_dir) / "anchor_graph"
                graph, look, anchor_prov = load_anchor_graph(graph_dir / "anchor_graph_live.json",
                                                             graph_dir / "anchor_look_live.npy", ANCHOR_GRAPH_SHA256)
                ablation_record["anchor_graph_sha256"] = anchor_prov["anchor_graph_sha256"]
                look_heading = look_provider(look)
                anchor_fn = anchor_fn_for(graph, ANCHOR_RADIUS_M, lambda: (
                    None if follower.cruise is None else ANCHOR_SLOW_FRACTION * float(follower.cruise)))
            fsm = NavFSM(sim,gm,primitives,follower,localizer,target_observer=observer,
                         free_objects=objects,target_body=target_name,target_source=target_source,
                         look_heading=look_heading, anchor_fn=anchor_fn, active_look=ablation["look"])
            fsm.start(goal, task=kind, target_xy=target, direction=task.get("direction", 0.0))
            last_state, obs = None, None
            last_cmd = np.zeros(3)                  # command last passed to sim.step (what the localizer integrates)
            timeout = max(float(self.max_time), float(task.get("timeout_s", 0.0)),
                          25.0 * float(task.get("geodesic", 0.0)))
            while sim.t < timeout:
                estimate_time = sim.t
                if hasattr(localizer, "set_command"):
                    localizer.set_command(*last_cmd)   # no-op unless mode is learned+fiducial; must precede tick()
                cmd, state = fsm.tick(gt, obs)
                if state != last_state:
                    transitions.append({"t": sim.t, "state": state, "reason": fsm.reason})
                    last_state = state
                estimates.append([estimate_time, *np.asarray(localizer.pose, float).tolist()])
                if state in ("done", "fail"):
                    # Referee requires the full final second inside the goal tolerance.
                    for _ in range(50):
                        gt, obs = sim.step(np.zeros(3))
                    break
                gt, obs = sim.step(cmd)
                last_cmd = np.asarray(cmd, float).reshape(3)
            else:
                transitions.append({"t": sim.t, "state": "fail", "reason": "episode_timeout"})
        except Exception as exc:
            error, error_tb = repr(exc), traceback.format_exc()
        finally:
            if localizer is not None:
                try:
                    localizer.close()
                except Exception as exc:
                    if error is None:
                        error, error_tb = repr(exc), traceback.format_exc()
        return collect()


@lru_cache(maxsize=16)
def static_geometry_reference(scene_xml, closure_hash):
    """Compile the original scene independently of Sim; compare static geometry only."""
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    return [{"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i),
             "type": int(model.geom_type[i]), "pos": model.geom_pos[i].tolist(),
             "size": model.geom_size[i].tolist(), "quat": model.geom_quat[i].tolist(),
             "contype": int(model.geom_contype[i]), "conaffinity": int(model.geom_conaffinity[i])}
            for i in range(model.ngeom) if model.geom_bodyid[i] == 0 and
            (model.geom_contype[i] or model.geom_conaffinity[i])]


def failure_video(task, trace, path):
    """Bounded-length time-lapse; the adjacent JSON retains every 50 Hz sample."""
    from .video import render_trace
    path_model = trace["model_path"]
    model = (mujoco.MjModel.from_binary_path(path_model) if path_model.endswith(".mjb")
             else mujoco.MjModel.from_xml_path(path_model))
    qpos = np.asarray(trace["qpos"])
    stride = max(1, math.ceil(len(qpos) / 500))
    frames = qpos[::stride]
    # Always include the terminal state.
    if len(qpos) > 1:
        frames = np.vstack([frames, qpos[-1]])
    output = render_trace(model, frames, path, fps=25, width=320, height=240)
    if str(output) != str(path):
        raise RuntimeError(f"video unavailable; saved {output}")
    return output


def artifact_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
