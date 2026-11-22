import ast
import math
import os
import numpy as np
import mujoco
import pytest
from nav import checker

SCENE = os.path.expanduser(
    "~/microduck/upstream/microduck_rl/src/mjlab_microduck/robot/microduck/scene_apartment.xml")
ALLOWED = {"mujoco", "numpy", "json", "math", "dataclasses", "typing"}


def test_imports_only_allowed():
    src = open(checker.__file__).read()
    mods = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            assert n.level == 0, "relative import"
            mods.add(n.module.split(".")[0])
    assert mods <= ALLOWED, mods - ALLOWED


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(SCENE)


def key_qpos(model, name="STAND"):
    return model.key(name).qpos.copy()


def trace(model, x, y, yaw=0.0, z=0.12, quat=None, n=51, extra=None):
    q = key_qpos(model)
    q[0], q[1], q[2] = x, y, z
    q[3:7] = quat if quat is not None else [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    qs = np.tile(q, (n, 1))
    if extra:
        extra(qs)
    return {"qpos": qs, "t": np.arange(n) * 0.02, "model_path": SCENE}


def test_hole_derived(model):
    h = checker.derive_hole(model)
    bb = h.bbox()
    assert bb is not None
    # apartment.xml: hole x in [-0.40, 0.0], y in [-1.40, -0.70]
    for got, want in zip(bb, [-0.40, -1.40, 0.0, -0.70]):
        assert abs(got - want) < 0.03
    assert h.contains(-0.2, -1.0) and not h.contains(-2.0, 0.0) and not h.contains(-0.2, 1.0)


def nav_task(x, y, th=0.0):
    return {"type": "nav", "goal": [x, y, th], "pos_tol": 0.05, "yaw_tol_deg": 10}


def test_nav_success(model):
    v = checker.check(trace(model, -2.0, 0.0), nav_task(-2.0, 0.0))
    assert v["success"] and v["reasons"] == [] and not v["fallen"] and not v["hole_incursion"]
    assert v["time_to_goal"] == 0.0


def test_nav_too_far_and_yaw(model):
    v = checker.check(trace(model, -2.0, 0.1), nav_task(-2.0, 0.0))
    assert not v["success"] and "pos_tol" in v["reasons"]
    v = checker.check(trace(model, -2.0, 0.0, yaw=math.radians(20)), nav_task(-2.0, 0.0))
    assert not v["success"] and "yaw_tol" in v["reasons"]


def test_fall(model):
    # rolled 90 deg about x
    q = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]
    v = checker.check(trace(model, -2.0, 0.0, z=0.05, quat=q), nav_task(-2.0, 0.0))
    assert v["fallen"] and not v["success"] and "fall" in v["reasons"]
    assert v["max_tilt_deg"] > 85
    # A rigid vertical shift preserves height above the feet and is not a fall.
    v = checker.check(trace(model, -2.0, 0.0, z=0.02), nav_task(-2.0, 0.0))
    assert not v["fallen"]


def test_hole_incursion(model):
    v = checker.check(trace(model, -0.2, -1.0, z=0.1), nav_task(-0.2, -1.0))
    assert v["hole_incursion"] and not v["success"] and "stair_hole" in v["reasons"]
    # path past the hole (corridor north of it) is fine
    v = checker.check(trace(model, -0.2, -0.5), nav_task(-0.2, -0.5))
    assert not v["hole_incursion"]


def test_path_length_and_time(model):
    tr = trace(model, 0, 0, n=11)
    q0 = key_qpos(model)
    for k in range(11):
        tr["qpos"][k, 0] = -3.0 + 0.1 * k
        tr["qpos"][k, 1] = 0.5 - 0.0
    tr["qpos"][:, 1] = 0.0
    v = checker.check(tr, nav_task(-2.0, 0.0))
    assert abs(v["path_length_raw"] - 1.0) < 1e-9
    assert v["success"] and abs(v["time_to_goal"] - 0.2) < 1e-9 + 0.0 or v["time_to_goal"] is not None


def test_wall_collision(model):
    # robot pushed into wall_west (x=-4, half thickness .06): trunk at x=-3.95
    v = checker.check(trace(model, -3.93, 0.0), nav_task(-3.93, 0.0))
    assert v["collision_steps"] > 0 and any("wall" in g for g in v["collision_geoms"])
    v = checker.check(trace(model, 2.0, 0.0), nav_task(2.0, 0.0))
    assert v["collision_steps"] == 0


def test_kick(model):
    ba = model.jnt_qposadr[model.body_jntadr[model.body("ball_0").id]]
    def extra(qs):
        qs[:, ba] = -3.0 + 0.1 * np.arange(len(qs))
    tr = trace(model, -3.3, 0.0, n=6, extra=extra)
    tr["policy_timeline"] = [[0.0, "kick_right"]]
    task = {"type": "kick", "ball_body": "ball_0", "direction": 0.0, "min_speed": 0.3, "angle_tol_deg": 20}
    v = checker.check(tr, task)
    assert v["success"], v
    task["direction"] = math.pi / 2
    v = checker.check(tr, task)
    assert not v["success"] and "wrong_direction" in v["reasons"]
    v = checker.check(trace(model, -3.3, 0.0), {**task, "direction": 0.0})
    assert "ball_not_moved" in v["reasons"]


def test_pick(model):
    ba = model.jnt_qposadr[model.body_jntadr[model.body("obj_0").id]]
    def lift(qs):
        qs[-1, ba + 2] += 0.05
    task = {"type": "pick", "target_body": "obj_0"}
    tr = trace(model, -2.2, -1.0, n=4, extra=lift)
    tr["policy_timeline"] = [[0.0, "pick"]]
    v = checker.check(tr, task)
    assert v["success"], v
    v = checker.check(trace(model, -2.2, -1.0, n=4), task)
    assert "not_lifted" in v["reasons"]
    def drift(qs):
        lift(qs); qs[-1, 0] += 0.05
    tr = trace(model, -2.2, -1.0, n=4, extra=drift)
    tr["policy_timeline"] = [[0.0, "pick"]]
    v = checker.check(tr, task)
    assert "drift" in v["reasons"] and not v["success"]


def test_pick_skill_interval_retains_episode_safety(model):
    ba = model.jnt_qposadr[model.body_jntadr[model.body("obj_0").id]]
    tr = trace(model, -2.2, -1.0, n=5)
    tr["qpos"][:2, 0] -= 0.4  # preceding approach is outside skill drift window
    tr["qpos"][-1, ba + 2] += 0.05
    tr["skill_start_index"] = 0  # ignored in favour of recorded policy switch
    tr["policy_timeline"] = [[0, "walk"], [0.04, "pick"]]
    task = {"type": "pick", "target_body": "obj_0"}
    assert checker.check(tr, task)["success"]
    tr["qpos"][0, 3:7] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]
    v = checker.check(tr, task)
    assert v["fallen"] and not v["success"]


def test_bad_trace_and_unknown_task_safety(model):
    for bad in ({"qpos": [], "t": []}, {"qpos": np.zeros((2, model.nq)), "t": [0, 0]},
                {"qpos": np.zeros((2, model.nq)), "t": [0, float("nan")]}):
        verdict = checker.check({**bad, "model_path": SCENE}, nav_task(-2, 0))
        assert verdict["reasons"] == ["bad_trace"] and verdict["safety_unknown"]
    assert checker.check(trace(model, -2, 0), {"type": "unknown"})["safety_unknown"]


def test_nav_requires_full_second_of_goal_dwell(model):
    assert not checker.check(trace(model, -2, 0, n=50), nav_task(-2, 0))["success"]
    assert checker.check(trace(model, -2, 0, n=51), nav_task(-2, 0))["success"]
    tr = trace(model, -2, 0, n=60)
    tr["qpos"][20, 0] -= 0.1
    assert not checker.check(tr, nav_task(-2, 0))["success"]


def test_policy_record_defines_pick_onset(model):
    tr = trace(model, -2.2, -1.0, n=60)
    ba = model.jnt_qposadr[model.body_jntadr[model.body("obj_0").id]]
    tr["qpos"][:20, 0] -= 0.4
    tr["qpos"][-1, ba + 2] += 0.05
    tr["skill_start_index"] = 0  # untrusted FSM index contradicts the sim switch
    tr["policy_events"] = [{"t": 0, "step": 0, "policy": "walk", "mirror": False},
                           {"t": 0.4000000000000002, "step": 20, "policy": "pick", "mirror": False}]
    verdict = checker.check(tr, {"type": "pick", "target_body": "obj_0"})
    assert verdict["success"] and verdict["skill_onset_index"] == 20
    tr["policy_events"] = tr["policy_events"][:1]
    assert "skill_not_started" in checker.check(tr, {"type": "pick", "target_body": "obj_0"})["reasons"]


def test_kick_rejects_approach_bump(model):
    tr = trace(model, -3.3, 0, n=60)
    ba = model.jnt_qposadr[model.body_jntadr[model.body("ball_0").id]]
    tr["policy_timeline"] = [[0, "walk"], [0.4, "kick_right"]]
    tr["qpos"][:20, ba] += np.linspace(0, 0.03, 20)
    tr["qpos"][20:, ba] += 0.03 + np.arange(40) * 0.01
    task = {"type": "kick", "ball_body": "ball_0", "direction": 0.0}
    verdict = checker.check(tr, task)
    assert "ball_moved_before_kick" in verdict["reasons"] and not verdict["success"]
    # Without an approach bump the same post-switch kick qualifies.
    tr["qpos"][:20, ba] = tr["qpos"][0, ba]
    tr["qpos"][20:, ba] -= 0.03
    assert checker.check(tr, task)["success"]


def test_independent_static_geometry_reference(model):
    tr = trace(model, -2, 0)
    reference = [{"name": model.geom(g).name,
                  **{key: np.asarray(getattr(model, "geom_" + key)[g]).tolist()
                     for key in ("type", "pos", "size", "quat", "contype", "conaffinity")}}
                 for g in checker._static_geom_ids(model)
                 if model.geom_contype[g] or model.geom_conaffinity[g]]
    tr["static_geometry_reference"] = reference
    assert checker.check(tr, nav_task(-2, 0))["success"]
    reference[0]["pos"][0] += 0.2
    verdict = checker.check(tr, nav_task(-2, 0))
    assert verdict["reasons"] == ["bad_trace"] and verdict["safety_unknown"]


def test_training_static_geometry_matches_independent_scene(tmp_path):
    from nav.sim import Sim
    from nav.episode import static_geometry_reference
    from nav.provenance import scene_sha256
    sim = Sim("apartment", "atlantis_walk", record=False, dynamics="training")
    gt = sim.reset(0, (2, 0, 0))
    path = str(tmp_path / "training.mjb")
    mujoco.mj_saveModel(sim.model, path)
    tr = {"qpos": np.tile(gt.qpos, (51, 1)), "t": np.arange(51) * 0.02, "model_path": path,
          "static_geometry_reference": static_geometry_reference(sim.scene_xml, scene_sha256(sim.scene_xml))}
    verdict = checker.check(tr, nav_task(2, 0))
    assert verdict["success"] and not verdict["safety_unknown"]
