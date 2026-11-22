"""Regression tests for the twelve repaired benchmark-referee audit findings.

Defect IDs match reports/audit_benchmark.md. These are ordinary passing tests;
real-Sim spawn/target checks run on one CPU process without freezing seed data.
"""
import json
import math
import os
import shutil
import sys

import mujoco
import numpy as np
import pytest

from nav import bench, checker

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from tools import make_seeds  # noqa: E402

SCENE = os.path.expanduser(
    "~/microduck/upstream/microduck_rl/src/mjlab_microduck/robot/microduck/scene_apartment.xml")
FROZEN_MAP = os.path.join(ROOT, "data", "map_59977bef39ede44c.npz")  # sigma 0.0197 (atlantis_walk gate), step 0
POLICY = "atlantis_walk"


def _stand_trace(model, xy_fn, yaw_fn, n, dt=0.02):
    q = model.key("STAND").qpos.copy()
    qs = np.tile(q, (n, 1))
    for k in range(n):
        qs[k, 0], qs[k, 1] = xy_fn(k * dt)
        a = yaw_fn(k * dt)
        qs[k, 3:7] = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
    return {"qpos": qs, "t": np.arange(n) * dt, "model_path": SCENE}


@pytest.fixture(scope="module")
def xml_model():
    return mujoco.MjModel.from_xml_path(SCENE)


@pytest.fixture(scope="module")
def sim():
    from nav.sim import Sim
    return Sim("apartment", POLICY, record=False, dynamics="training")


@pytest.fixture(scope="module")
def sim_mjb(sim, tmp_path_factory):
    p = str(tmp_path_factory.mktemp("mjb") / "train.mjb")
    mujoco.mj_saveModel(sim.model, p)
    return p


# ---------------------------------------------------------------- A-01 SPL path length
def test_A01_spl_path_length_not_inflated_by_gait_sway(xml_model):
    # 1.0 m straight walk at 0.1 m/s (pilot net speed), with lateral gait sway of 14 mm RMS
    # (20 mm amplitude) at 1.8 Hz, the measured spectrum peak of pilot_corridor_straight.
    amp, f, v = 0.02, 1.8, 0.1
    tr = _stand_trace(xml_model, lambda t: (-3.0 + v * min(t, 10.0), 0.5 + amp * math.sin(2 * math.pi * f * min(t, 10.0))),
                      lambda t: 0.0, n=551)
    vd = checker.check(tr, {"type": "nav", "goal": [-2.0, 0.5, 0.0]})
    assert vd["success"]
    from nav import stats
    term = stats.spl([True], [1.0], [vd["path_length"]])
    assert term >= 0.9, (vd["path_length"], term)


# ---------------------------------------------------------------- A-02 seeds spawnability
def test_A02_every_frozen_start_is_a_valid_spawn(tmp_path, sim):
    from nav.mapping import GridMap
    gm = GridMap.load(FROZEN_MAP)
    doc = make_seeds.generate(lambda scene: gm, SCENE, str(tmp_path / "seeds.json"))
    bad = []
    for kind in ("nav", "kick", "pick"):
        for i, t in enumerate(doc[kind]):
            try:
                sim.reset(t["seed"], t["start"])
            except ValueError as e:
                bad.append((kind, i, str(e)[:60]))
    assert not bad, bad


# ---------------------------------------------------------------- A-03 kick/pick targets
def test_A03_task_bodies_are_reachable_and_stable(sim, tmp_path):
    from nav.mapping import GridMap
    from nav.taskscene import apply_task_objects, body_pose
    from nav.skills import kick_pre_pose, pick_pre_pose
    gm = GridMap.load(FROZEN_MAP)
    doc = make_seeds.generate(lambda scene: gm, SCENE, str(tmp_path / "seeds.json"))
    for kind in ("kick", "pick"):
        for task in doc[kind]:
            sim.reset(task["seed"], task["start"])
            apply_task_objects(sim, task)
            body = task.get("ball_body", task.get("target_body"))
            initial = body_pose(sim, body)[:3]
            assert not gm.is_lethal(*initial[:2])
            pre = (kick_pre_pose(initial[:2], task["direction"], "right") if kind == "kick"
                   else pick_pre_pose(initial[:2], task["direction"]))
            assert not gm.is_lethal(*pre[:2])
            for _ in range(50):
                sim.step(np.zeros(3))
                assert np.linalg.norm(body_pose(sim, body)[:3] - initial) <= .01


# ---------------------------------------------------------------- A-04 hole incursion by geom extent
def test_A04_foot_overhanging_hole_is_counted(sim, sim_mjb):
    gt = sim.reset(0, (-0.2, -0.40, -math.pi / 2))
    q = gt.qpos.copy()
    yaw = math.radians(-35.0)
    q[1] = -0.656
    q[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    m = checker.load_model(sim_mjb)
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    g = m.geom("right_foot_collision").id
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    hole = checker.derive_hole(m)
    over = [hole.contains(px, py) for px, py, _ in w]
    assert np.mean(over) > 0.2  # precondition: >20% of the foot's collision vertices are over the hole
    vd = checker.check({"qpos": np.tile(q, (3, 1)), "t": np.arange(3) * 0.02, "model_path": sim_mjb},
                       {"type": "nav", "goal": [-0.2, -0.656, yaw]})
    assert vd["hole_incursion"]
    q[2] += 0.2  # elevated swing extents do not count while the trunk remains outside
    elevated = checker.check({"qpos": np.tile(q, (3, 1)), "t": np.arange(3) * 0.02, "model_path": sim_mjb},
                             {"type": "nav", "goal": [-0.2, -0.656, yaw]})
    assert not elevated["hole_incursion"]


# ---------------------------------------------------------------- A-05 checker-model cache key
def test_A05_checker_model_tracks_included_geometry(tmp_path, monkeypatch):
    import nav.sim as navsim
    from nav.episode import EpisodeRunner
    data = tmp_path / "data"
    data.mkdir()
    for f in (f"envelope_{POLICY}.json", f"primitives_{POLICY}.json"):
        shutil.copy(os.path.join(ROOT, "data", f), data / f)
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    (scene_dir / "scene.xml").write_text(
        '<mujoco><include file="room.xml"/></mujoco>')

    class Stop(Exception):
        pass

    class StubSim:
        def __init__(self, *a, **k):
            self.scene_xml = str(scene_dir / "scene.xml")
            self.model = mujoco.MjModel.from_xml_path(self.scene_xml)

        def reset(self, *a, **k):
            raise Stop

    monkeypatch.setattr(navsim, "Sim", StubSim)
    runner = EpisodeRunner(policy=POLICY, dynamics="training", data_dir=str(data))
    for wall_x in (1.0, 2.0):  # the included geometry moves a wall
        (scene_dir / "room.xml").write_text(
            f'<mujoco><worldbody><geom name="wall" type="box" size=".1 1 .2" pos="{wall_x} 0 .2"/>'
            '</worldbody></mujoco>')
        with pytest.raises(Stop):
            runner({"seed": 0, "start": [0, 0, 0], "type": "nav", "goal": [0, 0, 0]})
    mjbs = sorted((data / "checker_models").glob("*.mjb"))
    xs = {round(float(mujoco.MjModel.from_binary_path(str(p)).geom("wall").pos[0]), 3) for p in mjbs}
    assert xs == {1.0, 2.0}, xs


# ---------------------------------------------------------------- A-06 latest report ordering
def test_A06_lock_changed_uses_most_recent_report(tmp_path):
    lock = tmp_path / "policies.lock.json"
    lock.write_text('{"a": 1}')
    old_md5 = bench.md5_file(str(lock))
    rep = tmp_path / "rep"
    rep.mkdir()
    # older run, alphabetically LATER name, recorded with the old lock
    (rep / "bench_walk_gt_training_20261101_000000.json").write_text(json.dumps({"lock_md5": old_md5, "timestamp": "20261101_000000",
                    "gate": {"checks": {"full_frozen_suite": True}}}))
    lock.write_text('{"a": 2}')
    # newer run, alphabetically earlier name, recorded with the current lock
    (rep / "bench_atlantis_walk_gt_training_20261109_000000.json").write_text(
        json.dumps({"lock_md5": bench.md5_file(str(lock)), "timestamp": "20261109_000000",
                    "gate": {"checks": {"full_frozen_suite": True}}}))
    assert not bench.lock_changed(str(rep), str(lock))


# ---------------------------------------------------------------- A-07 bad_trace safety
def test_A07_bad_trace_marks_safety_unknown(xml_model):
    q = xml_model.key("STAND").qpos.copy()
    tr = {"qpos": np.tile(np.r_[q, np.zeros(14)], (3, 1)), "t": np.arange(3) * 0.02, "model_path": SCENE}
    vd = checker.check(tr, {"type": "nav", "goal": [0, 0, 0]})
    assert vd["reasons"] == ["bad_trace"]
    rep = bench.aggregate([{"type": "nav"}], [vd])
    assert bench.target_gate(rep)["checks"]["safety_observed"] is False


# ---------------------------------------------------------------- A-08 stopped at goal
def test_A08_nav_success_requires_robot_stopped(xml_model):
    # passes through the goal at 0.3 m/s; the last frame is 4 cm short of the goal
    tr = _stand_trace(xml_model, lambda t: (-3.0 + 0.3 * t, 0.0), lambda t: 0.0, n=161)
    goal_x = float(tr["qpos"][-1, 0]) + 0.04
    vd = checker.check(tr, {"type": "nav", "goal": [goal_x, 0.0, 0.0]})
    assert not vd["success"]


# ---------------------------------------------------------------- A-09 fall definition
def test_A09_fall_matches_sim_ground_truth(sim, sim_mjb):
    from nav import sim as navsim
    gt = sim.reset(0, (-0.2, -0.40, -math.pi / 2))
    q = gt.qpos.copy()
    # same upright posture standing on stair_1 (top z = -0.10): rigid shift down by 0.10 m
    q[1], q[2] = -0.79, q[2] - 0.10
    m = checker.load_model(sim_mjb)
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    foot_z = min(d.geom_xpos[m.geom(f"{s}_foot_collision").id][2] for s in ("left", "right"))
    sim_fallen = (q[2] - foot_z) < navsim.FALL_HEIGHT_M  # tilt is 0
    vd = checker.check({"qpos": np.tile(q, (3, 1)), "t": np.arange(3) * 0.02, "model_path": sim_mjb},
                       {"type": "nav", "goal": [-0.2, -0.79, -math.pi / 2]})
    assert vd["fallen"] == sim_fallen, (vd["fallen"], sim_fallen, q[2], foot_z)


# ---------------------------------------------------------------- A-10 compare() provenance
def test_A10_compare_refuses_different_seed_sets():
    from nav import stats
    def rep(k, n, sha):
        return {"seeds_sha256": sha, "by_task": {"nav": {"n": n, "success_rate": k / n,
                                                         "success_ci": list(stats.wilson_ci(k, n))}}}
    with pytest.raises((ValueError, RuntimeError)):
        bench.compare(rep(50, 200, "aaa"), rep(180, 200, "bbb"))


# ---------------------------------------------------------------- A-11 geodesic metric
def test_A11_geodesic_close_to_euclidean_in_open_space():
    class Open:
        res = 0.02
        lethal = np.zeros((200, 200), bool)
        def world_to_cell(self, x, y):
            return int(y / self.res), int(x / self.res)
    gm = Open()
    dist = make_seeds.geodesic_from(gm, (0.01, 0.01))
    i, j = 50, 100  # direction atan(0.5) = 26.6 deg; Euclidean = 2.236 m
    eu = math.hypot(i, j) * gm.res
    assert dist[i, j] / eu < 1.02, dist[i, j] / eu


# ---------------------------------------------------------------- A-12 crash retention
def _raising_runner(task):
    raise RuntimeError("boom inside episode")


def test_A12_crashed_episode_retains_debug_artifacts(tmp_path):
    seeds = str(tmp_path / "s.json")
    doc = {"version": "test", "nav": [{"seed": 0, "start": [0, 0, 0], "goal": [0, 0, 0]}]}
    doc["sha256"] = bench.seeds_digest(doc)
    with open(seeds, "w") as handle:
        json.dump(doc, handle)
    lock = tmp_path / "lock.json"
    lock.write_text("{}")
    bench.run_bench(_raising_runner, "crash", seeds_path=seeds, lock_path=str(lock), procs=1, limit=1,
                    reports_dir=str(tmp_path / "rep"))
    rec = json.load(open(tmp_path / "rep" / "failures" / "crash" / "nav_000.json"))
    assert "Traceback" in json.dumps(rec)
