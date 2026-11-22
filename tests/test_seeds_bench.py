import json, os, sys, types
import numpy as np
import pytest
from nav import bench

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from tools import make_seeds


class StubMap:
    res = 0.05
    def __init__(self):
        self.lethal = np.zeros((120, 160), bool)
        self.lethal[0, :] = self.lethal[-1, :] = True
        self.lethal[:, 0] = self.lethal[:, -1] = True
        self.lethal[50:70, 30:34] = True  # a wall with gaps
        self.rooms = {"a": [-4, -3, -0.5, 0], "b": [-4, 0, -0.5, 3], "c": [0, -3, 4, 0],
                      "d": [0, 0, 4, 3], "e": [-0.5, -3, 0, 3], "f": [-3, 0.2, -2, 0.8]}
    def world_to_cell(self, x, y):
        return int((y + 3) / self.res), int((x + 4) / self.res)
    def is_lethal(self, x, y):
        i, j = self.world_to_cell(x, y)
        return bool(self.lethal[i, j])


def test_generate_stub(tmp_path):
    out = str(tmp_path / "s.json")
    d = make_seeds.generate(lambda scene: StubMap(), "scene_apartment.xml", out)
    assert (len(d["nav"]), len(d["kick"]), len(d["pick"])) == (200, 50, 50)
    assert all(isinstance(t["seed"], int) for k in ("nav", "kick", "pick") for t in d[k])
    assert {t["category"] for t in d["nav"]} >= {"same_room", "cross_room", "near_hole"}
    assert all(t["geodesic"] >= 0.5 for t in d["nav"])
    assert bench.load_seeds(out)["sha256"] == d["sha256"]
    # deterministic
    d2 = make_seeds.generate(lambda scene: StubMap(), "scene_apartment.xml", str(tmp_path / "s2.json"))
    assert d2["sha256"] == d["sha256"]
    # tamper -> refuse
    doc = json.load(open(out)); doc["nav"][0]["goal"][0] += 0.01
    json.dump(doc, open(out, "w"))
    with pytest.raises(RuntimeError):
        bench.load_seeds(out)


def test_compare_flags():
    def rep(k, n):
        from nav import stats
        lo, hi = stats.wilson_ci(k, n)
        return {"seeds_sha256": "same", "mode": "gt", "by_task": {"nav": {"n": n, "success_rate": k / n, "success_ci": [lo, hi]}}}
    assert bench.compare(rep(90, 100), rep(92, 100)) == []        # overlap
    fl = bench.compare(rep(50, 200), rep(180, 200))
    assert fl and fl[0]["metric"] == "success_rate"
    assert bench.compare(rep(180, 200), rep(50, 200)) == []        # improvement


def test_geodesic_cannot_cut_lethal_corner():
    gm = StubMap()
    gm.lethal[:] = True
    gm.lethal[1, 1] = gm.lethal[2, 2] = False
    dist = make_seeds.geodesic_from(gm, (-4 + 1.5 * gm.res, -3 + 1.5 * gm.res))
    assert np.isinf(dist[2, 2])


def dummy_runner(task):
    import mujoco
    from tests.test_checker import SCENE
    m = mujoco.MjModel.from_xml_path(SCENE)
    q = m.key("STAND").qpos.copy()
    if task["type"] == "nav":
        q[0], q[1] = task["goal"][0], task["goal"][1]
        yaw = task["goal"][2]; q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    return {"qpos": np.tile(q, (3, 1)), "t": np.arange(3) * 0.02, "model_path": SCENE, "plan_time_s": 0.1}


def test_run_bench_smoke(tmp_path):
    seeds = str(tmp_path / "s.json")
    make_seeds.generate(lambda scene: StubMap(), "scene_apartment.xml", seeds)
    lock = tmp_path / "policies.lock.json"; lock.write_text(json.dumps({"walk": {"md5": "abc", "path": "p"}}))
    out = bench.run_bench(dummy_runner, "t", seeds_path=seeds, lock_path=str(lock), procs=1, limit=3,
                          reports_dir=str(tmp_path / "rep"))
    rep = json.load(open(out))
    assert rep["policies"]["walk"]["md5"] == "abc" and rep["by_task"]["nav"]["n"] == 3
    assert rep["lock_md5"] == bench.md5_file(str(lock))
    assert bench.lock_changed(str(tmp_path / "rep"), str(lock))  # limited smoke run is not latest full suite
    lock.write_text("{}")
    assert bench.lock_changed(str(tmp_path / "rep"), str(lock))


def test_compare_requires_matching_provenance_and_detects_rate_regression():
    from nav import stats
    def report(k=190, n=200, seed="same", mode="gt"):
        return {"mode": mode, "seeds_sha256": seed, "by_task": {"nav": {
            "n": n, "success_rate": k / n, "success_ci": list(stats.wilson_ci(k, n))}}}
    base = report()
    for other in (report(seed="changed"), report(mode="odom"), report(n=201)):
        with pytest.raises(ValueError):
            bench.compare(other, base)
    flags = bench.compare(report(k=176), base)
    assert len(flags) == 1 and flags[0]["p_value"] < 0.05


def test_latest_ignores_partial_and_uses_report_timestamp(tmp_path):
    def save(name, timestamp, full=True, limit=None):
        path = tmp_path / name
        path.write_text(json.dumps({"timestamp": timestamp, "limit": limit,
                                    "gate": {"checks": {"full_frozen_suite": full}}}))
        return str(path)
    latest = save("bench_aaa_oldfilename.json", "20261109_010000")
    save("bench_zzz_newfilename.json", "20261101_010000")
    save("bench_zzz_smoke.json", "20261109_020000", False, 3)
    save("bench_zzz_limitedfull.json", "20261109_030000", True, 300)
    save("bench_bad_timestamp.json", "invalid")
    assert bench.latest_report(str(tmp_path)) == latest


def test_aggregate_success_time_and_every_plan_call():
    tasks = [{"type": "nav", "geodesic": 1, "room_pair": ["office", "bedroom"]},
             {"type": "nav", "geodesic": 1, "room_pair": ["office", "bedroom"]}]
    verdicts = [{"success": True, "time_to_goal": 2, "plan_times_s": [0.01, 0.03], "path_length": 1},
                {"success": False, "time_to_goal": 100, "plan_times_s": [0.02], "path_length": 1}]
    report = bench.aggregate(tasks, verdicts)
    metrics = report["by_task"]["nav"]
    assert report["by_room_pair"]["office->bedroom"]["n"] == 2
    assert metrics["time_to_goal"]["p50"] == 2
    assert metrics["plan_time_s"]["p50"] == 0.02


def test_crash_partial_trace_and_video_attempt(tmp_path):
    from tests.test_checker import SCENE
    import mujoco
    m = mujoco.MjModel.from_xml_path(SCENE)
    q = m.key("STAND").qpos.copy()
    class CrashRunner:
        def __call__(self, task):
            self.partial_trace = {"qpos": np.tile(q, (2, 1)), "t": [0, 0.02], "model_path": SCENE}
            raise RuntimeError("partial simulation crash")
    doc = {"nav": [{"seed": 0, "start": [0, 0, 0], "goal": [0, 0, 0]}]}
    doc["sha256"] = bench.seeds_digest(doc)
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps(doc))
    calls = []
    def video(task, tr, path):
        calls.append(path)
        raise RuntimeError("test renderer unavailable")
    report_path = bench.run_bench(CrashRunner(), "partial", seeds_path=str(path), procs=1,
                                  reports_dir=str(tmp_path / "reports"), video_hook=video)
    report = json.load(open(report_path))
    verdict = report["episodes"]["nav_000"]
    failure = json.load(open(tmp_path / "reports/failures/partial/nav_000.json"))
    assert "Traceback" in verdict["traceback"] and verdict["safety_unknown"]
    assert failure["trace"]["qpos"] and os.path.exists(verdict["trace_file"])
    assert calls and "renderer unavailable" in verdict["video_error"]


def test_policy_preflight_preserves_provenance_for_comparison(tmp_path, monkeypatch):
    from nav import episode, mapsigma, provenance
    env = {"policy_md5": "verified-md5", "dynamics": "training", "dynamics_version": "current"}
    monkeypatch.setattr(episode, "read_artifacts", lambda policy, data: (env, dict(env)))
    monkeypatch.setattr(episode, "artifact_digest", lambda path: "artifact-sha")
    monkeypatch.setattr(mapsigma, "planning_lateral_sigma", lambda policy, env: (0.02, {"follower_gate_passed": True}))
    monkeypatch.setattr(provenance, "scene_sha256", lambda scene, dynamics: "scene-sha")
    doc = {"nav": [{"seed": 0, "start": [0, 0, 0], "goal": [0, 0, 0]}],
           "kick": [{"seed": 1, "start": [0, 0, 0]}],
           "provenance": {"policy": "primary", "policy_md5": "verified-md5", "dynamics": "training",
                          "dynamics_version": "current", "scene_sha256": "scene-sha",
                          "envelope_sha256": "artifact-sha", "lateral_sigma": 0.02,
                          "path_smoothing_steps": 30, "goal_dwell_s": 1.0,
                          "geodesic_method": "footprint_16_dijkstra_los_v1"}}
    doc["sha256"] = bench.seeds_digest(doc)
    seed_path = tmp_path / "seeds.json"
    seed_path.write_text(json.dumps(doc))
    class Runner:
        policy, dynamics, mode, scene = "alternate", "training", "gt", "apartment"
        data_dir, comparison_policy = str(tmp_path), False
        def __call__(self, task):
            return dummy_runner(task)
    runner = Runner()
    kwargs = dict(seeds_path=str(seed_path), reports_dir=str(tmp_path / "reports"), procs=1)
    with pytest.raises(RuntimeError, match="measurements"):
        bench.run_bench(runner, "wrong-primary", **kwargs)
    runner.comparison_policy = True
    with pytest.raises(RuntimeError, match="only.*navigation"):
        bench.run_bench(runner, "wrong-types", task_types=["nav", "kick"], **kwargs)
    report = json.load(open(bench.run_bench(runner, "alternate", **kwargs)))
    assert report["seeds_sha256"] == doc["sha256"] and report["task_counts"] == {"nav": 1}
    assert not report["gate"]["checks"]["full_frozen_suite"]
    monkeypatch.setattr(provenance, "scene_sha256", lambda scene, dynamics: "changed-scene")
    with pytest.raises(RuntimeError, match="scene include/asset closure"):
        bench.run_bench(runner, "changed-scene", **kwargs)
    monkeypatch.setattr(provenance, "scene_sha256", lambda scene, dynamics: "scene-sha")
    env["dynamics_version"] = "stale"
    with pytest.raises(RuntimeError, match="dynamics/version"):
        bench.run_bench(runner, "stale-dynamics", **kwargs)
    env["dynamics_version"] = "current"
    for key, bad in (("path_smoothing_steps", 50), ("goal_dwell_s", 0), ("geodesic_method", "octile")):
        original = doc["provenance"][key]
        doc["provenance"][key] = bad
        doc["sha256"] = bench.seeds_digest(doc)
        seed_path.write_text(json.dumps(doc))
        with pytest.raises(RuntimeError, match="checker contract"):
            bench.run_bench(runner, "changed-metric", **kwargs)
        doc["provenance"][key] = original
