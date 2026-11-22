"""Same-apartment mjlab replay cross-check: harness tests (nav Python; the mjlab part is checked via its outputs).

  cd ~/microduck/nav && PYTHONPATH=. $PY -m pytest tests/test_replay.py -q
Set REPLAY_SLOW=1 to also run a 25-step live mjlab replay through the reference Python (~1 min).
"""
import ast
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools import replay_compare as rc
from tools import replay_mjlab as rm

NAV = Path(__file__).resolve().parents[1]
PILOTS = NAV / "reports/replay_pilot_v2/traces"   # v1 pilots (reports/replay_pilot) predate sim training dynamics v2
PILOT = PILOTS / "pilot_corridor_straight.npz"


def _truncate(tr, n):
    out = dict(tr)
    out["qpos"], out["t"] = tr["qpos"][: n + 1], tr["t"][: n + 1]
    return out


def test_step_schedule_alignment_and_switches():
    meta = {"commands": [[0.0, 0, 0, 0], [0.02, 0.1, 0, 0.2], [0.1, 0.3, 0, 0], [0.12, 0, 0, 0]],
            "policy_timeline": [[0.0, "walk"], [0.1, "pick"]]}
    pol, cmd, missing = rm.step_schedule(meta, 7)
    assert pol == ["walk"] * 5 + ["pick"] * 2
    assert cmd[1] == [0.1, 0.0, 0.2] and cmd[5] == [0.3, 0.0, 0.0]
    assert cmd[2] == cmd[3] == cmd[4] == [0.0, 0.0, 0.0]        # hand-off steps: no control -> zero
    assert missing == 3                                           # steps 2,3,4


def test_command_block_modes():
    tw = rm.command_block({"command_mode": "twist"}, [0.1, -0.2, 0.3], 1)
    assert np.allclose(tw[:3], [0.1, -0.2, 0.3]) and not tw[3:].any()
    assert not rm.command_block({"command_mode": "zero"}, [1, 1, 1], 5).any()
    ph = rm.command_block({"command_mode": "phase", "period_s": 4.0}, [0, 0, 0], 50)   # phase 0.25
    assert np.allclose(ph[:2], [0.0, 1.0], atol=1e-6)
    assert rm.command_block({"command_mode": "posture_flag"}, [1, 0, 0], 3)[0] == 1.0


def test_exact_replay_preserves_frozen_object_placements():
    from nav.sim import Sim
    from nav.taskscene import apply_task_objects
    sim = Sim("apartment", "walk", dynamics="training", record=True)
    sim.reset(7, [2., 0., 0.])
    apply_task_objects(sim, {"object_poses": {"obj_1": [-3., 2.2, 0., 1., 0., 0., 0.]}})
    commands = []
    for k in range(10):
        commands.append([k * .02, 0., 0., 0.])
        sim.step(np.zeros(3))
    trace = {"qpos": np.array(sim.trace), "meta": {"provenance": sim.provenance(),
             "scene_xml": sim.scene_xml, "policy": "walk", "dynamics": "training",
             "commands": commands, "policy_timeline": [[0., "walk"]]}}
    assert np.array_equal(rc.nav_replay(trace), trace["qpos"])


@pytest.mark.skipif(not PILOT.exists(), reason="pilot traces not generated")
def test_exact_nav_replay_is_bitwise():
    tr = _truncate(rm.load_trace(PILOT), 200)
    q = rc.nav_replay(tr)
    assert q.shape == tr["qpos"].shape
    assert np.array_equal(q, tr["qpos"])


@pytest.mark.skipif(not PILOT.exists(), reason="pilot traces not generated")
def test_perturbed_replay_differs_but_starts_close():
    tr = _truncate(rm.load_trace(PILOT), 100)
    q = rc.nav_replay(tr, 0, [1e-6, 0.0])
    assert abs(q[0, 0] - tr["qpos"][0, 0] - 1e-6) < 1e-12
    assert not np.array_equal(q, tr["qpos"])


@pytest.mark.skipif(not PILOT.exists(), reason="pilot traces not generated")
def test_joint_name_layout_covers_training_model():
    import mujoco
    tr = rm.load_trace(PILOT)
    m = mujoco.MjModel.from_binary_path(tr["meta"]["model_path"])
    lay = rm.nav_joint_layout(m)
    assert sum(w for _, w in lay.values()) == m.nq == tr["qpos"].shape[1]
    assert "passive_left_knee_backlash" in lay and "trunk_base_freejoint" in lay
    assert {f"body:obj_{i}" for i in range(4)} | {f"body:ball_{i}" for i in range(3)} <= set(lay)
    # qpos addresses are disjoint (mapping is by name, never by index)
    used = sorted(a + i for a, w in lay.values() for i in range(w))
    assert used == list(range(m.nq))


def test_mjlab_layout_strips_entity_prefix():
    import mujoco
    child = mujoco.MjSpec()
    b = child.worldbody.add_body(name="trunk_base")
    b.add_freejoint(name="trunk_base_freejoint")
    b.add_geom(size=[0.05, 0, 0])
    lb = b.add_body(name="leg", pos=[0, 0, -0.1])
    lb.add_joint(name="left_knee", type=mujoco.mjtJoint.mjJNT_HINGE)
    lb.add_geom(size=[0.02, 0, 0])
    parent = mujoco.MjSpec()
    parent.attach(child, prefix="robot/", frame=parent.worldbody.add_frame())
    ob = parent.worldbody.add_body(name="obj_0")
    ob.add_freejoint()
    ob.add_geom(size=[0.02, 0, 0])
    lay = rm.mj_joint_layout(parent.compile())
    assert set(lay) == {"trunk_base_freejoint", "left_knee", "body:obj_0"}
    assert lay["trunk_base_freejoint"][2] == 7 and lay["left_knee"][2] == 1


def test_reference_tool_is_independent_of_nav_package():
    tree = ast.parse((NAV / "tools/replay_mjlab.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] in ("nav", "tools") for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in ("nav", "tools")


def _crit():
    return {"position_rule": {"floor_m": 0.10, "factor": 2.0}}


def _o(fell=False, hole=False, coll=False, xy=(0.0, 0.0)):
    return {"fell": fell, "stair_hole": hole, "static_collision": coll, "final_xy": list(xy)}


def test_judge_rules():
    cloud = {"original": _o(), "replicates": [{"outcome": _o(), "divergence": {"end": 0.02}},
                                              {"outcome": _o(coll=True), "divergence": {"end": 0.30}}]}
    j = rc.judge(_crit(), cloud, _o(xy=(0.55, 0.0)))
    assert j["final_position"]["X_m"] == pytest.approx(0.60) and j["agree"]
    assert not rc.judge(_crit(), cloud, _o(xy=(0.65, 0.0)))["agree"]
    assert rc.judge(_crit(), cloud, _o(coll=True))["static_collision"]["agree"]      # inside nav set
    j = rc.judge(_crit(), cloud, _o(hole=True))
    assert not j["stair_hole"]["agree"] and not j["agree"]
    j = rc.judge(_crit(), cloud, _o(fell=True, xy=(5, 5)))
    assert not j["fell"]["agree"] and j["final_position"]["agree"] and not j["final_position"]["applies"]
    tight = {"original": _o(), "replicates": [{"outcome": _o(), "divergence": {"end": 0.0}}]}
    assert rc.judge(_crit(), tight, _o(xy=(0.09, 0.0)))["agree"]                       # floor 0.10 m


def test_select_skips_unreplayable(tmp_path):
    base = {"commands": [[0.0, 0, 0, 0]], "policy_timeline": [[0.0, "walk"]], "dynamics": "training"}
    def put(name, task, meta):
        np.savez_compressed(tmp_path / f"{name}.npz", qpos=np.zeros((2, 3)), t=np.zeros(2),
                            metadata=json.dumps(meta), task=json.dumps(task))
    put("a", {"ep": "a", "type": "nav"}, base)
    put("b", {"ep": "b", "type": "kick"}, base)
    put("c", {"ep": "c", "type": "nav"}, dict(base, policy_timeline=[[0.0, "walk"], [3.0, "kick_right"]]))
    put("d", {"ep": "d", "type": "nav"}, dict(base, dynamics="infer"))
    keep, skipped = rc.select(tmp_path, 20)
    assert [Path(p).name for p in keep] == ["a.npz"]
    assert skipped == {"b.npz": "not_nav_task", "c.npz": "kick_policy:kick_right", "d.npz": "not_training_dynamics"}


def test_criterion_frozen_and_complete():
    crit = json.loads((NAV / "reports/replay_criterion.json").read_text())
    for k in ("outcomes", "binary_rule", "position_rule", "episode_rule", "verdict_rule", "sample_rule",
              "nav_reference_set", "calibration_sha256", "frozen_at"):
        assert k in crit
    assert crit["position_rule"]["factor"] == 2.0 and crit["position_rule"]["floor_m"] == 0.10
    assert rc.sha256(NAV / crit["calibration_file"]) == crit["calibration_sha256"]
    with pytest.raises(SystemExit):
        rc.freeze()                       # refuses to overwrite a frozen criterion


@pytest.mark.skipif(not (NAV / "reports/replay_pilot_check.json").exists(), reason="pilot check not run")
def test_pilot_report_consistency():
    rep = json.loads((NAV / "reports/replay_pilot_check.json").read_text())
    assert rep["criterion_sha256"] == rc.sha256(NAV / "reports/replay_criterion.json")
    assert rep["exact_replays_bitwise"] is True
    assert rep["mjlab_init_state_max_err"] < 1e-6
    for g in rep["mjlab_summary"]["geometry_check"].values():
        assert g["missing_in_mjlab"] == [] and g["type_mismatch"] == []
        assert g["max_pos_err"] < 1e-9 and g["max_size_err"] < 1e-9 and g["robot_max_body_mass_err"] < 1e-9
    assert rep["n_disagree"] == sum(not e["judgement"]["agree"] for e in rep["episodes"])


@pytest.mark.skipif(os.environ.get("REPLAY_SLOW") != "1" or not PILOT.exists(), reason="set REPLAY_SLOW=1")
def test_live_mjlab_replay_short(tmp_path):
    tr = rm.load_trace(PILOT)
    short = tmp_path / "short.npz"
    np.savez_compressed(short, qpos=tr["qpos"][:26], t=tr["t"][:26], metadata=json.dumps(tr["meta"]),
                        task=json.dumps(tr["task"]))
    subprocess.run([str(rc.REF_PY), str(NAV / "tools/replay_mjlab.py"), "--out", str(tmp_path / "o"), str(short)],
                   check=True, cwd=str(NAV), env=dict(os.environ, OMP_NUM_THREADS="1"), capture_output=True)
    s = json.loads((tmp_path / "o/replay_mjlab_summary.json").read_text())
    e = s["episodes"][0]
    assert e["init_state_max_err"] < 1e-6 and e["nan_step"] is None
    q = np.load(e["file"])["qpos"]
    assert q.shape == (26, tr["qpos"].shape[1]) and np.isfinite(q).all()
    assert np.abs(q[-1, :2] - tr["qpos"][25, :2]).max() < 0.05      # 0.5 s: still near the nav state
