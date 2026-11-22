"""Phase A tests: obs parity with infer_policy.py, determinism, RTF, smoke walks.

Run: cd ~/microduck/nav && $PY -m pytest tests/test_sim.py -q
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

NAV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NAV))

from nav import policies as pol  # noqa: E402
from nav import sim as simmod  # noqa: E402
from nav.sim import Sim  # noqa: E402

UPSTREAM = pol.microduck_root() / "upstream/microduck_rl"
SCENES = UPSTREAM / "src/mjlab_microduck/robot/microduck"
OBS_TOL = 1e-6


def test_training_dynamics_is_explicit_and_seeded():
    with _quiet():
        a = Sim("flat", "walk", dynamics="training", record=False)
        b = Sim("flat", "walk", dynamics="training", record=False)
    assert a.model.nq == 35  # free base + 14 motor and 14 passive backlash hinges
    assert len(a.backlash_qpos_indices) == 14
    assert a.bam_model.actuator.max_current == 1.75
    for s in (a, b):
        s.reset(4, (0., 0., 0.))
    for _ in range(30):
        ga, oa = a.step([.3, 0., .5])
        gb, ob = b.step([.3, 0., .5])
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(ga.qpos, gb.qpos)
    assert a.provenance()["dynamics"] == "training"
    assert a.params["dynamics"] == "training"


def test_training_encoder_sees_passive_backlash_and_observe_is_idempotent():
    with _quiet():
        s = Sim("flat", "walk", dynamics="training", record=False)
    s.reset(0, (0., 0., 0.))
    s.data.qpos[s.backlash_qpos_indices] = .01
    mujoco.mj_forward(s.model, s.data)
    obs = s.observe()
    np.testing.assert_allclose(obs[6:20], .01, atol=1e-7)
    np.testing.assert_array_equal(obs, s.observe())
    assert len(s._obs_history) == 1


def test_default_dynamics_preserves_infer_model():
    with _quiet():
        s = Sim("flat", record=False)
    assert s.dynamics == "infer"
    assert s.model.nq == 21
    assert s.bam_model.actuator.max_current is None
    with pytest.raises(ValueError, match="dynamics"):
        Sim("flat", dynamics="unknown")


@pytest.fixture(scope="module")
def ip():
    """infer_policy.py as a module (reference runtime, independent code path)."""
    spec = importlib.util.spec_from_file_location("infer_policy_ref", UPSTREAM / "scripts" / "infer_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def test_constants_mirror_infer_policy(ip):
    for name in ["BAM_MOTOR_NAME", "BAM_MODEL", "BAM_KP_FW", "BAM_VIN_RANGE", "BAM_VIN_DROP_GAIN_RANGE",
                 "BAM_VIN_MIN", "BAM_MAX_CURRENT", "BAM_STIFF_SOLREF_FRICTION", "BAM_STIFF_SOLIMP_FRICTION",
                 "BALL_OFFSET_X", "BALL_OFFSET_ABS_Y", "BALL_RADIUS"]:
        assert getattr(simmod, name) == getattr(ip, name), name
    assert simmod.DEFAULT_POSE.dtype == ip.DEFAULT_POSE.dtype
    assert np.array_equal(simmod.DEFAULT_POSE, ip.DEFAULT_POSE)


# ---------------------------------------------------------------------------
# Reference: infer_policy.PolicyInference driven exactly like infer_policy.main()
# ---------------------------------------------------------------------------
class Ref:
    def __init__(self, ip, scene: str, policy_key: str, vin=7.4, gain=0.1, pose=(0.0, 0.0, 0.0)):
        self.ip = ip
        lock = pol.load_lock()
        entry = lock[policy_key]
        self.mode = entry["command_mode"]
        onnx = str(pol.resolve(entry))
        with _quiet():
            bm = ip.load_bam_model(ip.BAM_KP_FW, vin, 0.0)
            self.model, self.data, self.bam, _ = ip.load_mujoco_with_bam(
                str(SCENES / scene), bm, 0.005, gain if gain > 0 else None, ip.BAM_VIN_MIN)
            kw = dict(model=self.model, data=self.data, bam_ctrl=self.bam,
                      use_projected_gravity=True, new_cmd_obs=True)
            if self.mode == "twist":
                kw["walking_onnx_path"] = onnx
            elif self.mode == "phase":
                kw["walking_onnx_path"] = str(pol.resolve(lock["walk"]))
                kw["ground_pick_onnx_path"] = onnx
                kw["ground_pick_period"] = entry["period_s"]
            elif self.mode == "posture_flag":
                kw["sitstand_onnx_path"] = onnx
            else:
                raise NotImplementedError(self.mode)
            self.p = ip.PolicyInference(**kw)
            self.p.set_vel_cmd(0.0, 0.0, 0.0)
        m, d, p = self.model, self.data, self.p
        fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        a = m.jnt_qposadr[fj]
        x, y, th = pose
        d.qpos[a:a + 3] = [x, y, 0.125]
        d.qpos[a + 3:a + 7] = [math.cos(th / 2), 0, 0, math.sin(th / 2)]
        for i, q in enumerate(p.joint_qpos_indices):
            d.qpos[q] = p.default_pose[i]
        self.bam.reset(d.qpos)
        p.set_position_targets(p.default_pose)
        mujoco.mj_forward(m, d)
        if self.mode == "phase":
            p.trigger_ground_pick()

    def set_command(self, cmd):
        with _quiet():
            if self.mode == "twist":
                self.p.set_vel_cmd(*[float(c) for c in cmd])
            elif self.mode == "posture_flag":
                want = bool(cmd[0] > 0.5)
                if want != self.p.sit_mode:
                    self.p.toggle_sit()
            elif self.mode == "phase":
                self.p.update_ground_pick_phase(0.02)

    def obs_and_step(self):
        obs = self.p.get_observations()
        action = self.p.infer()
        self.p.apply_action(action)
        for _ in range(4):
            self.bam.update()
            mujoco.mj_step(self.model, self.data)
        return obs


def _cmd_schedule(k: int, mode: str):
    if mode == "posture_flag":
        return np.array([1.0 if 100 <= k < 300 else 0.0, 0, 0])
    if k < 50:
        return np.zeros(3)
    if k < 250:
        return np.array([0.3, 0.0, 0.0])
    if k < 400:
        return np.array([0.25, 0.1, 0.5])
    return np.array([0.0, 0.0, -0.8])


@pytest.mark.parametrize("scene,key,pose", [
    ("scene.xml", "walk", (0.0, 0.0, 0.0)),
    ("scene_apartment.xml", "walk", (2.0, 0.0, 0.7)),
    ("scene.xml", "pick", (0.0, 0.0, 0.0)),
    ("scene.xml", "sitstand", (0.0, 0.0, 0.0)),
    ("scene.xml", "velstand_rough", (0.0, 0.0, -1.2)),
])
def test_obs_matches_infer_policy(ip, scene, key, pose):
    """500 steps: (a) state-synced: the reference reads the Sim's exact MuJoCo state
    (mj_copyData) and must build the same obs (|d| <= 1e-6); (b) free-running: both
    sims evolve independently from the same init + commands; obs must still agree."""
    n = 500
    sim = Sim(scene, key)
    sim.reset(seed=0, start_pose=pose, extra={"vin": 7.4, "vin_drop_gain": 0.1})
    synced = Ref(ip, scene, key, pose=pose)
    free = Ref(ip, scene, key, pose=pose)
    mode = sim.command_mode
    # same initial state
    assert np.array_equal(sim.data.qpos, free.data.qpos)
    worst_sync, worst_free = 0.0, 0.0
    for k in range(n):
        if mode == "phase" and sim.skill_done:
            n = k          # infer_policy hands back to walking at phase 0.7 (139 steps)
            break
        cmd = _cmd_schedule(k, mode)
        # (a) state-synced reference: copy Sim state into the reference's MjData.
        mujoco.mj_copyData(synced.data, sim.model, sim.data)
        synced.set_command(cmd)
        ref_obs_sync = synced.p.get_observations()
        synced.p.infer()                          # updates the reference's own last_action
        free.set_command(cmd)
        ref_obs_free = free.obs_and_step()
        _, obs = sim.step(cmd)
        assert obs.shape == (61,) and obs.dtype == np.float32
        worst_sync = max(worst_sync, float(np.abs(obs - ref_obs_sync).max()))
        worst_free = max(worst_free, float(np.abs(obs - ref_obs_free).max()))
        assert worst_sync <= OBS_TOL, f"step {k}: synced obs diff {worst_sync}"
        assert worst_free <= OBS_TOL, f"step {k}: free-running obs diff {worst_free}"
    assert np.abs(sim.data.qpos - free.data.qpos).max() <= 1e-9
    _record("parity", f"{scene}:{key}", {"steps": n, "max_obs_diff_synced": worst_sync,
                                         "max_obs_diff_free": worst_free,
                                         "final_qpos_diff": float(np.abs(sim.data.qpos - free.data.qpos).max())})


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def _rollout(sim, seed, n=300, start_pose=None):
    gts = [sim.reset(seed, start_pose=start_pose)]
    for k in range(n):
        gt, _ = sim.step(_cmd_schedule(k, "twist"))
        gts.append(gt)
    return np.array([g.qpos for g in gts]), sim.params


@pytest.mark.parametrize("scene", ["flat", "apartment"])
def test_same_seed_same_trajectory(scene):
    a = Sim(scene, "walk")
    q1, p1 = _rollout(a, seed=7)
    _rollout(a, seed=8, n=120)            # pollute state with another episode
    q2, p2 = _rollout(a, seed=7)
    b = Sim(scene, "walk")                # fresh instance
    q3, _ = _rollout(b, seed=7)
    assert p1 == p2
    assert np.abs(q1 - q2).max() <= 1e-9
    assert np.abs(q1 - q3).max() <= 1e-9
    q4, p4 = _rollout(a, seed=9, n=50)
    assert p4["vin"] != p1["vin"]          # different seed -> different draw
    assert 6.5 <= p1["vin"] <= 8.2 and 0.0 <= p1["vin_drop_gain"] <= 0.2
    _record("determinism", scene, {"max_diff_reset": float(np.abs(q1 - q2).max()),
                                   "max_diff_new_instance": float(np.abs(q1 - q3).max())})


def test_md5_mismatch_refused(tmp_path):
    lock = pol.load_lock()
    lock["walk"] = dict(lock["walk"], md5="0" * 32)
    bad = tmp_path / "lock.json"
    bad.write_text(json.dumps(lock))
    with pytest.raises(pol.PolicyLockError):
        Sim("flat", "walk", lock_path=bad)
    with pytest.raises(pol.PolicyLockError):
        Sim("flat", "velocity_hard_4250")   # status: rejected


def test_provenance_records_lock_entries():
    s = Sim("flat", "walk")
    s.reset(0)
    assert s.switch_policy("stand")
    prov = s.provenance()
    lock = pol.load_lock()
    assert set(prov["policies"]) == {"walk", "stand"}
    for k, v in prov["policies"].items():
        assert v["md5"] == lock[k]["md5"]


# ---------------------------------------------------------------------------
# Hand-off
# ---------------------------------------------------------------------------
def test_switch_policy_handoff():
    s = Sim("flat", "walk")
    s.reset(0, start_pose=(0, 0, 0))
    for _ in range(100):
        s.step([0.3, 0, 0])
    t0 = s.t
    assert s.switch_policy("stand")
    took = s.t - t0
    assert s.policy_key == "stand"
    assert s.is_settled()
    # last_action reproduces the held targets
    assert np.allclose(simmod.DEFAULT_POSE + s.last_action, s.bam_ctrl.q_target, atol=1e-6)
    for _ in range(100):
        gt, _ = s.step([0, 0, 0])
        assert not gt.fallen
    # timeout path: impossible threshold -> False and policy unchanged
    assert s.switch_policy("walk")
    old = simmod.HANDOFF_TILT_RAD
    try:
        simmod.HANDOFF_TILT_RAD = -1.0
        assert s.switch_policy("stand", timeout_s=0.2) is False
        assert s.policy_key == "walk"
    finally:
        simmod.HANDOFF_TILT_RAD = old
    _record("handoff", "walk->stand", {"settle_time_s": took})


# ---------------------------------------------------------------------------
# Smoke walks
# ---------------------------------------------------------------------------
def _walk(sim, cmd, seconds, start_pose, kick_s=0.0, seed=0):
    gt0 = sim.reset(seed, start_pose=start_pose)
    fell = False
    for _ in range(int(round(kick_s / simmod.CONTROL_DT))):
        gt, _ = sim.step([0.3, 0, 0])
        fell |= gt.fallen
    for _ in range(int(round(seconds / simmod.CONTROL_DT))):
        gt, _ = sim.step(cmd)
        fell |= gt.fallen
    th0 = gt0.pose2[2]
    d = gt.pose2[:2] - gt0.pose2[:2]
    fwd = d[0] * math.cos(th0) + d[1] * math.sin(th0)
    return fell, float(fwd), gt


@pytest.mark.xfail(strict=True, reason=(
    "alpha_walking does not start stepping from a standing start at |v|<=0.2 m/s "
    "(stands still; infer_policy.py's own loop reproduces it: x=0.01 m after 10 s)"))
def test_smoke_flat_walk_0p2():
    fell, fwd, _ = _walk(Sim("flat", "walk"), [0.2, 0, 0], 10.0, (0, 0, 0))
    _record("smoke", "flat_0.2_10s", {"fell": fell, "forward_m": fwd})
    assert not fell and fwd > 1.0


def test_smoke_flat_walk_0p2_kickstarted():
    """0.5 s at 0.3 to break the standing deadband, then 0.2 for 10 s."""
    fell, fwd, _ = _walk(Sim("flat", "walk"), [0.2, 0, 0], 10.0, (0, 0, 0), kick_s=0.5)
    _record("smoke", "flat_0.2_10s_kick0.5", {"fell": fell, "forward_m": fwd})
    assert not fell and fwd > 0.4


def test_smoke_flat_walk_0p4():
    fell, fwd, _ = _walk(Sim("flat", "walk"), [0.4, 0, 0], 10.0, (0, 0, 0))
    _record("smoke", "flat_0.4_10s", {"fell": fell, "forward_m": fwd})
    assert not fell and fwd > 1.0


def test_smoke_apartment_free_start():
    """Default apartment spawn (office, random yaw from seed), 5 s at 0.3 m/s."""
    s = Sim("apartment", "walk")
    res = []
    for seed in range(3):
        fell, fwd, gt = _walk(s, [0.3, 0, 0], 5.0, None, seed=seed)
        res.append({"seed": seed, "fell": fell, "forward_m": fwd, "start": s.params["start_pose"]})
        assert not fell and fwd > 0.3, res[-1]
    _record("smoke", "apartment_default_spawn_0.3_5s", res)


# ---------------------------------------------------------------------------
# Real-time factor
# ---------------------------------------------------------------------------
def test_rtf_report():
    out = {}
    for scene in ["flat", "apartment"]:
        s = Sim(scene, "walk", record=False)
        s.reset(0)
        for _ in range(20):
            s.step([0.3, 0, 0])
        out[scene] = simmod.measure_rtf(s, n_steps=500, command=(0.3, 0.0, 0.0))
    import platform
    out["machine"] = {"platform": platform.platform(), "processor": platform.processor(),
                      "python": platform.python_version(), "mujoco": mujoco.__version__,
                      "note": "single process, onnxruntime 1 thread; one Sim.step = 20 ms sim time"}
    path = NAV / "reports" / "sim_rtf.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    assert out["flat"]["rtf"] > 1.0 and out["apartment"]["rtf"] > 1.0


# ---------------------------------------------------------------------------
_RESULTS_PATH = NAV / "reports" / "phaseA_test_results.json"


def _record(section, name, value):
    try:
        data = json.loads(_RESULTS_PATH.read_text())
    except Exception:
        data = {}
    data.setdefault(section, {})[name] = value
    _RESULTS_PATH.write_text(json.dumps(data, indent=1, default=float) + "\n")
