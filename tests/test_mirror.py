"""Tests for nav.mirror (Phase G mirrored kick policy)."""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from nav import mirror as M  # noqa: E402

UP = Path.home() / "microduck" / "upstream"
SYM = UP / "microduck_rl/src/mjlab_microduck/tasks/symmetry.py"
POL = UP / "policies"
REPORT = HERE.parent / "reports" / "mirror_check.json"


def _upstream_tables():
    """Evaluate the table assignments in symmetry.py WITHOUT importing it (torch/mjlab)."""
    tree = ast.parse(SYM.read_text())
    wanted = ("_JOINT_PERM", "_JOINT_SIGN", "_OBS_PERM", "_OBS_SIGN")
    ns = {}
    for node in tree.body:
        tgt = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            tgt, val = node.target.id, node.value
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            tgt, val = node.targets[0].id, node.value
        if tgt in wanted:
            ns[tgt] = eval(compile(ast.Expression(val), str(SYM), "eval"), {}, dict(ns))
    assert set(ns) == set(wanted)
    return ns


def test_tables_equal_upstream():
    up = _upstream_tables()
    assert M._JOINT_PERM == up["_JOINT_PERM"]
    assert [float(x) for x in M._JOINT_SIGN] == [float(x) for x in up["_JOINT_SIGN"]]
    assert M._OBS_PERM == up["_OBS_PERM"]
    assert [float(x) for x in M._OBS_SIGN] == [float(x) for x in up["_OBS_SIGN"]]
    assert len(M._OBS_PERM) == len(M._OBS_SIGN) == 61


def test_perm_is_involutive_permutation():
    p = np.array(M._OBS_PERM)
    assert sorted(p) == list(range(61))
    assert (p[p] == np.arange(61)).all()
    # signs of swapped pairs agree, so perm+sign is an involution
    s = np.array(M._OBS_SIGN)
    assert (s[p] * s == 1).all()


def test_involution_random():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 61)).astype(np.float32)
    assert np.allclose(M.mirror_obs(M.mirror_obs(x)), x, atol=0)
    a = rng.normal(size=(50, 14)).astype(np.float32)
    assert np.allclose(M.mirror_action(M.mirror_action(a)), a, atol=0)
    # single vectors keep their shape
    assert M.mirror_obs(x[0]).shape == (61,) and M.mirror_action(a[0]).shape == (14,)


def test_semantic_slots():
    """Spot-check against the documented rules (symmetry.py docstring)."""
    x = np.arange(1, 62, dtype=np.float32)
    y = M.mirror_obs(x)
    assert (y[[0, 2]] == -x[[0, 2]]).all() and y[1] == x[1]          # ang vel: roll, yaw negated
    assert y[4] == -x[4] and y[3] == x[3] and y[5] == x[5]            # gravity gy negated
    assert y[48] == x[48] and y[49] == -x[49] and y[50] == -x[50]     # twist
    assert y[6 + 5] == x[6 + 5] and y[6 + 6] == x[6 + 6]              # neck/head pitch unchanged
    assert y[6 + 0] == -x[6 + 9] and y[6 + 13] == -x[6 + 4]           # leg swap+negate
    assert y[6 + 7] == -x[6 + 7] and y[6 + 8] == -x[6 + 8]            # head yaw/roll


def test_default_pose_mirror_symmetric():
    """Mirroring the raw action also mirrors joint targets iff DEFAULT_POSE is invariant."""
    import onnxruntime as ort
    meta = ort.InferenceSession(str(POL / "ball_kick_left.onnx")).get_modelmeta().custom_metadata_map
    d = np.array([float(v) for v in meta["default_joint_pos"].split(",")], dtype=np.float32)
    assert np.allclose(d[M._JOINT_PERM] * np.array(M._JOINT_SIGN), d, atol=1e-3)
    assert meta["action_scale"].startswith("1.0")


@pytest.mark.parametrize("name", ["ball_kick_left", "ball_kick_right"])
def test_onnx_dims(name):
    p = M.MirroredPolicy(POL / f"{name}.onnx")
    assert (p.obs_dim, p.act_dim) == (61, 14)
    rng = np.random.default_rng(1)
    o = rng.normal(size=61).astype(np.float32) * 0.1
    a = p.act(o)
    assert a.shape == (14,) and np.isfinite(a).all()
    assert p.act(np.stack([o, o])).shape == (2, 14)
    # act == mirror_action(raw(mirror_obs(.)))
    assert np.allclose(a, M.mirror_action(p.raw(M.mirror_obs(o))))


def test_mirror_of_mirror_policy_is_identity_on_actions():
    """pi(M(M(o))) path: act applied through two mirrors reproduces the raw policy."""
    p = M.MirroredPolicy(POL / "ball_kick_right.onnx")
    o = np.random.default_rng(2).normal(size=61).astype(np.float32) * 0.1
    assert np.allclose(M.mirror_action(p.act(M.mirror_obs(o))), p.raw(o), atol=1e-6)


def test_ball_target():
    assert M.kick_ball_target("right") == (0.09, -0.042)
    assert M.kick_ball_target("left") == (0.09, 0.042)


def test_kick_sweep_stub_signature():
    import inspect
    assert list(inspect.signature(M.kick_sweep).parameters) == ["sim_factory", "policy_key", "offsets", "seeds"]
    with pytest.raises(NotImplementedError):
        M.kick_sweep(None, "kick_left", [(0, 0)], [0])


# ----------------------------- offline comparison vs Pollen left -----------------------------
def _flat_corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def test_live_mirrored_right_vs_pollen_left():
    kr = pytest.importorskip("kick_rollout")
    try:
        Lp = M.MirroredPolicy(POL / "ball_kick_left.onnx")
        Rp = M.MirroredPolicy(POL / "ball_kick_right.onnx")
        rolls = [kr.rollout(Lp.raw, "left", s, (0.0, 0.0), 100) for s in range(3)]
    except Exception as e:  # bam/mujoco unavailable
        pytest.skip(f"rollout unavailable: {e}")
    O = np.concatenate([r["obs"] for r in rolls]); AL = np.concatenate([r["act"] for r in rolls])
    AM = Rp.act(O)
    assert not any(r["fell"] for r in rolls)
    assert _flat_corr(AL, AM) > 0.6
    # a sign-flipped hip_pitch must be clearly worse than the correct mirror
    s = np.array(M._JOINT_SIGN, np.float32).copy(); s[2] *= -1
    bad = Rp.raw(M.mirror_obs(O))[:, M._ACT_PERM_NP] * s
    assert _flat_corr(AL, bad) < _flat_corr(AL, AM) - 0.1


def test_closed_loop_mirrored_right_kicks_left_side():
    kr = pytest.importorskip("kick_rollout")
    Rp = M.MirroredPolicy(POL / "ball_kick_right.onnx")
    Lp = M.MirroredPolicy(POL / "ball_kick_left.onnx")
    try:
        r_m = kr.rollout(Rp.act, "left", 0, steps=150)
        r_l = kr.rollout(Lp.raw, "left", 0, steps=150)
        r_r = kr.rollout(Rp.raw, "right", 0, steps=150)
    except Exception as e:
        pytest.skip(f"rollout unavailable: {e}")
    for r in (r_m, r_l, r_r):
        assert not r["fell"]
    d_m = r_m["ball_xy"][-1] - r_m["ball_xy"][0]
    d_r = r_r["ball_xy"][-1] - r_r["ball_xy"][0]
    d_l = r_l["ball_xy"][-1] - r_l["ball_xy"][0]
    assert d_m[0] > 0.3 and d_l[0] > 0.3                  # ball goes forward
    assert d_m[1] > 0 and d_r[1] < 0                      # mirrored: leftwards drift vs rightwards
    assert abs(d_m[1] + d_r[1]) < 0.1                     # mirror symmetry of outcome (deterministic seed)


# ----------------------------- recorded report -----------------------------
def test_report_no_sign_errors():
    if not REPORT.exists():
        pytest.skip("run tests/run_mirror_check.py first")
    r = json.loads(REPORT.read_text())
    for key in ("offline_left_obs__pollen_left_vs_mirrored_right",
                "offline_right_obs__pollen_right_vs_mirrored_left"):
        sec = r[key]
        assert sec["corr_flat"] > 0.7, key
        # a sign error on a joint with real motion would show as strongly negative correlation
        for j, v in sec["per_joint"].items():
            if v["corr"] is not None and v["ref_std"] > 0.05:
                assert v["corr"] > -0.5, (key, j, v)
    for name, c in r["negative_controls"].items():
        assert c["joint_corr"] < 0 < c["joint_corr_correct"], name
