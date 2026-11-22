"""Generate nav/reports/mirror_check.json: offline + closed-loop comparison of
MirroredPolicy(ball_kick_right) vs Pollen ball_kick_left (and the reverse).

Test-only; uses tests/kick_rollout.py (own minimal MuJoCo loop, not nav.sim).
Run: cd nav && $PY tests/run_mirror_check.py [n_seeds]
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import kick_rollout as kr
from nav import mirror as M

JOINTS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle", "neck_pitch", "head_pitch",
          "head_yaw", "head_roll"]
JNAMES = ["left_" + j for j in JOINTS[:5]] + JOINTS[5:9] + ["right_" + j for j in JOINTS[:5]]
STEPS = 100
OFFS = [-0.015, 0.0, 0.015]


def corr(a, b):
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def compare(obs, ref_act, test_act):
    per = {}
    for j, n in enumerate(JNAMES):
        per[n] = dict(mean_abs_diff=float(np.abs(ref_act[:, j] - test_act[:, j]).mean()),
                      ref_std=float(ref_act[:, j].std()), test_std=float(test_act[:, j].std()),
                      corr=corr(ref_act[:, j], test_act[:, j]))
    return dict(n_samples=int(len(obs)),
                mean_abs_diff_all=float(np.abs(ref_act - test_act).mean()),
                mean_abs_ref=float(np.abs(ref_act).mean()),
                corr_flat=corr(ref_act.ravel(), test_act.ravel()),
                per_joint=per)


def ball_success(r):
    d = r["ball_xy"][-1] - r["ball_xy"][0]
    ang = math.degrees(math.atan2(d[1], d[0]))
    return (not r["fell"]) and np.linalg.norm(d) >= 0.3 and abs(ang) <= 30, d


def main(n_seeds=20):
    Lp = M.MirroredPolicy(kr.POL / "ball_kick_left.onnx")
    Rp = M.MirroredPolicy(kr.POL / "ball_kick_right.onnx")
    out = {"obs_layout": "61 = ang_vel3 | proj_grav3 | joint_pos14 | joint_vel14 | last_action14 | cmd13 (all zero for kicks)",
           "kick_policy_blind_to_ball": True,
           "ball_target_trunk_frame": {"right": list(M.kick_ball_target("right")),
                                        "left": list(M.kick_ball_target("left"))},
           "ball_pos_noise_xy_training": M.BALL_POS_NOISE_XY,
           "steps_per_rollout": STEPS, "n_seeds": n_seeds}
    # --- obs distributions from Pollen's own policies (teacher-forced comparison) ---
    roll_L = [kr.rollout(Lp.raw, "left", s, (dx, dy), STEPS, noise_ball=0.0)
              for s in range(n_seeds) for dx in OFFS for dy in OFFS]
    roll_R = [kr.rollout(Rp.raw, "right", s, (dx, dy), STEPS)
              for s in range(n_seeds) for dx in OFFS for dy in OFFS]
    OL = np.concatenate([r["obs"] for r in roll_L]); AL = np.concatenate([r["act"] for r in roll_L])
    OR = np.concatenate([r["obs"] for r in roll_R]); AR = np.concatenate([r["act"] for r in roll_R])
    # A) obs from left rollouts: Pollen-left vs mirrored-right
    out["offline_left_obs__pollen_left_vs_mirrored_right"] = compare(OL, AL, Rp.act(OL))
    # B) obs from right rollouts: Pollen-right vs mirrored-left
    out["offline_right_obs__pollen_right_vs_mirrored_left"] = compare(OR, AR, Lp.act(OR))
    # C) generated obs: mirror of right-rollout obs should be in-dist for left; compare actions
    OLm = M.mirror_obs(OR)
    out["offline_mirrored_right_obs__pollen_left_vs_mirrored_right"] = compare(OLm, Lp.raw(OLm), Rp.act(OLm))
    # D) negative controls: deliberately wrong mirrors must correlate worse
    def wrong(sign_fix):
        s = np.array(M._JOINT_SIGN, np.float32).copy(); s[sign_fix] *= -1
        return lambda o: Rp.raw(M.mirror_obs(o))[..., M._ACT_PERM_NP] * s
    ctrl = {}
    for name, idx in [("flip_left_hip_roll_sign", 1), ("flip_left_hip_pitch_sign", 2), ("flip_right_knee_sign", 12)]:
        pred = wrong(idx)(OL)
        ctrl[name] = dict(corr_flat=corr(AL.ravel(), pred.ravel()),
                          joint_corr=corr(AL[:, idx], pred[:, idx]),
                          joint_corr_correct=corr(AL[:, idx], Rp.act(OL)[:, idx]))
    out["negative_controls"] = ctrl
    # --- closed loop, quick (full sweep done later with nav.sim) ---
    cl = {}
    for name, fn, foot in [("pollen_left", Lp.raw, "left"), ("mirrored_right_as_left", Rp.act, "left"),
                           ("pollen_right", Rp.raw, "right"), ("mirrored_left_as_right", Lp.act, "right")]:
        ok, ds, falls = [], [], 0
        for s in range(n_seeds):
            r = kr.rollout(fn, foot, 1000 + s, steps=150, noise_ball=M.BALL_POS_NOISE_XY)
            s_ok, d = ball_success(r)
            ok.append(s_ok); ds.append(d); falls += r["fell"]
        ds = np.array(ds)
        cl[name] = dict(success_rate=float(np.mean(ok)), fall_rate=falls / n_seeds,
                        ball_disp_mean=ds.mean(0).tolist(), ball_disp_std=ds.std(0).tolist())
    out["closed_loop_quick_noise_0.015"] = cl
    p = HERE.parent / "reports" / "mirror_check.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    o = main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
    for k in ["offline_left_obs__pollen_left_vs_mirrored_right", "offline_right_obs__pollen_right_vs_mirrored_left",
              "offline_mirrored_right_obs__pollen_left_vs_mirrored_right"]:
        v = o[k]; print(k, "MAD", round(v["mean_abs_diff_all"], 4), "ref|a|", round(v["mean_abs_ref"], 4), "corr", v["corr_flat"])
        for n, j in v["per_joint"].items():
            print("   ", n, round(j["mean_abs_diff"], 4), j["ref_std"] and round(j["ref_std"], 3), j["corr"] and round(j["corr"], 3))
    print(json.dumps(o["negative_controls"], indent=1)); print(json.dumps(o["closed_loop_quick_noise_0.015"], indent=1))
