"""Evaluate the follower on the existing courses against the actual policy/physics backend.

Gate (unchanged): every run must have cross-track RMS < 0.03 m, reach the goal (status
"done", within 5 cm / 10 deg after a 1 s zero-command settle) and not fall.
Courses straight, s_bend, doorway, u_turn x seeds, flat scene, dynamics from the envelope.
"""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nav.courses import all_courses
from nav.episode import read_envelope
from nav.follower import RPPFollower, CommandModel, Envelope
from nav.follower_eval import run_course, sim_backend
from nav.sim import Sim

FOLLOWER_SHA256 = hashlib.sha256((Path(__file__).resolve().parents[1] / "nav" / "follower.py").read_bytes()).hexdigest()

RMS_LIMIT = 0.03


def _one(args):
    policy, dynamics, seed, ci, trace_dir, fkw = args
    env = read_envelope(policy)
    course = all_courses()[ci]
    sim = Sim("flat", policy, record=False, dynamics=dynamics)
    initial = sim.reset(seed, course.start)
    follower = RPPFollower(env, gridmap=course.gridmap, **(fkw or {}))
    step, pose = sim_backend(sim, initial)
    falls = []

    def checked_step(cmd):
        step(cmd)
        falls.append(sim.ground_truth().fallen)

    result = run_course(course, checked_step, pose, follower)
    if trace_dir:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(trace_dir) / f"{policy}_{course.name}_{seed}.npz",
                            traj=result["traj"], cmds=result["cmds"], path=course.poses)
    g = course.poses[-1]
    last = result["traj"][-1]
    done_pos = float(np.hypot(last[0] - g[0], last[1] - g[1]))
    done_yaw = float(np.degrees(abs((last[2] - g[2] + np.pi) % (2 * np.pi) - np.pi)))
    result = {k: v for k, v in result.items() if k not in ("cmds", "traj")}
    result.update(pos_err_at_status=done_pos, yaw_err_at_status_deg=done_yaw)
    result.update(seed=seed, fallen=any(falls), final_fixes=int(getattr(follower, "fixes", 0)),
                  passed=bool(result["cross_track_rms"] < RMS_LIMIT and result["reached"] and not any(falls)))
    return result


def run(policy, dynamics="training", seeds=range(5), workers=3, trace_dir=None, follower_kw=None):
    env = read_envelope(policy)
    if env.get("dynamics", "infer") != dynamics:
        raise RuntimeError("measurement/gate dynamics mismatch")
    jobs = [(policy, dynamics, s, ci, trace_dir, follower_kw) for s in seeds for ci in range(len(all_courses()))]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, 3)) as pool:
            rows = list(pool.map(_one, jobs))
    else:
        rows = [_one(j) for j in jobs]
    rms = [r["cross_track_rms"] for r in rows]
    courses = {}
    for c in all_courses():
        rs = [r for r in rows if r["course"] == c.name]
        courses[c.name] = {"passed": sum(r["passed"] for r in rs), "reached": sum(r["reached"] for r in rs), "n": len(rs),
                           "rms_worst": max(r["cross_track_rms"] for r in rs),
                           "rms_median": float(np.median([r["cross_track_rms"] for r in rs])),
                           "final_pos_err_worst": max(r["final_pos_err"] for r in rs),
                           "final_yaw_err_worst_deg": max(r["final_yaw_err_deg"] for r in rs),
                           "time_mean_s": float(np.mean([r["time_to_goal"] for r in rs]))}
    model = CommandModel(Envelope(env))
    return {"policy": policy, "dynamics": dynamics, "dynamics_version": env.get("dynamics_version"), "policy_md5": env["policy_md5"],
            "follower_sha256": FOLLOWER_SHA256,
            "follower": "envelope inverse-model tracker (nav/follower.py)", "command_model": model.summary(),
            "follower_kw": follower_kw or {}, "cross_track_rms_limit": RMS_LIMIT, "goal_tolerance": {"pos_m": 0.05, "yaw_deg": 10},
            "passed": all(r["passed"] for r in rows),
            "n_passed": sum(r["passed"] for r in rows),
            "reached": sum(r["reached"] for r in rows), "n": len(rows),
            "falls": sum(r["fallen"] for r in rows),
            "worst_cross_track_rms": max(rms), "median_cross_track_rms": float(np.median(rms)),
            "courses": courses, "runs": rows,
            "doorway_scope": "analytic clearance course on flat physics; apartment traversal is benchmarked separately"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--dynamics", choices=["infer", "training"], default="training")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=0, help="development: held-out seeds (gate uses 0)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--follower-kw", default=None, help="JSON dict of RPPFollower overrides (development only)")
    args = ap.parse_args()
    result = run(args.policy, args.dynamics, range(args.seed_offset, args.seed_offset + args.seeds), args.workers, args.trace_dir,
                 json.loads(args.follower_kw) if args.follower_kw else None)
    out = args.out or f"reports/follower_sim_{args.policy}.json"
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: result[k] for k in ("policy", "passed", "n_passed", "reached", "n", "falls",
                                      "worst_cross_track_rms", "median_cross_track_rms")}
    print(json.dumps(summary, indent=2))
    for r in result["runs"]:
        print(f"{r['course']:9s} s{r['seed']} {r['status']:8s} t={r['time_to_goal']:5.1f} rms={r['cross_track_rms']:.4f} "
              f"max={r['cross_track_max']:.3f} pos={r['final_pos_err']:.3f} yaw={r['final_yaw_err_deg']:5.1f} "
              f"fixes={r['final_fixes']} {'PASS' if r['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
