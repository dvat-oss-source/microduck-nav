"""Benchmark runner (Phase H).

run_episode(task) -> trace dict {"qpos","t","model_path", optional "plan_time_s","wall_s"}.
`runner` may be a callable (picklable, top-level) or a "module:function" string.
CLI:
  python -m nav.bench --runner pkg.mod:fn --run NAME [--scene SCENE_XML] [--limit N]
  python -m nav.bench --compare NEW.json BASE.json
  python -m nav.bench --if-lock-changed --runner ... --run NAME
"""
import argparse
import datetime
import glob
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from . import checker, stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEEDS = os.path.join(ROOT, "data", "seeds_v1.json")
LOCK = os.path.join(ROOT, "policies.lock.json")
REPORTS = os.path.join(ROOT, "reports")
DEFAULT_PROCS = min(9, max(1, (os.cpu_count() or 4) - 2))


def md5_file(p: str) -> Optional[str]:
    if not os.path.exists(p):
        return None
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def seeds_digest(doc: Dict[str, Any]) -> str:
    body = {k: v for k, v in doc.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_seeds(path: str = SEEDS) -> Dict[str, Any]:
    doc = json.load(open(path))
    if doc.get("sha256") != seeds_digest(doc):
        raise RuntimeError("seed file %s was modified (sha256 mismatch); refusing to run" % path)
    return doc


def all_tasks(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for kind in ("nav", "kick", "pick"):
        for i, t in enumerate(doc.get(kind, [])):
            t = dict(t); t.setdefault("type", kind); t["ep"] = "%s_%03d" % (kind, i)
            out.append(t)
    return out


def lock_entries_used(lock_path: str = LOCK, keys: Optional[List[str]] = None) -> Dict[str, Any]:
    if not os.path.exists(lock_path):
        return {}
    lock = json.load(open(lock_path))
    return {k: {"md5": v.get("md5"), "path": v.get("path")} for k, v in lock.items()
            if isinstance(v, dict) and "path" in v and (keys is None or k in keys)}


def _resolve(runner):
    if isinstance(runner, str):
        m, f = runner.split(":")
        return getattr(importlib.import_module(m), f)
    return runner


def _work(args):
    runner, task, scene = args
    trace = None
    try:
        fn = _resolve(runner)
        trace = fn(task)
        if trace.get("error"):
            raise RuntimeError(trace["error"])
        verdict = checker.check(trace, task["check"] if "check" in task else task, scene or trace.get("model_path"))
        verdict["plan_time_s"] = trace.get("plan_time_s")
        verdict["plan_times_s"] = trace.get("plan_times_s", [entry["plan_time_s"] for entry in trace.get("plan_log", [])
                                                               if "plan_time_s" in entry])
        verdict["error"] = None
        terminal = trace.get("transitions", [{}])[-1] if trace.get("transitions") else {}
        verdict["stack_reason"] = terminal.get("reason")
        if "bad_trace" in verdict.get("reasons", []):
            raise ValueError("checker rejected malformed trace or model nq mismatch")
        return task["ep"], verdict, trace
    except Exception as e:  # an episode crash is a failure, not a bench crash
        trace = trace or getattr(e, "partial_trace", None) or getattr(runner, "partial_trace", None)
        detail = (trace or {}).get("traceback") or traceback.format_exc()
        return task["ep"], {"success": False, "reasons": ["exception"], "error": repr(e),
                            "traceback": detail, "task_type": task.get("type"), "fallen": False,
                            "hole_incursion": False, "safety_unknown": True, "path_length": 0.0,
                            "time_to_goal": None, "plan_times_s": (trace or {}).get("plan_times_s", [])}, trace


def _video_from_trace_file(job):
    """Spawn-safe renderer: reconstruct the saved trace from its path."""
    ep, trace_path, video_path = job
    try:
        from .episode import failure_video
        with np.load(trace_path, allow_pickle=False) as saved:
            trace = json.loads(str(saved['metadata']))
            trace['qpos'], trace['t'] = saved['qpos'], saved['t']
            task = json.loads(str(saved['task']))
        failure_video(task, trace, video_path)
        return ep, None
    except Exception as e:
        return ep, repr(e)


def _serializable(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)


def aggregate(tasks, verdicts, _include_room_pairs=True) -> Dict[str, Any]:
    n = len(tasks)
    byk: Dict[str, Any] = {}
    for kind in ("nav", "kick", "pick"):
        idx = [i for i, t in enumerate(tasks) if t["type"] == kind]
        if not idx:
            continue
        vs = [verdicts[i] for i in idx]
        k = sum(v["success"] for v in vs)
        lo, hi = stats.wilson_ci(k, len(vs))
        m = {"n": len(vs), "successes": k, "success_rate": k / len(vs), "success_ci": [lo, hi]}
        kf = sum(bool(v.get("fallen")) for v in vs)
        m["fall_rate"] = kf / len(vs); m["fall_ci"] = list(stats.wilson_ci(kf, len(vs)))
        kh = sum(bool(v.get("hole_incursion")) for v in vs)
        m["hole_incursions"] = kh; m["hole_rate"] = kh / len(vs); m["hole_ci"] = list(stats.wilson_ci(kh, len(vs)))
        m["time_to_goal"] = stats.percentiles([v.get("time_to_goal") for v in vs if v["success"]])
        m["plan_time_s"] = stats.percentiles([value for v in vs for value in v.get("plan_times_s", [])])
        if kind == "nav":
            geo = [tasks[i].get("geodesic", 0.0) for i in idx]
            m["spl"] = stats.spl([v["success"] for v in vs], geo, [v.get("path_length", 0.0) for v in vs])
            terms = stats.spl_terms([v["success"] for v in vs], geo, [v.get("path_length", 0.0) for v in vs])
            m["spl_ci"] = list(stats.bootstrap_ci(terms))
        reasons: Dict[str, int] = {}
        for v in vs:
            for r in v.get("reasons", []):
                reasons[r] = reasons.get(r, 0) + 1
        m["reason_counts"] = reasons
        stack_reasons = {}
        for v in vs:
            if v.get("stack_reason"):
                stack_reasons[v["stack_reason"]] = stack_reasons.get(v["stack_reason"], 0) + 1
        m["stack_reason_counts"] = stack_reasons
        m["safety_unknown"] = sum(bool(v.get("safety_unknown")) for v in vs)
        byk[kind] = m
    report = {"n_episodes": n, "by_task": byk}
    if _include_room_pairs:
        groups = {}
        for i, task in enumerate(tasks):
            if task["type"] != "nav":
                continue
            pair = task.get("room_pair", ["unknown", "unknown"])
            label = "->".join(map(str, pair)) if isinstance(pair, (list, tuple)) else str(pair)
            groups.setdefault(label, []).append(i)
        report["by_room_pair"] = {
            label: aggregate([tasks[i] for i in idx], [verdicts[i] for i in idx], False)["by_task"]["nav"]
            for label, idx in sorted(groups.items())}
    return report


def target_gate(report, mode="gt"):
    """Point-estimate targets; CIs remain reported separately, without rounding."""
    nav = report.get("by_task", {}).get("nav", {})
    success, spl = (0.95, 0.8) if mode == "gt" else (0.85, 0.7)
    checks = {"success": nav.get("success_rate", 0.0) >= success,
              "spl": nav.get("spl", 0.0) >= spl,
              "falls": all(v["fall_rate"] <= 0.02 for v in report.get("by_task", {}).values()),
              "stair_hole": all(v["hole_incursions"] == 0 for v in report.get("by_task", {}).values()),
              "safety_observed": all(v.get("safety_unknown", 0) == 0 for v in report.get("by_task", {}).values()),
              "full_frozen_suite": not report.get("DEV", False) and not report.get("dev_subset", False) and all(report.get("by_task", {}).get(k, {}).get("n", 0) == n
                                       for k, n in (("nav", 200), ("kick", 50), ("pick", 50)))}
    return {"targets": {"success": success, "spl": spl, "fall_rate_max": 0.02, "hole_incursions_max": 0},
            "checks": checks, "passed": all(checks.values()),
            "reference_replay": "pending; numerical gate alone does not establish sim fidelity"}


def run_bench(runner, run: str, scene: Optional[str] = None, seeds_path: str = SEEDS,
              lock_path: str = LOCK, procs: int = DEFAULT_PROCS, limit: Optional[int] = None,
              reports_dir: str = REPORTS, video_hook: Optional[Callable] = None,
              lock_keys: Optional[List[str]] = None, task_types: Optional[List[str]] = None,
              episodes: Optional[List[str]] = None) -> str:
    doc = load_seeds(seeds_path)
    if not 1 <= procs <= 9:
        raise ValueError("local benchmark requires 1..9 processes")
    comparison = bool(getattr(runner, "comparison_policy", False))
    dev_subset = episodes is not None
    if dev_subset and limit is not None:
        raise ValueError("--episodes and --limit cannot be combined")
    dev_seed_dir = os.path.realpath(os.path.join(ROOT, "data", "dev"))
    seed_location = os.path.realpath(seeds_path)
    is_dev_seed = os.path.commonpath((dev_seed_dir, seed_location)) == dev_seed_dir
    is_dev = bool(doc.get("provenance", {}).get("dev")) or is_dev_seed or dev_subset
    if is_dev and not run.startswith("DEV_"):
        run = "DEV_" + run
    if is_dev and os.path.abspath(reports_dir) == os.path.abspath(REPORTS):
        reports_dir = os.path.join(REPORTS, "dev", "housekeeping")
    follower_reports_dir = getattr(runner, "follower_reports_dir", None)
    if follower_reports_dir and not is_dev:
        raise ValueError("custom follower gates are development only; use a dev fork or --episodes")
    comparison_runtime = None
    selected_types = set(task_types or (["nav"] if comparison else ["nav", "kick", "pick"]))
    if not selected_types or not selected_types <= {"nav", "kick", "pick"}:
        raise ValueError("task_types must select nav, kick and/or pick")
    if comparison and selected_types != {"nav"}:
        raise RuntimeError("comparison policies may only run the frozen navigation tasks")
    if hasattr(runner, "policy"):
        from .episode import artifact_digest, read_artifacts
        from .mapsigma import planning_lateral_sigma
        from .provenance import scene_sha256
        frozen = doc.get("provenance")
        if not frozen:
            raise RuntimeError("policy runner requires frozen benchmark provenance")
        if (frozen.get("path_smoothing_steps") != checker.PATH_SMOOTHING_STEPS
                or frozen.get("goal_dwell_s") != 1.0
                or frozen.get("geodesic_method") != "footprint_16_dijkstra_los_v1"):
            raise RuntimeError("frozen path smoothing, goal dwell or geodesic method differs from checker contract")
        if ("checker_sha256" in frozen and
                artifact_digest(checker.__file__) != frozen["checker_sha256"]):
            raise RuntimeError("independent checker differs from frozen benchmark provenance")
        for module in ('skills','episode'):
            if (module+'_sha256' in frozen and
                    artifact_digest(os.path.join(ROOT,'nav',module+'.py')) != frozen[module+'_sha256']):
                raise RuntimeError(module+' differs from frozen benchmark provenance')
        if ("fsm_sha256" in frozen and
                artifact_digest(os.path.join(ROOT, "nav", "fsm.py")) != frozen["fsm_sha256"]):
            raise RuntimeError("FSM differs from frozen benchmark provenance")
        for module in ("planner", "mapping", "mapsigma"):
            key = module + "_sha256"
            if key in frozen and artifact_digest(os.path.join(ROOT, "nav", module + ".py")) != frozen[key]:
                raise RuntimeError(module + " differs from frozen benchmark provenance")
        if "localization_sha256" in frozen:
            if artifact_digest(os.path.join(ROOT, "nav", "localization.py")) != frozen["localization_sha256"]:
                raise RuntimeError("localization differs from frozen benchmark provenance")
        elif getattr(runner, "mode", "gt") != "gt":
            # Seed files that predate localization provenance (v1/v2) were run with FIDUCIAL_RENDER_VERSION 1 (raw
            # qpos-prefix camera pose). Refuse non-GT runs rather than silently mixing the fixed camera into them;
            # the v1 render source is preserved in reports/bak_2026-11-09_camfix/localization.py.
            from .localization import FIDUCIAL_RENDER_VERSION
            if FIDUCIAL_RENDER_VERSION != 1:
                raise RuntimeError("seed file predates localization provenance; non-GT runs need a fork recording "
                                   "localization_sha256 (fiducial render v2)")
        env, primitives = read_artifacts(runner.policy, runner.data_dir)
        if (runner.dynamics != frozen["dynamics"] or env.get("dynamics") != runner.dynamics
                or primitives.get("dynamics") != runner.dynamics
                or env.get("dynamics_version") != frozen.get("dynamics_version")):
            raise RuntimeError("runner dynamics/version differ from frozen benchmark provenance")
        actual_scene = scene or getattr(runner, "scene", "apartment")
        if not frozen.get("scene_sha256") or scene_sha256(actual_scene, runner.dynamics) != frozen["scene_sha256"]:
            raise RuntimeError("scene include/asset closure differs from frozen benchmark provenance")
        sigma, sigma_provenance = (planning_lateral_sigma(runner.policy, env, follower_reports_dir)
                                   if follower_reports_dir else planning_lateral_sigma(runner.policy, env))
        if not sigma_provenance.get("follower_gate_passed"):
            raise RuntimeError("benchmark policy requires a passing follower gate")
        if comparison:
            comparison_runtime = {
                "policy": runner.policy, "policy_md5": env["policy_md5"], "sigma": sigma,
                "follower_sha256": sigma_provenance.get("follower_sha256"),
                "follower_gate_sha256": artifact_digest(os.path.join(
                    follower_reports_dir or REPORTS, sigma_provenance.get("file", f"follower_sim_{runner.policy}.json"))),
            }
        if not comparison:
            if (runner.policy != frozen["policy"] or env["policy_md5"] != frozen.get("policy_md5")
                    or artifact_digest(os.path.join(runner.data_dir, f"envelope_{runner.policy}.json")) != frozen["envelope_sha256"]):
                raise RuntimeError("runner measurements differ from frozen benchmark provenance")
            if "primitives_sha256" in frozen and artifact_digest(os.path.join(runner.data_dir, f"primitives_{runner.policy}.json")) != frozen["primitives_sha256"]:
                raise RuntimeError("primitives differ from frozen benchmark provenance")
            if abs(sigma - frozen["lateral_sigma"]) > 1e-12:
                raise RuntimeError("map inflation sigma differs from frozen benchmark provenance")
            if ("follower_sha256" in frozen.get("lateral_sigma_provenance", {}) and
                    sigma_provenance.get("follower_sha256") != frozen["lateral_sigma_provenance"]["follower_sha256"]):
                raise RuntimeError("follower differs from the frozen benchmark controller")
            if ("skill_contracts_sha256" in frozen and
                    artifact_digest(os.path.join(runner.data_dir, "skill_contracts.json")) != frozen["skill_contracts_sha256"]):
                raise RuntimeError("skill contracts differ from frozen benchmark provenance")
    tasks = [task for task in all_tasks(doc) if task["type"] in selected_types]
    if episodes is not None:
        if not episodes or len(set(episodes)) != len(episodes):
            raise ValueError("episodes must be nonempty, unique episode IDs")
        available = {task["ep"] for task in tasks}
        missing = set(episodes) - available
        if missing:
            raise ValueError("unknown or unselected episode IDs: " + ", ".join(sorted(missing)))
        selected = set(episodes)
        tasks = [task for task in tasks if task["ep"] in selected]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        tasks = tasks[:limit]
    jobs = [(runner, t, scene) for t in tasks]
    if procs <= 1:
        results = [_work(j) for j in jobs]
    else:
        by_episode = {}
        with mp.get_context("spawn").Pool(procs) as pool:
            for result in pool.imap_unordered(_work, jobs, chunksize=1):
                by_episode[result[0]] = result
                print(json.dumps({"completed": len(by_episode), "total": len(jobs),
                                  "ep": result[0], "success": result[1]["success"],
                                  "reasons": result[1].get("reasons", [])}), flush=True)
        results = [by_episode[task["ep"]] for task in tasks]
    verdicts = [r[1] for r in results]
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    fdir = os.path.join(reports_dir, "failures", run)
    n_fail = 0
    trace_dir = os.path.join(reports_dir, "traces", run)
    video_jobs = []
    failures = {}
    for task, (ep, v, tr) in zip(tasks, results):
        if tr is not None:
            os.makedirs(trace_dir, exist_ok=True)
            metadata = {k: value for k, value in tr.items() if k not in ("qpos", "t")}
            np.savez_compressed(os.path.join(trace_dir, ep + ".npz"),
                                qpos=tr["qpos"], t=tr["t"],
                                metadata=json.dumps(metadata, default=_serializable),
                                task=json.dumps(task))
            v["trace_file"] = os.path.join(trace_dir, ep + ".npz")
        if not v["success"]:
            os.makedirs(fdir, exist_ok=True)
            n_fail += 1
            failure = {"ep": ep, "task": task, "verdict": v, "trace": tr}
            failures[ep] = failure
            with open(os.path.join(fdir, ep + ".json"), "w") as handle:
                json.dump(failure, handle, indent=1, default=_serializable)
            if video_hook is not None and tr is not None:
                video_jobs.append((ep, v["trace_file"], os.path.join(fdir, ep + ".mp4")))

    # Episode workers have closed above; save every trace and failure JSON before rendering.
    # Only the built-in renderer is sent to spawn workers. Custom hooks, including
    # local/unpickleable test hooks, keep their original in-process behavior.
    parallel_video = False
    if procs > 1 and video_hook is not None:
        from .episode import failure_video
        parallel_video = video_hook is failure_video

    finished_videos = set()

    def record_video(ep, error):
        verdict = failures[ep]['verdict']
        if error is None:
            verdict['failure_video'] = os.path.join(fdir, ep + '.mp4')
        else:
            verdict['video_error'] = error
            print('video hook failed for', ep, error, file=sys.stderr)
        with open(os.path.join(fdir, ep + '.json'), 'w') as handle:
            json.dump(failures[ep], handle, indent=1, default=_serializable)
        finished_videos.add(ep)
        print(json.dumps({'video_export_completed': len(finished_videos),
                          'total': len(video_jobs), 'ep': ep, 'success': error is None}), flush=True)

    if video_jobs and parallel_video:
        try:
            with ProcessPoolExecutor(max_workers=min(3, procs), mp_context=mp.get_context('spawn')) as pool:
                futures = {pool.submit(_video_from_trace_file, job): job[0] for job in video_jobs}
                for future in as_completed(futures):
                    ep = futures[future]
                    try:
                        _, error = future.result()
                    except Exception as e:
                        error = repr(e)
                    record_video(ep, error)
        except Exception as e:
            for ep, _, _ in video_jobs:
                if ep not in finished_videos:
                    record_video(ep, repr(e))
    else:
        for ep, _, video_path in video_jobs:
            failure = failures[ep]
            try:
                video_hook(failure['task'], failure['trace'], video_path)
                error = None
            except Exception as e:
                error = repr(e)
            record_video(ep, error)
    report = {
        "run": run, "timestamp": ts, "seeds_sha256": doc["sha256"], "seeds_version": doc.get("version"),
        "lock_md5": md5_file(lock_path), "policies": lock_entries_used(lock_path, lock_keys),
        "scene": scene, "n_failures": n_fail, **aggregate(tasks, verdicts),
        "episodes": {r[0]: r[1] for r in results},
        "frozen_provenance": doc.get("provenance"),
        "runner_config": {k: value for k, value in vars(runner).items() if k != "partial_trace"}
                         if hasattr(runner, "__dict__") else str(runner),
        "mode": getattr(runner, "mode", "gt"), "limit": limit,
        "task_counts": {kind: sum(task["type"] == kind for task in tasks) for kind in sorted(selected_types)},
        "comparison_policy": comparison,
        "dev_subset": dev_subset,
        "DEV": is_dev,
    }
    if comparison_runtime is not None:
        report["comparison_runtime"] = comparison_runtime
    report["gate"] = target_gate(report, getattr(runner, "mode", "gt"))
    if limit is not None or is_dev:
        report["gate"]["checks"]["full_frozen_suite"] = False
        report["gate"]["passed"] = False
    os.makedirs(reports_dir, exist_ok=True)
    out = os.path.join(reports_dir, "bench_%s_%s.json" % (run, ts))
    json.dump(report, open(out, "w"), indent=1)
    return out


# ---- regression -----------------------------------------------------------------------------
# (metric key, higher_is_better, ci key or None)
_METRICS = [("success_rate", True, "success_ci"), ("fall_rate", False, "fall_ci"),
            ("hole_rate", False, "hole_ci"), ("spl", True, "spl_ci")]


def compare(new: Dict[str, Any], base: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare matched frozen tasks; flag worsening rates at two-sided alpha .05."""
    if not new.get("seeds_sha256") or new.get("seeds_sha256") != base.get("seeds_sha256"):
        raise ValueError("cannot compare different or missing frozen seed hashes")
    def mode(report):
        return report.get("mode", report.get("runner_config", {}).get("mode")
                          if isinstance(report.get("runner_config"), dict) else None)
    if mode(new) is None or mode(new) != mode(base):
        raise ValueError("cannot compare different or missing localization modes")
    counts = lambda report: {kind: values.get("n") for kind, values in report.get("by_task", {}).items()}
    if counts(new) != counts(base) or any(not n for n in counts(new).values()):
        raise ValueError("cannot compare different or missing task counts")
    flags = []
    for kind, nm in new.get("by_task", {}).items():
        bm = base["by_task"][kind]
        for key, hib, cik in _METRICS:
            if key not in nm or key not in bm:
                continue
            worse = nm[key] < bm[key] if hib else nm[key] > bm[key]
            if not worse:
                continue
            a, b = nm[cik], bm[cik]
            if key != "spl":
                pvalue = stats.two_proportion_pvalue(nm[key] * nm["n"], nm["n"],
                                                       bm[key] * bm["n"], bm["n"])
                significant = pvalue < 0.05
            else:
                pvalue = None
                significant = not (a[0] <= b[1] and b[0] <= a[1])
            if significant:
                flags.append({"task": kind, "metric": key, "new": nm[key], "base": bm[key],
                              "new_ci": a, "base_ci": b, "p_value": pvalue,
                              "test": "two_proportion_z" if key != "spl" else "bootstrap_ci_nonoverlap",
                              "alpha": 0.05})
    return flags


def latest_report(reports_dir: str = REPORTS, run: Optional[str] = None) -> Optional[str]:
    candidates = []
    for path in glob.glob(os.path.join(reports_dir, "bench_%s_*.json" % (run or "*"))):
        try:
            with open(path) as handle:
                report = json.load(handle)
            if (not report.get("gate", {}).get("checks", {}).get("full_frozen_suite")
                    or report.get("limit") is not None):
                continue
            timestamp = report["timestamp"]
            # Current reports use YYYYmmdd_HHMMSS. Parse ISO as well, without using filenames/mtime.
            try:
                stamp = datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S").replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                stamp = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=datetime.timezone.utc)
            candidates.append((stamp.timestamp(), path))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return max(candidates)[1] if candidates else None


def lock_changed(reports_dir: str = REPORTS, lock_path: str = LOCK) -> bool:
    last = latest_report(reports_dir)
    if last is None:
        return True
    return json.load(open(last)).get("lock_md5") != md5_file(lock_path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nav.bench")
    ap.add_argument("--runner"); ap.add_argument("--run", default="default"); ap.add_argument("--scene")
    ap.add_argument("--seeds", default=SEEDS); ap.add_argument("--limit", type=int)
    ap.add_argument("--procs", type=int, default=DEFAULT_PROCS)
    ap.add_argument("--compare", nargs=2, metavar=("NEW", "BASE"))
    ap.add_argument("--if-lock-changed", action="store_true")
    a = ap.parse_args(argv)
    if a.compare:
        flags = compare(json.load(open(a.compare[0])), json.load(open(a.compare[1])))
        print(json.dumps(flags, indent=1))
        return 1 if flags else 0
    if a.if_lock_changed and not lock_changed():
        print("policies.lock.json unchanged since last report; nothing to do")
        return 0
    if not a.runner:
        ap.error("--runner required")
    print(run_bench(a.runner, a.run, a.scene, a.seeds, procs=a.procs, limit=a.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
