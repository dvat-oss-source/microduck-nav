"""Same-apartment outcome cross-check: nav Sim vs independent mjlab (Warp CPU) replays.

One-command entry point (nav Python, from ~/microduck/nav):
  PYTHONPATH=. $PY tools/replay_compare.py --traces reports/traces/<run> --n 20

Pipeline for each selected trace (nav-type, replayable policies, sample frozen in
reports/replay_criterion.json):
  1. nav open-loop replays of the recorded command sequence in the nav Sim (planner/follower NOT re-run):
     an exact replay (same seed; must reproduce the trace bit-for-bit = harness sanity) and the
     criterion's replicate set (1e-6 m initial perturbation and/or different internal delay-RNG seed).
     Their spread is the "chaotic noise" an outcome comparison must tolerate.
  2. tools/replay_mjlab.py (reference Python, subprocess) replays the same commands in mjlab.
  3. The SAME independent checker (nav/checker.py) scores the original trace, the nav replicates and
     the mjlab replay; per-episode agreement follows the frozen criterion; overall disagreement rate
     with a Wilson 95% CI; verdict PASS iff disagreement <= 10%.
Outputs: <out>.json / <out>.md (default reports/replay_check.*).

Other subcommands (development / calibration, run before freezing):
  --make-pilots DIR          generate self-made apartment episodes with the real stack (nav/episode.py)
  --calibrate DIR            nav-vs-nav divergence of open-loop replays -> reports/replay_calibration.json
  --freeze                   write reports/replay_criterion.json from the calibration (refuses to overwrite)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

NAV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NAV))
from nav import checker, stats  # noqa: E402
from tools.replay_mjlab import (REFUSED_KINDS, load_trace, lock_entry, step_schedule)  # noqa: E402

ROOT = NAV.parent
REF_PY = ROOT / ".venv-ref/bin/python"
REPORTS = NAV / "reports"
CRITERION = REPORTS / "replay_criterion.json"
CALIBRATION = REPORTS / "replay_calibration.json"

# Self-made development episodes (NOT benchmark tasks). Start/goal checked free on the walk map.
# (office -> corridor through the office doorway is unplannable: the filing cabinet narrows it.)
PILOT_TASKS = [
    {"ep": "pilot_office_short", "type": "nav", "seed": 0, "start": [2.0, 0.0, 0.0], "goal": [1.5, 0.0, 0.0],
     "note": "handoff pilot geometry (office, turn + short walk)"},
    {"ep": "pilot_corridor_straight", "type": "nav", "seed": 1, "start": [-0.2, 0.45, 1.5708],
     "goal": [-0.2, 2.3, 1.5708], "note": "straight corridor walk north"},
    {"ep": "pilot_bedroom_doorway", "type": "nav", "seed": 2, "start": [0.9, 1.8, 3.1416],
     "goal": [-0.2, 1.2, -1.5708], "note": "bedroom -> corridor through the bedroom doorway, turn south"},
    {"ep": "pilot_bath_doorway", "type": "nav", "seed": 5, "start": [2.6, -1.8, 3.1416],
     "goal": [-0.2, -2.0, 1.5708], "note": "bathroom -> corridor through the bathroom doorway, south of the hole"},
    {"ep": "pilot_toward_hole", "type": "nav", "seed": 3, "start": [-0.2, 1.6, -1.5708],
     "goal": [-0.2, -0.35, -1.5708], "note": "corridor south, stop 0.35 m north of the stair hole"},
    {"ep": "pilot_living_to_corridor", "type": "nav", "seed": 4, "start": [-1.85, -1.95, 0.0],
     "goal": [-0.2, -2.0, 0.0], "note": "living room -> corridor south via doorway next to the hole"},
]


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------- pilots
def make_pilots(out_dir, policy="walk", max_time=60.0, tasks=PILOT_TASKS, only=None):
    from nav.episode import EpisodeRunner
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = EpisodeRunner(policy=policy, mode="gt", dynamics="training", max_time=max_time)
    rows = []
    for task in tasks:
        if only and task["ep"] not in only:
            continue
        t0 = time.time()
        tr = runner(task)
        metadata = {k: v for k, v in tr.items() if k not in ("qpos", "t")}
        np.savez_compressed(out / f"{task['ep']}.npz", qpos=tr["qpos"], t=tr["t"],
                            metadata=json.dumps(metadata, default=lambda v: v.tolist()), task=json.dumps(task))
        v = checker.check(tr, task)
        rows.append({"ep": task["ep"], "steps": len(tr["qpos"]) - 1, "wall_s": time.time() - t0,
                     "transitions": tr["transitions"], "verdict": {k: v.get(k) for k in (
                         "success", "fallen", "hole_incursion", "collision_steps", "final_pose", "pos_err", "reasons")}})
        print(json.dumps(rows[-1]), flush=True)
    old = json.loads((out / "pilots.json").read_text()) if (out / "pilots.json").exists() else []
    rows = [r for r in old if r["ep"] not in {x["ep"] for x in rows}] + rows
    (out / "pilots.json").write_text(json.dumps(rows, indent=1))
    return rows


# ----------------------------------------------------------------------------- nav open-loop replay
def replayable(tr):
    """(ok, reason). Nav tasks only; kick policies refused (mirroring is not in the trace)."""
    if tr["task"].get("type", "nav") != "nav":
        return False, "not_nav_task"
    if tr["meta"].get("dynamics") != "training":
        return False, "not_training_dynamics"
    for _, k in tr["meta"]["policy_timeline"]:
        if lock_entry(k)["kind"] in REFUSED_KINDS:
            return False, f"kick_policy:{k}"
    if len(tr["qpos"]) < 2:
        return False, "empty_trace"
    return True, ""


def nav_replay(tr, seed_offset=0, perturb=None):
    """Open-loop replay of the recorded per-step commands in the nav Sim. Returns qpos (T, nq)."""
    import mujoco
    from nav.sim import Sim, TRAINING_DYNAMICS_VERSION
    meta = tr["meta"]
    if meta["provenance"].get("dynamics_version") != TRAINING_DYNAMICS_VERSION:
        raise RuntimeError("trace dynamics_version differs from current training Sim")
    prm = meta["provenance"]["params"]
    n_steps = len(tr["qpos"]) - 1
    pol, cmd, _ = step_schedule(meta, n_steps)
    sim = Sim(meta["scene_xml"], pol[0] if pol else meta["policy"], record=True, dynamics=meta["dynamics"])
    if sim.model.nq != tr["qpos"].shape[1]:
        raise RuntimeError("nav model layout differs from the trace")
    sim.reset(int(prm["seed"]) + int(seed_offset), start_pose=prm["start_pose"],
              extra={"vin": prm["vin"], "vin_drop_gain": prm["vin_drop_gain"], "init": prm.get("init", "stand"),
                     "policy": pol[0]})
    # Frozen episodes explicitly relocate free objects; replay their complete initial state.
    sim.data.qpos[:] = tr["qpos"][0]
    sim.data.qvel[:] = 0.
    mujoco.mj_forward(sim.model, sim.data)
    sim.trace[0] = sim.data.qpos.copy()
    if perturb is not None:
        a = sim.fj_qadr
        sim.data.qpos[a:a + 2] += np.asarray(perturb, float)
        mujoco.mj_forward(sim.model, sim.data)
        sim.trace[0] = sim.data.qpos.copy()
    for k in range(n_steps):
        if pol[k] != sim.policy_key:
            sim.force_switch(pol[k])
        sim.step(np.asarray(cmd[k], float))
    return np.asarray(sim.trace)


def outcome(qpos, t, model_path, task):
    v = checker.check({"qpos": qpos, "t": t, "model_path": model_path}, task)
    xy = v["final_pose"][:2]
    return {"fell": bool(v["fallen"]), "stair_hole": bool(v["hole_incursion"]),
            "static_collision": int(v["collision_steps"]) > 0, "collision_geoms": v["collision_geoms"],
            "final_xy": xy, "final_yaw": v["final_pose"][2], "success": bool(v["success"]),
            "pos_err": v.get("pos_err"), "fall_step": v["fall_step"]}


def trunk_xy(model_path, qpos):
    import mujoco
    m = checker.load_model(model_path)
    a = int(m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")]])
    return qpos[:, a:a + 2], a


def divergence(model_path, q_ref, q_other):
    xy0, _ = trunk_xy(model_path, q_ref)
    xy1, _ = trunk_xy(model_path, q_other)
    d = np.linalg.norm(xy0 - xy1, axis=1)
    return {"end": float(d[-1]), "max": float(np.nanmax(d)), "mean": float(np.nanmean(d)),
            "t_1cm": next((i * 0.02 for i, x in enumerate(d) if x > 0.01), None)}


def nav_cloud(tr, replicates):
    """Exact replay (sanity) + replicate set; outcomes and divergences vs the original trace."""
    mp_ = tr["meta"]["model_path"]
    out = {"ep": tr["ep"], "steps": len(tr["qpos"]) - 1}
    q_exact = nav_replay(tr, 0, None)
    out["exact_replay_max_abs_qpos_diff"] = float(np.abs(q_exact - tr["qpos"]).max()) if q_exact.shape == tr["qpos"].shape else None
    out["original"] = outcome(tr["qpos"], tr["t"], mp_, tr["task"])
    out["replicates"] = []
    for spec in replicates:
        q = nav_replay(tr, spec.get("seed_offset", 0), spec.get("perturb"))
        out["replicates"].append({"spec": spec, "outcome": outcome(q, tr["t"], mp_, tr["task"]),
                                  "divergence": divergence(mp_, tr["qpos"], q)})
    return out


# ----------------------------------------------------------------------------- calibration / criterion
CAL_REPLICATES = ([{"name": f"perturb_x_{i}", "seed_offset": 0, "perturb": [1e-6 * s, 0.0]}
                   for i, s in enumerate((1, -1))] +
                  [{"name": "perturb_y", "seed_offset": 0, "perturb": [0.0, 1e-6]}] +
                  [{"name": f"rng_{o}", "seed_offset": o, "perturb": None} for o in (1000, 2000, 3000, 4000)])


def calibrate(trace_dir, out=CALIBRATION):
    traces = [load_trace(p) for p in sorted(Path(trace_dir).glob("*.npz"))]
    rows = []
    for tr in traces:
        ok, why = replayable(tr)
        if not ok:
            continue
        t0 = time.time()
        c = nav_cloud(tr, CAL_REPLICATES)
        c["wall_s"] = time.time() - t0
        rows.append(c)
        print(tr["ep"], "exact", c["exact_replay_max_abs_qpos_diff"],
              [(r["spec"]["name"], round(r["divergence"]["end"], 3)) for r in c["replicates"]], flush=True)
    pert = [r["divergence"]["end"] for c in rows for r in c["replicates"] if r["spec"]["seed_offset"] == 0]
    rng = [r["divergence"]["end"] for c in rows for r in c["replicates"] if r["spec"]["seed_offset"] != 0]
    per_s = [r["divergence"]["end"] / max(c["steps"] * 0.02, 1e-9) for c in rows for r in c["replicates"]]
    summary = {"traces": str(trace_dir), "n_episodes": len(rows), "replicates": CAL_REPLICATES,
               "perturb_1e-6_end_div_m": stats.percentiles(pert, (50, 90, 100)),
               "rng_end_div_m": stats.percentiles(rng, (50, 90, 100)),
               "end_div_per_second": stats.percentiles(per_s, (50, 90, 100)),
               "exact_replay_bitwise": all(c["exact_replay_max_abs_qpos_diff"] == 0.0 for c in rows),
               "outcome_flips_within_nav": {
                   k: sum(any(r["outcome"][k] != c["original"][k] for r in c["replicates"]) for c in rows)
                   for k in ("fell", "stair_hole", "static_collision")},
               "episodes": rows}
    Path(out).write_text(json.dumps(summary, indent=1))
    return summary


def loo_false_alarms(cal, factor, floor_m):
    """Treat each RNG replicate as a 'perfect simulator with its own delay draws'; tolerance from the others."""
    n = k = 0
    for c in cal["episodes"]:
        reps = c["replicates"]
        for j, r in enumerate(reps):
            if r["spec"]["seed_offset"] == 0:
                continue
            others = [x["divergence"]["end"] for i, x in enumerate(reps) if i != j]
            n += 1
            k += r["divergence"]["end"] > max(floor_m, factor * max(others))
    return {"false_alarms": k, "n": n}


def freeze(calibration=CALIBRATION, out=CRITERION, floor_m=0.10, factor=2.0):
    if Path(out).exists():
        raise SystemExit(f"{out} exists: the criterion is frozen; refusing to overwrite")
    cal = json.loads(Path(calibration).read_text())
    loo = {f"factor{f}_floor{fl}": loo_false_alarms(cal, f, fl) for f in (1.0, 1.5, 2.0) for fl in (0.05, 0.10)}
    crit = {
        "version": 1,
        "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "frozen_before": "any 20-episode benchmark replay; pilot mjlab results other than the single "
                         "harness-development run of pilot_office_short were not inspected before freezing",
        "comparison_unit": "open-loop replay of the recorded per-step command sequence (planner/follower/localizer "
                           "not re-run); same ONNX file (md5-checked); same initial state mapped by joint name; "
                           "same battery vin/sag; mjlab nominal play cfg (no DR, no pushes, no obs corruption)",
        "scorer": "nav/checker.py applied identically to nav original, nav replicates and mjlab replay "
                  "(mjlab qpos mapped back into the trace's qpos layout by joint/body name)",
        "outcomes": {
            "fell": "checker.fallen (tilt > 60 deg or trunk z < 0.05 m at any step)",
            "stair_hole": "checker.hole_incursion (trunk or any robot geom centre over the derived stair hole)",
            "static_collision": "checker.collision_steps > 0 (robot contact with a static non-floor geom)",
            "final_position": "trunk xy at the final recorded step",
        },
        "nav_reference_set": {"original": "the recorded trace",
                              "replicates": [r for r in CAL_REPLICATES]},
        "binary_rule": "a binary outcome AGREES iff the mjlab value equals the value of the original trace OR the "
                       "value of at least one nav replicate (i.e. mjlab lies inside the nav chaotic-noise set)",
        "position_rule": {
            "statistic": "d = |xy_mjlab(T) - xy_nav_original(T)|",
            "tolerance": "X_ep = max(floor_m, factor * max_r |xy_replicate_r(T) - xy_nav_original(T)|)",
            "floor_m": floor_m, "factor": factor,
            "applies_when": "neither the nav original nor the mjlab replay fell (otherwise 'fell' decides)",
            "agrees": "d <= X_ep",
        },
        "episode_rule": "an episode AGREES iff all four outcomes agree; mjlab rep 0 is primary (extra reps diagnostic)",
        "verdict_rule": "disagreement rate = disagreeing / n; PASS iff rate <= 0.10 (point estimate, n = 20); "
                        "Wilson 95% CI reported; if FAIL the nav sim must be fixed before any benchmark result is trusted",
        "sample_rule": "from --traces DIR take *.npz sorted by name, keep replayable nav-type traces (training dynamics, "
                       "no kick policy), then numpy default_rng(0).choice(n, replace=False) if more than n remain; "
                       "fewer than n => verdict INSUFFICIENT_SAMPLE",
        "mjlab": {"task_id": "Mjlab-Velocity2-Rough-Backlash-MicroDuck", "play_cfg": True, "seed": 0, "reps": 1},
        "calibration_file": str(Path(calibration).relative_to(NAV)) if Path(calibration).is_relative_to(NAV) else str(calibration),
        "calibration_sha256": sha256(calibration),
        "calibration_summary": {k: cal[k] for k in ("n_episodes", "perturb_1e-6_end_div_m", "rng_end_div_m",
                                                    "end_div_per_second", "exact_replay_bitwise",
                                                    "outcome_flips_within_nav")},
        "leave_one_out_false_alarms": loo,
        "rationale": "Open-loop replay diverges even for a perfect simulator. The nav sim replayed against itself "
                     "with 1e-6 m initial perturbations and with different internal actuator/obs-delay RNG streams "
                     "(the reference necessarily draws its own delays) measures that divergence per episode. "
                     "factor 2 compensates for estimating the spread from 7 replicates (leave-one-out over the pilot "
                     "RNG replicates: factor 1 -> 3-4/20 false alarms, 1.5 -> 1-2/20, 2 -> 0/20); the 0.10 m floor guards "
                     "against episodes where the replicates happen to coincide. Binary outcomes can flip inside the nav "
                     "set itself (pilot_living_to_corridor: 2/4 RNG replicates fall, 4/4 touch wall wB_s in the doorway); "
                     "the binary rule therefore accepts any value the nav set produces.",
    }
    Path(out).write_text(json.dumps(crit, indent=1))
    return crit


# ----------------------------------------------------------------------------- comparison
def select(trace_dir, n):
    paths = sorted(Path(trace_dir).glob("*.npz"))
    keep, skipped = [], {}
    for p in paths:
        tr = load_trace(p)
        ok, why = replayable(tr)
        (keep.append(str(p)) if ok else skipped.__setitem__(p.name, why))
    if len(keep) > n:
        idx = np.random.default_rng(0).choice(len(keep), size=n, replace=False)
        keep = [keep[i] for i in sorted(idx)]
    return keep, skipped


def judge(crit, cloud, mj):
    """Per-episode agreement under the frozen criterion."""
    orig = cloud["original"]
    reps = [r["outcome"] for r in cloud["replicates"]]
    res = {}
    for k in ("fell", "stair_hole", "static_collision"):
        allowed = {orig[k]} | {r[k] for r in reps}
        res[k] = {"nav": orig[k], "nav_set": sorted(allowed), "mjlab": mj[k], "agree": mj[k] in allowed}
    spread = max([r["divergence"]["end"] for r in cloud["replicates"]] or [0.0])
    pr = crit["position_rule"]
    X = max(pr["floor_m"], pr["factor"] * spread)
    d = float(np.hypot(mj["final_xy"][0] - orig["final_xy"][0], mj["final_xy"][1] - orig["final_xy"][1]))
    applies = not orig["fell"] and not mj["fell"]
    res["final_position"] = {"d_m": d, "X_m": X, "nav_spread_m": spread, "applies": applies,
                             "agree": (d <= X) if applies else True}
    res["agree"] = all(res[k]["agree"] for k in ("fell", "stair_hole", "static_collision", "final_position"))
    return res


def write_md(rep, path):
    L = [f"# Same-apartment mjlab replay cross-check ({rep['label']})", "",
         f"Verdict: **{rep['verdict']}** — {rep['n_disagree']}/{rep['n']} episodes disagree "
         f"(rate {rep['disagreement_rate']:.3f}, Wilson 95% CI [{rep['wilson95'][0]:.3f}, {rep['wilson95'][1]:.3f}]). "
         f"Threshold: rate <= 0.10 (criterion frozen {rep['criterion_frozen_at']}, sha256 {rep['criterion_sha256'][:12]}).", "",
         "| episode | steps | fell nav/set/mj | hole nav/set/mj | coll nav/set/mj | end d / X (m) | nav success / mj | agree |",
         "|---|---|---|---|---|---|---|---|"]
    def b(x):
        return "".join("T" if v else "F" for v in x) if isinstance(x, list) else ("T" if x else "F")
    for e in rep["episodes"]:
        j = e["judgement"]
        if j.get("reference_nonfinite"):
            L.append(f"| {e['ep']} | {e['steps']} | — | — | — | — | {b(e['nav_success'])} / — | "
                     f"**NO** ({j['reason']}; counted as disagreement) |")
            continue
        L.append(f"| {e['ep']} | {e['steps']} | {b(j['fell']['nav'])}/{b(j['fell']['nav_set'])}/{b(j['fell']['mjlab'])} | "
                 f"{b(j['stair_hole']['nav'])}/{b(j['stair_hole']['nav_set'])}/{b(j['stair_hole']['mjlab'])} | "
                 f"{b(j['static_collision']['nav'])}/{b(j['static_collision']['nav_set'])}/{b(j['static_collision']['mjlab'])} | "
                 f"{j['final_position']['d_m']:.3f} / {j['final_position']['X_m']:.3f} | "
                 f"{b(e['nav_success'])} / {b(e['mjlab_success'])} | {'yes' if j['agree'] else '**NO**'} |")
    L += ["", f"Per-outcome disagreements: {rep['per_outcome_disagreements']}. Non-finite reference episodes counted as disagreements: {rep.get('nonfinite_reference_counted_as_disagreement')}.",
          f"Exact nav replays bit-identical to traces: {rep['exact_replays_bitwise']}. "
          f"mjlab initial-state max error (mapped by name): {rep['mjlab_init_state_max_err']:.2e}.",
          f"mjlab extra reps (diagnostic): {rep.get('mjlab_rep_spread')}.",
          f"Skipped traces: {rep['skipped']}.", "",
          "Scorer definition update: " + rep.get("checker_definition_update", "historical scorer"),
          f"Exact goal-success agreement (separate from frozen physics criterion): {rep.get('exact_goal_success_agreement')}.", "",
          "Notes: " + rep["notes"]]
    Path(path).write_text("\n".join(L) + "\n")


def run_check(trace_dir, n, out_base, label=None, mjlab_dir=None, reps=1, skip_mjlab=False):
    if not CRITERION.exists():
        raise SystemExit("criterion not frozen: run --calibrate and --freeze first")
    crit = json.loads(CRITERION.read_text())
    paths, skipped = select(trace_dir, n)
    if not paths:
        raise SystemExit("no replayable traces")
    mjlab_dir = Path(mjlab_dir or (Path(out_base).parent / (Path(out_base).name + "_mjlab")))
    proc = None
    if not skip_mjlab:
        cmd = [str(REF_PY), str(NAV / "tools/replay_mjlab.py"), "--out", str(mjlab_dir),
               "--reps", str(max(reps, crit["mjlab"]["reps"])), "--seed", str(crit["mjlab"]["seed"]), *paths]
        log = open(str(mjlab_dir) + ".log", "w") if mjlab_dir.parent.exists() else subprocess.DEVNULL
        env = dict(os.environ, OMP_NUM_THREADS="1")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(NAV), env=env)
        print("mjlab replay started:", " ".join(cmd[:6]), "...", flush=True)
    clouds = {}
    replicates = crit["nav_reference_set"]["replicates"]
    for p in paths:   # nav replicates run in this process while the reference runs (2 processes max)
        tr = load_trace(p)
        clouds[tr["ep"]] = (tr, nav_cloud(tr, replicates))
        print("nav cloud", tr["ep"], [round(r["divergence"]["end"], 3) for r in clouds[tr["ep"]][1]["replicates"]], flush=True)
    if proc is not None:
        if proc.wait() != 0:
            raise SystemExit(f"replay_mjlab failed; see {mjlab_dir}.log")
    summ = json.loads((mjlab_dir / "replay_mjlab_summary.json").read_text())
    by = {}
    for e in summ["episodes"]:
        by.setdefault(e["ep"], {})[e["rep"]] = e
    episodes = []
    rep_spread = []
    for ep, (tr, cloud) in clouds.items():
        mjs = by[ep]
        z = np.load(mjs[0]["file"])
        zq = np.asarray(z["qpos"], float)
        if mjs[0].get("nan_step") is not None or not np.all(np.isfinite(zq)):
            # The frozen criterion does not define non-finite reference traces. Conservative handling: the episode
            # is not adjudicated and counts as a DISAGREEMENT, so a numerically failing reference can never
            # improve the verdict. (The criterion file itself is unchanged.)
            mj = {"success": None, "final_xy": None, "collision_geoms": None}
            why = f"mjlab reference non-finite from step {mjs[0].get('nan_step')}"
            j = {o: {"not_evaluated": why} for o in ("fell", "stair_hole", "static_collision", "final_position")}
            j.update(agree=False, reference_nonfinite=True, reason=why)
            episodes.append({"ep": ep, "trace": tr["path"], "steps": cloud["steps"], "task": tr["task"],
                             "nav_success": cloud["original"]["success"], "mjlab_success": None,
                             "nav_final_xy": cloud["original"]["final_xy"], "mjlab_final_xy": None,
                             "exact_replay_max_abs_qpos_diff": cloud["exact_replay_max_abs_qpos_diff"],
                             "nav_replicates": cloud["replicates"], "mjlab_extra_reps": [],
                             "mjlab_nan_step": mjs[0].get("nan_step"), "judgement": j})
            continue
        mj = outcome(zq, np.asarray(z["t"], float), tr["meta"]["model_path"], tr["task"])
        j = judge(crit, cloud, mj)
        extra = []
        for r, e in sorted(mjs.items()):
            if r == 0:
                continue
            zr = np.load(e["file"])
            o = outcome(np.asarray(zr["qpos"], float), np.asarray(zr["t"], float), tr["meta"]["model_path"], tr["task"])
            dd = float(np.hypot(o["final_xy"][0] - mj["final_xy"][0], o["final_xy"][1] - mj["final_xy"][1]))
            extra.append({"rep": r, "fell": o["fell"], "stair_hole": o["stair_hole"],
                          "static_collision": o["static_collision"], "end_dist_to_rep0_m": dd,
                          "judgement_agree": judge(crit, cloud, o)["agree"]})
            rep_spread.append(dd)
        episodes.append({"ep": ep, "trace": tr["path"], "steps": cloud["steps"], "task": tr["task"],
                         "nav_success": cloud["original"]["success"], "mjlab_success": mj["success"],
                         "nav_final_xy": cloud["original"]["final_xy"], "mjlab_final_xy": mj["final_xy"],
                         "mjlab_collision_geoms": mj["collision_geoms"],
                         "nav_collision_geoms": cloud["original"]["collision_geoms"],
                         "exact_replay_max_abs_qpos_diff": cloud["exact_replay_max_abs_qpos_diff"],
                         "nav_replicates": cloud["replicates"], "mjlab_extra_reps": extra,
                         "mjlab_nan_step": mjs[0]["nan_step"], "judgement": j})
    n_eff = len(episodes)
    k = sum(not e["judgement"]["agree"] for e in episodes)
    rate = k / n_eff
    verdict = ("INSUFFICIENT_SAMPLE" if n_eff < n else ("PASS" if rate <= 0.10 else "FAIL"))
    rep = {"label": label or str(trace_dir), "generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "traces": str(trace_dir), "n_requested": n, "n": n_eff, "n_disagree": k, "disagreement_rate": rate,
           "wilson95": list(stats.wilson_ci(k, n_eff)), "verdict": verdict,
           "criterion_sha256": sha256(CRITERION), "criterion_frozen_at": crit["frozen_at"],
           "checker_sha256": sha256(checker.__file__),
           "checker_definition_update": "User-approved pre-benchmark audit repairs A04/A09: near-floor collision extents over hole; fall tilt>60deg or trunk height above lowest foot<0.02m. Frozen replicate/position/agreement rules unchanged; criterion file retains its historical scorer descriptions.",
           "exact_goal_success_agreement": sum(e["nav_success"] == e["mjlab_success"] for e in episodes) / n_eff,
           "per_outcome_disagreements": {o: sum(not e["judgement"][o].get("agree", True) for e in episodes)
                                         for o in ("fell", "stair_hole", "static_collision", "final_position")},
           "nonfinite_reference_counted_as_disagreement": [e["ep"] for e in episodes
                                                           if e["judgement"].get("reference_nonfinite")],
           "exact_replays_bitwise": all(e["exact_replay_max_abs_qpos_diff"] == 0.0 for e in episodes),
           "mjlab_init_state_max_err": max(e["init_state_max_err"] for e in summ["episodes"]),
           "mjlab_rep_spread": stats.percentiles(rep_spread, (50, 100)) if rep_spread else None,
           "mjlab_summary": {k2: summ[k2] for k2 in ("version", "task_id", "seed", "reps", "n_worlds", "steps",
                                                     "wall_s", "env_report", "policies", "geometry_check")},
           "skipped": skipped, "episodes": episodes,
           "notes": ("Outcomes compare open-loop command replays, not closed-loop navigation; the agreement is "
                     "about simulator physics under identical commands. mjlab terrain geoms carry the training "
                     "_soften_terrain_contacts solref/solimp (nav Sim training mode applies the same since dynamics_version cb8561f6_nominal_v2; earlier traces did not), see geometry_check.")}
    Path(str(out_base) + ".json").write_text(json.dumps(rep, indent=1, default=float))
    write_md(rep, str(out_base) + ".md")
    print(f"{verdict}: {k}/{n_eff} disagree -> {out_base}.json/.md", flush=True)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traces", help="directory of benchmark trace NPZ files")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=str(REPORTS / "replay_check"), help="output base path (no extension)")
    ap.add_argument("--label")
    ap.add_argument("--mjlab-dir")
    ap.add_argument("--reps", type=int, default=1, help="mjlab worlds per trace (rep 0 is judged)")
    ap.add_argument("--skip-mjlab", action="store_true", help="reuse an existing --mjlab-dir")
    ap.add_argument("--make-pilots", metavar="DIR")
    ap.add_argument("--policy", default="walk")
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--only", nargs="*", help="--make-pilots: restrict to these pilot names")
    ap.add_argument("--calibrate", metavar="DIR")
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args(argv)
    if a.make_pilots:
        make_pilots(a.make_pilots, a.policy, a.max_time, only=a.only)
    if a.calibrate:
        s = calibrate(a.calibrate)
        print(json.dumps({k: v for k, v in s.items() if k != "episodes"}, indent=1))
    if a.freeze:
        freeze()
    if a.traces:
        run_check(a.traces, a.n, a.out, a.label, a.mjlab_dir, a.reps, a.skip_mjlab)


if __name__ == "__main__":
    main()
