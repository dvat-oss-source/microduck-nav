"""Independent same-apartment replay of recorded nav episodes in Pollen's real mjlab env (Warp CPU).

RUN WITH THE REFERENCE PYTHON (never the nav venv; this file imports nothing from nav/nav):
  OMP_NUM_THREADS=1 ~/microduck/.venv-ref/bin/python tools/replay_mjlab.py \
      --out <dir> [--reps M] [--seed S] trace1.npz [trace2.npz ...]

Input traces are the benchmark's per-episode NPZ files (nav/bench.py: qpos, t, metadata JSON, task
JSON) written by nav/episode.py. For every trace this script:
  1. builds Mjlab-Velocity2-Rough-Backlash-MicroDuck (play cfg, cb8561f6 code) with the procedural
     terrain replaced by the apartment: every top-level static geom of upstream apartment.xml is added
     to mjlab's static "terrain" body (so the training contact sensor 'secondary=terrain' still
     matches the floors), the free objects/balls are added as free bodies, and the training
     _soften_terrain_contacts spec_fn is applied afterwards exactly as the play cfg does;
  2. switches off every domain-randomization / perturbation / observation-corruption term
     (nominal), every termination and curriculum; keeps the structural training dynamics
     (backlash, encoder feedback, BAM, current limit, 3-6 substep actuator delay, 0-1 step obs delays,
     solver settings) as they are in mjlab;
  3. sets battery vin / sag per env to the values recorded in the trace provenance;
  4. writes the recorded initial qpos into mjlab BY JOINT NAME (robot joints and backlash joints by
     name, free objects by body name; layouts differ between the deployment, training and mjlab
     models, so qpos indices are never assumed);
  5. replays OPEN LOOP the recorded per-step command sequence through the SAME ONNX file (md5
     checked against the lock and the trace provenance), writing the 13-D command block of the 61-D
     actor obs directly (twist / zero / phase / posture flag by the policy's lock command_mode) -
     planner, follower and localization are NOT re-run;
  6. maps the mjlab qpos of every step back into the trace's own qpos layout (by name) so the SAME
     independent nav checker can score both, and saves <out>/<ep>.mjlab.npz (+ summary JSON).

Episodes are batched into one env (one Warp world per episode x replicate). Policy switches inside a
trace are replayed by swapping the ONNX file at the recorded time; traces that use a kick policy are
refused (the trace does not record whether the kick was mirrored, see reports/REQUESTS.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("MICRODUCK_ROOT", Path.home() / "microduck")).expanduser()
NAV = ROOT / "nav"
TASK_ID = "Mjlab-Velocity2-Rough-Backlash-MicroDuck"
APARTMENT_XML = ROOT / "upstream/microduck_rl/src/mjlab_microduck/robot/microduck/apartment.xml"
CONTROL_DT = 0.02
CMD_SLOT = (48, 61)
DR_EVENTS = ("push_robot", "foot_friction", "encoder_bias", "base_com", "randomize_com",
             "randomize_head_com", "randomize_mass_inertia", "randomize_joint_friction",
             "randomize_armature")
REFUSED_KINDS = ("kick_left", "kick_right")
VERSION = "replay_mjlab_v1"


# ----------------------------------------------------------------------------- trace handling
def md5_file(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def load_trace(path):
    import numpy as np
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["metadata"]))
    task = json.loads(str(z["task"])) if "task" in z.files else {}
    return {"path": str(path), "qpos": np.asarray(z["qpos"], float), "t": np.asarray(z["t"], float),
            "meta": meta, "task": task, "ep": task.get("ep", Path(path).stem)}


def step_schedule(meta, n_steps):
    """Per-step (policy key, 3-vector command) from the trace.

    Command k is the control recorded at t = 0.02 k (nav/episode.py logs post-tick sim.t, i.e. the time
    of the step about to be executed). Steps without a control (inside a policy hand-off) get zero.
    Policy at step k is the last timeline entry with t <= 0.02 k.
    """
    by_step = {}
    for row in meta["commands"]:
        by_step[int(round(row[0] / CONTROL_DT))] = [float(v) for v in row[1:4]]
    timeline = sorted(((float(t), k) for t, k in meta["policy_timeline"]), key=lambda r: r[0])
    pol, cmd = [], []
    j = 0
    for k in range(n_steps):
        while j + 1 < len(timeline) and timeline[j + 1][0] <= k * CONTROL_DT + 1e-9:
            j += 1
        pol.append(timeline[j][1])
        cmd.append(by_step.get(k, [0.0, 0.0, 0.0]))
    missing = sum(1 for k in range(n_steps) if k not in by_step)
    return pol, cmd, missing


def nav_joint_layout(model):
    """{key: (qposadr, width)} keyed by joint name, or 'body:<name>' for unnamed free joints."""
    import mujoco
    out = {}
    for j in range(model.njnt):
        name = model.joint(j).name
        if not name:
            name = "body:" + model.body(model.jnt_bodyid[j]).name
        width = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(model.jnt_type[j], 1)
        out[name] = (int(model.jnt_qposadr[j]), width)
    return out


def mj_joint_layout(model):
    """Same keys for the mjlab model: strip the entity prefix 'robot/'."""
    import mujoco
    out = {}
    for j in range(model.njnt):
        name = model.joint(j).name
        if not name:
            name = "body:" + model.body(model.jnt_bodyid[j]).name.split("/")[-1]
        name = name.split("/", 1)[1] if name.startswith("robot/") else name
        width = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(model.jnt_type[j], 1)
        out[name] = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]), width)
    return out


# ----------------------------------------------------------------------------- scene construction
def _copy_geom(dst_body, g, name=None):
    import numpy as np
    ng = dst_body.add_geom()
    ng.name = name if name is not None else g.name
    for attr in ("type", "size", "pos", "quat", "contype", "conaffinity", "condim", "friction",
                 "solref", "solimp", "solmix", "priority", "margin", "gap", "group", "mass", "density"):
        setattr(ng, attr, getattr(g, attr))
    ng.rgba = np.array([0.6, 0.6, 0.6, 1.0]) if g.material else g.rgba   # materials are visual only
    return ng


def apartment_spec_fn(apartment_xml, training_spec_fn, report):
    """Replace the plane terrain with the apartment; keep the play cfg's terrain treatment after."""
    import mujoco
    import numpy as np
    apt = mujoco.MjSpec.from_file(str(apartment_xml))

    def fn(spec):
        terrain = spec.body("terrain")
        plane = [g for g in terrain.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
        for g in plane:
            spec.delete(g)
        n_static = 0
        for g in apt.worldbody.geoms:
            _copy_geom(terrain, g)
            n_static += 1
        extra_q = []
        free_bodies = []
        for b in apt.worldbody.bodies:
            joints = list(b.joints)
            if not joints:
                continue   # dock: visual-only, contype 0 geoms
            if len(joints) != 1 or joints[0].type != mujoco.mjtJoint.mjJNT_FREE:
                raise RuntimeError(f"unsupported apartment body {b.name}")
            nb = spec.worldbody.add_body(name=b.name, pos=b.pos, quat=b.quat)
            nb.add_freejoint()
            for g in b.geoms:
                _copy_geom(nb, g)
            extra_q += list(b.pos) + list(b.quat)
            free_bodies.append(b.name)
        for key in spec.keys:   # merged scene keyframe must cover the added free joints
            key.qpos = np.concatenate([np.asarray(key.qpos), np.asarray(extra_q)])
        report.update(n_static_geoms=n_static, removed_plane=len(plane), free_bodies=free_bodies)
        if training_spec_fn is not None:
            training_spec_fn(spec)     # _soften_terrain_contacts: every geom of body "terrain"
            report["training_spec_fn"] = getattr(training_spec_fn, "__name__", str(training_spec_fn))
    return fn


def geometry_check(nav_model, mj_model):
    """Static collidable geoms: identical geometry by name; report contact-parameter differences."""
    import numpy as np
    out = {"n_compared": 0, "missing_in_mjlab": [], "max_pos_err": 0.0, "max_size_err": 0.0,
           "max_quat_err": 0.0, "type_mismatch": [], "param_diffs": {}}
    mj_names = {mj_model.geom(i).name: i for i in range(mj_model.ngeom)}
    for g in range(nav_model.ngeom):
        if nav_model.geom_bodyid[g] != 0:
            continue
        if nav_model.geom_contype[g] == 0 and nav_model.geom_conaffinity[g] == 0:
            continue
        name = nav_model.geom(g).name
        if name not in mj_names:
            out["missing_in_mjlab"].append(name)
            continue
        h = mj_names[name]
        out["n_compared"] += 1
        out["max_pos_err"] = max(out["max_pos_err"], float(np.abs(nav_model.geom_pos[g] - mj_model.geom_pos[h]).max()))
        out["max_size_err"] = max(out["max_size_err"], float(np.abs(nav_model.geom_size[g] - mj_model.geom_size[h]).max()))
        out["max_quat_err"] = max(out["max_quat_err"], float(np.abs(nav_model.geom_quat[g] - mj_model.geom_quat[h]).max()))
        if nav_model.geom_type[g] != mj_model.geom_type[h]:
            out["type_mismatch"].append(name)
        for f in ("geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_priority", "geom_margin"):
            a, b = np.atleast_1d(getattr(nav_model, f)[g]), np.atleast_1d(getattr(mj_model, f)[h])
            if np.abs(a - b).max() > 1e-9:
                d = out["param_diffs"].setdefault(f, {"n": 0, "nav": a.tolist(), "mjlab": b.tolist()})
                d["n"] += 1
    # Robot: per-body mass by name (trunk subtree).
    import mujoco
    tr = mujoco.mj_name2id(nav_model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    errs = []
    for b in range(nav_model.nbody):
        if nav_model.body_rootid[b] != tr:
            continue
        nm = "robot/" + nav_model.body(b).name
        j = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if j >= 0:
            errs.append(abs(float(nav_model.body_mass[b] - mj_model.body_mass[j])))
    out["robot_bodies_compared"] = len(errs)
    out["robot_max_body_mass_err"] = max(errs) if errs else None
    return out


def build_env(n_envs, seed, report):
    sys.path.insert(0, str(ROOT / "upstream/rl_cb8561f6/src"))
    import torch
    torch.set_num_threads(1)
    import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs = n_envs
    cfg.scene.env_spacing = 0.0               # every world at the apartment's own origin
    terrain = cfg.scene.terrain
    terrain.terrain_type = "plane"
    terrain.terrain_generator = None
    terrain.env_spacing = 0.0
    training_spec_fn = cfg.scene.spec_fn
    cfg.scene.spec_fn = apartment_spec_fn(APARTMENT_XML, training_spec_fn, report)
    removed = [k for k in DR_EVENTS if cfg.events.pop(k, None) is not None]
    report["removed_events"] = removed
    report["kept_events"] = sorted(cfg.events)
    report["removed_curriculum"] = sorted(cfg.curriculum)
    cfg.curriculum.clear()
    report["removed_terminations"] = sorted(cfg.terminations)
    cfg.terminations.clear()
    cfg.episode_length_s = 1.0e6
    actor = cfg.observations["actor"]
    actor.enable_corruption = False
    for t in ("base_ang_vel", "projected_gravity"):
        if "max_angle_deg" in actor.terms[t].params:
            actor.terms[t].params["max_angle_deg"] = 0.0
    report["actor_terms"] = list(actor.terms)
    # Initial state is written by name after reset; neutralise the random reset ranges.
    cfg.events["reset_base"].params["pose_range"] = {"z": (0.125, 0.125)}
    cfg.events["reset_base"].params["velocity_range"] = {}
    report["sim_cfg"] = {"timestep": cfg.sim.mujoco.timestep, "integrator": cfg.sim.mujoco.integrator,
                         "cone": cfg.sim.mujoco.cone, "solver": cfg.sim.mujoco.solver,
                         "iterations": cfg.sim.mujoco.iterations, "ls_iterations": cfg.sim.mujoco.ls_iterations,
                         "nconmax": cfg.sim.nconmax, "njmax": cfg.sim.njmax, "decimation": cfg.decimation}
    torch.manual_seed(seed)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    return env


# ----------------------------------------------------------------------------- policies
def lock_entry(key):
    lock = json.loads((NAV / "policies.lock.json").read_text())
    if key not in lock:
        raise KeyError(f"policy {key} not in lock")
    e = lock[key]
    if e.get("status") == "rejected":
        raise RuntimeError(f"policy {key} is rejected in the lock")
    return e


class PolicyBank:
    def __init__(self):
        sys.path.insert(0, str(ROOT))
        self._p = {}
        self.used = {}

    def get(self, key, trace_md5=None):
        if key not in self._p:
            from duckmac.onnx_mlp import NumpyOnnxMLP
            e = lock_entry(key)
            path = ROOT / e["path"]
            md5 = md5_file(path)
            if md5 != e["md5"] or (trace_md5 is not None and trace_md5 != md5):
                raise RuntimeError(f"{key}: md5 {md5} != lock {e['md5']} / trace {trace_md5}")
            if e["obs_dim"] != 61 or e["action_dim"] != 14 or float(e.get("action_scale", 1.0)) != 1.0:
                raise RuntimeError(f"{key}: unexpected policy interface")
            pol = NumpyOnnxMLP(path)
            self._p[key] = (pol, e)
            self.used[key] = {"path": e["path"], "md5": md5, "numpy_vs_ort_max_err": pol.check_err}
        return self._p[key]


def command_block(entry, cmd3, steps_since_switch):
    import numpy as np
    c = np.zeros(13, np.float32)
    mode = entry["command_mode"]
    if mode == "twist":
        c[0:3] = cmd3
    elif mode == "posture_flag":
        c[0] = cmd3[0]
    elif mode == "phase":
        ph = steps_since_switch * CONTROL_DT / float(entry.get("period_s", 4.0))
        c[0], c[1] = math.cos(2 * math.pi * ph), math.sin(2 * math.pi * ph)
    elif mode != "zero":
        raise RuntimeError(f"unknown command_mode {mode}")
    return c


# ----------------------------------------------------------------------------- main replay
def replay(trace_paths, out_dir, reps=1, seed=0, log=print):
    import mujoco
    import numpy as np
    import torch

    traces = [load_trace(p) for p in trace_paths]
    for tr in traces:
        kinds = {k for _, k in tr["meta"]["policy_timeline"]}
        bad = [k for k in kinds if lock_entry(k)["kind"] in REFUSED_KINDS]
        if bad:
            raise RuntimeError(f"{tr['ep']}: kick policies {bad} cannot be replayed (mirroring not recorded)")
        if tr["meta"].get("dynamics") != "training":
            raise RuntimeError(f"{tr['ep']}: reference replays only 'training'-dynamics traces")
    worlds = [(i, r) for i in range(len(traces)) for r in range(reps)]
    N = len(worlds)
    report = {}
    env = build_env(N, seed, report)
    mjm = env.sim.mj_model
    lay_mj = mj_joint_layout(mjm)
    robot = env.scene["robot"]
    bam = [a for a in robot.actuators if getattr(a, "vin_tensor", None) is not None]
    if len(bam) != 1:
        raise RuntimeError("expected one BAM actuator group")
    bam = bam[0]

    # Per-trace mapping, schedules and the static-geometry cross-check.
    nav_models, maps, scheds = {}, [], []
    for tr in traces:
        mp = tr["meta"]["model_path"]
        if mp not in nav_models:
            nm = (mujoco.MjModel.from_binary_path(mp) if mp.endswith(".mjb") else mujoco.MjModel.from_xml_path(mp))
            nav_models[mp] = (nm, nav_joint_layout(nm), geometry_check(nm, mjm))
        nm, lay_nav, _ = nav_models[mp]
        if tr["qpos"].shape[1] != nm.nq:
            raise RuntimeError(f"{tr['ep']}: trace nq {tr['qpos'].shape[1]} != model nq {nm.nq}")
        common = sorted(set(lay_nav) & set(lay_mj))
        unmapped_nav = sorted(set(lay_nav) - set(lay_mj))
        unmapped_mj = sorted(set(lay_mj) - set(lay_nav))
        if unmapped_nav:
            raise RuntimeError(f"{tr['ep']}: nav joints without an mjlab counterpart: {unmapped_nav}")
        for k in common:
            if lay_nav[k][1] != lay_mj[k][2]:
                raise RuntimeError(f"joint {k}: width mismatch")
        maps.append({"common": common, "unmapped_mjlab": unmapped_mj, "nav": lay_nav})
        n_steps = len(tr["qpos"]) - 1
        pol, cmd, missing = step_schedule(tr["meta"], n_steps)
        scheds.append((pol, np.asarray(cmd, np.float32), missing))
    T = max(len(tr["qpos"]) - 1 for tr in traces)

    bank = PolicyBank()
    for tr in traces:
        pmd5 = {k: v["md5"] for k, v in tr["meta"]["provenance"]["policies"].items()}
        for k in {k for _, k in tr["meta"]["policy_timeline"]}:
            bank.get(k, pmd5.get(k))

    # Battery per world, from the trace's seeded draw (nominal otherwise).
    for w, (i, r) in enumerate(worlds):
        prm = traces[i]["meta"]["provenance"]["params"]
        bam.vin_tensor[w] = float(prm["vin"])
        if bam.vin_drop_gain is not None:
            bam.vin_drop_gain[w] = float(prm["vin_drop_gain"])

    # Reset, then write each world's recorded initial state by name and recompute observations.
    env.reset()
    sim = env.sim
    qpos = sim.data.qpos
    qvel = sim.data.qvel
    for w, (i, r) in enumerate(worlds):
        q0 = traces[i]["qpos"][0]
        for k in maps[i]["common"]:
            a_nav, width = maps[i]["nav"][k]
            a_mj, v_mj, _ = lay_mj[k]
            qpos[w, a_mj:a_mj + width] = torch.as_tensor(q0[a_nav:a_nav + width], dtype=qpos.dtype)
        qvel[w, :] = 0.0
    sim.forward()
    env.scene.update(dt=0.0)
    obs = env.observation_manager.compute(update_history=True)
    # The action history/delay buffers were reset by env.reset(); re-reset them now that the state is
    # final so nothing computed from the pre-write state survives (actuator targets are re-derived).
    ids = torch.arange(N)
    env.action_manager.reset(ids)
    env.observation_manager.reset(ids)
    obs = env.observation_manager.compute(update_history=True)
    actor = obs["actor"]
    if actor.shape[1] != 61:
        raise RuntimeError(f"actor obs dim {actor.shape[1]}")

    def snapshot():
        return np.asarray(sim.data.qpos.numpy() if hasattr(sim.data.qpos, "numpy") else sim.data.qpos.cpu().numpy(), float).copy()

    q_log = np.full((T + 1, N, mjm.nq), np.nan)
    q_log[0] = snapshot()
    init_err = []
    for w, (i, r) in enumerate(worlds):
        q0 = traces[i]["qpos"][0]
        e = max(float(np.abs(q_log[0, w, lay_mj[k][0]:lay_mj[k][0] + maps[i]["nav"][k][1]] -
                             q0[maps[i]["nav"][k][0]:maps[i]["nav"][k][0] + maps[i]["nav"][k][1]]).max())
                for k in maps[i]["common"])
        init_err.append(e)
    cur_key = [None] * N
    since = np.zeros(N, int)
    nan_at = [None] * N
    t0 = time.time()
    for k in range(T):
        ob = actor.numpy().astype(np.float32).copy() if hasattr(actor, "numpy") else actor.cpu().numpy().astype(np.float32)
        keys = []
        for w, (i, r) in enumerate(worlds):
            pol, cmd, _ = scheds[i]
            kk = pol[min(k, len(pol) - 1)]
            if kk != cur_key[w]:
                cur_key[w], since[w] = kk, 0
            since[w] += 1    # nav Sim increments policy_steps before building the obs
            c3 = cmd[k] if k < len(cmd) else np.zeros(3, np.float32)
            ob[w, CMD_SLOT[0]:CMD_SLOT[1]] = command_block(bank.get(kk)[1], c3, since[w])
            keys.append(kk)
        act = np.zeros((N, 14), np.float32)
        for kk in set(keys):
            rows = [w for w in range(N) if keys[w] == kk]
            act[rows] = bank.get(kk)[0](ob[rows]).astype(np.float32)
        obs, _, _, _, _ = env.step(torch.from_numpy(act))
        actor = obs["actor"]
        q_log[k + 1] = snapshot()
        bad = ~np.isfinite(q_log[k + 1]).all(axis=1)
        for w in np.nonzero(bad)[0]:
            if nan_at[w] is None:
                nan_at[w] = k + 1
        if k % 250 == 0:
            log(f"step {k}/{T} wall {time.time() - t0:.1f}s")
    wall = time.time() - t0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geom_checks = {mp: v[2] for mp, v in nav_models.items()}
    summary = {"version": VERSION, "task_id": TASK_ID, "play_cfg": True, "seed": seed, "reps": reps,
               "n_worlds": N, "steps": T, "wall_s": wall, "env_report": report,
               "policies": bank.used, "geometry_check": geom_checks, "episodes": []}
    for w, (i, r) in enumerate(worlds):
        tr = traces[i]
        Ti = len(tr["qpos"])
        nav_q = np.array(tr["qpos"], float)          # keeps the trace's own layout; overwritten by name
        mapped = np.repeat(nav_q[:1], Ti, axis=0)
        for kname in maps[i]["common"]:
            a_nav, width = maps[i]["nav"][kname]
            a_mj = lay_mj[kname][0]
            mapped[:, a_nav:a_nav + width] = q_log[:Ti, w, a_mj:a_mj + width]
        name = f"{tr['ep']}.mjlab.r{r}.npz"
        meta = {"source_trace": tr["path"], "ep": tr["ep"], "rep": r, "model_path": tr["meta"]["model_path"],
                "init_state_max_err": init_err[w], "nan_step": nan_at[w],
                "unmapped_mjlab_joints": maps[i]["unmapped_mjlab"],
                "missing_command_steps": scheds[i][2], "policies": sorted(set(scheds[i][0])),
                "vin": float(tr["meta"]["provenance"]["params"]["vin"]),
                "vin_drop_gain": float(tr["meta"]["provenance"]["params"]["vin_drop_gain"])}
        np.savez_compressed(out_dir / name, qpos=mapped, t=tr["t"], metadata=json.dumps(meta),
                            task=json.dumps(tr["task"]))
        meta["file"] = str(out_dir / name)
        summary["episodes"].append(meta)
    (out_dir / "replay_mjlab_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log(f"DONE {N} worlds x {T} steps in {wall:.1f}s -> {out_dir}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=1, help="independent mjlab worlds per trace")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    replay(a.traces, a.out, reps=a.reps, seed=a.seed, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
