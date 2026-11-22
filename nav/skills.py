"""Phase G skills: measured kick / pick / recover contracts + helpers to run them in nav.sim.

Everything here is *simulation* only and drives `nav.sim.Sim` through its public API
(reset / step / switch_policy / force_switch / place_ball / ball_pos / skill_done).
Contracts are measured by tools/kick_sweep.py and tools/skill_sweeps.py and stored in
nav/data/skill_contracts.json; the FSM (nav/fsm.py) only *reads* that file.

Frames: trunk frame = trunk yaw frame (x forward, y left), origin = trunk_base position.

KICK success (our definition, see reports/kick_sweep.md for the comparison with checker.py):
    no fall (tilt <= 60 deg, trunk z >= 0.05) AND ball peak speed >= 0.30 m/s AND ball net displacement
    over the 2.0 s window >= 0.30 m AND net direction within +-30 deg of the trunk's forward axis.
checker.py (type "kick") scores peak per-step ball speed >= min_speed (0.3) and the direction *at the
peak-speed step* within angle_tol (30 deg), but has no displacement criterion; both are evaluated on every
trial and agreement is reported.

PICK robot postconditions are no fall, upright at the end and xy drift under 1 cm.
They do not establish grasp or lift; the old geometric reach radius is unvalidated,
and generated contracts mark pick unsupported.

RECOVER success = after the stand-up policy runs from a settled fallen state: not fallen, tilt < 15 deg,
height >= 0.09 m (standing 0.110) within the time limit.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import mirror as _mirror

NAV_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_PATH = NAV_ROOT / "data" / "skill_contracts.json"

CONTROL_DT = 0.02
KICK_WINDOW_STEPS = 100            # 2.0 s after the kick policy starts
KICK_MIN_DISP = 0.30
KICK_MIN_SPEED = 0.30
KICK_ANGLE_TOL = math.radians(30.0)
GRID_1CM = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
UPRIGHT_TILT = math.radians(15.0)
STAND_HEIGHT_OK = 0.09


# --------------------------------------------------------------------------- frames
def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def trunk_to_world(pose, p):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (pose[0] + c * p[0] - s * p[1], pose[1] + s * p[0] + c * p[1])


def world_to_trunk(pose, p):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx, dy = p[0] - pose[0], p[1] - pose[1]
    return (c * dx + s * dy, -s * dx + c * dy)


# --------------------------------------------------------------------------- skill-morphology sim
# The kick / ground-pick / stand-up tasks are trained on MICRODUCK_STANDUP_ROBOT_CFG = robot_allcollisions*.xml
# (trunk, hips, shins, head shell and jaw collide with the floor / ball).  nav.sim's dynamics="training" path
# builds robot_walk_backlash.xml (the Velocity2 walker model): only the two foot pads collide with the floor and
# the trunk/shins are contype 2 (self-collision only).  A fallen robot there sinks through the floor
# (trunk z -0.106 m) and the ball can only be touched by the soles.  Until nav.sim exposes the robot model
# (REQUESTS.md, Phase G 2026-11-09), skill_sim() builds the allcollisions_backlash variant by rewriting the
# robot include during Sim construction only.  Infer dynamics already uses an allcollisions-equivalent model.
SKILL_ROBOT_XML = "robot_allcollisions_backlash.xml"


_diagnostic_sim_class = None


def _skill_diagnostic_sim_class():
    """Isolated copy of Sim for causal diagnostics, never benchmark contracts.

    The frozen training checkout calls its ground-contact variant allcollisions.
    Loading the copy in a separate module avoids changing ElementTree or the
    benchmark Sim process-wide. Fail closed if the model builder changes.
    """
    global _diagnostic_sim_class
    if _diagnostic_sim_class is None:
        import sys
        import types
        from . import sim as benchmark_sim

        source = Path(benchmark_sim.__file__).read_text()
        old = 'str(robot_dir / "robot_walk_backlash.xml")'
        new = 'str(robot_dir / "robot_allcollisions_backlash.xml")'
        if source.count(old) != 1:
            raise RuntimeError("diagnostic Sim model selection changed; review before use")
        module = types.ModuleType("nav._skill_diagnostic_sim")
        module.__file__ = benchmark_sim.__file__
        module.__package__ = "nav"
        sys.modules[module.__name__] = module
        try:
            exec(compile(source.replace(old, new), module.__file__, "exec"), module.__dict__)
        except Exception:
            del sys.modules[module.__name__]
            raise
        _diagnostic_sim_class = module.Sim
    return _diagnostic_sim_class


def skill_sim(scene="flat", policy_key="walk", dynamics="training", robot="allcollisions", **kw):
    """nav.sim.Sim with the skill-task collision morphology.  robot: "allcollisions" | "walk" (sim default)."""
    from .sim import Sim
    if dynamics != "training" or robot == "walk":
        return Sim(scene, policy_key, dynamics=dynamics, **kw)
    sim = _skill_diagnostic_sim_class()(scene, policy_key, dynamics=dynamics, **kw)
    sim.robot_model = SKILL_ROBOT_XML
    return sim


def benchmark_ball_sim(policy_key, dynamics="training"):
    """Flat benchmark-morphology Sim with the apartment target ball geometry.

    Unlike the training ball, this sphere is offset above a floor-level body
    origin. Placement must preserve that origin convention.
    """
    if dynamics != "training":
        raise ValueError("benchmark ball scene requires training dynamics")
    from .sim import Sim
    sim = Sim(str(NAV_ROOT / "data" / "scene_skill_apartment_ball.xml"), policy_key,
              dynamics=dynamics, ball=True)

    def place_ball(x, y, for_kick=None):
        if for_kick:
            x, y = sim._ball_in_front(for_kick)
        sim.data.qpos[sim.ball_qadr:sim.ball_qadr + 7] = [x, y, 0., 1, 0, 0, 0]
        sim.data.qvel[sim.ball_vadr:sim.ball_vadr + 6] = 0.

    sim.place_ball = place_ball
    return sim


# --------------------------------------------------------------------------- mirrored session
class MirroredSession:
    """Drop-in for an onnxruntime session inside nav.sim.Sim: act = M_a(pi(M_o(obs)))."""

    def __init__(self, base_session):
        self._mp = _mirror.MirroredPolicy(session=base_session)
        self.base = base_session

    def run(self, out_names, feed):
        obs = np.asarray(next(iter(feed.values())), dtype=np.float32).reshape(-1)
        return [self._mp.act(obs)[None, :].astype(np.float32)]


@dataclass(frozen=True)
class KickSpec:
    """name: label; key: lock key whose ONNX is run; side: foot that kicks; mirror: wrap in MirroredPolicy."""
    name: str
    key: str
    side: str
    mirror: bool = False


KICK_SPECS = {
    "right_pollen": KickSpec("right_pollen", "kick_right", "right"),
    "right_user": KickSpec("right_user", "kick_right_user", "right"),
    "left_pollen": KickSpec("left_pollen", "kick_left", "left"),
    "left_mirror_pollen": KickSpec("left_mirror_pollen", "kick_right", "left", True),
    "left_mirror_user": KickSpec("left_mirror_user", "kick_right_user", "left", True),
}


def activate_kick_policy(sim, spec: KickSpec) -> bool:
    """Hand off from the current (standing) policy to the kick policy (mirrored if asked)."""
    if not sim.switch_policy(spec.key):
        return False
    if spec.mirror:
        sim._sess = MirroredSession(sim._sess)
    return True


# --------------------------------------------------------------------------- kick trial
def _stand_settle(sim, seed, start_pose, n_steps, policy="walk", extra=None):
    ex = {"policy": policy}
    ex.update(extra or {})
    sim.reset(seed, start_pose=start_pose, extra=ex)
    gt = sim.ground_truth()
    for _ in range(n_steps):
        gt, _ = sim.step(np.zeros(3))
    return gt


def run_kick_trial(sim, spec: KickSpec, offset=(0.0, 0.0), seed=0, settle_steps=75,
                   window=KICK_WINDOW_STEPS, with_checker=False, scene_path=None, stand_policy="walk"):
    """One closed-loop kick: settled-standing -> hand-off -> ball at nominal+offset (trunk frame) -> kick.

    Returns a dict with success, fell, ball displacement (trunk frame at kick start), direction error,
    peak speed, and (optionally) the independent checker verdict on the same trace.
    """
    sim.reset(seed, start_pose=(0.0, 0.0, 0.0), extra={"policy": stand_policy, "ball_xy": (5.0, 5.0)})
    for _ in range(settle_steps):
        sim.step(np.zeros(3))
    ok = activate_kick_policy(sim, spec)
    gt = sim.ground_truth()
    out = {"handoff_ok": bool(ok), "seed": int(seed), "offset": [float(offset[0]), float(offset[1])]}
    if not ok:
        out.update(success=False, fell=bool(gt.fallen), reason="handoff_timeout")
        return out
    tx, ty = _mirror.kick_ball_target(spec.side)
    pose = gt.pose2.copy()
    bw = trunk_to_world(pose, (tx + offset[0], ty + offset[1]))
    sim.place_ball(*bw)
    b0 = sim.ball_pos()[:2].copy()
    n_trace0 = len(sim.trace)
    start_row = sim.data.qpos.copy()          # includes the just-placed ball
    peak = 0.0
    fell = False
    tilt_max = 0.0
    for _ in range(window):
        gt, _ = sim.step(np.zeros(3))
        v = sim.data.qvel[sim.ball_vadr:sim.ball_vadr + 2]
        peak = max(peak, float(np.hypot(v[0], v[1])))
        tilt_max = max(tilt_max, gt.tilt_rad)
        if gt.fallen:
            fell = True
    b1 = sim.ball_pos()[:2]
    d = b1 - b0
    disp = float(np.hypot(*d))
    ang = wrap(math.atan2(d[1], d[0]) - pose[2]) if disp > 1e-4 else float("nan")
    success = bool((not fell) and peak >= KICK_MIN_SPEED and disp >= KICK_MIN_DISP
                   and abs(ang) <= KICK_ANGLE_TOL)
    out.update(success=success, fell=fell, disp=disp, angle_deg=math.degrees(ang) if disp > 1e-4 else None,
               peak_speed=peak, tilt_max_deg=math.degrees(tilt_max),
               ball_dx=float(np.dot(d, [math.cos(pose[2]), math.sin(pose[2])])),
               ball_dy=float(np.dot(d, [-math.sin(pose[2]), math.cos(pose[2])])),
               vin=sim.params.get("vin"))
    if with_checker:
        from . import checker as _ck
        trace = {"qpos": np.asarray([start_row] + list(sim.trace[n_trace0:])),
                 "model_path": str(scene_path or sim.scene_xml)}
        trace["t"] = np.arange(len(trace["qpos"])) * CONTROL_DT
        v = _ck.check(trace, {"type": "kick", "ball_body": "ball", "direction": float(pose[2]),
                              "min_speed": KICK_MIN_SPEED, "angle_tol_deg": 30.0})
        out["checker_success"] = bool(v["success"])
        out["checker_reasons"] = v["reasons"]
        out["checker_peak_speed"] = v.get("ball_peak_speed")
        out["checker_angle_err_deg"] = v.get("angle_err_deg")
    return out


def kick_sweep(sim, spec: KickSpec, offsets=None, seeds=range(20), with_checker=False, progress=None,
               stand_policy="walk"):
    """Closed-loop sweep of one kick policy over a (dx, dy) grid (trunk frame, relative to nominal).

    Returns dict(offsets=[(dx,dy)...], success (n_off,), fall_rate, ball_dx_mean, ball_dy_mean, angle_mean_deg,
    n_seeds, trials=[...]).
    """
    if offsets is None:
        offsets = [(dx, dy) for dx in GRID_1CM for dy in GRID_1CM]
    seeds = list(seeds)
    rows = []
    succ, fall, bdx, bdy, angm, spd = [], [], [], [], [], []
    agree, chk = [], []
    for k, off in enumerate(offsets):
        tr = [run_kick_trial(sim, spec, off, s, with_checker=with_checker, stand_policy=stand_policy) for s in seeds]
        rows.append(tr)
        succ.append(np.mean([t["success"] for t in tr]))
        fall.append(np.mean([t["fell"] for t in tr]))
        good = [t for t in tr if t.get("disp") is not None]
        bdx.append(float(np.mean([t["ball_dx"] for t in good])) if good else float("nan"))
        bdy.append(float(np.mean([t["ball_dy"] for t in good])) if good else float("nan"))
        angs = [t["angle_deg"] for t in good if t["angle_deg"] is not None]
        angm.append(float(np.mean(angs)) if angs else float("nan"))
        spd.append(float(np.mean([t["peak_speed"] for t in good])) if good else float("nan"))
        if with_checker:
            for t in tr:
                agree.append(bool(t["success"]) == bool(t["checker_success"]))
                chk.append(bool(t["checker_success"]))
        if progress:
            progress(spec.name, k + 1, len(offsets))
    res = dict(spec=spec.name, key=spec.key, side=spec.side, mirror=spec.mirror,
               offsets=[list(map(float, o)) for o in offsets], n_seeds=len(seeds),
               success=np.array(succ), fall_rate=np.array(fall),
               ball_dx_mean=np.array(bdx), ball_dy_mean=np.array(bdy),
               angle_mean_deg=np.array(angm), peak_speed_mean=np.array(spd),
               overall_success=float(np.mean(succ)), trials=rows)
    if with_checker:
        res["checker_agreement"] = float(np.mean(agree))
        res["checker_success_rate"] = float(np.mean(chk))
    return res


def tolerance_region(offsets, success, thresh=0.9):
    """Cells with success >= thresh; plus the largest all-cells-pass axis-aligned rectangle (grid cells)
    and the connected-region summary.  Returns dict (JSON-serialisable)."""
    offs = np.asarray(offsets, float)
    succ = np.asarray(success, float)
    xs = np.unique(np.round(offs[:, 0], 6))
    ys = np.unique(np.round(offs[:, 1], 6))
    grid = np.full((len(xs), len(ys)), np.nan)
    for (dx, dy), s in zip(offs, succ):
        grid[np.searchsorted(xs, round(dx, 6)), np.searchsorted(ys, round(dy, 6))] = s
    ok = grid >= thresh - 1e-9
    best = None
    for i0 in range(len(xs)):
        for i1 in range(i0, len(xs)):
            for j0 in range(len(ys)):
                for j1 in range(j0, len(ys)):
                    if ok[i0:i1 + 1, j0:j1 + 1].all():
                        area = (i1 - i0 + 1) * (j1 - j0 + 1)
                        # tie-break: smaller offset of box centre from the nominal (0,0)
                        cx = 0.5 * (xs[i0] + xs[i1]); cy = 0.5 * (ys[j0] + ys[j1])
                        key = (area, -math.hypot(cx, cy))
                        if best is None or key > best[0]:
                            best = (key, (i0, i1, j0, j1))
    region = {"threshold": thresh, "n_cells_ok": int(ok.sum()), "n_cells": int(ok.size),
              "cells_ok": [[float(xs[i]), float(ys[j])] for i, j in zip(*np.nonzero(ok))],
              "best_cell": None, "box": None}
    k = int(np.nanargmax(succ))
    region["best_cell"] = {"offset": [float(offs[k][0]), float(offs[k][1])], "success": float(succ[k])}
    if best is not None:
        i0, i1, j0, j1 = best[1]
        region["box"] = {"dx": [float(xs[i0]), float(xs[i1])], "dy": [float(ys[j0]), float(ys[j1])],
                         "center": [float(0.5 * (xs[i0] + xs[i1])), float(0.5 * (ys[j0] + ys[j1]))],
                         "half": [float(0.5 * (xs[i1] - xs[i0])), float(0.5 * (ys[j1] - ys[j0]))],
                         "n_cells": int((i1 - i0 + 1) * (j1 - j0 + 1))}
    return region


# --------------------------------------------------------------------------- pick
PICK_KEYS = {"pollen": "pick", "user": "atlantis_pick_flat"}


def _mouth_site(sim):
    import mujoco
    return mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")


def run_pick_trial(sim, key, seed=0, obj_xy_trunk=None, t_pre=100, t_post=150, back_to="walk",
                   stand_policy="walk"):
    """Evaluator-style pick trial (tools/eval_ground_pick.py protocol): stand 2.0 s on the walker,
    pick (phase 0 -> 0.7, 139 steps), stand 3.0 s.  obj_xy_trunk: (x, y) of a proxy ball in the trunk frame
    at the pick start (None: no object).  Returns robot terms + mouth-tip trajectory summary + ball effect.
    """
    import mujoco
    sim.reset(seed, start_pose=(0.0, 0.0, 0.0), extra={"policy": stand_policy, "ball_xy": (5.0, 5.0)})
    mouth = _mouth_site(sim)
    m, d = sim.model, sim.data
    start_xy = sim.ground_truth().pose2[:2].copy()
    for _ in range(t_pre):
        sim.step(np.zeros(3))
    ok = sim.switch_policy(key)
    gt = sim.ground_truth()
    pose = gt.pose2.copy()
    if obj_xy_trunk is not None:
        sim.place_ball(*trunk_to_world(pose, obj_xy_trunk))
        ball0 = sim.ball_pos()[:2].copy()
    fell = bool(gt.fallen)
    best_h, best_tr, best_ball_d = 1e9, None, None
    contact_steps = 0
    n = 0
    while not sim.skill_done and n < 300:
        gt, _ = sim.step(np.zeros(3))
        n += 1
        fell |= bool(gt.fallen)
        h = float(d.site_xpos[mouth][2])
        if h < best_h:
            best_h = h
            best_tr = world_to_trunk(gt.pose2, d.site_xpos[mouth][:2])
            if obj_xy_trunk is not None:
                best_ball_d = float(np.hypot(*(np.array(d.site_xpos[mouth][:2]) - sim.ball_pos()[:2])))
        if obj_xy_trunk is not None:
            bgeom = sim.model.geom("ball_geom").id
            for c in d.contact[:d.ncon]:
                if (c.geom1 == bgeom) != (c.geom2 == bgeom):
                    other = c.geom2 if c.geom1 == bgeom else c.geom1
                    if sim.robot_geoms[other] and other not in sim.foot_geoms.values():
                        contact_steps += 1
                        break
    sim.force_switch(back_to)
    for _ in range(t_post):
        gt, _ = sim.step(np.zeros(3))
        fell |= bool(gt.fallen)
    drift = float(np.hypot(*(gt.pose2[:2] - start_xy)))
    upright = bool((not fell) and gt.tilt_rad < UPRIGHT_TILT)
    out = dict(seed=int(seed), handoff_ok=bool(ok), fell=fell, end_upright=upright, drift_m=drift,
               min_mouth_h=best_h, mouth_trunk_xy=[float(best_tr[0]), float(best_tr[1])],
               vin=sim.params.get("vin"), steps=n, post_ok=bool(upright and drift < 0.01))
    if obj_xy_trunk is not None:
        b = sim.ball_pos()[:2]
        out.update(ball_push_m=float(np.hypot(*(b - ball0))), mouth_ball_d=best_ball_d,
                   head_ball_contact_steps=int(contact_steps))
    return out


# --------------------------------------------------------------------------- recover
FALL_ORIENTS = {
    # name: (axis, angle) rotation applied to the upright trunk before dropping with joints holding DEFAULT_POSE
    "back": ("y", -math.pi / 2),    # body x (forward) points up: lying on its back
    "front": ("y", math.pi / 2),    # face down
    "side_left": ("x", math.pi / 2),
    "side_right": ("x", -math.pi / 2),
}


def _quat_axis_angle(axis, ang):
    ax = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([math.cos(ang / 2)] + [math.sin(ang / 2) * a for a in ax])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def make_fallen(sim, orient, seed, drop_height=0.15, hold_s=1.5, jitter_rad=0.15, stand_policy="walk"):
    """Reset the sim and let the robot collapse into a settled fallen pose (motors hold DEFAULT_POSE, no policy).
    Returns the GroundTruth after settling.  Deterministic in (orient, seed)."""
    import mujoco
    rng = np.random.default_rng(1000 + seed)
    sim.reset(seed, start_pose=(0.0, 0.0, 0.0), extra={"policy": stand_policy})
    axis, ang = FALL_ORIENTS[orient]
    q = _quat_axis_angle(axis, ang + rng.uniform(-jitter_rad, jitter_rad))
    q = _quat_mul(_quat_axis_angle("z", rng.uniform(-math.pi, math.pi)), q)
    q = _quat_mul(q, _quat_axis_angle("x", rng.uniform(-jitter_rad, jitter_rad)))
    q /= np.linalg.norm(q)
    a = sim.fj_qadr
    sim.data.qpos[a + 2] = drop_height
    sim.data.qpos[a + 3:a + 7] = q
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(int(hold_s / 0.005)):
        sim.bam_ctrl.update()
        mujoco.mj_step(sim.model, sim.data)
    sim.t = float(sim.data.time)
    return sim.ground_truth()


def run_recover_trial(sim, key, orient, seed, max_steps=500, init_from="fallen", command=None,
                      stand_policy="walk"):
    """Settled fallen state -> stand-up policy `key` -> success if standing (tilt<15 deg, h>=0.09) before
    max_steps (10 s default).  Returns dict(success, t_stand_s, fell_again, ...)."""
    gt0 = make_fallen(sim, orient, seed, stand_policy=stand_policy)
    tilt0 = gt0.tilt_rad
    h0 = gt0.height
    sim.force_switch(key)
    t_stand = None
    hold = 0
    gt = gt0
    cmd = np.zeros(3) if command is None else np.asarray(command)
    for k in range(max_steps):
        gt, _ = sim.step(cmd)
        if gt.tilt_rad < UPRIGHT_TILT and gt.height >= STAND_HEIGHT_OK:
            hold += 1
            if hold >= 25 and t_stand is None:        # stays up for 0.5 s
                t_stand = (k + 1 - 24) * CONTROL_DT
                break
        else:
            hold = 0
    ok = t_stand is not None
    return dict(seed=int(seed), orient=orient, success=bool(ok), t_stand_s=t_stand,
                start_tilt_deg=math.degrees(tilt0), start_height=float(h0),
                end_tilt_deg=math.degrees(gt.tilt_rad), end_height=float(gt.height),
                start_fallen=bool(gt0.fallen))


# --------------------------------------------------------------------------- contracts
_CONTRACT_CACHE: dict = {}


def load_contracts(path=None) -> dict:
    p = Path(path or CONTRACTS_PATH)
    key = (str(p), p.stat().st_mtime if p.exists() else 0)
    if key not in _CONTRACT_CACHE:
        if not p.exists():
            _CONTRACT_CACHE[key] = default_contracts()
        else:
            _CONTRACT_CACHE[key] = json.loads(p.read_text())
    return _CONTRACT_CACHE[key]


def default_contracts() -> dict:
    """Fallback (nominal training-distribution numbers) used only when skill_contracts.json is absent."""
    return {
        "version": 0, "fallback": True,
        "kick": {
            "right": {"policy_key": "kick_right", "mirror": False, "target_trunk": [0.09, -0.042],
                      "tol_half": [0.015, 0.015], "direction_bias_rad": 0.0, "duration_s": 2.0},
            "left": {"policy_key": "kick_right", "mirror": True, "target_trunk": [0.09, 0.042],
                     "tol_half": [0.015, 0.015], "direction_bias_rad": 0.0, "duration_s": 2.0},
        },
        "pick": {"policy_key": "pick", "mouth_trunk_xy": [0.108, 0.0], "reach_radius": 0.02,
                 "tol_half": [0.02, 0.02], "duration_s": 2.8, "drift_max_m": 0.01},
        "recover": {"policy_key": "atlantis_standup_flat", "tilt_trigger_deg": 60.0, "max_time_s": 10.0},
    }


def choose_kick_side(ball_xy_trunk) -> str:
    """Kick with the foot on the same side as the ball (y>0 -> left)."""
    return "left" if ball_xy_trunk[1] > 0 else "right"


def kick_pre_pose(ball_xy, direction, side, contracts=None):
    """Robot pose (x, y, yaw) from which the contracted kick is expected to work.

    The ball must sit at the contract target in the trunk frame; the robot heading is the intended
    direction minus the policy's measured direction bias (the kicks are not exactly straight)."""
    c = (contracts or load_contracts())["kick"][side]
    yaw = wrap(direction - c.get("direction_bias_rad", 0.0))
    tx, ty = c["target_trunk"]
    cs, sn = math.cos(yaw), math.sin(yaw)
    return (ball_xy[0] - (cs * tx - sn * ty), ball_xy[1] - (sn * tx + cs * ty), yaw)


def pick_pre_pose(obj_xy, heading, contracts=None):
    """Pose from which the pick's mouth-tip lowest point lands on obj_xy; robot faces `heading`."""
    c = (contracts or load_contracts())["pick"]
    mx, my = c["mouth_trunk_xy"]
    cs, sn = math.cos(heading), math.sin(heading)
    return (obj_xy[0] - (cs * mx - sn * my), obj_xy[1] - (sn * mx + cs * my), wrap(heading))


def kick_precondition(pose, ball_xy, side, contracts=None, margin=1.0):
    """(ok, err_trunk(dx,dy)): ball (world) vs the contract target in the trunk frame, within the tolerance box."""
    c = (contracts or load_contracts())["kick"][side]
    bx, by = world_to_trunk(pose, ball_xy)
    ex, ey = bx - c["target_trunk"][0], by - c["target_trunk"][1]
    hx, hy = c["tol_half"]
    return (abs(ex) <= hx * margin + 1e-12 and abs(ey) <= hy * margin + 1e-12), (ex, ey)


def pick_precondition(pose, obj_xy, contracts=None, margin=1.0):
    c = (contracts or load_contracts())["pick"]
    bx, by = world_to_trunk(pose, obj_xy)
    ex, ey = bx - c["mouth_trunk_xy"][0], by - c["mouth_trunk_xy"][1]
    return (math.hypot(ex, ey) <= c["reach_radius"] * margin), (ex, ey)


KICK_BALL_RADIUS = 0.04
KICK_STAGE_BACKOFF = 0.25
KICK_RETARGET_M = 0.02


def kick_direction_budget(contract):
    """Remaining direction budget from successful measured contract-box trials."""
    measured = contract.get('measured', {})
    box = measured.get('region', {}).get('box')
    if not box:
        return None
    angles = [abs(float(t['angle_deg'])) for t in measured.get('trials', [])
              if t.get('success') and t.get('angle_deg') is not None
              and all(box[axis][0] - 1e-9 <= t['offset'][i] <= box[axis][1] + 1e-9
                      for i, axis in enumerate(('dx', 'dy')))]
    return KICK_ANGLE_TOL - math.radians(max(angles)) if angles else None


def kick_stage_plan(gridmap, pose, ball_xy, direction, contracts=None):
    """Plan to a stage outside the fully inflated ball; reserve a straight run-in.

    The target ball is removed only from the final run-in's obstacle set. Other
    free objects and static obstacles retain their full measured inflation.
    """
    from .mapping import overlay_discs
    contracts = contracts or load_contracts()
    ball = np.asarray(ball_xy, float)[:2]
    preferred = choose_kick_side(world_to_trunk(pose, ball))
    for side in (preferred, 'left' if preferred == 'right' else 'right'):
        c = contracts['kick'][side]
        if not c.get('supported') or min(c['tol_half']) <= 0:
            continue
        pre = kick_pre_pose(ball, direction, side, contracts)
        axis = np.array([math.cos(pre[2]), math.sin(pre[2])])
        stage_distance = KICK_STAGE_BACKOFF + float(c['target_trunk'][0])
        if gridmap is not None:
            stage_distance = max(stage_distance,KICK_STAGE_BACKOFF+float(gridmap.params['r_inflate']))
        backoff = max(.02,stage_distance-float(c['target_trunk'][0]))
        stage_xy = np.asarray(pre[:2]) - backoff * axis
        stage = (*stage_xy.tolist(), pre[2])
        if gridmap is not None:
            runin = np.asarray(pre[:2])[None, :] - np.linspace(0., backoff,
                int(math.ceil(backoff/(gridmap.res/2)))+1)[:,None] * axis
            if np.any(gridmap.is_lethal(runin[:,0], runin[:,1])):
                continue
            planning_map = overlay_discs(gridmap, [{'xy':ball.tolist(), 'radius':KICK_BALL_RADIUS,
                                                    'name':'kick_target'}])
            if planning_map.is_lethal(*stage[:2]) or planning_map.is_lethal(*pose[:2]):
                continue
        else:
            planning_map = None
        return dict(ok=True, side=side, pre_pose=pre, stage_pose=stage,
                    ball=ball.tolist(), map=planning_map, runin_map=gridmap,
                    backoff=backoff)
    return dict(ok=False, reason='kick_staging_blocked', ball=ball.tolist())


class KickApproach:
    """Straight world-line run-in, with moving steering and current-ball handoff.

    Large turns finish at the stage; the run-in keeps positive forward motion. A miss
    returns to that stage using an exact measured reverse command before a new
    alignment. No settling or in-place alignment is attempted beside the ball.
    """
    def __init__(self, contracts, side, direction, follower, gridmap=None):
        self.contracts, self.side = contracts, side
        self.direction, self.follower, self.gridmap = float(direction), follower, gridmap
        self.contract = contracts['kick'][side]
        self.yaw_tol = kick_direction_budget(self.contract)
        self.events = []
        self.reason = None
        self.state = 'idle'

    def begin(self, pose, ball_xy, t):
        self.ref_ball = np.asarray(ball_xy, float)[:2].copy()
        self.started = float(t)
        self.state = 'advance'
        self.backoff = KICK_STAGE_BACKOFF
        if self.gridmap is not None:
            self.backoff = max(self.backoff,KICK_STAGE_BACKOFF+float(self.gridmap.params['r_inflate'])
                              -float(self.contract['target_trunk'][0]))
        # The FSM already chose the side and reached its stage.
        model, env = self.follower.model, self.follower.env
        safe = env.safe
        self.forward = self.reverse = None
        self.boost = None
        self._pose_history = []
        self._next_command_t = float(t)
        self._held_command = np.zeros(3)
        if safe is not None:
            straight = [c for c in safe if abs(c[1])+abs(c[2]) < 1e-9]
            forward = [c for c in straight if c[0] > 0 and model.v_at(c[0]) > 0.01]
            reverse = [c for c in straight if c[0] < 0 and model.v_at(c[0]) < -.005]
            if forward:
                self.forward = min(forward, key=lambda c:model.v_at(c[0])).copy()
                starts = [c for c in forward if c[0] >= getattr(model,'vx_start',self.forward[0])-1e-9]
                self.boost = min(starts,key=lambda c:c[0]).copy() if starts else None
            if reverse:
                self.reverse = max(reverse, key=lambda c:c[0]).copy()
        if self.forward is None or self.boost is None or self.yaw_tol is None or self.yaw_tol <= 0:
            self.state, self.reason = 'fail', 'kick_approach_unmeasured'
        self.events.append(dict(t=float(t),event='begin',side=self.side))

    def step(self, pose, ball_xy, t):
        zero = np.zeros(3)
        if self.state in ('idle','fail','ready'):
            return zero, self.state
        ball = np.asarray(ball_xy, float)[:2]
        self._pose_history.append((float(t),np.asarray(pose[:2],float).copy(),float(pose[2])))
        self._pose_history = [p for p in self._pose_history if t-p[0] <= .3+1e-9]
        dt = float(t)-self._pose_history[0][0]
        velocity = (np.asarray(pose[:2])-self._pose_history[0][1])/dt if dt > 1e-9 else np.zeros(2)
        moving = float(velocity @ np.array([math.cos(pose[2]),math.sin(pose[2])])) > .02
        if np.linalg.norm(ball-self.ref_ball) > KICK_RETARGET_M:
            self.events.append(dict(t=float(t),event='retarget',ball=ball.tolist()))
            self.state = 'retarget'
            return zero, self.state
        pre = kick_pre_pose(ball,self.direction,self.side,self.contracts)
        yaw_error = abs(wrap(pose[2]-pre[2]))
        valid, error = kick_precondition(pose,ball,self.side,self.contracts)
        if self.state == 'advance' and valid and yaw_error <= self.yaw_tol:
            self.state = 'ready'
            self.events.append(dict(t=float(t),event='ready',error=list(map(float,error)),
                                    yaw_error=float(yaw_error)))
            return zero, self.state
        if t-self.started > 20.:
            self.state, self.reason = 'fail', 'kick_approach_timeout'
            return zero, self.state
        # Heading and lateral misses cannot be repaired by turning beside the ball.
        hx, hy = self.contract['tol_half']
        # Walking yaw and lateral sway are not settled kick-direction errors.
        # Continue the straight run-in until the current box is reached or the
        # robot has passed its longitudinal tolerance; never turn to chase sway.
        if self.state == 'advance' and error[0] < -hx:
            self.state = 'retreat'
        if self.state == 'retreat':
            axis = np.array([math.cos(pre[2]),math.sin(pre[2])])
            along = float((np.asarray(pose[:2])-np.asarray(pre[:2])) @ axis)
            if along <= -self.backoff:
                return zero, 'restage'
            if self.reverse is None:
                self.state, self.reason = 'fail', 'kick_approach_no_reverse'
                return zero, self.state
            cmd = self.reverse.copy()
        else:
            if t >= self._next_command_t-1e-9:
                model = self.follower.model
                # Track a straight WORLD line with moving steering. Never ask
                # for an in-place turn inside the ball inflation. Match the
                # follower's cadence, sway averaging and measured gait startup.
                window = float(getattr(self.follower,'sway_window',.1))
                history = [p for p in self._pose_history if t-p[0] <= window+1e-9]
                mean_xy = np.mean([p[1] for p in history],axis=0)
                heading = math.atan2(np.mean([math.sin(p[2]) for p in history]),
                                     np.mean([math.cos(p[2]) for p in history]))
                axis_yaw = pre[2]
                lateral = float((mean_xy-np.asarray(pre[:2])) @ np.array([-math.sin(axis_yaw),math.cos(axis_yaw)]))
                omega,zeta = float(getattr(self.follower,'omega',1.3)),float(getattr(self.follower,'zeta',.9))
                lookahead = max(.1,model.v_at(model.vx_min)/omega)
                desired = axis_yaw-float(np.clip(math.atan2(lateral,lookahead),-math.pi/6,math.pi/6))
                wz = float(np.clip(2.*omega*zeta*wrap(desired-heading),-model.w_turn_max,model.w_turn_max))
                vx,turn,_,_ = model.inverse(model.v_at(model.vx_min),wz,
                    vx_lo=model.vx_min if moving else model.vx_start)
                command = np.array([vx,0.,turn])
                clamp = getattr(self.follower.env,'clamp',lambda c:c)
                self._held_command = np.asarray(clamp(command),float)
                if self._held_command[0] <= 0.:
                    self.state,self.reason = 'fail','kick_approach_no_forward'
                    return zero,self.state
                self._next_command_t = float(t)+float(getattr(self.follower,'dt',.1))
            cmd = self._held_command.copy()
        if self.gridmap is not None:
            axis = np.array([math.cos(pose[2]),math.sin(pose[2])])
            speed = self.follower.model.v_at(cmd[0])
            end = np.asarray(pose[:2]) + axis * speed * (self.follower.model.stop_per_v + CONTROL_DT)
            segment = np.linspace(pose[:2],end,max(2,int(np.linalg.norm(end-pose[:2])/(self.gridmap.res/2))+1))
            if np.any(self.gridmap.is_lethal(segment[:,0],segment[:,1])):
                self.state, self.reason = 'fail', 'kick_runin_obstacle'
                return zero, self.state
        return cmd, self.state


def run_kick_approach_trial(sim, gridmap, envelope, primitives, seed, distance, bearing,
                            checker_model, trace_path=None, max_time=90.):
    """Full flat approach, handoff and kick; ball is placed before any motion.

    Uses the production FSM and independent, unchanged checker. This is a DEV
    sweep on the canonical walking morphology and apartment ball geometry.
    """
    from .follower import RPPFollower
    from .localization import Localizer
    from .fsm import NavFSM
    from . import checker
    import json
    ball = np.array([0.,0.])
    start_xy = ball + float(distance)*np.array([math.cos(bearing),math.sin(bearing)])
    start = (*start_xy,wrap(bearing+math.pi))
    gt = sim.reset(int(seed),start_pose=start,
                   extra=dict(policy=sim.initial_policy_key,ball_xy=ball.tolist()))
    sim.trace[-1] = sim.data.qpos.copy()
    follower = RPPFollower(envelope,gridmap=gridmap)
    loc = Localizer(sim,'gt',seed=int(seed))
    loc.reset(start)
    fsm = NavFSM(sim,gridmap,primitives,follower,loc,
                 target_observer=lambda:sim.ball_pos()[:2].copy(),target_body='ball',
                 target_source='current simulator object observation (GT mode)')
    fsm.start(None,task='kick',target_xy=ball,direction=0.)
    events,commands,transitions = [],[],[]
    step = sim.step
    def recorded(command):
        event = dict(t=float(sim.t),step=len(sim.trace)-1,policy=sim.policy_key,
                     mirror=hasattr(sim._sess,'_mp'))
        if not events or (event['policy'],event['mirror']) != (events[-1]['policy'],events[-1]['mirror']):
            events.append(event)
        commands.append(dict(t=float(sim.t),command=np.asarray(command,float).tolist(),
                             state=fsm.state,ball_distance=float(np.linalg.norm(sim.ground_truth().pose2[:2]-sim.ball_pos()[:2]))))
        return step(command)
    sim.step = recorded
    last = None
    try:
        while sim.t < max_time:
            command,state = fsm.tick(gt)
            if state != last:
                transitions.append(dict(t=float(sim.t),state=state,reason=fsm.reason))
                last = state
            if state in ('done','fail'):
                for _ in range(50):
                    gt,_ = sim.step(np.zeros(3))
                break
            gt,_ = sim.step(command)
        else:
            transitions.append(dict(t=float(sim.t),state='fail',reason='episode_timeout'))
    finally:
        sim.step = step
        loc.close()
    task = dict(type='kick',ball_body='ball',direction=0.,min_speed=KICK_MIN_SPEED,angle_tol_deg=30.)
    trace = dict(qpos=np.asarray(sim.trace),t=np.arange(len(sim.trace))*CONTROL_DT,
                 model_path=str(checker_model),policy_events=events,commands=commands,
                 transitions=transitions,kick_events=fsm.kick_events,
                 runin_events=list(getattr(fsm._kick_approach,'events',[])))
    verdict = checker.check(trace,task,str(checker_model))
    if trace_path is not None:
        metadata = {k:v for k,v in trace.items() if k not in ('qpos','t')}
        with open(trace_path,'xb') as handle:
            np.savez_compressed(handle,qpos=trace['qpos'],t=trace['t'],metadata=json.dumps(metadata),task=json.dumps(task))
    inflation = float(gridmap.params['r_inflate'])+KICK_BALL_RADIUS+gridmap.res/np.sqrt(2)
    forbidden = [c for c in commands if c['state'] in ('follow','kick_align','kick_runin') and
                 c['ball_distance'] <= inflation and abs(c['command'][2])>1e-9 and
                 np.linalg.norm(c['command'][:2])<1e-9]
    return dict(seed=int(seed),distance=float(distance),bearing=float(bearing),side=fsm.side,
                duration_s=float(sim.t),verdict=verdict,stack_reason=transitions[-1]['reason'],
                transitions=transitions,forbidden_turns=len(forbidden),
                provenance=sim.provenance(),trace_path=str(trace_path) if trace_path else None)
