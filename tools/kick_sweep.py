"""Measure skill contracts locally. No success is inferred for unsampled offsets."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from nav.sim import Sim
from nav.skills import (KICK_SPECS, kick_sweep, tolerance_region, run_pick_trial,
                        run_recover_trial, FALL_ORIENTS, default_contracts)


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0., center-half), min(1., center+half)]


# --------------------------------------------------------------------------- protocol diagnosis
TRAIN_BALL_NOISE = 0.015          # BALL_POS_NOISE_XY, ball_kick_env_cfg
TRAIN_JOINT_NOISE = 0.05          # reset_robot_joints position_range
TRAIN_TILT = math.radians(5)      # set_ground_state sitting_tilt_max (standing bucket)


def kick_trial_train_protocol(sim, spec, seed, window=100, offset=(0.0, 0.0), ball_noise=TRAIN_BALL_NOISE):
    """BallKick env reset replicated: HOME + U(+-0.05) joints, z U(0.11,0.12), +-5 deg tilt, random yaw, ball at
    nominal + offset + U(+-noise) in the root yaw frame, kick policy from t=0 with zero action history."""
    import mujoco
    from nav import skills as sk
    from nav import mirror as mr
    rng = np.random.default_rng(7000 + seed)
    yaw = rng.uniform(-math.pi, math.pi)
    sim.reset(seed, start_pose=(0.0, 0.0, yaw), extra={"policy": spec.key, "ball_xy": (5.0, 5.0)})
    if spec.mirror:
        sim._sess = sk.MirroredSession(sim._sess)
    d, a = sim.data, sim.fj_qadr
    for qi in sim.joint_qpos_indices:
        d.qpos[qi] += rng.uniform(-TRAIN_JOINT_NOISE, TRAIN_JOINT_NOISE)
    p_, r_ = rng.uniform(-1, 1, 2) * TRAIN_TILT
    q = sk._quat_mul(sk._quat_mul(sk._quat_axis_angle("z", yaw), sk._quat_axis_angle("y", p_)),
                     sk._quat_axis_angle("x", r_))
    d.qpos[a + 2] = rng.uniform(0.11, 0.12)
    d.qpos[a + 3:a + 7] = q
    d.qvel[:] = 0.0
    sim.bam_ctrl.reset(d.qpos)
    sim.bam_ctrl.q_target[:] = sim.default_pose
    sim.last_action[:] = 0.0
    tx, ty = mr.kick_ball_target(spec.side)
    nx, ny = rng.uniform(-ball_noise, ball_noise, 2) if ball_noise > 0 else (0.0, 0.0)
    pose = (0.0, 0.0, yaw)
    sim.place_ball(*sk.trunk_to_world(pose, (tx + offset[0] + nx, ty + offset[1] + ny)))
    mujoco.mj_forward(sim.model, d)
    return _measure_kick(sim, pose, window, dict(seed=int(seed), offset=[float(offset[0] + nx), float(offset[1] + ny)]))


def _measure_kick(sim, pose, window, out):
    from nav import skills as sk
    b0 = sim.ball_pos()[:2].copy()
    fwd = np.array([math.cos(pose[2]), math.sin(pose[2])])
    peak = peak_fwd = 0.0
    fell = False
    for _ in range(window):
        gt, _ = sim.step(np.zeros(3))
        v = sim.data.qvel[sim.ball_vadr:sim.ball_vadr + 2]
        peak = max(peak, float(np.hypot(v[0], v[1])))
        peak_fwd = max(peak_fwd, float(v @ fwd))
        fell |= bool(gt.fallen)
    dd = sim.ball_pos()[:2] - b0
    disp = float(np.hypot(*dd))
    ang = sk.wrap(math.atan2(dd[1], dd[0]) - pose[2]) if disp > 1e-4 else float("nan")
    ok = bool((not fell) and peak >= sk.KICK_MIN_SPEED and disp >= sk.KICK_MIN_DISP and abs(ang) <= sk.KICK_ANGLE_TOL)
    out.update(success=ok, fell=fell, disp=disp, angle_deg=math.degrees(ang) if disp > 1e-4 else None,
               peak_speed=peak, peak_fwd_speed=peak_fwd, ball_dx=float(dd @ fwd))
    return out


def kick_trial_nav_protocol(sim, spec, seed, window=100, offset=(0.0, 0.0), ball_noise=0.0):
    """skills.run_kick_trial protocol (settle 1.5 s on walk, hand-off, ball at nominal+offset), optional U(+-noise)."""
    from nav import skills as sk
    rng = np.random.default_rng(7000 + seed)
    nx, ny = rng.uniform(-ball_noise, ball_noise, 2) if ball_noise > 0 else (0.0, 0.0)
    r = sk.run_kick_trial(sim, spec, (offset[0] + nx, offset[1] + ny), seed, window=window)
    fwd_ok = r.get("disp") is not None
    r.setdefault("peak_fwd_speed", None)
    return r if fwd_ok else r


def diag(args):
    from nav import skills as sk
    specs = args.specs.split(",")
    res = {"seeds": args.seeds, "window_s": 2.0, "runs": []}
    for cfg in args.configs.split(","):
        dyn, robot = cfg.split(":")
        sim = sk.skill_sim("flat", "walk", dynamics=dyn, robot=robot, ball=True)
        for proto in args.protocols.split(","):
            for name in specs:
                spec = sk.KICK_SPECS[name]
                if proto == "train":
                    tr = [kick_trial_train_protocol(sim, spec, s) for s in range(args.seeds)]
                elif proto == "train_nonoise":
                    tr = [kick_trial_train_protocol(sim, spec, s, ball_noise=0.0) for s in range(args.seeds)]
                elif proto == "nav_noise":
                    tr = [kick_trial_nav_protocol(sim, spec, s, ball_noise=TRAIN_BALL_NOISE) for s in range(args.seeds)]
                else:
                    tr = [kick_trial_nav_protocol(sim, spec, s) for s in range(args.seeds)]
                n_ok = int(sum(t["success"] for t in tr))
                row = dict(dynamics=dyn, robot=robot, protocol=proto, spec=name, success=n_ok, n=len(tr),
                           wilson_95=wilson(n_ok, len(tr)), falls=int(sum(t["fell"] for t in tr)),
                           disp_median=float(np.median([t.get("disp") or 0 for t in tr])),
                           peak_speed_median=float(np.median([t.get("peak_speed") or 0 for t in tr])),
                           angle_median=float(np.nanmedian([t["angle_deg"] if t.get("angle_deg") is not None else np.nan for t in tr])),
                           trials=tr)
                res["runs"].append(row)
                print(f"{dyn:8s} {robot:13s} {proto:13s} {name:20s} {n_ok:2d}/{len(tr)} falls {row['falls']} "
                      f"disp {row['disp_median']:.2f} peak {row['peak_speed_median']:.2f} ang {row['angle_median']:+.0f}",
                      flush=True)
    Path(args.diag_out).write_text(json.dumps(res, indent=1, allow_nan=True) + "\n")


def approach_sweep(args):
    import hashlib
    import mujoco
    from nav.skills import benchmark_ball_sim, run_kick_approach_trial
    from nav.episode import read_artifacts
    from nav.mapping import GridMap, robot_dimensions
    from nav.mapsigma import planning_lateral_sigma
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('approach sweep refuses to overwrite evidence or skill contracts')
    output.parent.mkdir(parents=True,exist_ok=True)
    traces = output.with_suffix('')
    traces.mkdir()
    envelope,primitives = read_artifacts(args.stand_policy)
    sigma,sigma_prov = planning_lateral_sigma(args.stand_policy,envelope,args.follower_reports_dir)
    if not sigma_prov['follower_gate_passed']:
        raise RuntimeError('approach sweep requires a matching passing follower gate')
    sim = benchmark_ball_sim(args.stand_policy)
    # This diagnostic scene is an infinite flat plane. Its source include is
    # resolved by Sim's training-scene loader; derive the footprint from that
    # exact compiled model and represent a bounded portion of the plane directly.
    _,radius,_,_ = robot_dimensions(sim.model,sim.data)
    shape = (200,200)
    z = np.zeros(shape,bool)
    gridmap = GridMap(.02,(-2.,-2.),z.copy(),z.copy(),z.copy(),
        np.zeros(shape,np.float32),np.full(shape,4.,np.float32),
        params=dict(robot_radius=radius,lateral_sigma=sigma,r_inflate=radius+3*sigma,decay_k=5.))
    model = traces/'checker.mjb'
    mujoco.mj_saveModel(sim.model,str(model))
    cases = [(d,math.radians(b)) for d in (.3,.5,.75,1.) for b in (-180.,-90.,-60.,60.,90.)]
    if args.approach_case_limit:
        cases = cases[:args.approach_case_limit]
    rows = []
    for index,(distance,bearing) in enumerate(cases):
        for seed in range(args.approach_seed_start,args.approach_seed_start+args.seeds):
            result = run_kick_approach_trial(sim,gridmap,envelope,primitives,seed,distance,bearing,
                model,traces/f'case_{index:02d}_seed_{seed:02d}.npz')
            rows.append(result)
            print(json.dumps(dict(completed=len(rows),total=len(cases)*args.seeds,
                case=index,seed=seed,success=result['verdict']['success'],
                reasons=result['verdict']['reasons'],stack_reason=result['stack_reason'])),flush=True)
    n = len(rows)
    successes = sum(bool(r['verdict']['success']) for r in rows)
    out = dict(DEV=True,protocol='ball placed before motion; distances 0.3–1.0 m; varied bearings; production FSM and unchanged checker',
        stand_policy=args.stand_policy,seeds=list(range(args.approach_seed_start,args.approach_seed_start+args.seeds)),
        n=n,successes=successes,wilson_95=wilson(successes,n),
        falls=sum(bool(r['verdict']['fallen']) for r in rows),
        pre_kick_disturbed=sum(r['verdict']['ball_pre_kick_displacement']>.02 for r in rows),
        not_started=sum('skill_not_started' in r['verdict']['reasons'] for r in rows),
        forbidden_turns=sum(r['forbidden_turns'] for r in rows),sigma_provenance=sigma_prov,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path('nav')/f'{name}.py' for name in ('skills','fsm','mapping','localization','follower','planner','checker')]
            +[Path(__file__),Path('data/skill_contracts.json')]},trials=rows)
    with output.open('x') as handle:
        json.dump(out,handle,indent=2)
    print(json.dumps({k:v for k,v in out.items() if k not in ('trials','source_sha256')},indent=2),flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', action='store_true', help='protocol diagnosis instead of the contract sweep')
    ap.add_argument('--approach',action='store_true',help='approach-inclusive DEV sweep; never writes contracts')
    ap.add_argument('--approach-case-limit',type=int,default=0,help='DEV smoke only; zero runs all twenty geometric cases')
    ap.add_argument('--approach-seed-start',type=int,default=0,help='first seed of an explicitly partitioned approach sweep')
    ap.add_argument('--follower-reports-dir',default='reports',help='matching measured follower gates for approach sweep')
    ap.add_argument('--specs', default='right_pollen,right_user,left_mirror_pollen,left_pollen')
    ap.add_argument('--configs', default='infer:walk,training:walk,training:allcollisions')
    ap.add_argument('--protocols', default='nav,nav_noise,train')
    ap.add_argument('--diag-out', default='reports/skills_diag_kick.json')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--grid', choices=['coarse', 'full'], default='coarse')
    ap.add_argument('--dynamics', choices=['training'], default='training')
    ap.add_argument('--stand-policy', default='atlantis_walk', help='benchmark navigator used during handoff')
    ap.add_argument('--output', default='data/skill_contracts.json')
    args = ap.parse_args()
    if args.approach:
        if args.seeds < 1 or args.approach_case_limit < 0 or args.approach_seed_start < 0:
            ap.error('approach sweep requires positive seeds and nonnegative case limit')
        return approach_sweep(args)
    if args.diag:
        return diag(args)
    if args.seeds < 1:
        ap.error('--seeds must be positive')
    offsets = [(x, y) for x in ([-.03, 0., .03] if args.grid == 'coarse' else [-.03, -.02, -.01, 0., .01, .02, .03])
               for y in ([-.03, 0., .03] if args.grid == 'coarse' else [-.03, -.02, -.01, 0., .01, .02, .03])]
    from nav.skills import benchmark_ball_sim
    sim = benchmark_ball_sim(args.stand_policy, dynamics=args.dynamics)
    out = default_contracts()
    out.update(version=1, fallback=False, protocol={'scene':'flat', 'dynamics':args.dynamics,
        'dynamics_version': sim.provenance()['dynamics_version'],
        'robot_morphology': 'robot_walk_backlash.xml' if args.dynamics == 'training' else 'infer_policy',
        'stand_policy': args.stand_policy, 'stand_policy_md5': sim._lock[args.stand_policy]['md5'],
        'ball_model': 'apartment_sphere_40mm_25g_origin_floor_v1',
        'seeds':list(range(args.seeds)),
        'offset_grid_m': [float(x) for x in sorted(set(x for x, _ in offsets))],
        'kick_success':'no fall; displacement >=0.30m, peak speed >=0.30m/s, direction within 30deg over 2s',
        'scope':'sampled offsets and seeds only; tolerance is a heuristic between sampled cells, not a guarantee'})
    for side, name in [('right', 'right_pollen'), ('left', 'left_mirror_pollen')]:
        out['kick'][side]['policy_md5'] = sim._lock[out['kick'][side]['policy_key']]['md5']
        result = kick_sweep(sim, KICK_SPECS[name], offsets=offsets, seeds=range(args.seeds),
                            stand_policy=args.stand_policy)
        region = tolerance_region(result['offsets'], result['success'], 1.0)
        rows = [t for row in result['trials'] for t in row]
        nominal = result['offsets'].index([0., 0.])
        out['kick'][side].update(measured={'success':result['overall_success'], 'fall_rate':float(np.mean(result['fall_rate'])),
            'nominal_success':float(result['success'][nominal]), 'fall_count':sum(t['fell'] for t in rows),
            'fall_wilson_95':wilson(sum(t['fell'] for t in rows), len(rows)), 'region':region, 'trials':rows})
        # The box is a controller heuristic fitted to sampled successes, not a guarantee
        # between cells or beyond these seeds. Preserve all individual trials above.
        box = region['box']
        out['kick'][side]['supported'] = bool(box is not None and args.grid == 'full')
        if box is not None and args.grid == 'full':
            out['kick'][side]['target_trunk'] = [a + b for a, b in zip(out['kick'][side]['target_trunk'], box['center'])]
            out['kick'][side]['tol_half'] = box['half']
        else:
            out['kick'][side]['tol_half'] = [0., 0.]
        out['kick'][side]['direction_bias_rad'] = 0.0
    picks = [run_pick_trial(sim, 'pick', seed=s, stand_policy=args.stand_policy,
                            back_to=args.stand_policy) for s in range(args.seeds)]
    out['pick'].update(supported=False, policy_md5=sim._lock[out['pick']['policy_key']]['md5'])
    out['pick']['mouth_trunk_xy'] = list(np.mean([p['mouth_trunk_xy'] for p in picks], axis=0))
    out['pick']['measured'] = {'robot_post_ok_rate': float(np.mean([p['post_ok'] for p in picks])),
        'trials': picks, 'object_lift_tested':False, 'note':'blind policy; no grasp mechanism established; reach radius is unvalidated'}
    rec = [run_recover_trial(sim, 'atlantis_standup_flat', o, s, stand_policy=args.stand_policy)
           for o in FALL_ORIENTS for s in range(args.seeds)]
    out['recover'].update(supported=False, policy_md5=sim._lock[out['recover']['policy_key']]['md5'])
    out['recover']['measured'] = {'success_rate':float(np.mean([r['success'] for r in rec])), 'trials':rec}
    p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print('wrote', p)

if __name__ == '__main__':
    main()
