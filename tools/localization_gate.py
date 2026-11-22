"""Measure 50 Hz odometry and image fiducial localization on identical trajectories.

Requires offscreen MuJoCo rendering. GT is used only to initialize and evaluate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from nav.sim import Sim
from nav.localization import Localizer, FiducialCamera, wrap

ROOT = Path(__file__).resolve().parents[1]
MODES = ('odom', 'odom+fiducial')
ROOM_ORDER = ('living', 'kitchen', 'office', 'bedroom', 'bathroom', 'corridor')


def room_waypoints(gridmap, primitives):
    """Choose deterministic, well-cleared room interiors on the planner lattice."""
    from nav.planner import get_planner
    planner = get_planner(gridmap, primitives)
    points = []
    rooms = list(ROOM_ORDER) + sorted(set(gridmap.rooms) - set(ROOM_ORDER))
    for room in rooms:
        if room not in gridmap.rooms:
            continue
        candidates = gridmap.sample_free(room, n=30)
        for x, y in candidates:
            state = planner.snap((x, y, 0.))
            if state < 0:
                continue
            pose = planner.state_pose(state)
            if (not gridmap.is_lethal(*pose[:2])
                    and gridmap.room_of(*pose[:2]) == room):
                points.append({'room': room, 'pose': list(pose)})
                break
        else:
            raise RuntimeError(f'no safe planner-lattice waypoint in {room}')
    if not points:
        raise RuntimeError('apartment map contains no rooms')
    # Face the next waypoint; the follower still uses its ordinary goal tolerances.
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]['pose']
        point['pose'][2] = float(np.arctan2(nxt[1] - point['pose'][1],
                                          nxt[0] - point['pose'][0]))
    return points


def pose_error(estimate, truth):
    return [float(np.linalg.norm(estimate[:2] - truth[:2])),
            abs(float(wrap(estimate[2] - truth[2])))]


def summarize_errors(errors):
    e = np.asarray(errors)
    return {'final_xy_m': float(e[-1, 0]),
            'rms_xy_m': float(np.sqrt(np.mean(e[:, 0] ** 2))),
            'max_xy_m': float(e[:, 0].max()), 'final_yaw_rad': float(e[-1, 1])}


def fix_acceptance(stats, camera_ticks, accepted_ticks):
    """Measurement acceptance and camera frames with at least one accepted fix."""
    detected, used = int(stats['det']), int(stats['used'])
    return dict(stats, camera_ticks=int(camera_ticks), accepted_fix_ticks=int(accepted_ticks),
                acceptance_fraction=used / detected if detected else None,
                accepted_tick_fraction=accepted_ticks / camera_ticks if camera_ticks else None)


def run_long_route(sim, camera, gridmap, primitives, envelope, waypoints, seed,
                   min_seconds=60., room_timeout=180.):
    """One physical trajectory; only the GT controller may choose commands.

    Odom and fiducial estimates are passive observers, updated on every same
    50 Hz state. Coverage means reaching each room waypoint, rather than merely
    issuing a plan toward it. Repeat the room circuit if needed to reach 60 s.
    """
    from nav.follower import RPPFollower
    from nav.planner import plan
    if min_seconds < 60. or room_timeout <= 0:
        raise ValueError('long routes require >=60 s and a positive room timeout')
    gt = sim.reset(seed, start_pose=waypoints[0]['pose'])
    controller = Localizer(sim, 'gt', seed=seed)
    observers = {'odom': Localizer(sim, 'odom', seed=seed),
                 'odom+fiducial': Localizer(sim, 'odom+fiducial', seed=seed, camera=camera)}
    controller.reset(gt.pose2)
    for loc in observers.values():
        loc.reset(gt.pose2)
    follower = RPPFollower(envelope, gridmap=gridmap)
    errors = {name: [] for name in MODES}
    series, fixes, terminals, plans = [], [], [], []
    trajectory = hashlib.sha256()
    renders0 = camera.n_renders
    start_t = float(gt.t)
    step_count = accepted_ticks = camera_ticks = 0
    fell = False
    failure = None
    # Starting room also gets a terminal measurement after a closed-loop return.
    destinations = waypoints[1:] + waypoints[:1]
    leg = 0
    while leg < len(destinations) or float(gt.t) - start_t < min_seconds:
        target = destinations[leg % len(destinations)]
        result = plan(gridmap, primitives, tuple(gt.pose2), target['pose'],
                      budget_s=2., max_expansions=30000)
        plans.append({'room': target['room'], 't_s': float(gt.t) - start_t,
                      'complete': bool(result.complete), 'reason': result.reason,
                      'expansions': result.expansions})
        if not result.complete or not len(result.poses):
            failure = f"plan_incomplete:{target['room']}:{result.reason}"
            break
        path = list(result.poses)
        path[-1] = tuple(target['pose'])
        follower.set_path(path)
        leg_start = float(gt.t)
        command, status = np.zeros(3), 'tracking'
        leg_steps = 0
        while float(gt.t) - leg_start < room_timeout:
            if leg_steps % 5 == 0:
                pose = controller.update(sim, gt_for_eval_only=gt)
                command, status = follower.compute(pose, None)
                if status in ('done', 'stuck'):
                    break
            gt, _ = sim.step(command)
            step_count += 1
            leg_steps += 1
            fell = fell or bool(gt.fallen)
            elapsed = float(gt.t) - start_t
            trajectory.update(np.asarray([elapsed, *command, *gt.pose2], dtype='<f8').tobytes())
            current = {}
            fid = observers['odom+fiducial']
            before = dict(fid.stats)
            for name, loc in observers.items():
                estimate = loc.update(sim)  # no evaluation truth enters either estimator
                current[name] = pose_error(estimate, gt.pose2)
                errors[name].append(current[name])
            if fid._ticks % fid._cam_period == 0:
                camera_ticks += 1
                delta = {k: int(fid.stats[k] - before[k]) for k in before}
                accepted_ticks += int(delta['used'] > 0)
                fixes.append({'t_s': elapsed, **delta})
            if step_count == 1 or step_count % 50 == 0:
                series.append({'t_s': elapsed, 'room': gridmap.room_of(*gt.pose2[:2]),
                               'gt_pose2': list(map(float, gt.pose2)),
                               'errors': current})
            if fell:
                failure = 'fall'
                break
        terminal = {'room': target['room'], 't_s': float(gt.t) - start_t,
                    'status': status, 'reached': status == 'done',
                    'gt_goal_error': pose_error(gt.pose2, np.asarray(target['pose'])),
                    'errors': {name: pose_error(loc.pose, gt.pose2)
                               for name, loc in observers.items()}}
        # A follower status alone is insufficient evidence of room coverage.
        terminal['reached'] = bool(terminal['reached']
                                   and terminal['gt_goal_error'][0] <= .05
                                   and terminal['gt_goal_error'][1] <= np.deg2rad(10)
                                   and gridmap.room_of(*gt.pose2[:2]) == target['room'])
        terminals.append(terminal)
        if failure or not terminal['reached']:
            failure = failure or f"route_{status if status == 'stuck' else 'timeout_or_goal'}:{target['room']}"
            break
        leg += 1
    elapsed = float(gt.t) - start_t
    if errors['odom']:
        endpoint = {'t_s': elapsed, 'room': gridmap.room_of(*gt.pose2[:2]),
                    'gt_pose2': list(map(float, gt.pose2)),
                    'errors': {name: errors[name][-1] for name in MODES}}
        if not series or series[-1]['t_s'] != elapsed:
            series.append(endpoint)
    required = {p['room'] for p in waypoints}
    reached = {p['room'] for p in terminals if p['reached']}
    coverage = required <= reached
    fid = observers['odom+fiducial']
    localization_ok = bool(coverage and all(
        t['errors']['odom+fiducial'][0] <= .05
        and t['errors']['odom+fiducial'][1] <= np.deg2rad(10)
        for t in terminals))
    return {'seed': seed, 'duration_s': elapsed, 'steps': step_count, 'fell': fell,
            'route_complete': bool(coverage and elapsed >= min_seconds and not failure),
            'failure': failure, 'required_rooms': sorted(required), 'reached_rooms': sorted(reached),
            'passed': bool(coverage and elapsed >= min_seconds and not failure and localization_ok),
            'trajectory_sha256': trajectory.hexdigest(), 'renders': camera.n_renders - renders0,
            'per_room_terminal': terminals, 'error_vs_time': series, 'fix_acceptance_vs_time': fixes,
            'fiducial_stats': fix_acceptance(fid.stats, camera_ticks, accepted_ticks),
            'plans': plans, **{name: summarize_errors(e) for name, e in errors.items() if e}}


def apartment_long(args):
    from nav.episode import read_artifacts
    from nav.mapping import build_map
    from nav.mapsigma import planning_lateral_sigma
    envelope, primitives = read_artifacts(args.policy)
    if any(a.get('dynamics') != args.dynamics for a in (envelope, primitives)):
        raise RuntimeError('measurement/gate dynamics mismatch')
    sigma, sigma_provenance = planning_lateral_sigma(args.policy, envelope)
    if not sigma_provenance['follower_gate_passed']:
        raise RuntimeError('long route requires a passing follower gate')
    gridmap = build_map(lateral_sigma=sigma, step_over_height=envelope['step_over_height'])
    waypoints = room_waypoints(gridmap, primitives)
    sim = Sim('apartment', args.policy, record=False, dynamics=args.dynamics)
    camera = FiducialCamera(sim.scene_xml)
    try:
        runs = []
        for seed in range(args.seeds):
            runs.append(run_long_route(sim, camera, gridmap, primitives, envelope, waypoints, seed,
                                       args.min_seconds, args.room_timeout))
            print(f"seed={seed} duration={runs[-1]['duration_s']:.2f}s "
                  f"rooms={len(runs[-1]['reached_rooms'])}/{len(waypoints)} "
                  f"passed={runs[-1]['passed']}", flush=True)
        hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                  ('tools/localization_gate.py', 'nav/localization.py', 'nav/planner.py',
                   'nav/follower.py', 'nav/mapping.py', 'nav/mapsigma.py',
                   f'data/envelope_{args.policy}.json', f'data/primitives_{args.policy}.json')}
        return {'route': 'apartment_long', 'dev': True, 'scene': 'apartment', 'policy': args.policy,
                'dynamics': args.dynamics, 'dynamics_version': sim.provenance()['dynamics_version'],
                'provenance': sim.provenance(), 'source_sha256': hashes,
                'sigma_provenance': sigma_provenance, 'waypoints': waypoints,
                'min_seconds': args.min_seconds, 'room_timeout_s': args.room_timeout,
                'terminal_limits': {'xy_m': .05, 'yaw_rad': float(np.deg2rad(10))},
                'note': 'One GT planner+follower trajectory per seed; both estimators observe identical '
                        '50 Hz states. GT controls commands and evaluates errors only. '
                        'Error series sampled at 1 Hz plus endpoints; aggregate errors use every tick. '
                        'DEV infrastructure evidence, not a frozen benchmark or a new F threshold.',
                'passed': bool(runs and all(r['passed'] for r in runs)), 'runs': runs}
    finally:
        camera.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', choices=['short', 'apartment_long'], default='short')
    ap.add_argument('--policy', default='atlantis_walk', help='walker for apartment_long')
    ap.add_argument('--min-seconds', type=float, default=60.)
    ap.add_argument('--room-timeout', type=float, default=180.)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--dynamics', default='training', choices=['training', 'infer'])
    ap.add_argument('--output')
    args = ap.parse_args()
    if args.route == 'apartment_long':
        if args.seeds < 1 or args.min_seconds < 60. or args.room_timeout <= 0:
            ap.error('apartment_long requires seeds >=1, min-seconds >=60, room-timeout >0')
        output = Path(args.output or 'reports/dev/housekeeping/localization/apartment_long.json')
        if output.exists():
            ap.error(f'refusing to overwrite existing long-route evidence: {output}')
        out = apartment_long(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as f:
            json.dump(out, f, indent=2)
        print(f'DEV apartment_long passed={out["passed"]}: {output}')
        return
    args.output = args.output or 'reports/localization_gate.json'
    sim = Sim('apartment', dynamics=args.dynamics)
    camera = FiducialCamera(sim.scene_xml)
    runs = []
    try:
        for seed in range(args.seeds):
            gt = sim.reset(seed, start_pose=(2., 0., (-.8 + .4*seed)))
            od = Localizer(sim, 'odom', seed=seed)
            fid = Localizer(sim, 'odom+fiducial', seed=seed, camera=camera)
            od.reset(gt.pose2); fid.reset(gt.pose2)
            errs = {'odom': [], 'odom+fiducial': []}
            renders0 = camera.n_renders
            measurements = []
            for i in range(args.steps):
                command = np.array([.3, 0., 0.]) if i < args.steps//2 else np.array([.25, 0., .25])
                gt, _ = sim.step(command)
                for name, loc in [('odom', od), ('odom+fiducial', fid)]:
                    est = loc.update(sim, gt_for_eval_only=gt)
                    errs[name].append([float(np.linalg.norm(est[:2]-gt.pose2[:2])), abs(wrap(est[2]-gt.pose2[2]))])
                if fid._ticks % fid._cam_period == 0:
                    import mujoco
                    cam_id = mujoco.mj_name2id(camera.model, mujoco.mjtObj.mjOBJ_CAMERA, 'head_camera')
                    cp = camera.data.cam_xpos[cam_id]
                    for det in fid.last_dets:
                        delta = camera.markers[det.id].center[:2] - cp[:2]
                        measurements.append({'id':det.id, 'range_est':det.range_h,
                            'range_gt':float(np.linalg.norm(delta)), 'bearing_est':det.bearing,
                            'bearing_gt':wrap(float(np.arctan2(delta[1], delta[0])) - gt.pose2[2]), 'h_px':det.h_px})
            result = {'seed':seed, 'fell':bool(gt.fallen), 'renders':camera.n_renders-renders0,
                      'fiducial_stats':fid.stats, 'measurements_eval_only':measurements}
            for name in errs:
                e = np.asarray(errs[name]); result[name] = {'final_xy_m':float(e[-1,0]),
                    'rms_xy_m':float(np.sqrt(np.mean(e[:,0]**2))), 'max_xy_m':float(e[:,0].max()),
                    'final_yaw_rad':float(e[-1,1])}
            runs.append(result)
        out = {'dynamics':args.dynamics, 'scene':'apartment', 'steps':args.steps,
               'dynamics_version':sim.provenance()['dynamics_version'], 'provenance':sim.provenance(),
               'note':'GT used for initialization and error evaluation only; orientation sensor is simulated fused orientation with declared sensor noise',
               'runs':runs}
        with open(args.output, 'w') as f: json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
    finally:
        camera.close()

if __name__ == '__main__': main()
