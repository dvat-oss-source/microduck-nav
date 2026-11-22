"""Navigation and skill orchestration; objective success is checked independently.

Each tick is one 50 Hz policy step; the caller executes the returned command.
The follower runs at 10 Hz. Sim.switch_policy can itself advance the simulator
while settling: callers must retain the entire Sim trace, including handoff steps.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from .planner import plan, get_planner

# Deterministic planner budget (reports/planner_gate_atlantis_walk.json, section e): 30000
# expansions, idle wall p95 0.12-0.15 s. The wall cap is a safety net only.
PLAN_MAX_EXPANSIONS = 30000
PLAN_WALL_CAP_S = 2.0
MAX_PARTIAL_SEGMENTS = 20      # Baritone-style best-partial replans allowed per goal
MIN_PROGRESS_S = 0.05          # required drop of the admissible heuristic per partial plan
from .localization import quat_to_mat
from .skills import (load_contracts, choose_kick_side, kick_pre_pose, pick_pre_pose,
                     kick_precondition, pick_precondition, KickSpec, activate_kick_policy)
from . import policies as _pol

# ---------------------------------------------------------------- active look (Stage 1c)
# Enabled ONLY when the localizer advertises `active_look = True` and its mode is not one of
# the existing modes. For every other localizer the code paths below are never entered.
# All numbers live in convergence_threshold.json (PLACEHOLDER until derived from apartment_long).
ROOT = Path(__file__).resolve().parents[1]
ACTIVE_LOOK_JSON = ROOT / 'data' / 'convergence_threshold.json'
ACTIVE_LOOK_EXCLUDED_MODES = ('gt', 'odom', 'odom+fiducial')
_ACTIVE_LOOK_KEYS = {
    'convergence': ('xy_trace_m2', 'yaw_var_rad2', 'max_fix_age_s'),
    'look': ('sweep_increment_rad', 'sweep_max_rad', 'tol_rad', 'burst_s', 'settle_s',
             'max_bursts_per_look', 'max_looks', 'max_reapproaches', 'budget_s'),
}
_ACTIVE_LOOK_CFG = {}


def active_look_config(path=None):
    """Load and validate the active-look constants. Fails closed (raises) on any problem."""
    path = Path(path) if path is not None else ACTIVE_LOOK_JSON
    key = str(path)
    if key not in _ACTIVE_LOOK_CFG:
        cfg = json.loads(path.read_text())
        for section, names in _ACTIVE_LOOK_KEYS.items():
            for name in names:
                value = cfg[section][name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                    raise ValueError(f'active look constant {section}.{name} must be a positive finite number')
        _ACTIVE_LOOK_CFG[key] = cfg
    return _ACTIVE_LOOK_CFG[key]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _sweep_radius(primitives):
    """Largest trunk sweep of any measured primitive (same quantity the existing localization probe used)."""
    items = primitives.get('primitives', []) if isinstance(primitives, dict) else []
    radii = [float(np.linalg.norm(np.asarray(p['swept'], float)[:, :2], axis=1).max())
             for p in items if len(p.get('swept', []))]
    return max(radii) if radii else float('inf')


def sensor_safety(frame, kinematics):
    """Tilt and approximate trunk height from IMU, FK and virtual foot contacts.

    Height is available only while at least one foot contacts the floor; it is
    an estimate from the lowest contacting foot site, not simulator truth.
    """
    R = quat_to_mat(frame.imu_quat) @ kinematics.R_trunk_imu.T
    tilt = float(np.arccos(np.clip(R[2, 2], -1., 1.)))
    feet = kinematics.feet_in_trunk(frame.q)
    heights = [-(R @ feet[i][0])[2] for i, contact in enumerate(frame.foot_contacts or ()) if contact]
    return tilt, (float(max(heights)) if heights else None)


class NavFSM:
    def __init__(self, sim, gridmap, primitives, follower, localizer, contracts=None,
                 target_observer=None, free_objects=(), target_body=None,
                 target_source='frozen surveyed object placement', look_heading=None, anchors=(),
                 active_look=None, anchor_fn=None):
        self.target_observer, self.free_objects = target_observer, list(free_objects)
        # Active look only (ignored unless the localizer enables it): look_heading(pose) -> yaw or None
        # (None = default sweep); anchors = annotated waypoint list from the fiducial graph (see README).
        self.look_heading, self.anchors = look_heading, [dict(a) for a in anchors]
        # active_look: None = decided by the localizer (candidate default); True/False = explicit ablation switch.
        # anchor_fn(poses) -> anchor dicts for the accepted plan (nav only, active look only); None = no anchors.
        self.look_switch, self.anchor_fn = active_look, anchor_fn
        self.localization_events = []
        self.target_body, self.target_source = target_body, target_source
        self.sim, self.gridmap, self.primitives = sim, gridmap, primitives
        self.follower, self.localizer = follower, localizer
        self._follower_default_tolerances = {
            name: getattr(follower, name) for name in ('tol_pos', 'tol_yaw') if hasattr(follower, name)
        }
        self.contracts = contracts or load_contracts()
        if gridmap is not None:
            get_planner(gridmap, primitives)   # one-time per-map lattice precompute (D1)
        self.state = 'idle'
        self.reason = None
        self.planner_times = []
        self.plan_log = []
        self._partial_segments, self._segment_complete, self._goal_fixed = 0, True, False
        self.skill_start_index = None
        self._cmd = np.zeros(3)
        self._tick = 0
        self._follower_default_cruise = getattr(follower, 'cruise', None)
        self._reset_look_state()

    def start(self, goal, task='nav', target_xy=None, direction=0):
        self._restore_follower_tolerances()
        if task not in ('nav', 'kick', 'pick'):
            raise ValueError('task must be nav, kick or pick')
        if task != 'nav' and target_xy is None:
            raise ValueError('skill task requires target_xy')
        self.task, self.target_xy, self.direction = task, target_xy, float(direction)
        self.goal = tuple(goal) if goal is not None else None
        self.reason, self.skill_start_index = None, None
        self.planner_times = []
        self.plan_log = []
        self._partial_segments, self._segment_complete, self._goal_fixed = 0, True, False
        self._tick, self._cmd = 0, np.zeros(3)
        self.side = None
        self._planning_map = self.gridmap
        self.follower.gridmap = self.gridmap
        self._kick_plan = self._kick_approach = None
        self._kick_attempts, self._kick_retarget_retreat = 0,False
        self.kick_events = []
        self.localization_events = []
        self._reset_look_state()
        if self._follower_default_cruise is not None:
            self.follower.cruise = self._follower_default_cruise
        self.state = 'plan'
        if task != 'nav':
            error = self._skill_contract_error(task)
            if error:
                self._fail(error)

    def _skill_contract_error(self, task, side=None):
        """Refuse unsupported or stale contracts before any policy activation."""
        proto = self.contracts.get('protocol', {})
        dynamics = getattr(self.sim, 'dynamics', 'infer')
        if proto.get('dynamics') != dynamics:
            return 'skill_dynamics_mismatch'
        provenance = self.sim.provenance()
        if not proto.get('dynamics_version') or proto['dynamics_version'] != provenance.get('dynamics_version'):
            return 'skill_dynamics_version_mismatch'
        morphology = getattr(self.sim, 'robot_model',
                             'robot_walk_backlash.xml' if dynamics == 'training' else 'infer_policy')
        if proto.get('robot_morphology') != morphology:
            return 'skill_morphology_mismatch'
        if task == 'kick':
            if proto.get('ball_model') != 'apartment_sphere_40mm_25g_origin_floor_v1':
                return 'skill_ball_model_mismatch'
            try:
                model = self.sim.model
                for name in ((self.target_body,) if self.target_body else ('ball_0', 'ball_1', 'ball_2')):
                    body, geom = model.body(name).id, model.geom(name + '_geom').id
                    if not (np.isclose(model.body_mass[body], .025) and
                            np.allclose(model.body_inertia[body], [1.6e-5] * 3) and
                            np.allclose(model.body_ipos[body], [0., 0., .04]) and
                            np.allclose(model.body_iquat[body], [1., 0., 0., 0.]) and
                            model.geom_type[geom] == 2 and model.geom_contype[geom] == 1 and
                            model.geom_conaffinity[geom] == 1 and model.geom_priority[geom] == 0 and
                            np.allclose(model.geom_size[geom], [.04, 0., 0.]) and
                            np.allclose(model.geom_pos[geom], [0., 0., .04]) and
                            np.allclose(model.geom_quat[geom], [1., 0., 0., 0.]) and
                            np.isclose(model.geom_margin[geom], 0.) and np.isclose(model.geom_gap[geom], 0.) and
                            np.allclose(model.geom_friction[geom], [1., .005, .0001]) and
                            np.allclose(model.geom_solref[geom], [.02, 1.]) and
                            np.allclose(model.geom_solimp[geom], [.9, .95, .001, .5, 2.])):
                        return 'skill_ball_model_mismatch'
            except (AttributeError, KeyError, ValueError, IndexError):
                return 'skill_ball_model_mismatch'
            stand_key = getattr(self.sim, 'initial_policy_key', None)
            if proto.get('stand_policy') != stand_key:
                return 'skill_stand_policy_mismatch'
            try:
                stand = _pol.verified_entry(stand_key, lock=self.sim._lock, allow_rejected=False)
            except (KeyError, OSError, TypeError, _pol.PolicyLockError):
                return 'skill_policy_invalid'
            if proto.get('stand_policy_md5') != stand.get('md5'):
                return 'skill_stand_policy_mismatch'
            entries = [self.contracts.get('kick', {}).get(s, {}) for s in ([side] if side else ('left', 'right'))]
        else:
            entries = [self.contracts.get(task, {})]
        for entry in entries:
            if entry.get('supported') is not True:
                return 'skill_unsupported_' + task
            key = entry.get('policy_key')
            try:
                verified = _pol.verified_entry(key, lock=self.sim._lock, allow_rejected=False)
            except (KeyError, OSError, TypeError, _pol.PolicyLockError):
                return 'skill_policy_invalid'
            if entry.get('policy_md5') != verified.get('md5'):
                return 'skill_policy_mismatch'
        return None

    def _fail(self, reason):
        self.reason, self.state = reason, 'fail'
        self._cmd = np.zeros(3)

    def _restore_follower_tolerances(self):
        for name, value in self._follower_default_tolerances.items():
            setattr(self.follower, name, value)

    def _configure_kick_precision(self, side):
        """Split the sampled position box between translation and yaw error.

        For target radius r, a yaw error theta moves its trunk-frame position
        by at most r*abs(theta). The follower subtracts its own stop margins
        from these limits; neither those margins nor its stop spread are waived.
        """
        try:
            contract = self.contracts['kick'][side]
            half = np.asarray(contract['tol_half'], dtype=float)
            target = np.asarray(contract['target_trunk'], dtype=float)
            if (contract.get('supported') is not True or half.shape != (2,) or target.shape != (2,)
                    or not np.isfinite(half).all() or not np.isfinite(target).all()
                    or np.any(half <= 0)):
                return False
            budget = float(np.min(half))
            radius = float(np.linalg.norm(target))
            default_pos = float(self._follower_default_tolerances['tol_pos'])
            default_yaw = float(self._follower_default_tolerances['tol_yaw'])
            pos_margin = float(self.follower.pos_margin)
            yaw_margin = float(self.follower.yaw_margin)
            spread = float(self.follower.model.stop_spread)
            if not np.isfinite([default_pos, default_yaw, pos_margin, yaw_margin, spread]).all():
                return False
            if min(default_pos, default_yaw) <= 0 or min(pos_margin, yaw_margin, spread) < 0:
                return False
            pos = min(default_pos, budget / 2.0)
            yaw = min(default_yaw, budget / (2.0 * radius)) if radius > 0 else default_yaw
            if pos <= pos_margin or yaw <= yaw_margin or (pos - pos_margin) / np.sqrt(2.0) <= spread:
                return False
            self.follower.tol_pos, self.follower.tol_yaw = pos, yaw
            return True
        except (KeyError, AttributeError, TypeError, ValueError):
            return False

    def _to_exact_goal(self, poses):
        """The lattice path ends within the planner's goal tolerance of the goal; the follower's own
        tolerance would stack on top of it. End the path at the exact goal pose (the planner
        guarantees the last lattice pose is within its goal tolerance, so this is a short,
        collision-checked-by-construction step only if the goal itself is free)."""
        P = [tuple(map(float, p)) for p in poses]
        g = self.goal
        if g is None or len(g) < 3 or g[2] is None:
            return P
        free = self.gridmap is None or not self.gridmap.is_lethal(float(g[0]), float(g[1]))
        if not free:
            return P
        P[-1] = (float(g[0]), float(g[1]), float(g[2]))
        return P

    def _partial_progress(self, pose, result):
        """Drop of the admissible time-to-goal heuristic from pose to the partial path end."""
        pl = get_planner(self._planning_map, self.primitives)
        h = pl.goal_spec(self.goal)[2]
        s0, se = pl.snap(pose), pl.snap(result.end_state_pose)
        if s0 < 0 or se < 0:
            return 0.0
        return float(h[s0 // pl.nh] - h[se // pl.nh])


    def _kick_target(self):
        if self.target_observer is not None:
            value = np.asarray(self.target_observer(),float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError('invalid kick target observation')
            self.target_xy = value.copy()
        return np.asarray(self.target_xy,float)[:2]

    def _prepare_kick(self, pose):
        from .mapping import overlay_discs
        from .skills import kick_stage_plan
        for contract in self.contracts['kick'].values():
            half = np.asarray(contract.get('tol_half',[]),float)
            if half.shape != (2,) or not np.isfinite(half).all() or np.any(half <= 0):
                self._fail('skill_precision_unachievable')
                return False
        ball = self._kick_target()
        self._kick_base_map = overlay_discs(self.gridmap,self.free_objects) if self.gridmap is not None else None
        info = kick_stage_plan(self._kick_base_map,pose,ball,self.direction,self.contracts)
        self.kick_events.append(dict(t=float(self.sim.t),event='stage_plan',
            ok=info['ok'],ball=ball.tolist(),target_source=self.target_source,
            side=info.get('side'),stage_pose=info.get('stage_pose'),reason=info.get('reason')))
        if not info['ok']:
            self._fail(info['reason'])
            return False
        self._kick_plan = info
        self.side, self.goal = info['side'], tuple(info['stage_pose'])
        self._planning_map = info['map']
        self.follower.gridmap = self._planning_map
        return True

    def _begin_kick_runin(self, pose):
        from .skills import KickApproach
        self.follower.gridmap = self._kick_base_map
        self._kick_approach = KickApproach(self.contracts,self.side,self.direction,
                                         self.follower,self._kick_base_map)
        self._kick_approach.begin(pose,self._kick_target(),self.sim.t)
        self.state = 'kick_runin'



    def _kick_control(self, pose):
        """Return a command only when the staged skill controller owns this tick."""
        from .skills import KICK_RETARGET_M
        if self.task != 'kick' or self.state not in ('plan','follow','kick_runin','kick_wait_ball'):
            return None
        ball = self._kick_target()
        if (self.state in ('follow','kick_runin') and self._kick_plan is not None
                and np.linalg.norm(ball-np.asarray(self._kick_plan['ball'])) > KICK_RETARGET_M
                and not self._kick_retarget_retreat):
            self.state = 'kick_wait_ball'
            self._ball_wait_start = self._ball_still_t = float(self.sim.t)
            self._ball_still_xy = ball.copy()
            self.kick_events.append(dict(t=float(self.sim.t),event='ball_moved',ball=ball.tolist()))
        if self.state == 'kick_wait_ball':
            if self.sim.t-self._ball_wait_start > 4.:
                self._fail('kick_ball_unstable')
            elif np.linalg.norm(ball-self._ball_still_xy) > .002:
                self._ball_still_t, self._ball_still_xy = float(self.sim.t),ball.copy()
            elif self.sim.t-self._ball_still_t >= .5:
                self._kick_attempts += 1
                if self._kick_attempts > 3:
                    self._fail('kick_retarget_limit')
                elif self._kick_approach is not None:
                    self._kick_approach.ref_ball = ball.copy()
                    self._kick_approach.state = 'retreat'
                    self._kick_approach.started = float(self.sim.t)
                    self._kick_retarget_retreat = True
                    self.state = 'kick_runin'
                else:
                    self._goal_fixed, self.state = False,'plan'
            return np.zeros(3)
        if self.state != 'kick_runin':
            return None
        cmd, status = self._kick_approach.step(pose,ball,self.sim.t)
        if status == 'ready':
            self.state = 'approach'
            return None
        if status == 'restage':
            self._kick_attempts += int(not self._kick_retarget_retreat)
            if self._kick_attempts > 3:
                self._fail('kick_approach_exhausted')
            else:
                self._goal_fixed, self.state = False,'plan'
                self._kick_retarget_retreat = False
            return np.zeros(3)
        if status == 'fail':
            self._fail(self._kick_approach.reason)
        elif status == 'retarget':
            self.state = 'kick_wait_ball'
            self._ball_wait_start = self._ball_still_t = float(self.sim.t)
            self._ball_still_xy = ball.copy()
        return np.zeros(3) if self.state == 'fail' else cmd




    # ------------------------------------------------------------ active look (Stage 1c)
    def _reset_look_state(self):
        self._localize_deadline = None
        self._localize_spent = 0.0          # seconds spent in the 'localize' state (looks and dwell), not travel
        self._last_tick_t = None
        self._sweep_origin = 0.0
        self._last_fix_t, self._fix_count = None, 0
        self._looks, self._reapproaches, self._bursts = 0, 0, 0
        self._look_target, self._look_phase, self._look_dwelt = 0.0, None, False
        self._burst_end = self._settle_until = 0.0
        self._burst_sign = 0
        self._anchor_visited, self._cruise_overridden = set(), False

    def _active_look_enabled(self):
        if self.look_switch is not None:
            return bool(self.look_switch)
        loc = self.localizer
        return (getattr(loc, 'active_look', False) is True
                and getattr(loc, 'mode', None) not in ACTIVE_LOOK_EXCLUDED_MODES)

    def _note_fixes(self, now):
        """Time of the last accepted fiducial update (from the localizer's own counter)."""
        stats = getattr(self.localizer, 'stats', None)
        if not isinstance(stats, dict):
            return
        used = int(stats.get('used', 0))
        if used > self._fix_count:
            self._last_fix_t = now
        self._fix_count = used

    def _converged(self, now):
        """Converged = an accepted fix exists, is recent, and the fused covariance is below threshold.

        The covariance alone is not evidence: its initial value and odometry drift are small
        for a long time, so a fiducial fix must have been accepted within max_fix_age_s.
        """
        cfg = active_look_config()['convergence']
        P = np.asarray(self.localizer.P, float)
        xy, yaw = float(np.trace(P[:2, :2])), float(P[2, 2])
        age = None if self._last_fix_t is None else now - self._last_fix_t
        ok = bool(age is not None and age <= cfg['max_fix_age_s'] + 1e-9
                  and np.isfinite([xy, yaw]).all() and xy <= cfg['xy_trace_m2'] and yaw <= cfg['yaw_var_rad2'])
        return ok, dict(xy_trace_m2=xy, yaw_var_rad2=yaw, fix_age_s=age)

    def _nav_done(self, pose):
        """Follower reached the goal. Accept done only with a converged fix in active-look modes."""
        if not self._active_look_enabled():
            self.state = 'done'
            return
        now = float(self.sim.t)
        ok, quality = self._converged(now)
        self.localization_events.append(dict(t=now, event='completion_check', ok=ok, quality=quality))
        if ok:
            self.state = 'done'
        else:
            self._begin_localization(pose)

    def _begin_localization(self, pose, target=None):
        now = float(self.sim.t)
        if self._localize_deadline is None:          # one budget per episode, from the first unconverged check
            self._localize_deadline = now + active_look_config()['look']['budget_s']
            self._sweep_origin = float(pose[2])
            self.localization_events.append(dict(t=now, event='localize_begin', deadline=self._localize_deadline))
        self._start_look(pose, target)

    def _look_target_for(self, pose):
        if self.look_heading is not None:
            value = self.look_heading(np.array(pose[:3], float, copy=True))
            if value is not None:
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError('look_heading returned a non-finite yaw')
                return value
        # default sweep about the heading at localization start: +inc, -inc, +2inc, -2inc, ... (clipped)
        cfg = active_look_config()['look']
        k = self._looks
        mag = min((k // 2 + 1) * cfg['sweep_increment_rad'], cfg['sweep_max_rad'])
        return self._sweep_origin + (mag if k % 2 == 0 else -mag)

    def _start_look(self, pose, target=None):
        cfg = active_look_config()['look']
        now = float(self.sim.t)
        if self._looks >= cfg['max_looks']:
            self.localization_events.append(dict(t=now, event='looks_exhausted', looks=self._looks))
            self._fail('localization_looks_exhausted')
            return
        if target is None:
            target = self._look_target_for(pose)
        self._looks += 1
        self._look_target = float(_wrap(float(target)))
        self._look_phase, self._look_dwelt, self._bursts = 'turn', False, 0
        self.state = 'localize'
        self.localization_events.append(dict(t=now, event='look_start', index=self._looks,
                                             target_yaw=self._look_target, pose=[float(v) for v in pose[:3]]))

    def _reapproach(self, pose, event):
        cfg = active_look_config()['look']
        now = float(self.sim.t)
        self._reapproaches += 1
        if self._reapproaches > cfg['max_reapproaches']:
            self.localization_events.append(dict(t=now, event='reapproach_exhausted'))
            self._fail('localization_reapproach_exhausted')
            return
        self.localization_events.append(dict(t=now, event=event, reapproach=self._reapproaches))
        self._look_phase = None
        self._segment_complete = True
        self.state = 'plan'          # the existing planner + follower re-approach the unchanged goal

    def _turn_is_clear(self, pose):
        """In-place turns only where the footprint clears every inflated obstacle (fail closed)."""
        gm = self._planning_map if self._planning_map is not None else self.gridmap
        if gm is None or not hasattr(gm, 'clearance'):
            return False
        x, y = float(pose[0]), float(pose[1])
        if gm.is_lethal(x, y):
            return False
        required = float(gm.params['r_inflate']) + _sweep_radius(self.primitives)
        return float(gm._lookup(gm.clearance, x, y, np.float32(0.0))) > required

    def _rot_command(self, sign):
        """Measured in-place yaw command from the follower's safe-rotation model (None if unmeasured)."""
        rot = getattr(getattr(self.follower, 'model', None), 'rot', None)
        if not rot or sign not in rot:
            return None
        return np.array([0., 0., float(rot[sign][0])])

    def _localize_tick(self, pose):
        now = float(self.sim.t)
        cfg = active_look_config()['look']
        ok, quality = self._converged(now)
        if ok:
            self.localization_events.append(dict(t=now, event='fix_replan', quality=quality))
            self._reapproach(pose, 'reapproach_after_fix')
            return np.zeros(3)
        if self._look_phase == 'burst':
            if not self._turn_is_clear(pose):
                self.localization_events.append(dict(t=now, event='look_blocked', pose=[float(v) for v in pose[:3]]))
                self._reapproach(pose, 'reapproach_after_block')
                return np.zeros(3)
            if now < self._burst_end - 1e-9:
                cmd = self._rot_command(self._burst_sign)
                if cmd is None:
                    self.localization_events.append(dict(t=now, event='look_unavailable'))
                    self._reapproach(pose, 'reapproach_after_unavailable')
                    return np.zeros(3)
                return cmd
            # Burst finished: keep turning. A zero command between bursts stops the walker's in-place yaw
            # (measured: 0.25 s pulses with 0.4 s gaps turn ~0.014 rad per cycle), so the next decision
            # falls through to the turn logic below without a settle; settle happens only at look end.
            self._look_phase = 'turn'
        if self._look_phase == 'settle':
            if now < self._settle_until - 1e-9:
                return np.zeros(3)                   # dwell: let the camera observe
            self._look_phase = 'turn'
        err = float(_wrap(self._look_target - float(pose[2])))
        if abs(err) <= cfg['tol_rad'] or self._bursts >= cfg['max_bursts_per_look']:
            if not self._look_dwelt:
                self._look_dwelt = True
                self._look_phase, self._settle_until = 'settle', now + cfg['settle_s']
                return np.zeros(3)
            self.localization_events.append(dict(t=now, event='look_end', err=err, bursts=self._bursts))
            self._start_look(pose)
            return np.zeros(3)
        if not self._turn_is_clear(pose):
            self.localization_events.append(dict(t=now, event='look_blocked', pose=[float(v) for v in pose[:3]]))
            self._reapproach(pose, 'reapproach_after_block')
            return np.zeros(3)
        self._bursts += 1
        self._burst_sign = 1 if err > 0 else -1
        self._burst_end = now + cfg['burst_s']
        self._look_phase = 'burst'
        cmd = self._rot_command(self._burst_sign)
        if cmd is None:
            self.localization_events.append(dict(t=now, event='look_unavailable'))
            self._reapproach(pose, 'reapproach_after_unavailable')
            return np.zeros(3)
        return cmd

    def _anchor_hook(self, pose):
        """Annotated anchors: slow down inside an anchor, and request one look at it if unconverged.

        Returns a yaw to look at (or None). Only used when the active look is enabled.
        """
        now = float(self.sim.t)
        request, slow = None, None
        xy = np.asarray(pose[:2], float)
        for i, anchor in enumerate(self.anchors):
            inside = float(np.hypot(*(np.asarray(anchor['xy'], float) - xy))) <= float(anchor.get('radius_m', 0.3))
            if not inside:
                continue
            if anchor.get('slow_cruise_mps') is not None:
                v = float(anchor['slow_cruise_mps'])
                slow = v if slow is None else min(slow, v)
            if i not in self._anchor_visited:
                self._anchor_visited.add(i)
                self.localization_events.append(dict(t=now, event='anchor_pass', index=i))
                if (request is None and anchor.get('request_fix') and anchor.get('look_yaw') is not None
                        and not self._converged(now)[0]):
                    request = float(anchor['look_yaw'])
                    self.localization_events.append(dict(t=now, event='anchor_fix_request', index=i, target_yaw=request))
        default = self._follower_default_cruise
        if default is not None:
            if slow is not None:
                model = getattr(self.follower, 'model', None)
                if model is not None and hasattr(model, 'v_at'):   # never below the measured minimum speed
                    slow = max(slow, float(model.v_at(model.vx_min)))
                self.follower.cruise = min(default, slow)
                self._cruise_overridden = True
            elif self._cruise_overridden:
                self.follower.cruise = default
                self._cruise_overridden = False
        return request

    def _deadline_expired(self):
        """Budget = localization time only (looks and dwell). Re-approach travel is not charged to the budget."""
        return self._localize_spent >= active_look_config()['look']['budget_s'] - 1e-9

    def tick(self, gt, obs=None):
        if self.state in ('idle', 'done', 'fail'):
            return np.zeros(3), self.state
        pose = self.localizer.update(self.sim, gt_for_eval_only=gt)
        if self._active_look_enabled():
            self._note_fixes(float(self.sim.t))
        frame = self.localizer.last_sensor_frame
        tilt, height = sensor_safety(frame, self.localizer.kin)
        if tilt > np.deg2rad(60) and self.state not in ('recover',):
            error = self._skill_contract_error('recover')
            if error:
                self._fail('recovery_unsupported')
                return np.zeros(3), self.state
            self.state = 'recover'
            self._recover_started = False
        if self.state == 'recover':
            if not self._recover_started:
                self._recover_started = True
                self.sim.force_switch(self.contracts['recover']['policy_key'])
                self._recover_t = self.sim.t
                self._recover_hold = 0
            if tilt < np.deg2rad(15) and height is not None and height >= .08:
                self._recover_hold += 1
            else:
                self._recover_hold = 0
            if self._recover_hold >= 25:
                self._fail('fall_recovered')  # recovery is not navigation success
            elif self.sim.t - self._recover_t >= self.contracts['recover']['max_time_s']:
                self._fail('recovery_timeout')
            return np.zeros(3), self.state
        control = self._kick_control(pose)
        if control is not None:
            return control, self.state
        if self._active_look_enabled():
            now = float(self.sim.t)
            if self._last_tick_t is not None and self.state == 'localize':
                self._localize_spent += now - self._last_tick_t
            self._last_tick_t = now
            if self._deadline_expired():          # localization time is bounded by budget_s
                self.localization_events.append(dict(t=float(self.sim.t), event='localization_timeout'))
                self._fail('localization_timeout')
                return np.zeros(3), self.state
            if self.state == 'localize':
                return self._localize_tick(pose), self.state
        if self.state == 'plan':
            if self.task != 'nav'  and not self._goal_fixed:   # don't re-pick the kick side on replans
                self._goal_fixed = True
                if self.task == 'kick':
                    if not self._prepare_kick(pose):
                        return np.zeros(3), self.state
                else:
                    self.goal = pick_pre_pose(self.target_xy, self.direction, self.contracts)
            if self.goal is None:
                self._fail('missing_goal')
            else:
                result = plan(self._planning_map, self.primitives, pose, self.goal,
                              budget_s=PLAN_WALL_CAP_S, max_expansions=PLAN_MAX_EXPANSIONS)
                self.planner_times.append(result.plan_time_s)
                self.plan_log.append({'t': getattr(self.sim, 't', None), 'complete': result.complete,
                                      'plan_time_s': result.plan_time_s,
                                      'reason': getattr(result, 'reason', ''),
                                      'expansions': getattr(result, 'expansions', None),
                                      'cost': getattr(result, 'cost', None),
                                      'eps': getattr(result, 'eps_achieved', None),
                                      'start_snap': getattr(result, 'start_snap', None),
                                      'goal_snap': getattr(result, 'goal_snap', None)})
                if result.complete and len(result.poses) > 0:
                    if self.anchor_fn is not None and self.task == 'nav' and self._active_look_enabled():
                        self.anchors = [dict(a) for a in self.anchor_fn(result.poses)]   # annotate this path
                        self._anchor_visited = set()
                    self._segment_complete = True
                    if self.task == 'kick' and not self._configure_kick_precision(self.side):
                        self._fail('skill_precision_unachievable')
                    else:
                        if self.task == 'kick':
                            self._restore_follower_tolerances()  # ordinary stage; precise heading is aligned outside inflation
                            self.follower.tol_pos = min(self.follower.tol_pos,min(self.contracts['kick'][self.side]['tol_half']))
                        self.follower.set_path(self._to_exact_goal(result.poses))
                        self.state = 'follow'
                elif (getattr(result, 'reason', '') == 'timeout_partial' and len(result.poses) > 1
                      and self._partial_segments < MAX_PARTIAL_SEGMENTS
                      and self._partial_progress(pose, result) >= MIN_PROGRESS_S):
                    self._partial_segments += 1
                    self._segment_complete = False
                    self._restore_follower_tolerances()
                    self.follower.set_path(result.poses)
                    self.state = 'follow'
                else:
                    self._fail('plan_incomplete')   # cause kept in plan_log[-1]['reason']
            return np.zeros(3), self.state
        if self.state == 'follow':
            if self._active_look_enabled() and self.anchors and self.task == 'nav':
                request = self._anchor_hook(pose)
                if request is not None:
                    self._begin_localization(pose, target=request)
                    return np.zeros(3), self.state
            if self._tick % 5 == 0:
                self._cmd, status = self.follower.compute(pose, None)
                if (self.task == 'kick' and self._planning_map is not None
                        and self._planning_map.is_lethal(*pose[:2])
                        and np.linalg.norm(self._cmd[:2]) < 1e-9 and abs(self._cmd[2]) > 1e-9):
                    self._fail('kick_stage_clearance')
                    return np.zeros(3),self.state
                if status == 'stuck':
                    self._fail('follower_stuck')
                elif status == 'done':
                    self._cmd = np.zeros(3)
                    if not self._segment_complete:
                        self.state = 'plan'          # end of a partial segment: replan from here
                    else:
                        if self.task == 'nav':
                            self._nav_done(pose)
                        elif self.task == 'kick':
                            self._begin_kick_runin(pose)
                        else:
                            self.state = 'approach' 
            self._tick += 1
            return self._cmd.copy(), self.state
        if self.state == 'approach':
            if self.task == 'kick':
                self._kick_target()
            error = self._skill_contract_error(self.task, self.side)
            if error:
                self._fail(error)
                return np.zeros(3), self.state
            valid = (kick_precondition(pose, self.target_xy, self.side, self.contracts)[0]
                     if self.task == 'kick' else pick_precondition(pose, self.target_xy, self.contracts)[0])
            if not valid:
                self._fail('skill_precondition')
                return np.zeros(3), self.state
            c = self.contracts['kick'][self.side] if self.task == 'kick' else self.contracts['pick']
            if self.task == 'kick':
                ok = activate_kick_policy(self.sim, KickSpec(self.side, c['policy_key'], self.side, c.get('mirror', False)))
            else:
                ok = self.sim.switch_policy(c['policy_key'])
            if not ok:
                self._fail('handoff_timeout')
            else:
                if self.task == 'kick':
                    # Handoff runs the walker internally; recheck before its first
                    # kick action, on the latest observation and encoder estimate.
                    pose_after = self.localizer.update(self.sim)
                    self._kick_target()
                    from .skills import kick_direction_budget,wrap,KICK_RETARGET_M
                    moved = (self._kick_approach is not None and
                        np.linalg.norm(np.asarray(self.target_xy)-self._kick_approach.ref_ball) > KICK_RETARGET_M)
                    direction_ok = abs(wrap(pose_after[2]+c.get('direction_bias_rad',0.)-self.direction)) <= kick_direction_budget(c)
                    if moved or not direction_ok or not kick_precondition(pose_after,self.target_xy,self.side,self.contracts)[0]:
                        self.sim.force_switch(self.sim.initial_policy_key)
                        if moved:
                            self.state = 'kick_wait_ball'
                            self._ball_wait_start = self._ball_still_t = float(self.sim.t)
                            self._ball_still_xy = np.asarray(self.target_xy,float).copy()
                        else:
                            self._kick_attempts += 1
                            if self._kick_attempts > 3:
                                self._fail('kick_handoff_precondition')
                            else:
                                self._kick_approach.state = 'retreat'
                                self._kick_approach.started = float(self.sim.t)
                                self._kick_retarget_retreat = True
                                self.state = 'kick_runin'
                        return np.zeros(3),self.state
                self.skill_start_index = len(self.sim.trace) - 1
                self._skill_t = self.sim.t
                self._duration = c['duration_s']
                self.state = 'skill'
            return np.zeros(3), self.state
        if self.state == 'skill':
            if self.sim.skill_done or self.sim.t - self._skill_t >= self._duration:
                self.sim.force_switch(self.sim.initial_policy_key)
                self._settle_t = self.sim.t
                self.state = 'settle'
            return np.zeros(3), self.state
        if self.state == 'settle':
            if self.sim.t - self._settle_t >= 1.0:
                self.state = 'done'
            return np.zeros(3), self.state
        self._fail('invalid_state')
        return np.zeros(3), self.state
