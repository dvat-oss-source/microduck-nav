# Baritone-Microduck nav — shared contracts (read before writing code)

Project root: ~/microduck/nav  (package `nav/nav/`, tests `nav/tests/`, data `nav/data/`, reports `nav/reports/`)
Python: ~/microduck/upstream/microduck_rl/.venv/bin/python  (has mujoco 3.10, bam, onnxruntime, scipy 1.18, numpy, pytest)
Run tests: cd ~/microduck/nav && $PY -m pytest tests/<file> -q
Upstream (READ ONLY, never edit): ~/microduck/upstream/microduck_rl  (scripts/infer_policy.py = reference runtime,
  src/mjlab_microduck/robot/microduck/*.xml, src/mjlab_microduck/tasks/*.py)
Policies: ~/microduck/upstream/policies/*.onnx (Pollen stand-ins), ~/microduck/runs/finished_atlantis/*/policy.onnx (user's)
Simulation only. CPU MuJoCo. No GPU, no cluster (do NOT ssh anywhere).

Each module owns its files only. Do not edit another phase's module; if you need a change, write it in
nav/reports/REQUESTS.md (append, with your phase letter).

## Frames & units
World frame = MuJoCo world, z up, meters, radians. Pose2 = (x, y, theta), theta = trunk yaw.
Velocity command = (vx, vy, wz) in body frame, m/s, rad/s.

## Phase A — nav/nav/sim.py, nav/nav/policies.py, nav/policies.lock.json
  class Sim:
    __init__(scene_xml: str, policy_key: str = "walk", lock_path=...)
    reset(seed: int, start_pose: Pose2 | None = None, extra: dict | None = None) -> GroundTruth
    step(command: np.ndarray(3)) -> (GroundTruth, obs: np.ndarray)   # one 50 Hz policy step = 4 x 5 ms substeps
    switch_policy(key: str) -> bool          # runs hand-off protocol; returns False on timeout
    model, data (mujoco handles, read-only for others), t (sim seconds)
  @dataclass GroundTruth: t, pose2 (x,y,theta), pos3, quat (wxyz), lin_vel_body(3), ang_vel_body(3),
    tilt_rad, qpos, qvel, foot_contacts (dict name->bool), fallen (bool: tilt > 60 deg or trunk z < threshold)
  Seed randomizes: battery voltage U(6.5, 8.2), sag, initial yaw (only if start_pose None) — via np.random.default_rng(seed).
  policies.lock.json: {key: {path, md5, kind, obs_slots, ...}}. Sim refuses to run if md5 mismatch.

## Phase B — nav/nav/measure.py -> nav/data/envelope_<policykey>.json, nav/data/primitives_<policykey>.json
  envelope JSON: {"policy_md5":..., "vx":[min,max], "vy":[min,max], "wz":[min,max], "safe_cmds":[[vx,vy,wz],...],
                  "lateral_sigma":float, "stop_dist":{...}, "step_over_height": float, "fol": {...}}
  primitives JSON: {"policy_md5":..., "n_headings":16, "primitives":[{"id", "start_heading_idx", "cmd_sequence":[[vx,vy,wz,dur_s],...],
                  "end_mean":[dx,dy,dtheta] (world-aligned, relative to start), "end_cov": 3x3, "duration": s, "fall_rate": float,
                  "swept":[[dx,dy],...] (mean intermediate points, start-relative)}]}

## Phase C — nav/nav/mapping.py -> nav/data/map_<scenehash>.npz
  class GridMap: res (0.02), origin (x0,y0), occ (bool HxW), cliff (bool), lethal (bool = inflated occ|cliff), cost (float32, 0..1, inf for lethal),
    clearance (m, float32), world_to_cell(x,y)->(i,j), cell_to_world(i,j)->(x,y), is_lethal(x,y)->bool, cost_at(x,y)
    rooms: dict name -> polygon or bbox (from geometry / geom names), used by benchmark sampling
  build_map(scene_xml, robot_radius=None, lateral_sigma=0.0, step_over_height=0.02) -> GridMap (cached by scene hash + params)
  GridMap.save(path) / GridMap.load(path)

## Phase D — nav/nav/planner.py
  plan(gridmap, primitives, start: Pose2, goal: Pose2, budget_s=0.2, eps0=3.0) -> PlanResult
  PlanResult: poses [Pose2...] (dense, ~2 cm spacing), primitive_ids, cost, eps_achieved, complete (bool), plan_time_s
  dijkstra_reference(gridmap, primitives, start, goal) -> cost (brute force, for tests)

## Phase E — nav/nav/follower.py
  class RPPFollower(envelope): set_path(poses); compute(pose2_est, gt_vel_or_None) -> (cmd(3), status in {"tracking","rotate","done","stuck"})
  Runs at 10 Hz; caller holds command across 5 policy steps.

## Phase F — nav/nav/localization.py
  class Localizer: reset(pose2), update(sim_obs, gt_for_eval_only=None) -> Pose2 estimate; modes "gt", "odom", "odom+fiducial"

## Phase G — nav/nav/skills.py, nav/nav/fsm.py, nav/nav/mirror.py
  Skill contracts in nav/data/skill_contracts.json (measured). FSM states: idle, plan, follow, approach, skill, settle, recover, fail.
  A7 adds kick_runin and
  kick_wait_ball: plan outside fully inflated free objects to a staging pose, track a straight run-in
  with positive forward motion, and hand off only on the current contracted ball pose. GT mode may
  observe the current object position; non-GT mode declares a surveyed-target assumption and cannot
  detect later object motion without an object detector. Contracts and checker thresholds are unchanged.

## Phase H — nav/nav/checker.py (imports ONLY mujoco/numpy/json — never planner/localization), nav/nav/bench.py, nav/data/seeds_v1.json
  Metrics: Wilson 95% CI, SPL, fall rate, stair-hole incursions, time-to-goal, plan-time percentiles. Failures -> nav/reports/failures/<run>/<ep>.{mp4,json}

## Status file
Each agent appends a short entry to nav/reports/STATUS.md at the end: phase, what passed (with numbers), what failed, what's left. Report failures plainly.
