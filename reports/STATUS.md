
## Phase E (follower) — DONE on kinsim
Files: nav/nav/follower.py, kinsim.py, courses.py, follower_eval.py; tests/test_follower.py (6 pass, 1 skip until envelope_walk.json exists); reports/follower_kinsim.json.
Kinsim (tau .3, delay .1, gain .9, lateral drift 0.01 m/s, 5 seeds): cross-track RMS worst/course: straight 1.8 mm, s_bend 2.6 mm, doorway 1.9 mm, u_turn 3.9 mm (limit 30 mm); max err 12 mm (u_turn); all reach goal within 1 cm / 0.5 deg; time 8.8/13.6/14.2/19.2 s (cruise 0.25 m/s). Stuck on blocked course detected at ~8 s. All cmds within envelope; safe_cmds projection tested with a synthetic set.
Caveats: tuned only against kinsim (default envelope fixture, lag model supplied); real envelope_walk.json / MuJoCo Sim not yet exercised. vy used only in final approach (<=0.06) and optional lateral_gain (default 0). Rerun gate with Sim: follower_eval.run_course(course, *sim_backend(sim, initial_gt), follower).

## Phase H part 1 (checker/stats/bench/seeds)
- nav/nav/checker.py: independent (mujoco/numpy/json/math/dataclasses/typing only, AST-tested). Derives stair hole from apartment geometry (= x[-0.40,-0.01] y[-1.40,-0.71], matches XML). Verdict: success, fall, hole incursion, path length, time_to_goal, static collisions, reason codes for nav/kick/pick.
- nav/nav/stats.py: Wilson, SPL, percentiles, bootstrap CI. nav/nav/bench.py: multiprocessing (cpu-2), reports, failures/<run>/<ep>.json + video hook, --compare regression (CI non-overlap), --if-lock-changed, seed-file sha256 enforcement.
- nav/tools/make_seeds.py: needs nav.mapping.build_map (+ gm.rooms, lethal, world_to_cell, is_lethal, res); tested on stub only. seeds_v1.json NOT yet generated (waits on Phase C). Run: python -m tools.make_seeds from nav/.
- Tests: 25 pass (test_checker, test_stats, test_seeds_bench).
- Not done: real video hook, real runner (needs sim/planner); kick/pick checker semantics are my own definitions (peak ball speed/direction; object lift >= 2 cm, trunk drift < 1 cm) - confirm vs skill contracts. Checker "collision" ignores static geoms with top < 1.2 cm (floors).

## Phase G (mirror) -- nav/nav/mirror.py  [mirrored left-kick]
- Done: `MirroredPolicy` (act = mirror_action(pi(mirror_obs(obs)))), `mirror_obs/mirror_action`, `kick_ball_target(foot)`, `kick_sweep` stub (NotImplementedError; closed-loop sweep deferred to a nav.sim wave).
- Tables (_JOINT_PERM/_JOINT_SIGN/_OBS_PERM/_OBS_SIGN) copied verbatim from upstream symmetry.py (it imports torch/mjlab, so not imported); tests/test_mirror.py asserts equality via ast. No action-specific sign table upstream: actions use _JOINT_PERM/_JOINT_SIGN.
- Kick obs = unified 61-D (ang_vel3|grav3|q14|qd14|last_act14|cmd13 all zeros), same as walking; symmetry.py covers all 61 indices. Policies are BLIND to the ball (no ball slot), so nothing extra to mirror. ONNX batch dim is fixed at 1 (MirroredPolicy loops rows). Ball spawn in trunk yaw frame: right (0.09,-0.042), left (0.09,+0.042), +-0.015 uniform DR in training.
- Tests: 13 passed (tables==upstream, involution, dims 61->14, default pose mirror-invariant, live offline comparison, closed-loop sanity, report check).
- Offline (obs from Pollen-left rollouts, 20 seeds x 9 ball offsets x 100 steps): MirroredPolicy(right) vs ball_kick_left: MAD 0.049 (mean|a| 0.10), flat corr 0.87; per-joint corr 0.53-0.97, none negative. Reverse direction (Pollen right vs mirrored left): MAD 0.057, flat corr 0.77, per-joint 0.35-0.97 except right_ankle -0.27 (low-variance, signs flipping any other joint gives strong negative corr, so not a table sign error; separately trained policies). Negative controls (flip sign of a joint) give corr -0.77/-0.96/-0.89 on that joint.
- Quick closed loop (20 seeds, ball noise 0.015, own mujoco loop in tests/kick_rollout.py, BAM actuators): no falls any policy; success (ball >=0.3 m within +-30 deg, no fall): Pollen-left 0.55, mirrored-right 0.65, Pollen-right 0.65, mirrored-left 0.55. Ball lateral drift mirrors in sign (left-kick +0.07/+0.19, right-kick -0.22/-0.11). Full +-3 cm grid x 20 seeds sweep NOT done yet.
- Report: nav/reports/mirror_check.json; regenerate via `$PY tests/run_mirror_check.py 20`.

## Phase C — planning map (nav/nav/mapping.py)  [agent C]
PASSED (tests/test_map.py, 6 tests, ~1 s; map build 0.7 s, cached to nav/data/map_<hash>.npz):
- Vertical truth (10,000 random points, mj_ray from top of robot band, 3-class free/obstacle/cliff): strict point-vs-cell
  agreement 96.8% (2 cm quantisation of conservative raster, all errors are false-positive boundary cells);
  1-cell-tolerant agreement 99.990%; missed obstacle/cliff points = 0; missed cliff points = 0 of 1091 cliff pts
  (also 0 missed in the inflated lethal map).
- Horizontal rays (10,000 random heights in band, random headings from free floor): hit-surface cell occupied 100.0%.
- Stair hole (2000 random points in x[-0.4,0] y[-1.4,-0.7]) all cliff+lethal.
Numbers: res 0.02 m, grid 326x426 cells, bounds x[-4.26,4.26] y[-3.26,3.26] (origin (-4.26,-3.26)).
  robot_height 0.2744 m (collision geoms at STAND), footprint radius 0.0827 m (about trunk), feet z-min 0.003.
  Default inflation r = 0.0827 + 3*sigma. Cost = 0.99*exp(-5*(d-r_inflate)), inf if lethal. 52 static geoms in band.
Rooms (bbox xmin,ymin,xmax,ymax; from visual overlay geoms floor_kitchen/living/office/bedroom/bath; corridor =
  x between kitchen/living east edge and office/bedroom/bath west edge, y = union of floor_cor_n/s):
  kitchen (-4,0.5,-1,3) living (-4,-3,-1,0.5) corridor (-1,-3,0.5,3) bedroom (0.5,1,4,3) office (0.5,-1,4,1) bathroom (0.5,-3,4,-1)
Doorways (6, found as free runs along shared room boundaries, width 0.76 m raster, nominal 0.8): kitchen-living (-2.6,0.5),
  kitchen-corridor (-1,2.0), living-corridor (-1,-1.4), bedroom-corridor (0.5,1.8), office-corridor (0.5,0.2), bathroom-corridor (0.5,-2.1).
Geometry quirks: (1) there is NO ground plane; floor = 4 collidable boxes (top z=0): floor_w, floor_e, floor_corridor_n/s.
  The stair hole (x[-0.4,0], y[-1.4,-0.7]) is simply missing floor; stairs are 4 boxes with tops at z=-0.1..-0.4, i.e.
  they DESCEND (below floor), all treated as cliff. Everything beyond +-4 x / +-3 y is void = cliff (margin ring 0.2 m).
  (2) Per-room floor overlays, rugs, landmarks, dock, bed pillow/blanket, mirror, towel rail are contype=0 -> not obstacles.
  (3) Desk top (z 0.34-0.38) is above robot height 0.274 -> only the 4 desk legs are obstacles (robot can walk under).
  Coffee table top (z .14-.22) IS in band; legs too. (4) obj_*/ball_* (free bodies) excluded. (5) Cliff flag is not set on
  cells already occupied (walls straddle the floor edge). (6) Occupancy is conservative (any touched cell).
Not done / left: step_over_height is a parameter (default 0.02); Phase B value must be passed to build_map (cache key includes it).

## Phase A — headless sim harness (nav/nav/sim.py, policies.py, video.py, policies.lock.json)  [agent A]
PASSED (tests/test_sim.py: 15 passed, 1 xfailed (strict, see below), 5.4 s total):
- Parity vs infer_policy.py (reference = PolicyInference + load_mujoco_with_bam imported from the upstream script,
  driven like main(); independent code path). 500 steps each, vin 7.4 / sag 0.1, mixed commands
  (0 -> 0.3 fwd -> (0.25,0.1,0.5) -> spin -0.8). Two checks per step: state-synced (mj_copyData of Sim state into the
  reference) and free-running (both sims evolve on their own). Max |obs diff| = 0.0 for walk/flat, walk/apartment
  (yaw 0.7), sitstand (flag 1 then 0), velstand_rough; 2.2e-15 for pick (phase float rounding, 139 steps = until
  infer_policy ends the pick). Final qpos diff 0.0 (bit-identical physics).
- Determinism: same seed -> identical qpos (diff 0.0) after reset-in-between-other-episode and in a fresh instance (flat + apartment).
- md5 mismatch refused; lock status "rejected" refused (velocity_hard_4250); provenance lists every lock entry used.
- RTF (M3 Pro, 1 process, ORT 1 thread): flat 68x (3400 steps/s), apartment 40x (1975 steps/s). reports/sim_rtf.json.
- Hand-off walk(0.3 m/s)->stand: settles in 0.40 s; last_action == held targets; no fall over 2 s after.
- Smoke: flat walk 0.4 m/s 10 s: 1.53 m, no fall. Apartment default spawn (office (2.0,0.0), seeded yaw), 0.3 m/s 5 s,
  seeds 0-2: 0.46-0.49 m forward, no fall. Video: mujoco.Renderer offscreen works on macOS (250-step apartment clip in 3.1 s).
FAILED (reported plainly):
- Spec smoke "alpha_walking cmd (0.2,0,0) 10 s flat > 1 m": moves 0.01 m. NOT a harness bug: infer_policy.py's own loop
  gives x=0.01 m too. From a standing start the walkers ignore |vx| <= 0.2 (walk, velstand_rough, atlantis_walk,
  velocity_hard: dx 0.00-0.01 m in 10 s). Kick-started with 0.5 s at 0.3 they keep walking at 0.2 (0.64 m / 10 s).
  Kept as xfail(strict) test_smoke_flat_walk_0p2.
QUIRKS other phases must know:
- Speed tracking in CPU MuJoCo is ~30-55% of command (alpha_walking body-frame vx: cmd 0.2->0.067, 0.3->0.111,
  0.4->0.174 m/s; velocity_hard 0.3->0.158, 0.4->0.220; same in infer_policy). Large yaw drift at pure-vx commands
  (walk 0.3: -0.7 rad / 10 s; Pollen velstand 0.2: -1.6 rad / 10 s). Phase B must measure, not assume, the envelope.
  Solver = XML defaults as infer_policy (not mjlab's training MujocoCfg), forcerange = vin*kt/R per episode (infer_policy),
  training uses max(vin_range)=8.2 V for forcerange.
- Sim.step(command) returns (GT AFTER the 4 substeps, obs FED to the policy at the START of the step). obs gyro and
  projected gravity come from the previous substep's forward pass (5 ms stale vs qpos), exactly as infer_policy.
  Sim.observe() gives the obs for the current state; Sim.last_obs the last fed obs.
- Obs layout (all 20 lock entries are 61D in / 14D out, same DEFAULT_POSE, action_scale 1): ang_vel[0:3] proj_grav[3:6]
  joint_pos-DEFAULT[6:20] joint_vel[20:34] last_action[34:48] twist[48:51] head_pose[51:55]=0 body_pose[55:61]=0.
  command_mode per kind: walk="twist" (cmd[0:3]=vx,vy,wz; a 13-vector is also accepted), stand/kick/roulade/standup="zero"
  (arg ignored), sitstand="posture_flag" (command[0]: 1 sit, 0 stand), pick="phase" (Sim writes cos/sin(2 pi phase),
  phase = steps_since_switch*0.02/4.0; sim.skill_done when next phase >= 0.7, i.e. after 139 steps = 2.78 s).
  Kick/roulade sim.skill_done after duration_s (3.0/2.0 s, infer_policy defaults; manifest says 0.5/1.0).
- Sim never auto-switches back after an episodic skill (infer_policy does): caller uses force_switch()/switch_policy().
- fallen = tilt > 60 deg OR height < 0.02 m, height = trunk z - lowest foot-collision-geom center z (works on stairs).
  Standing height ~0.110, sitstand sit ~0.032, standup sit start 0.034. Roulade trips "fallen" by design.
- Measured skills (flat, seed 0): pick (Pollen + atlantis) from walk ends standing, h 0.110; sitstand sit 3 s -> h .032,
  stand back -> .110; kick_right/kick_left/kick_right_user move the ball 0.45/0.35/0.25 m in 1.5 s, no fall;
  atlantis_standup_flat and standup_rough rise from sit to h .110 in 4 s (yaw drift 0.5-0.7 rad).
- Spawn: z = floor + 0.125 (infer_policy); reset raises ValueError if (x,y) is not on the z=0 floor (stairs hole,
  furniture top, void) or the robot overlaps a non-floor geom. Default spawn: flat (0,0); apartment (2.0, 0.0) office,
  0.92 m clearance; yaw = seeded U(-pi,pi) when start_pose is None.
  Free floor (max clearance to colliding geoms/hole, 5 cm grid): office (2.00,0.00) 0.92 m; living (-1.85,-1.95) 0.78 m;
  corridor (-0.20,1.80) 0.75 m; bath (2.60,-1.80) 0.73 m; kitchen (-3.25,1.75) 0.65 m; bedroom (0.90,1.80) 0.50 m.
  Apartment has 7 free bodies (obj_0..3 boxes, ball_0..2) in qpos after the robot (nq 70); ball_2 starts at
  (-0.30,-1.40), at the stair-hole edge.
- Ball: Sim(scene, key, ball=True) adds the ball.xml body ("ball"/"ball_free") to any scene; reset places it in front of
  the right foot unless extra={"ball_xy":(x,y)} or {"ball_for":"kick_left"}; sim.place_ball(x,y,for_kick=...).
- Lock: keys walk(alpha_walking) velstand stand sitstand pick kick_left kick_right roulade atlantis_walk
  atlantis_sitstand_flat/rough atlantis_standup_flat atlantis_pick_flat atlantis_velstand_rough_wip velstand_flat
  velstand_rough kick_right_user standup_rough velocity_hard(=partial_4750) velocity_hard_4250(rejected).
  Regenerate only deliberately: `$PY -m nav.policies --write`. Runs must store sim.provenance().
- Linux batch node: sim.py/policies.py import only mujoco, numpy, bam, onnxruntime (no infer_policy/viewer/termios).
  Paths resolve against $MICRODUCK_ROOT (default ~/microduck): lock paths and scenes are relative to it.
  ORT runs 1 thread per Sim -> parallelize with processes. Offscreen video on headless Linux needs MUJOCO_GL=egl|osmesa;
  render_trace falls back to <out>.qpos.npz if GL fails.

## Phase D — lattice planner verification
- Re-ran `tests/test_planner.py` serially: 14 passed. This covers swept path collision clearance, brute-force Dijkstra cost and ARA* epsilon bounds, admissible/consistent heuristics, partial timeout behavior, unreachable/lethal start and goal, and segment/replan helpers.
- Timing report refreshed and valid JSON: 100 synthetic queries, 100% complete, planner p95 0.1874 s (max below 0.2 s); max achieved epsilon 1.318, worst cost / optimum 1.037. Real Phase C map + synthetic primitives: 40 queries, 100% complete, planner p95 0.1868 s, max epsilon 1.448. First solution p95 0.025 s synthetic / 0.025 s real.
- No code repairs needed in this pass. The real primitives dataset was unavailable, so performance and boundedness with measured Phase B primitives remain unverified.

## Phases F/G — localization, skill measurements and FSM
- Localizer supports gt, contact/leg odometry and camera-image fiducial EKF. Non-gt modes ignore gt_for_eval_only. Binary virtual foot-contact sensors use actual simulated foot collision events, not GT pose; kinematic anchor updates only use contacting feet and freeze translation while airborne. IMU framequat is simulated fused orientation (declared noise), not an independently validated inertial fusion stack. One flat infer-dynamics seed, 500 steps at 0.3 m/s: final xy error 0.0893 m, RMS 0.0504 m, 498 contact ticks. Five-seed training apartment camera gate is tools/localization_gate.py; rendering requires an approved CoreGraphics context. No camera accuracy/coverage is claimed until this gate is run.
- Training-dynamics flat sweep: 49 placement offsets at 1 cm spacing × 5 seeds, per side; right kick 87/245 (35.5%), mirrored-left 92/245 (37.6%), nominal 3/5 and 4/5. Falls 0/245 each, Wilson 95% upper bound 1.54% (correlation among offsets/seeds limits population inference). All-success sampled boxes have zero width in one dimension; their tolerance is a controller heuristic, not proven between cells. data/skill_contracts.json records all trials and dynamics; infer results preserved in data/skill_contracts_infer.json. Five pick trials passed robot postconditions; grasp/lift and reach radius unvalidated. Settled fallen recovery 0/20 (4 orientations × 5 seeds). No claim that recovery works.
- FSM advances localization at 50 Hz, follower at 10 Hz; safety uses IMU tilt and FK/virtual foot contact, with no GT-based fall/recover decisions. Handoff advances sim internally. Skill start index records the actual onset after handoff; task success requires independent checker. Dynamics mismatch rejects skill use. Training sampled kick rectangles are effectively zero-width in one axis, and skill benchmark preconditions may therefore fail. No camera-rendering error is suppressed.
- Verification: tests/test_localization.py tests/test_skills.py tests/test_fsm.py tests/test_mirror.py: 23 passed. Camera image segmentation tested synthetically; end-to-end camera render gate pending parent run. Tests and sweep use single process, ORT one thread.

## USER PAUSE — 2026-11-09, latest session checkpoint

All agents interrupted and task simulation processes terminated at the user's explicit request. Work is paused, not complete. Detailed current source of truth for this checkpoint: [CLAUDE_HANDOFF_2026-11-09.md](CLAUDE_HANDOFF_2026-11-09.md). It supersedes earlier "pending" claims where results are now available.

- A/B: explicit infer/training dynamics flag added; infer parity preserved. Sim scoped tests 18 passed + existing expected failure; measurement tests 4 passed. Six accepted walkers have completed 245-command × 8-seed grids and measured primitives, zero observed grid falls; velocity_hard was interrupted around 40/245. No final walker_gate.json or selected navigator. See measure_training.log and handoff for exact artifacts.
- Fidelity: 20 independent flat mjlab probes completed, but trajectory/initiation mismatch remains. Partial agreement, NOT certified; no same-scene apartment outcome replay. reports/sim_fidelity.md/.json.
- C/H integration: fixed zero-step-height floor classification; map cache version 4. Independent checker loads exact compiled MJB for training qpos layout. Map/checker combined 18 tests passed. H scoped suite 15 passed. New episode runner, benchmark CLI, trace/video/provenance/target-gate wiring added; full latest integration remains unverified.
- F: actual five-seed 300-step rendered camera gate completed and projection bias corrected. Latest calibrated final fiducial xy errors about 8/17/10/15/9 mm vs odom 7/19/11/11/11 mm, 85 detections/85 updates; short development gate only. reports/localization_gate_calibrated.json. Focused F/G suite reached 24 passing tests before later E edits.
- E: actual Sim baseline 7/20 reached, worst RMS 77.8 mm (target 30 mm). Interrupted development changes in follower.py address measured static/lateral deadzones; latest dev3 reached 15/20, worst RMS 84.5 mm, still FAIL. Review changes and rerun tests; no claim of E completion on real Sim.
- G shortfalls unchanged: training right/mirrored-left kick 87/245 and 92/245, zero observed falls; practical sampled tolerance degeneracy. Pick lift unvalidated, recovery 0/20. FSM restored selected navigator policy and uses sensor safety.
- No frozen seeds, full H run, success/SPL benchmark CIs, or independent apartment replay yet. All original targets retained. No upstream edits, registry edits, or cluster jobs.

## Sim fidelity — same-apartment mjlab replay cross-check (2026-11-09, Claude)
Built (files owned): tools/replay_mjlab.py (reference Python, imports nothing from nav), tools/replay_compare.py,
tests/test_replay.py (12 passed incl. live 25-step mjlab replay, REPLAY_SLOW=1; log reports/replay_tests.log),
reports/replay_criterion.json (FROZEN, sha256 983bd10d…), reports/replay_calibration.json,
reports/replay_pilot/traces/*.npz (5 self-made episodes), reports/replay_pilot_check.{json,md} (+ _mjlab/ replays).
- mjlab side: Mjlab-Velocity2-Rough-Backlash play cfg; plane removed, the 82 top-level apartment geoms added to the
  static "terrain" body (contact-sensor naming preserved), 7 free objects added; training _soften_terrain_contacts
  applied after; DR/push/obs-corruption/IMU misalignment off (incl. base_com, which fid_mjlab_ref kept), all
  terminations/curricula off; per-world vin/sag from trace. State mapped by joint/body NAME (nav training nq 84);
  initial-state error 9.5e-8. Geometry vs nav model: 61 collidable static geoms identical (pos/size/quat 0 error),
  robot body masses identical; only solref/solimp differ (training softening). Same ONNX (md5-checked; numpy MLP vs ORT 1.4e-6).
  Open-loop replay of recorded per-step commands, planner/follower not re-run. ~80 ms/step for 10 worlds on Warp CPU.
- nav side: exact open-loop replays reproduce every trace bit-for-bit (harness sanity). Nav-vs-nav divergence
  (5 pilots, 961–2481 steps): 1e-6 m perturbation endpoint div p50 0.13 / max 0.73 m; different delay-RNG p50 0.21 /
  max 0.98 m; inside the nav set itself 'fell' flipped on 1/5 episodes and 'static collision' on 1/5.
- Frozen criterion: per episode, fell / stair-hole / static-collision agree iff the mjlab value occurs in {nav
  original + 7 nav replicates (3 × 1e-6 m perturbations, 4 × other delay RNG)}; final xy agrees iff
  d ≤ X = max(0.10 m, 2 × max nav-replicate endpoint spread) (leave-one-out on pilots: 0/20 false alarms;
  factor 1 would give 3–4/20). Episode agrees iff all four agree; PASS iff disagreement ≤ 10% over n = 20, Wilson CI reported.
- Pilot result (n = 5, NOT the benchmark sample, policy walk, gt mode): 4/5 agree, 1/5 disagree (pilot_toward_hole:
  mjlab rep 0 drifts right, hits cor_console and falls at step 503; mjlab rep 1 does not fall). Pilot "verdict" FAIL
  0.20 (Wilson [0.04, 0.62]) — too few episodes to conclude, but a real warning: mjlab ends to the right of nav on 4 of
  5 pilots, matching the flat-probe yaw-drift sign difference. Requests to A/H in REQUESTS.md.
- One command for the benchmark: `PYTHONPATH=. $PY tools/replay_compare.py --traces reports/traces/<run> --n 20`
  → reports/replay_check.{json,md} (runs ≤ 2 processes: nav replicates in-process + one mjlab subprocess).
Limitations: kick traces refused (mirroring not recorded); pick/recover policy switches are replayed by ONNX swap but
unvalidated; nav-type tasks only by the sample rule; the pilot_office_short (pilot0) mjlab replay was inspected
during harness development before freezing; mjlab is itself stochastic (rep spread p50 0.24 / max 1.12 m), and only
rep 0 is judged. No sim edits, no upstream edits, no cluster.

## Phase E — path follower on the REAL Sim (training dynamics), 2026-11-09  [agent E2]
Gate unchanged (tools/follower_sim_gate.py, flat, dynamics='training', courses straight/s_bend/doorway/u_turn x seeds 0-4;
per run cross-track RMS < 0.03 m, status done and within 5 cm / 10 deg after 1 s zero-command settle, no fall).
One command per policy: `$PY -m tools.follower_sim_gate --policy <key>` -> reports/follower_sim_<key>.json (+ .log).
Results (final follower version; envelopes regenerated 15:23-15:33):
- atlantis_walk (selected navigator): PASS 20/20, worst RMS 23.6 mm, median 19.6 mm, 0 falls, 0 final fixes.
  Per course worst: straight 23.6 / s_bend 21.4 / doorway 21.0 / u_turn 20.1 mm; mean time 13.4/20.5/19.5/28.7 s.
- walk: PASS 20/20, worst 19.4 mm, median 17.1 mm, 0 falls.  (Supersedes baseline 7/20, worst 77.8 mm, and dev3 15/20.)
- velocity_hard: PASS 20/20, worst 15.5 mm, median 14.4 mm, 0 falls.
- velstand_flat: FAIL 18/20 (19 reached), worst 40.1 mm, median 21.1 mm, 0 falls. Failures doorway s4 (RMS 32.4 mm,
  reached) and u_turn s4 (heading-repair timeout). Both from post-stop repairs: its envelope has no consistent
  |wz| < 1.0 turn-in-place (0.5 bistable), so turns run at 1.4-1.6 rad/s and each in-place turn shifts the trunk
  5-10 cm; lateral stop spread 2.4 cm sd. Tracking itself is fine (median 21 mm).
- Held-out seeds 5-14 (development check, not the gate): atlantis_walk 40/40 worst 25.0 mm, walk 40/40 worst 23.7 mm,
  velocity_hard 40/40 worst 16.7 mm.
What changed (nav/follower.py rewritten; API unchanged): envelope inverse model from measurement.rows (vy=0 slice):
bilinear (vx,wz)->(v,w) map inverted over the "gait-on" region (lowest vx level that walks from standstill for every
wz: walk 0.3, atlantis_walk 0.2, velocity_hard 0.3), next level as start kick, per-sign turn-in-place command =
smallest |wz| that rotated consistently (walk +/-1.5, atlantis_walk/velocity_hard +/-1.0), stop distance from
envelope stop_dist scaled by speed. Cascaded tracking law (lateral -> saturated heading ref -> yaw rate, curvature
feed-forward, bias integrator, 0.1 s prediction), heading cap shrinks over last 0.5 m, slowest gait-on speed over last
0.3 m. vy never used. Post-stop: settle detection, heading fix with learned anticipation and learned minimum burst
(posture-twist relaxation), latched face+walk position repair. Internal tolerances = gate minus measured post-done
drift (<= 2.2 mm, <= 2.1 deg). No per-walker numbers: all walker-specific values come from the envelope file.
Measured facts behind it (scratch experiments, 4 seeds): trunk gait sway ~13.5-14 mm lateral RMS (4 cm p-p, ~0.6 s
period) is irreducible and dominates the RMS floor; gait start throws 3-8 cm / up to 16 deg in the first 2 s for any
start command; while walking the wz response is monotonic and close to the envelope rows (walk vx .3: wz .3->.22,
.6->.45 rad/s) — curvature control with the gait on works; sub-threshold rotation bursts produce a posture twist that
relaxes back; stop transient changes heading 4-7.5 deg sd (so ~10% of stops need a heading fix).
Other edits: nav/courses.py doorway clearance bug fixed (reported 0 m clearance in the middle of the opening, which
forced proximity slow-down); tools/follower_sim_gate.py adds --workers (<=3), per-course summary, median RMS,
--seed-offset and --follower-kw (development only), pos/yaw at status. Kinsim tests: tests/test_follower.py 7 passed
(+ test_fsm 4 passed); no fixture changes needed (envelopes without rows get a linear model from FOL gains; with no
FOL/stop_dist a 0.3 s stop lag is assumed). reports/follower_kinsim.json regenerated.
Caveats: gate uses ground-truth pose; under odom/fiducial estimates the settle/still detection (4 mm / 1.1 deg over
0.3 s) falls back to a 2 s timeout if the estimate jitters. Envelopes will change if the sim contact-softening port
lands: rerun the four gate commands (and nav/mapsigma.py, which reads follower_sim_atlantis_walk.json).

## Phase D — planner gate on REAL map + MEASURED primitives (atlantis_walk), 2026-11-09  [Claude, planner_gate]
One command: `$PY -m tools.planner_gate` → reports/planner_gate_atlantis_walk.{json,md} (~2.5 min, 1 process).
Fast subset: tests/test_planner_measured.py (5 passed, 3.7 s; deterministic, no timing asserts).
Setup: map r_inflate 0.1418 m (closed-loop σ 0.0197), 1 free component, 256 measured primitives (16/heading, none
synthesized), 555,504 lattice states, precompute 0.18–0.21 s. CPU during measurement: load 4–5 on 11 cores with 2 other
Python procs at ~100% (not idle); cpu/wall 0.99 (single-threaded).
Gate (plan text) PASS:
- (a) 100 queries, all 36 ordered room pairs + 64 random (83 cross-room): 100/100 complete, all bounds hold
  (cost ≤ eps·ref), eps max 1.5 (eps=1 on 64–67%), cost/ref max 1.023–1.032, no eps=inf; all end within 5 cm / heading
  bin of the requested goal; every segment matches its measured end_mean within 2.7 cm / half a heading bin.
- (b) 136k densified swept points re-rasterized vs gm.lethal: 0 lethal; min clearance on paths 0.144 m (> r_inflate).
- (c) plan_time p95 0.188 s, wall p95 0.191–0.192 s (max 0.1996), first solution p95 0.025 s.
- (d) 7 goals (one per room + any-heading): h ≤ exact cost-to-go on all 250k reachable states, 0 reachable states
  pruned, 56/56 random states h ≤ dijkstra_reference; median h/ctg 0.59–0.70.
Extra:
- (e) the wall budget is load-sensitive: same 100 queries rerun change result 7–9% idle-repeat, 8–9% with 2 busy loops
  (cost/path only; complete flag never flipped). max_expansions=30000 (budget_s large): 0/100 changes idle and under
  load, 100% complete, wall p95 0.121–0.150 s (max 0.176) idle, 0.123–0.133 s loaded. 50000 is the largest with p95 <
  0.2 s but max 0.25 s. Proposed in REQUESTS.md (D2).
- (f) cross-room 83/83 complete within 0.2 s (0 partial). Forced partial (30 expansions): 71/71 partial plans make
  progress (independent geodesic, p50 3.0 m), 0 lethal.
- (g) kitchen↔bathroom far halves (geodesic 8.2–9.0 m): 20/20 complete, wall p95 0.193–0.199 s (max 0.205 s; right at
  budget); at 30000 expansions p95 0.127–0.153 s.
DEFECT found (h): the first plan() per GridMap object pays the per-map precompute (~0.2 s) inside its 0.2 s budget →
ARA* stops at 128 expansions → timeout_partial; cold first plan complete 4/10 vs warm 10/10. episode.py builds a fresh
GridMap per episode, so this hits the first plan of every episode (reproduces the coordinator's bath→living
`plan_incomplete` at t=0; warm the same query is optimal in 17 ms). Fix diffs for planner.py (D1) and fsm.py (warm
planner, deterministic budget, follow partial plans + replan; D2/D3) in REQUESTS.md. Not applied (not my files).

## Coordinator (Claude, 2026-11-09) — integration fixes
- Phase B rerun for all 7 accepted walkers (training dynamics, `python -m nav.measure --all-walkers`, log reports/measure_training_v2.log). walker_gate.json: selected **atlantis_walk** (232 safe cmds, 0/1960 falls, vmax 0.282); velocity_hard 228 / 0.286.
- Map inflation σ: open-loop B lateral_sigma closed the doorways for 6/7 walkers. Frozen map now uses closed-loop follower cross-track σ (nav/mapsigma.py; atlantis_walk σ=19.7 mm → r_inflate 0.142 m; one free component). Enforced in make_seeds provenance + bench. Rationale in REQUESTS.md.
- Applied planner-gate proposals D1–D3: planner clock starts after per-map precompute; FSM warms planner, uses deterministic 30000-expansion budget (2 s wall cap), follows Baritone-style partial plans with progress check, records plan_log in traces.
- FSM: followed path now ends at the exact goal pose (planner goal tolerance + follower tolerance were stacking: pilot stopped 6.4 cm from goal).
- tests/test_planner.py equivalence test given an unconstrained budget (it was load-sensitive).
- Unfrozen pilots (atlantis_walk, gt, apartment): office→bedroom, living→kitchen, bath→living all succeed (pos 1.3–4.7 cm, yaw ≤ 6.4°, no fall/collision/hole). Before fixes: 1/3.

## H referee audit (2026-11-09, independent auditor)
reports/audit_benchmark.md + tests/test_audit.py (12 xfail(strict) defect tests, 39 s, 1 process). No audited file edited, seeds_v1.json not generated.
High: A-01 SPL p_i from raw 50 Hz trunk path is 1.6-2.4x the 0.6 s-smoothed path (pilot SPL terms 0.35 mean raw vs 0.69 smoothed); A-02 3/200 candidate nav starts make Sim.reset raise (under off_desk, on obj_2) -> safety_unknown -> gate can never pass; A-03 obj_1 starts inside kit_island and leaves the world, ball_2 is on the stair-hole edge (13/50 pick, 16/50 kick infeasible).
Medium/low: A-04 hole check misses a 24 mm foot overhang; A-05 checker MJB cache ignores included XML; A-06 latest_report lexical order; A-07..A-12 see report.

## Sim fidelity v2 — training dynamics `cb8561f6_nominal_v2` (2026-11-09, Claude)
Report: [sim_fidelity_v2.md](sim_fidelity_v2.md) / .json. sim.py md5 d07e522d → 8b5c40b9; `provenance()["dynamics_version"]`
= `nav.sim.TRAINING_DYNAMICS_VERSION` = cb8561f6_nominal_v2. Infer path bit-identical (test_sim 18 passed + 1 xfail; infer fid log identical).
- Kept: (1) exact `_soften_terrain_contacts` port (rl_cb8561f6 microduck_velocity_env_cfg.py:151-169/:778) on all static
  worldbody geoms — bit-identical on flat (feet priority 1 override floor solref/solimp), replay geometry diffs 122 → 0;
  (2) mjlab BamActuator friction budget (`sim.training_frictionloss`, bam/mjlab.py:425-481,:580-676): DOF-indexed friction
  subtraction, no qfrc_constraint (mujoco_warp sparse solver leaves it 0 for converged worlds, solver.py:1955-1985/:2187-2195),
  unsigned m6 quad term; frictionloss err vs mjlab 7.2e-3 → 8.5e-7.
- Rejected after one-factor isolation: reset zero-target init (mjlab drives toward absolute 0 for 3–6 substeps) — fixes the
  reset transient but worsens yaw initiation / endpoints. max_current: mjlab runtime is 1.75 A too. Flat reference was already
  DR-free (base_com has body_names=() → no bodies); added 6-rep reference fid_reference_nominal_x6.json.
- Flat forward yaw drift (box floor = mjlab tiles, 6 reps × 3 seed sets): v1 +0.021 → v2 −0.019 rad/s vs mjlab −0.020;
  paired endpoint XY median/p90 0.026/0.383 → 0.020/0.233 m. Constant-forward 8 reps box: vx .4 +0.001 → −0.026 (mjlab −0.022),
  vx .3 +0.053 → +0.036 (+0.026). Plane keeps an engine gap (warp ~4 contacts/foot vs C 1–2).
- Pilots (5, regenerated reports/replay_pilot_v2, check reports/replay_pilot_check_v2): FAIL 1/5 → PASS 0/5; same-command paired
  nav↔mjlab end distance mean/median 0.467/0.355 → 0.308/0.130 m, clockwise bias +0.19 → −0.01 m, fall agreement 3/5 → 4/5.
  v2 nav replicate spread is larger (≤1.3 m) so the frozen X is looser; not the 20-episode benchmark.
- Residual (not portable): mujoco_warp drops penetrating foot–box contacts in ~6% of substeps; plane manifolds differ.
- INVALIDATED (training mode; none record dynamics_version): Phase B envelopes/primitives (data/envelope_*, primitives_*,
  walker_gate.json, measure_training*.log), follower gate reports/follower_sim_*.json, data/skill_contracts.json, fid_nav_training.json /
  sim_fidelity.md training columns, v1 pilots reports/replay_pilot*/check, any trace with dynamics_version cb8561f6_nominal_v1.
- Also edited: tools/fid_nav.py (--scene, --seed-offset, provenance), tests/test_replay.py → v2 pilots (15 passed + 1 skipped),
  replay_compare.py note text; added data/sim_fidelity/scene_flat_box.xml. No upstream edits, no ssh, ≤2 processes.

## PAUSED 2026-11-09 (Claude) — see reports/HANDOFF_2026-11-09_claude.md
All measured artifacts (envelopes, primitives, walker_gate, follower gates) are STALE under dynamics v2 and were
moved to reports/stale_v1/. Audit fixes and skills diagnosis were interrupted mid-work. Tests: 106 passed,
2 skipped, 13 xfailed (excluding test_planner_measured, which needs regenerated primitives).

## Skill diagnosis and contract safety (2026-11-09)
- Compared the pinned training checkout: skill-task full-collision morphology differs from benchmark walk morphology. Diagnostic-only simulator now avoids process-wide XML monkeypatch; benchmark Sim is unchanged. See reports/skills_diag.md for caveats and sampled outcomes.
- Skill activation checks dynamics, version, model morphology, verified non-rejected policy hash, and explicit support; recovery fails safely when unsupported. Pick lift and fallen recovery are unsupported; zero-width sampled kick tolerances remain zero. Existing contracts intentionally fail closed pending the ordered remeasurement. Focused tests: 10 passed.

## H audit repair — resumed 2026-11-09 (Codex audit_repair)
Backups: `reports/bak_2026-11-09_resume/audit_repair/`. Repaired A01/A04/A06/A07/A09/A10/A12 in
checker/bench/stats; checker also enforces A08 final one-second dwell with EpisodeRunner's settle tail.
SPL uses centred MA30 endpoint-padded xy and retains raw path length; near-floor collision AABBs overlap the hole;
fall uses trunk height above lowest foot centre. Invalid/unknown traces never count as observed safe. Static
collision geometry is checked against the independent scene manifest when supplied. Skill windows use sim policy
records (exact recorder step preferred), reject missing onset and >2 cm ball movement before kick onset.
Benchmark compares only matching seed hash/task counts/mode and uses a two-sided pooled two-proportion test at
alpha .05 for worsening rates. Latest report uses parsed report timestamp and full-suite status; every limited run
is partial. Primary policy artifacts/scene/version/sigma stay frozen; explicitly marked comparison walkers use
verified own artifacts and a passing follower gate on the same frozen navigation subset. Reports add room-pair
metrics, successful-only time percentiles, every recorded planner-call time, and crash traceback/partial trace/video
attempt retention. Removed the repaired audit xfails (including delegated A08); the root removed its own markers.
Validation: owned checker/stats/seeds_bench/audit subset **37/37 passed**; complete audit **12/12 passed**, one local
CPU process. Additional narrow integration checks passed for policy preflight/comparison safeguards and real
training-model agreement with an independently compiled static scene (a first fixture used an invalid wall spawn;
corrected to the established valid spawn at [2,0,0]). No benchmark or frozen seed generation run by this worker;
no upstream, SSH, cluster, GPU, or downloads. Remaining integration review findings sent to the coordinator.
- Identified the training sweep's ball as 35 mm radius / 15 g versus apartment targets 40 mm / 25 g. Prepared a flat scene matching apartment ball geometry and physical parameters, plus selected-walker handoff provenance; no remeasurement until prerequisite gates finish. One-seed smoke diagnostic passed, which is not a reliability estimate. Each FSM plan log now includes plan_time_s.
Follow-up: frozen metadata now refuses any MA30/dwell/geodesic-method mismatch
(`30`, `1 s`, `footprint_16_dijkstra_los_v1`); targeted comparison preflight regression passed.
Read-only integration review found missing skill pre-goal dry validation and non-target object stability checks;
the coordinator corrected both, adding target separation and validating all objects for one second.
Real training-MJB static collision geometry matches the independently compiled deployment-scene manifest.

## Coordinator resume — 2026-11-09 (Codex; v2 integration, seeds NOT frozen)
- Baseline exited 0: original 106 passed / 2 skipped / 13 xfailed configuration retained before edits.
- All 12 audit regressions now pass (audit worker log); root A02/A03 dry reset/object feasibility tests passed (2 in 21.86 s), stronger A03 skill-prepose + all-object stability check passed (13.35 s), A05/A11 passed (0.40 s). No audit xfail markers remain.
- EpisodeRunner records each actual Sim step including internal handoffs, exact policy_events with mirror flags, dynamics_version and complete scene-closure hash; final goal dwell is 1 s. Exceptions retain partial trace and traceback. Independent static scene geometry accompanies checker MJB traces.
- Seed generation now validates starts/goals in training Sim with object clearance, uses explicit stable free-object layouts and feasible skill targets, and computes footprint-only 16-connected Dijkstra + line-of-sight string-pulled distances. Freeze refuses overwrite and records metric definitions. No seeds_v1.json generated.
- Replay exact-start reconstruction now restores all recorded free-object poses; replay tests 12 passed / 1 skipped (live mjlab test remains opt-in).
- Native camera access approved and exercised: localization_gate_v2.json, 5x300 steps, 150 renders, 85 detections/updates; odom final XY 5.7–14.9 mm, odom+fiducial 1.9–14.0 mm. This is a short development gate, not H certification.
- Full B v2 regeneration ongoing with 3 workers; six policy pairs complete so far. Training/infer physics untouched. Full connected benchmark and independent 20-episode replay remain pending.

## Phase B v2 rerun + Phase E baseline gates — 2026-11-09 (measurement_v2)
Regenerated all 7 accepted walker envelope/primitive pairs with training dynamics under `cb8561f6_nominal_v2` (`reports/measure_training_v3.log`; 7 COMPLETE, exit 0). All 14 artifacts match `policies.lock.json` md5 and pair dynamics_version; no rejected policy included. `walker_gate.json` selects atlantis_walk (232 safe commands, 0/1960 grid falls, 256 usable primitives, max forward 0.2831 m/s), same selection as stale v1. Added dynamics_version to future measurement summaries and enforced consistent/non-missing versions in walker-gate selection; backup at `reports/bak_2026-11-09_resume/measure/measure.py`. Baseline v2 follower gates: walk 20/20 PASS; velocity_hard 20/20 PASS; atlantis_walk 19/20 FAIL (straight seed 4 RMS 34.7 mm); velstand_flat 15/20 FAIL (2 not reached; multiple S-bend/U-turn and doorway failures). Full JSON/logs archived under `reports/follower_v2_before_repair/`. No follower or simulation code changed. Awaiting follower repair before rerunning canonical gates.

## Phase E — terminal steering repair, training v2 (Codex, 2026-11-09)
Selected walker baseline: 19/20 gate pass, 20 reached, zero falls; straight seed 4 RMS 34.71 mm failed the 30 mm
requirement. Exact replay showed a growing terminal lateral offset constrained by the fixed 3-degree final
heading-reference limit, followed by large position/heading correction turns. Added a generic terminal steering
reserve outside the equal-axis goal-position budget minus measured stop spread. Steering authority uses path
geometry plus the envelope's stopping distance and forward response horizon; existing safe-command bounds and gate
tolerances remain in force. Source frozen at SHA256 `0d02083416f5aa4c98c3564ab1ee0688dfa31bd6db35180529584a0ce4a789d6`.
Backups: `reports/bak_2026-11-09_resume/follower/`; report: `reports/follower_repair_v2_2026-11-09.md`.
Validation: follower unit/KinSim tests **8/8 passed**; selected single-process development gate **20/20 passed**, zero
falls, worst/median RMS **26.45/18.41 mm**. Seed 4 final position/yaw **25.81 mm / 2.61 degrees**. Canonical reruns of
all four walkers are in progress under the coordinator; no held-out v2 seeds checked by this repair worker.

## Phase E canonical gates after follower repair 1 — 2026-11-09 (measurement_v2)
All four canonical gates ran against frozen `nav/follower.py` SHA-256 `0d02083416f5aa4c98c3564ab1ee0688dfa31bd6db35180529584a0ce4a789d6`, with 3 workers, defaults, and no overrides. atlantis_walk PASS 20/20 (0 falls, worst RMS 26.45 mm); velocity_hard PASS 20/20 (0 falls, worst 15.36 mm); walk FAIL 19/20 (0 falls; doorway seed 4 RMS 30.32 mm, max 87 mm, position 46 mm, six fixes); velstand_flat FAIL 17/20 reached (0 falls; u_turn seed 1, s_bend seed 4, u_turn seed 4 timed out, worst RMS 42.63 mm). Full reports/logs archived in `reports/follower_v2_terminal_repair1/`. The required selected+walk+velocity_hard gate set is not all passing, so Phase D planner gate and measured planner tests were held.

### Phase E final refinement — sustained bias, 2026-11-09
Supersedes the previous terminal-repair source hash and development figures. Extra terminal steering now triggers
from mean signed lateral error over the envelope forward delay + twice its response time constant, preserving
normal heading alignment when an individual gait-sway peak exceeds the lateral budget. This removes the walk
Doorway seed 4 regression seen with an instantaneous trigger; that run is exactly restored to its passing baseline.
Final source SHA256: `99128a5eabef8744b4caf768c8b914d5f1df2f810f7c846eb93182ac8f68f04a` (frozen for canonical reruns).
Tests **8/8 passed**. Final-hash development gates: selected atlantis_walk **20/20**, walk **20/20**, velocity_hard
**20/20**, each 20 reached and zero falls; worst RMS **23.11 / 23.20 / 15.47 mm** respectively. Selected failing
straight seed 4 now RMS **19.38 mm**, final position/yaw **21.05 mm / 0.286 degrees**. All thresholds unchanged.
Final report rewritten around the final implementation: `reports/follower_repair_v2_2026-11-09.md`.
Canonical all-four-policy reruns are coordinated separately; no held-out v2 seeds evaluated here.

### Phase E canonical final validation — 2026-11-09
Coordinator/Luna final reruns match frozen follower SHA256 `99128a5eabef8744b4caf768c8b914d5f1df2f810f7c846eb93182ac8f68f04a`:
atlantis_walk **20/20**, walk **20/20**, velocity_hard **20/20**, each 20 reached and zero falls. Worst RMS
**23.11 / 23.20 / 15.47 mm**. Velstand_flat remains **FAIL 16/20 passed/reached**, zero falls, worst RMS **43.92 mm**;
failed straight seed 0 and s_bend seeds 0, 1, 4. All tolerances unchanged; no further E edits.
Separate apartment smoke succeeded safely but took119.92 s plus dwell and 2.946 m MA30 path for a 0.5 m displacement;
this time/SPL limitation remains for honest benchmark evaluation, despite the passing selected course gate.
- Phase G matched apartment ball, atlantis_walk handoff, training-v2 walk-morphology full sweep completed: right 227/245 and mirrored-left 228/245, 0 falls; each all-success sampled region has 36 cells and 25 mm half-width per axis. Seeds/offsets are correlated and interpolation is unvalidated. Pick lift remains unvalidated and unsupported; fallen recovery 0/20 and unsupported. New contracts contain dynamics version, morphology, ball and both policy identities; real apartment validation accepted kick and refused pick/recovery. Focused tests 10 passed. Full context and limitations: reports/skills_diag.md.

## Frozen benchmark v1 — 2026-11-09 (Codex)
Created data/seeds_v1.json EXACTLY ONCE after B, main3 E gates, D and G regeneration: 200 nav / 50 kick / 50 pick, canonical SHA256 65d6fdee13bc36aeabdd7a0f9dea70ce7d247feb7c5f8b64b596094b10ff69f0. Selected atlantis_walk, training cb8561f6_nominal_v2. Closed-loop sigma 0.018955012692105925 m; step-over height0 conservatively unmeasured. All dry-reset/object-stability checks passed (generation resampled9 reset rejects,16 free-object-clearance rejects,0 stability rejects). Frozen metadata includes registry walker md5, envelope/primitives/skills/checker hashes, follower source hash, full scene closure, MA30, 1s dwell, footprint-only16-neighbor+LOS metric, mirror recorder and timeouts. This file must not be regenerated silently. GT benchmark starts next, followed by odom+fiducial and independent20episode replay; no H result yet.

## 2026-11-09/10 frozen-v1 GT result and explicit controller revision
The immutable first full GT benchmark completed: nav81/200 (40.5%, Wilson95%33.94–47.42%), SPL0.291196, one fall/200 and five frozen-checker hole incursions; kick2/50, pick0/50, all83/300. Report `bench_frozen_v1_atlantis_walk_gt_20261110_012319.json`; all300 NPZ and217 failure JSON/MP4 retained. This supersedes the earlier “GT starts next” entry. V1 data/checker remain frozen. Diagnosis found sustained pure-yaw stalls; a bounded controller revision and contract-derived kick alignment are in progress. No v2 frozen fork, v2 primary benchmark or20-episode independent replay is complete yet. Root coordinates; main_execution owns integration/explicit provenance fork, audit_repair owns controller, skills_v2 owns FSM. Full details in CODEX_COMPACTION_HANDOFF_2026-11-09_v2.md.

## Phase E — rotation-progress controller promoted for prospective v2, 2026-11-09
Canonical follower source frozen at SHA256 `c491455b732c5e3ed7469a4e1c4e0ab8bf1d7940f8dde81b0f25ceb3400b4c37`.
Pure-yaw stalls now latch the next exact measured-safe rotation level after an envelope-derived response window;
expected progress is capped by the requested angle, and a new path clears the latch. Tracking, physics, checker,
FSM, tolerances and frozen v1 seeds remain unchanged by this worker. Historical991/experimentalaf900sources backed
up under `reports/follower_candidate_dev_v1/`.
Validation: **15/15 follower tests passed**; required three-walker DEV gates each **20/20 reached/pass**, zero falls,
worst RMS selected/walk/hard **23.11 / 23.20 / 15.47 mm**. Additional unseen apartment turn scenarios seeds5–7/±90°:
promoted **6/6**, archived991 **3/6**, zero falls/holes, three original passes unchanged. One promoted held-out case
still takes63.7s/2.392m for0.5m displacement; this is not a claim of95%/SPL.80. All DEV workers exited before READY
handoff to Luna/main_execution/root for canonical E/D gates and explicitv2metadatafork. Source must stay immutable
through those runs. Report: `reports/follower_rotation_progress_promotion_v2.md`.

## Canonical E/D mechanical gates — 2026-11-09 (measurement_v2, follower c491455b)
Source SHA-256 `c491455b732c5e3ed7469a4e1c4e0ab8bf1d7940f8dde81b0f25ceb3400b4c37`; no E dev workers active before farm. Canonical Phase E: atlantis_walk PASS 20/20 (0 falls, σ 18.955 mm, worst RMS 23.111 mm), walk PASS 20/20 (0 falls, σ 18.073 mm, worst 23.197 mm), velocity_hard PASS 20/20 (0 falls, σ 14.616 mm, worst 15.467 mm), velstand_flat FAIL 16/20 reached (0 falls, σ 25.420 mm, worst 43.921 mm). Selected gate passed. Phase D atlantis_walk PASS all a–d: 100/100 complete, 0 lethal, p95 plan 0.1868 s, wall 0.1905 s, heuristic 7/7, cross-room 83/83, far pair 100%, cold start 100%; idle/busy2 result changes 7%/4%. `tests/test_planner_measured.py`: 5 passed. Artifact md5/dynamics_version and follower-source/report hashes match. Current E/D reports and logs archived in `reports/follower_v2_c491455/`; prior D report/log in `reports/planner_gate_before_final/`. No active E/Planner/test workers remain.

## 2026-11-09/10 H explicit v2 provenance fork and completed GT
Frozen v2 controller `c491455b...` required selected/walk/hard gates20/20 each,zero falls; flat16/20 FAIL and withheld. Current full D100/100 complete,zero lethal,p95plan0.1868s/wall0.1905s,heuristic7/7,cross83/83,far/cold100%,five measured tests PASS. Explicit seeds_v2 controller-only fork SHA7fc18235fce79ed717288c091cc7482d7a86d9eb4c64fd52e330915273248fed preserves every v1 task and masterseed; all300 start/goal/prepose checks and1s object stability passed,zero rejections. V1 remains immutable.
Full v2 GT completed: report `bench_frozen_v2_atlantis_walk_gt_20261110_020423.json`. Nav187/20093.5% (Wilson89.20–96.16%), SPL0.66610 (CI0.63755–0.69065),zero falls (Wilson upper1.8845%),five hole incursions. GT targets FAIL(success<95%,SPL<0.8,holes>0), despite improvement from v1's81/200. Kick0/50: tighter contract-derived approach often stuck (40) and ball disturbed before kick (49); actual precision reachability was not established by its necessary guard. Pick0/50 unsupported. All300NPZ,113failureJSON and113MP4 retained; no recorded export errors.
Full unchanged-v2 odom+fiducial run active session77770 (`bench_frozen_v2_atlantis_walk_odom_fiducial.log`), latest199/300 physics episodes. Independent20GT reference replay, walk/hard nav200 comparisons, full pytest and FINAL_REPORT remain pending. Separate user-requested bounded pick/recovery remedy agent prepares isolated exact70mm/50g-box>=20mm-lift and matched-morphology/handoff diagnostics; no diagnostic compute yet during8-process H lease. User permits genuine model gaps and future training; no new training or gate relaxation now.

## 2026-11-09/10 H odom+fiducial completed; independent replay begun
Full v2 odom+fiducial report `bench_frozen_v2_atlantis_walk_odom_fiducial_20261110_022933.json`: nav75/20037.5%(Wilson31.09–44.39%),SPL0.257415(CI0.21280–0.30545),threefalls1.5%,tenholeincursions; targetsFAIL(success,SPL,holes). Kick0/50,pick0/50. All300NPZ,225failureJSON and225nonemptyMP4 retained. Independent20GT reference replay active session79386 (`replay_frozen_v2_gt.log`) under unchanged frozen criterion. One serial CPUprocess lease granted isolated pick/recovery diagnostics concurrently; no shared source/contract edits during frozenv2. Policy comparisons and finaltests/report still pending; see compaction checkpoint for exact ownership and commands.


## 2026-11-09 PAUSED by user; latest authoritative Claude handoff

Read [CLAUDE_RESUME_2026-11-09_PAUSED.md](CLAUDE_RESUME_2026-11-09_PAUSED.md) before resuming. It supersedes older ACTIVE/RUNNING snapshots. All project jobs stopped: replay session79386 interrupted exit130; main verified zero project worker/render/ffmpeg/replay/trial/resource-tracker/spawn PIDs. No policy comparison launched. Independent replay has NO final output/verdict; trust remains UNRESOLVED. Partial logs preserved. User accepts failed existing models and future training, but no training now; integrate proven fixes with explicit qualification/versioning.

Completed v2 GT nav187/200 (93.5%, Wilson89.20–96.16%), SPL0.666099,0falls,5holes; odom+fiducial75/200 (37.5%, Wilson31.09–44.39%), SPL0.257415,3falls,10holes. Both fail success/SPL/zero-hole targets. Kick0/50,pick0/50 each mode. All600traces and338failureJSON/nonemptyMP4 retained. Frozen seeds_v1/v2, dynamics, scorer, skill contracts preserved.

New corrected isolated pick/recovery tools implemented and scoped-checked; report pick_recovery_remedy.md. Actual-box pick80persisted trials: both models0/20 in each morphology, all no falls; full-collision contact restored but lift still absent. Recovery210trials: canonical walk all3policies fail; fullcollision velstand35/35 with0/35falls during2s selected-walker handoff. This is a promising diagnostic, NOT a qualified deployment fix: no canonical skill-support flip. Standup models recover sitting states only (5/15 immediate each), not fallen states. Installed-reference zero-weight payload hook behavior verified, historical training manager unverified.

Pending on explicit resume: complete independent20replay; eligible same-seed walk/hard nav comparisons; qualify/version any proven geometry or integration fix; investigate GT efficiency/safety, odometry drift and kick approach; final full suite and FINAL_REPORT. Last broad suite140pass/1skip/1xfail predates latest changes and is NOT a fresh final result. Handoff contains exact paths, provenance, commands, results and ready-to-paste Claude prompt.

## 2026-11-10 RESUMED and COMPLETED (Claude coordinator); FINAL_REPORT.md written
Read [FINAL_REPORT.md](FINAL_REPORT.md). Supersedes the PAUSED snapshot. Targets still FAIL in both modes; no criterion changed.
- Provenance re-check at resume: all frozen hashes PASS (`provenance_check_resume.md`); baseline suite 161 passed/1 skipped/1 xfailed.
- Independent 20-episode mjlab replay: **PASS 2/20 disagree (0.10, threshold <=0.10; Wilson 0.028-0.301)**, criterion 983bd10d... unchanged (`replay_frozen_v2_gt_final.{json,md}`). Both disagreements are mjlab references that went non-finite after falling into the stair hole (nav_014, nav_142); `tools/replay_compare.py` now counts non-finite references as disagreements (conservative), previously it crashed with KeyError final_pose. 18/18 finite episodes agree.
- Policy comparisons (v2 seeds, GT nav, `policy_comparison_v2.md`): walk 173/200, SPL 0.604 (success worse, p=0.020); velocity_hard 181/200, SPL 0.632, 8 falls (falls worse, p=0.004). atlantis_walk stays selected. velstand_flat withheld.
- Shortfall analysis (`shortfall_analysis_v2.md`): GT failures = 5 start_lethal lattice-snap refusals + 5 hole flags (2 genuine) + 3 goal limit cycles; GT SPL dominated by in-place turning on kinked plans (cap 0.712 even at 200/200). Kick fails on pre-kick ball contact during approach.
- **Defect fixed: fiducial camera qpos copied as raw prefix** (training model has interleaved backlash joints; camera orientation error median 27.6 deg). Now mapped by joint name (FIDUCIAL_RENDER_VERSION 2; median 1.42 deg; infer images bit-identical). Backups `bak_2026-11-09_camfix/`. Versioning: `tools/fork_seeds.py` records `localization_sha256`; `nav/bench.py` enforces it and refuses non-GT runs on pre-v3 seed files. F gate rerun (`localization_gate_render_v2.json`): fiducial final XY 7-18 mm (was 49-134). New seeds `data/seeds_v3.json` canonical 49fd00a0... (identical tasks; 300/300 revalidated). v3 GT 10/10 traces bit-identical to v2 (GT unaffected).
- **v3 odom+fiducial** (`bench_frozen_v3_atlantis_walk_odom_fiducial_20261110_070315.json`): nav **121/200 = 60.5% (Wilson 53.6-67.0%)**, SPL **0.4188**, 1 fall, 8 holes, kick/pick 0/50; paired vs v2 72 gained / 26 lost, McNemar p=3.7e-6. Still FAILS 85%/0.7/0 holes. 300 NPZ, 179 failure JSON+MP4.
- Final full suite run last: **163 passed, 1 skipped, 1 xfailed** (`pytest_final_2026-11-10.log`).
- Next (not implemented): A1 lattice snap, A3 converged-fix done gate, A4 goal repair, A5 smoothing/turn-while-walking (main SPL lever), A6 cliff-aware motion, A7 kick approach; A8 full-collision velstand recovery needs new dynamics version + B-H. Training specs for pick/recovery/kick/turning in FINAL_REPORT section 10.

## 2026-11-09/10 Codex Prompt 0.5 housekeeping complete

Implemented planner/mapping/mapsigma SHA provenance guards (older seeds without keys unchanged), dev episode selection with DEV and full_frozen_suite=false, dev forks, actual comparison runtime policy/MD5/sigma/follower+gate hashes, and paired >=60s all-room localization route. Added exact deterministic trace parity tool.

- Core focused tests: 15 passed. Long-route focused: 5 passed. Trace comparator: 3 passed.
- Full serial suite: 175 passed, 1 skipped, 1 xfailed, 95.95s (`pytest_full_idle.log`). Earlier concurrent run failed only synthetic planner caller wall p95=0.200843864s at unchanged <0.2s; retained in `pytest_full.log`.
- v3 GT nav000..009 vs frozen_v2 GT: 10/10 byte-identical qpos/t; deterministic metadata including commands/estimates/transitions identical.
- v3 odom+fiducial nav000..009 vs frozen_v3: same 10/10 exact result. Only plan_time_s, plan_times_s, wall_s profiling excluded. Native graphics needed; failed sandbox attempt preserved.
- Real dev fork `data/dev/seeds_v4dev_housekeeping.json`: 300/300 unchanged tasks validated.
- Long route seed0: 380.8 simulated seconds, all6 rooms reached, no fall. Diagnostic FAIL on existing localization error: living terminal63.49mm >50mm. Full paired statistics and series in localization/apartment_long_seed0.json. Short300step JSON byte-identical to original backup. This is infrastructure evidence, not a localization behavior fix.
- All3307 protected frozen artifact hashes rechecked unchanged after housekeeping.

Backups: reports/bak_2026-11-09_housekeeping_core, _localization, _parity, _test_outputs, reports/bak_2026-11-09_orchestration; each has SHA256SUMS. Final source SHA256 in source_hashes.json. No benchmark thresholds changed.

Source hashes: {"nav/bench.py": "fcc6a6c5f6e27fee4e4668dfcee2d6512d74db28a4b930058ab2136168b90a68", "tests/test_housekeeping_core.py": "3f9542fb39aefb484835975f9a7bfa4322446855ce7410a43d2c7fc2a1017713", "tests/test_localization_long_gate.py": "7e91478e29d7de4076ed2885bed1184914d4aa0f56db7fad66714922d0bc86f9", "tests/test_trace_parity.py": "577194adedba63f43a8c56d0f0a6d38890d0dd17d5df2f0e06854bdea6bc18d8", "tools/check_trace_parity.py": "6720a30ed2e76056a901b528e2f6b0510d8d48977b6e9b893a046a85a2cbd196", "tools/fork_seeds.py": "2e0e4ae94b41bd95564f20dafd09feed7358048c706045354a3ce5f496f6ab81", "tools/localization_gate.py": "0c768dad92b351734f96711a839891d903984eb2c207292f56398e6eb8dcb527", "tools/run_benchmark.py": "c3932e37720c5818a9da6dfbe108c85741c4fe742ff7dd60252404bfa55cf5ef"}

## 2026-11-10 PAUSED by user — Codex housekeeping and A5 experiment

No active project jobs (pause_processes.json); current planner gate finished normally before pausing. Read CLAUDE_RESUME_2026-11-10_A5_PAUSED.md before resuming.

Prompt0.5 COMPLETE: planner/mapping/mapsigma guards, explicit DEV episodes/forks, actual comparison runtime, paired all-room long localization diagnostic and exact trace parity tool. Full serial housekeeping suite175passed/1skipped/1xfailed95.95s; GT10/10 and odom+fiducial10/10 qpos/t and deterministic metadata bit-identical. Long route380.8s6/6rooms0falls catches existing63.49mm terminal drift; short loop output byte-identical.

A5 diagnostics and fixed60 baseline complete:47/60success,SPL0.5555145243,0falls,5holes;nav125258flips117.166/min38.1spureyaw6305.596deg/net114.890deg16.112mmsway1.32005pathratio. Final combined controller REJECTED by complete E:Atlantis13/20,walk0/20,hard20/20,0falls. Source archived under reports/dev/A5/rejected_combined; canonical follower restored SHA256c491455b732c5e3ed7469a4e1c4e0ab8bf1d7940f8dde81b0f25ceb3400b4c37; rejected heading tests archived/removed.

Smoothing-only planner STILL ACTIVE for unfinished bounded component diagnostic, not accepted A5. Canonical follower fresh E20/20 each3walkers. Complete smoothing-only planner gate PASS100/100,0lethal,planp95~0.1944s/wall~0.1986s,heuristic7/7,cross83/83,far/cold100%,all load/deterministic sweeps completed;reports/dev/A5/smoothing_only/gates/. No dev A5 fork/60run/aftermetrics/comparisonMP4 yet. Planner/gate/test restoration and final full-suite result remain pending. Do NOT run canonical frozen benchmarks with active candidate planner.

All3307 protected artifact hashes rechecked unchanged at pause. Current source hashes/gates in reports/dev/A5/pause_state.json; backups and exact continuation commands in handoff. STATUS-before-pause backup reports/bak_2026-11-10_pause/SHA256SUMS.

## 2026-11-10 A5 FINAL (Claude): combined REJECTED; smoothing-only component FAILS steering criteria; tree restored
Read [dev/A5/A5_RESULTS_2026-11-10.md](dev/A5/A5_RESULTS_2026-11-10.md). Combined heading-aware A5 rejected by unchanged E (13/20, 0/20, 20/20; never benchmarked). Smoothing-only DEV component run (fixed60, GT, canonical follower c491455b, fork data/dev/seeds_v4dev_A5_smoothing_only.json): success 47→53/60, SPL 0.5555→0.7168, falls 0→0, holes 5→0, median flips/min −20.4% (needs −75%), pure-yaw fraction −32.2% (needs −50%), nav_125 148.1 flips/min (≤30 FAIL), 2.5 s pure yaw (PASS); lost nav_045 and nav_197 (pos_tol). steering_criteria_passed=false. Video reports/dev/A5/nav_125_before_after.mp4 is captioned smoothing-only and is not a combined comparison. Candidate archived (bak_2026-11-10_A5_smoothing_archive); planner/gate/measured test restored to a4eddaea/43aa2ae8/289d574c, smoothing test removed. Final suite 187 passed/1 skipped/1 xfailed (72.88 s). Immutable 3307/3307 unchanged. No frozen targets claimed met. STATUS backup bak_2026-11-10_A5_final_status.

## 2026-11-10 A1 start/goal lattice snap repair — ACCEPTED (DEV evidence only; no frozen target claimed)
Change: `nav/planner.py` bounded endpoint snap repair. When the true start is free but its nearest 4 cm lattice node is lethal, it uses the nearest non-lethal node within sqrt(2)·spacing (5.66 cm, derived from lattice geometry and not tuned on the seeds). That node must be joined by a straight connector whose points (≤ res/2 apart) are all non-lethal on the 2 cm inflated map, and it must have at least one usable edge at the true heading. The connector is prepended to the path. Same logic for goals, applied only when no free node lies inside goal_tol: the heuristic tolerance becomes the repair radius (consistency kept) and the connector is appended. A lethal true pose still fails closed. `PlanResult.start_snap/goal_snap` and connector counts are recorded, and `nav/fsm.py` adds them to the plan log. `nav/seed_validation.planner_start_check` is used by `tools/fork_seeds.py` for future forks (records `planner_start_nodes`).
- Hashes: planner `6ad1b67f…`, fsm `5a23235a…`, seed_validation `16b069ee…`, fork_seeds `125935e4…`. Full list: `reports/dev/A1/source_hashes.txt`. Backups with SHA256SUMS: `reports/bak_2026-11-10_A1/`, test outputs in `reports/bak_2026-11-10_A1_test_outputs/`. fork_seeds was changed again after the fork, only to pass primitives positionally; behaviour is identical.
- Tests: new `tests/test_planner_snap.py` 5 passed: lethal true start refused, free start with lethal node plans, connector swept check, goal repair ends on true goal. Full suite **192 passed, 1 skipped, 1 xfailed** (76.9 s).
- Planner gate (DEV candidate copy `reports/dev/A1/planner_gate_candidate.py`, `--reports-dir` with overwrite refusal; canonical gate not run): **PASS**. 100/100 complete, 0 lethal, plan p95 0.1870 s, wall p95 0.1911 s, heuristic 7/7, cross-room 83/83, far 100% (wall p95 0.193), cold start 100%. Deterministic N=60000 wall p95 0.2009 and N=80000 0.2467 are informational, as before.
- DEV fork `data/dev/seeds_v4dev_A1.json` (v4dev-A1, sha `9611168b…`): 300/300 unchanged tasks valid; planner start nodes 294 nearest, 6 repaired (exactly nav_063/066/103/115/137 and kick_036).
- DEV GT run (`reports/dev/A1/bench_DEV_A1_gt_20261110_175330.json`, `full_frozen_suite=false`; selection in `episode_selection.json`, rng 20261110), compared with frozen v2 GT:
  - 5 start_lethal targets: 0/5 → **5/5 success**. Final errors 44.3/31.1/45.5/47.6/38.9 mm, with 1.7/0.4/3.9/4.3/4.4°. All are inside the 50 mm tolerance but 3 of them by less than 7 mm.
  - kick_036: now plans and walks (3.69 m), still fails kick (`ball_moved_before_kick`, skill_not_started). That is the A7 cause, not A1.
  - Regression: 30/30 successful, final poses **identical** to frozen v2 (no repair triggered).
  - 0 falls, 0 hole incursions.
- Projected effect on frozen GT (not measured): 187 → 192/200 if the full run reproduces this. SPL is unmeasured on a full run.
- 3307/3307 immutable artifacts unchanged.

## 2026-11-10 A4 goal repair — NOT ACCEPTED; PAUSED by user during attempt 3. Follower restored to c491455b.
Read `reports/CLAUDE_RESUME_2026-11-10_A4_PAUSED.md`. Diagnosis (read-only Opus agent, offline replay reproduces 195/195 GT traces with 0 command mismatches): `reports/dev/A4/diagnosis.md`. `_check` repairs position before yaw, and `_approach` only faces the goal and walks. Each corrective turn moves the trunk a median 52.9 mm against a 48 mm budget, which causes the limit cycle. 184/187 GT successes also enter this repair.
The design: coarse yaw first at the smallest-turning-circle level, then translation with no turning using exact safe commands, a turn-displacement budget, and a reversal guard. All constants come from the envelope rows and the 0.6 s primitives.
- Attempt 1 (`attempt1/`, follower 18cbfc47): E gate 20/20 ×3 walkers and held-out 20/20 ×3, but closed-loop stop rule ended steps before the gait started. Targets 3/3, regression 23/30. **Rejected.**
- Attempt 2 (`attempt2/`, 5ecaab4a): open-loop quanta + fine turn when translation can't make room. E 20/20 ×3, held-out 20/20 ×3. Targets 3/3, regression **23/30** (nav_006/007/010/050/093/130/181 follower_stuck). Steps measured from a standing reset barely move or yaw the trunk once the robot has settled at the goal, and the yaw guard then stopped the episode. **Rejected.**
- Attempt 3 (`attempt3_interrupted/`, c0eb30e3): non-moving commands banned per path, guards back-to-back only and falling back to the legacy repair, original max_fixes. E 20/20 ×3 walkers and held-out 20/20 ×3 (0 falls, RMS ≤23.2 mm), dev fork OK (sigma 0.018843). The run was **interrupted** after 55 of 73 episodes. Targets 3/3, regression_A1 28/28 completed, fresh held-out regression 23/24 completed: **nav_094 failed with a new stair_hole**. 8 nav + 10 kick not run. Incomplete, with one new failure: **not accepted**.
- Tree: follower `c491455b`, test_follower unchanged, test_follower_repair archived (not in tests/). A1 stays accepted (planner 6ad1b67f). Full suite on the restored tree: **192 passed, 1 skipped, 1 xfailed** (83.1 s). Immutable 3307/3307 unchanged. No project processes running. Hashes in `reports/dev/A4/pause_source_hashes.txt`.


## 2026-11-10 USER PAUSED — Codex A5 smoothness follow-up

Read [Codex-to-Claude handoff](CLAUDE_RESUME_2026-11-10_CODEX_A5_SMOOTHNESS_PAUSED.md) and [continuation prompt](CLAUDE_CONTINUE_A5_CODEX_PROMPT_2026-11-10.md). Current working tree is candidate6 follower SHA256 `95611a14653297e73e7574bb0a9ab6130c630995a590cf8be8c905c6b2c84756`; candidate7 has NO edits. No accepted A5 fix. C6: full E20/20 each3walkers,0falls; full planner100/100 PASS; unchanged300-task DEV fork valid. Fixed60 DEV:58/60success,SPL.726693,0falls,0holes,median42.15flips/min;nav12554.85flips/min11.5spureyaw. Fails steering criteria and loses nav110/nav129. C1 preserved with60/60success,SPL.772714 but also fails reversal criteria. Do not promote either as accepted/frozen. Full final suite and new comparison video NOT done. All known simulation sessions completed and agents stopped. Pause snapshot `dev/A5_fix_20261110/pause_20261110T203107Z/`; all3307/3307 protected artifacts rechecked unchanged. Proposed next experiment is bounded measurement-uncertainty inverse-command continuity; unimplemented/unvalidated. Preserve thresholds,seeds,policies and evidence.


## 2026-11-10 ~16:10 PST v4 push PAUSED by user
Read [CLAUDE_RESUME_2026-11-10_v4_PUSH_PAUSED.md](CLAUDE_RESUME_2026-11-10_v4_PUSH_PAUSED.md). Tree follower restored to C1 (98b188d0); everything else identical to C6 pause snapshot; 3307/3307 immutable unchanged. Pick BLOCKED, recovery BLOCKED (reports/dev/v4/pickrec/DECISION.md). A5 C7 NOT ACCEPTED (E walk 19/20, RMS 37.6 mm). A3/A4/A6/A7 not completed. No v4 freeze.

## 2026-11-10 16:22 PST — continuation step 1 complete: A5 CLOSED FAILED for v4

Carry C1 forward as integration fallback; A5 steering criteria remain failed.
[Decision](dev/v4/nav/a5_close_20261111T0020Z/DECISION.md) and
[remaining plan](dev/v4/CONTINUATION_PLAN_20261110.md).
Offline saved-trace replay: 60 C1 + 20 C7 traces reproduce all commands exactly,
zero mismatches. C7 suppresses 50 canonical inverse choices on failing walk
S-bend seed 4; first C1/C7 command divergence 0.5 s. Historical C1 finishes that
course in 19.0 s / RMS 20.3 mm / zero repairs; C7 times out at 35 s / 37.6 mm /
five repairs. Full causal dynamics not isolated; no new candidate accepted.
Fresh focused tests **37 passed**; all **81 source/config files unchanged**;
protected hashes **3307/3307 unchanged**. No physics or benchmark jobs launched;
SIM_LOCK observed absent. No new full-suite/frozen benchmark claim. Next: review
partial A4/A6 work on C1, then complete remaining streams and validation.

## 2026-11-10 16:40 PST — continuation step 2 complete: A4 ACCEPTED, A6 candidate rejected

[Results/evidence](dev/v4/nav/a4a6_resume_20261110T1627/RESULTS.md).
Live follower is C1 plus measured translation goal repair and settled-turn
escalation, SHA256 `b3109d2468e2f7836878a3c0f3057502c7ee83138ed23cca5e821549e7c9b8a1`.
New standalone repair/settled-turn tests installed. Pre-edit backup:
`bak_2026-11-10_A4A6_resume_1627/` with verified SHA256SUMS. Other existing
source/config files unchanged; no A6 mapping/cliff promotion.

A4: 54 focused tests; E 20/20 each three walkers and held-out 20/20 each;
strict unchanged-task fork 300/300; full planner gate PASS 100/100, zero lethal,
wall p95 194.7 ms. nav_125 smoke success. Exact fixed-60: C1 60/60 / SPL
0.772714 -> **60/60 / 0.782622**, median goal time 34.79 -> 34.46 s, no lost
successes, zero falls/holes. Original A4 set: targets 3/3, regression_A1 30/30,
regression_fresh 30/30 (nav_094 succeeds), kick 1/10; all 73 zero falls/holes.
Six paired direct apartment turns pass for C1 and isolated settled-turn change,
identical durations (no demonstrated speedup there). Full serial suite:
**224 passed, 1 skipped, 1 xfailed**, 78.94 s. Atlantis planning sigma changes
0.0187023317 -> 0.0187380117 m; recorded in the new fork, not capped.

A6: refreshed p99.5 calibration from current gates, provenance-verified reuse of
independent policy turn probes. Candidate radius 159.570 mm, one component,
six open doorways, but **nav_070 start and nav_158 goal are lethal**. Rejected
at exhaustive unchanged-task static preflight; no resampling or smaller margin.
Isolated A6/map/follower checks: 51 passed. No A6 physics/benchmark acceptance;
live MAP_VERSION remains 4. Stop-and-replan integration also remains unfinished.

Matching gates: `dev/v4/nav/a4a6_resume_20261110T1627/repair_settled/gates`.
Matching DEV fork: `data/dev/seeds_v4dev_A4_resume_20261110T1627.json`.
All results are DEV. The 63-nav A4 run's planner p95 is 214.8 ms under three
workers, so the broader timing target remains unresolved. Next A3/A7, then
integrated validation and full GT/odom benchmarks. A5 still failed; pick and
canonical recovery still blocked. Protected-file and final worker audit:
`dev/v4/nav/a4a6_resume_20261110T1627/`.


## 2026-11-10 17:57 PST — A3 rejected/restored; A7 approach accepted in DEV

- A7 unchanged canonical kick tasks: **2/50 -> 21/50**, 20 gains / one regression (`kick_009`); pre-kick disturbances **48 -> 6**, not started **38 -> 29**, zero falls/holes.
- Approach-inclusive flat sweep: **150/200**, Wilson 95% **68.6–80.5%**, zero falls or in-place turns inside ball inflation. All successes also meet >=0.30 m displacement. Robust kicking is still below target.
- A3 v1 **22/60**, v2 **21/60** versus **21/60** baseline; rejected despite 30/30 long-route room terminals passing. V2 has 54 completion timeouts. Original localization restored exactly; A3 remains open. Saved pixel-bias hint is unvalidated and unexecuted.
- Final tree: accepted A4 follower plus A7. Strict unchanged **300/300** DEV fork `data/dev/seeds_v4dev_A7_final_20261110T1646.json`; **10 GT + 10 odom** trajectories bit-identical to accepted baseline.
- Full serial suite **232 passed / 1 skipped / 1 xfailed**; **3307/3307** protected hashes unchanged. All workers exited, SIM_LOCK released, no agents launched. Failed spawn startup excluded and cleanup verified; stale provenance fixture fixed, full suite rerun.
- Detailed evidence: [A3/A7 results](dev/v4/a3a7_20261110T1646/RESULTS.md). A5 remains failed, A6 unaccepted, pick/recovery blocked. Full v4 benchmarks/video/table are still pending.

## 2026-11-11 00:15 PST — v5 Stage 1 (A3 localization): ACCEPTED in DEV (learned odometry is the lever; anchors hurt)

New mode `learned+fiducial` with `--ablation {legacy(default),1a,1a+1c,all,1b+1c}`. Promoted: nav/localization.py, nav/fsm.py, nav/episode.py, nav/anchor_graph.py (new), tools/run_benchmark.py; data/learned_odom/ (ONNX c98c1452…, 162k-param 1D-CNN, cmd_{k-1}), data/anchor_graph/, data/convergence_threshold.json (derived on apartment_long only). Backup: `bak_2026-11-10_A3combined/` (SHA256SUMS verified). Training data: 4.9 h sim on independent routes (seeds ≥1e6), no benchmark episodes.
Offline val (6 held-out routes): per-0.1 s RMSE dx/dy 0.42/0.36 mm vs PyOdometry 2.14/3.34 mm; drift 0.17 vs 0.68 cm/m.
Gates: strict fork 300/300 (`data/dev/seeds_v5dev_A3_20261111T075140Z.json`); GT10 + odom10 trajectories bit-identical (only additive trace metadata — coordinator ruling per v4 restored_parity precedent, `dev/v5/A3/gates/gate2_ruling.md`); apartment_long 5 seeds 30/30 terminals, max 37.4 mm (odom+fiducial on same route: 50–129 mm, fails); full suite 232p/1s/1xf; protected 3307/3307.
Fixed A3 selection (60): baseline 21/60 SPL 0.2443, 36 false done, 3 falls, 2 holes →
| ablation | succ | SPL | false done | falls | holes | lost |
|---|---|---|---|---|---|---|
| 1a | **55/60** | **0.709** | 5 | 0 | 0 | 0 |
| 1a+1c | 54/60 | 0.693 | 5 | 0 | 0 | 0 |
| all (1a+1b+1c) | 33/60 | 0.441 | 3 | 0 | 0 | 10 |
`all` passes the stated gate, but the anchor graph (1b: look_heading + anchor slow-down) costs 21 successes vs 1a+1c; 1c adds nothing measurable here. Recommended carry-forward: `1a` (or `1a+1c`). Legacy modes unchanged. Caveats: DEV only, atlantis_walk only, thresholds in-sample on gate seeds (held-out route seeds pass). Evidence: [gates/RESULTS.md](dev/v5/A3/gates/RESULTS.md). Obsolete post-promotion tests: 2 in dev/v5/A3/combined/test_combined.py, 1 in integration/test_learned_localizer.py.

**2026-11-11 user decision:** carry forward **`--mode learned+fiducial --ablation 1a`** (learned odometry alone) for all later stages and the Stage 5 integrated benchmarks. Stop-and-look (1c) and anchor graph (1b) code stays in the tree but is not used; legacy modes remain the defaults.

## 2026-11-11 21:27 PST — v5 Stage 2 (A5 MPPI follower) PAUSED by user; not accepted

- Candidate 1 `reports/dev/v5/A5/follower_mppi.py` (SHA ff16b6e9…, Haiku-built): focused tests 55/55, kinsim course flips (|wz|>0.02) RPP 5.2 → MPPI 0.0/min. **FAILED E gate rung 2: walk 7/20, worst RMS 51.7 mm (limit 30), mostly goal-yaw failures + some tracking timeouts.** Rungs 3–7 not run. Evidence: `dev/v5/A5/ladder/rung2_egate/`, `dev/v5/A5/ladder/RUNLOG.md`.
- Ladder scripts ready in `dev/v5/A5/ladder/` (README; run_tool.py hash-substitution approved in COORDINATOR_RULING_run_tool.md). A4 follower steering on fixed-60: flips −25% (FAIL), nav_125 84.4/min (FAIL), pure-yaw/ratio criteria PASS.
- Opus rescue (candidate2) was stopped mid-diagnosis: `dev/v5/A5/candidate2/` holds diagnostic scripts and a draft follower, no DIAGNOSIS.md, no E-gate result. Its interrupted dev batch was killed; SIM_LOCK released; no sim processes remain.
- Live tree unchanged: nav/follower.py b3109d24 (A4). Nothing promoted. Resume: finish candidate2 diagnosis → E gate → rungs 3–7.
- 2026-11-11 ~21:45 PST: routing switched to Codex (GPT-6.1 Sol / GPT-6 Luna) per user. The Sol rescue job ended in 1m30s without doing any work: **the Codex sandbox treated ~/microduck/nav as read-only** (could not create candidate3/ or take SIM_LOCK). Fix before resuming: launch Codex with write access to the nav root. Stopped for the day; SIM_LOCK absent, no sim/Codex processes remain, nav/follower.py still b3109d24.


## 2026-11-12T07:49:04.172977+00:00 — v5 Stage 2 A5 MPPI ACCEPTED IN DEV; optional smoother REJECTED

Promoted exact follower `fe8c2f50…` after original E20/20 x3, strict300/300, full planner PASS100/100 (0 lethal; p95 187.7 ms), nav_125 and exact three-worker fixed60: **60/60, SPL0.797127 versus A4 0.782622, 0 falls/holes**. All original steering criteria pass: median19.82 flips/min (−79.0% vs original), pure-yaw fraction0.1059 (−66.4%); nav12523.99 flips/min and3.70s pureyaw. Full suite232p/1skip/1xf after fixing a stale expected-radius fixture without changing its tolerance or safety formula; protected3307/3307 unchanged. Optional A1-compatible smoothing transplant rejected at original primitive poses_consumed focused assertion (36p/1f), never promoted; live planner6ad1b67f unchanged. Initial spawned60 provenance exceptions are invalid infrastructure artifacts; corrected strict spawned loader rerun is authoritative. [Results](dev/v5/A5/RESULTS.md). Stages3–5 not run.

## 2026-11-12 PST — v5 Stage 3 (A6 heading-aware cliff footprint): ACCEPTED in DEV, promotion pending user approval

Measured oriented footprint (hull 72/82/113/108 mm front/back/left/right, full p99.5 cross-track 47.5 mm perpendicular to travel, 217 mm turn disk; no margin reduced) plus a planner with an ENav two-stage check (cheap check on every expansion, precomputed conservative swept-edge table, exact connectors, fail closed, verified partial paths only). Static preflight 650/650 poses over the unchanged 300 tasks, **nav_070 start and nav_158 goal now safe and plannable**. Gates: focused 151/151; E 20/20 x3; strict fork 300/300 (`data/dev/seeds_v5dev_A6_20261112T211130Z.json`); planner gate PASS a–d (p95 187 ms); nav_125 pass; fixed-60 GT 60/60 SPL 0.7949 (A5 0.7971), 0 falls/holes, no lost successes; steering criteria pass; 13 original hole episodes 0 holes in GT (13/13) and odom 1a (11/13; both misses identical on the live tree without A6); full suite 232p/1s/1xf; protected 3307/3307. One ruling for review: the planner-gate reference starts from the same A1-repaired node (`dev/v5/A6/COORDINATOR_RULING_reference_cost.md`). Live files backed up to `bak_2026-11-12_A6/`; the overwrite was blocked by the permission classifier, so the live tree is still MAP_VERSION 4. [Results](dev/v5/A6/RESULTS.md).

**2026-11-12 PROMOTED (user approved the planner-gate reference ruling and promotion).** Live: nav/planner.py 5b4f101d…, nav/mapping.py 9c5625db… (snapshot 58e7c435… plus a default hash-pinned calibration path `reports/dev/v5/A6/geometry/calibration_v1.json` 90d4de10…; env var override kept; fails closed on mismatch), new nav/a6_footprint.py e74fd85b…; **MAP_VERSION 5**. Backup `bak_2026-11-12_A6/`. Live tree without loader: full suite 232p/1s/1xf, protected 3307/3307. Current fork `data/dev/seeds_v5dev_A6_20261112T211130Z.json`; gates `dev/v5/A6/ladder/partial_v1_20261112/rung2_egate/gates`. Evidence `dev/v5/A6/promotion/`. Next: Stage 4 (skills plan doc), Stage 5 integrated validation.

## 2026-11-12 — v5 Stage 4 (skills training plan) DONE — document only
`reports/dev/v5/SKILLS_TRAINING_PLAN.md` (drafts in `dev/v5/skills/drafts/`, written by parallel Haiku 5.5 agents; coordinator spot-checked citations). OP3-style (Haarnoja et al. 2024) per-skill teacher plans for get-up, kick, box pick + shared setup; PPO/rsl_rl in mjlab; distillation optional. Nothing implemented, trained, simulated or promoted; no live files changed; no SIM_LOCK taken. Recommended order get-up → kick → pick. Owner decisions before training: (1) collision geometry — get-up/pick only work on full-collision geometry, adopting it invalidates nav envelopes/provenance; (2) BAM 1.75 A limiter ambiguity (sim.py:46 vs :239); (3) Atlantis Slurm association status. All new weights/DR ranges/compute are TBD — measure. Next: Stage 5 integrated validation.

## 2026-11-12 23:04 UTC — v5 Stage 5 integrated validation DONE (DEV)
Live combined tree (A3 learned+fiducial ablation 1a, A5 MPPI fe8c2f50, A6 planner 5b4f101d / mapping 9c5625db, MAP_VERSION 5); nothing promoted or changed in nav/. Ladder all pass: focused 88/88; E 20/20 ×3; strict fork 300/300 → `data/dev/seeds_v5dev_FINAL_20261112T215037Z.json`; planner gate PASS a–d (p95 188 ms); nav_125 ok; fixed-60 60/60 SPL 0.7949; suite 232p/1s/1xf; protected 3307/3307.
Full benchmarks (DEV): **GT nav 199/200 SPL 0.7976 (target 0.80: FAIL by 0.0024)**, 1 fall (nav_167, new), 0 holes. **Odom nav 179/200 (89.5%) SPL 0.7152: PASS** (all 21 misses 5.0–8.4 cm terminal error), 0 falls/holes. Kick GT 19/50 (A7 accept 21/50), odom 9/50; pick 0/50 (blocked). Video: `tools/run_comparison_serial_video.py` not used (hardcodes 4 workers > cap); before/after renders from traces instead. [FINAL_V5](dev/v5/FINAL_V5.md), evidence `dev/v5/final_20261112/`.

## 2026-11-13 v6 P2 c3 PROMOTED (user approved)
follower 98f0a56c (MPPI + path smoothing + final-approach tail), mapping 1ea6b54f (calibration reports/dev/v6/P2/c3_recal/calibration_c3_v1.json). DEV GT200 199/200 SPL 0.8183 (CI 0.8029-0.8327), 1 fall nav_143, 0 holes. Suite 232p/1s/1xf, immutable 3307 ok. Gates reports/dev/v6/P2/c3/rung2_egate/gates; fork data/dev/v6_p2_c3recal_20261114T034029Z.json. Backup reports/bak_2026-11-13_v6_c3recal_20261114T042051Z. Record reports/dev/v6/promotion_c3recal_20261114T042051Z/promotion.json.

## 2026-11-14 v6 skills: kick Phase 1 closed
Local kick training pipeline built (duckmac variant D, nav-runtime parity), pilot 10 min = no regression (914/916 of 980 vs Pollen 915/912). Nothing promoted; nav stack unchanged. See dev/v6/skills/kick/RESULTS.md.
