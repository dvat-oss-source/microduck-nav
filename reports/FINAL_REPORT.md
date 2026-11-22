# Microduck navigation: FINAL REPORT

Status: FINAL, 2026-11-10. Draft from existing reports, completed with the resumed-session results: independent replay verdict, policy comparisons, shortfall analysis, fiducial-camera fix and v3 odometry benchmark, final test suite.
Workspace: `~/microduck/nav` (simulation only, CPU MuJoCo, training dynamics `cb8561f6_nominal_v2`).
Authoritative state source: `reports/CLAUDE_RESUME_2026-11-09_PAUSED.md`. Numbers below were cross-checked against the frozen v2 JSON reports and the v3 odometry report.

---

## 1. Executive summary

**The frozen targets FAIL in both localization modes.** Stated plainly:

| Mode | Success (target) | Result | SPL (target) | Result | Falls (target <=2%) | Stair-hole incursions (target 0) |
|---|---|---|---|---|---|---|
| GT | >=95% | **187/200 = 93.5%, FAIL** | >=0.80 | **0.6661, FAIL** | 0/200 observed (Wilson upper 1.88%) | **5, FAIL** |
| odometry + fiducial, v2 (pre-fix camera) | >=85% | 75/200 = 37.5%, FAIL | >=0.70 | 0.2574, FAIL | 3/200 = 1.5% (Wilson upper 4.32%) | 10, FAIL |
| **odometry + fiducial, v3 (fixed camera, current)** | >=85% | **121/200 = 60.5% (Wilson 53.6-67.0%), FAIL** | >=0.70 | **0.4188 (CI 0.370-0.469), FAIL** | 1/200 = 0.5% (Wilson upper 2.78%) | **8, FAIL** |

- Kick 0/50 and pick 0/50 in both modes (Wilson upper bound 7.13% each). Pick is unsupported by design; kick fails in integration even though isolated envelopes look strong.
- **Independent 20-episode mjlab replay: PASS, 2/20 disagree (rate 0.10, exactly at the <=10% threshold; Wilson 2.8-30.1%).** Both disagreements are mjlab reference episodes that went numerically non-finite after falling into the stair hole; they were conservatively counted as disagreements. The other 18 agree on every outcome. Trust is established at the frozen threshold, narrowly. Simulator fidelity remains "improved, partial agreement, not certified" (reports/sim_fidelity_v2.md).
- **A real localization defect was found and fixed.** The fiducial camera was posed from a raw qpos prefix, which under training dynamics (backlash joints interleaved) put e.g. the left-knee angle into head pitch: camera orientation error median 27.6 deg. Fixed by copying qpos by joint name (median 1.42 deg; infer-mode images bit-identical to before). Properly versioned: new seed fork v3 recording the localization hash, guards, F gate rerun, full odometry rerun. Odometry nav went 75 -> 121/200 (paired exact McNemar p = 3.7e-6). Targets still fail.
- **Policy comparisons (same v2 seeds, GT nav):** walk 173/200 (significantly worse, p = 0.020); velocity_hard 181/200 with 8 falls (fall rate significantly worse, p = 0.004). The selected `atlantis_walk` remains the best eligible walker.
- The fall-rate target is met by the observed point estimates only. Upper confidence bounds: GT 1.88% (supports <=2%), odometry v3 2.78% (does not establish <=2%).
- The pipeline was not completed to the project's targets, but was moved a long way. Against the first frozen run (v1, same 300 tasks, same master seed), GT nav success went 81/200 (40.5%) to 187/200 (93.5%) and SPL 0.2912 to 0.6661. The v1 and v2 GT Wilson intervals do not overlap (40.5%: 33.94-47.42%; 93.5%: 89.20-96.16%). Falls stayed at or below 1 per 200, and holes stayed at 5.
- What was achieved: deterministic harness with policy/dynamics guards and bit-identical infer parity; 7 walkers remeasured on training v2 and `atlantis_walk` selected; 2 cm conservative map; measured-primitive planner passing 100/100; follower passing 20/20 on three walkers; a 12-defect benchmark audit fully repaired; frozen seeds v1/v2 with full provenance; every failing episode retained as trace plus video (113 GT + 225 odometry-v2 + 179 odometry-v3 failure JSON/MP4, plus 27 walk + 19 velocity_hard comparison failures).
- Honest limits: the v2 results come from a controller revision (follower `c491455b...`) forked from v1 after seeing v1 failures. Both v1 and v2 are reported. Localization assumptions (simulated fused orientation, virtual foot contacts, surveyed object targets) flatter the odometry result. Skills (kick, pick, recovery) are not delivered in the benchmark.

---

## 2. Frozen targets and definitions

### 2.1 Targets (unchanged throughout)

- GT: nav success >=95%, SPL >=0.8. Odometry+fiducial: success >=85%, SPL >=0.7. Falls <=2%. Zero stair-hole incursions. Wilson 95% CIs on success.
- Follower gate: 20/20 runs, each cross-track RMS <0.03 m, final within 5 cm / 10 deg after a 1 s settle, no falls.
- Planner gate: plan time p95 <0.2 s, 100 queries, all complete, zero lethal.
- Independent mjlab replay (20 episodes): >10% outcome disagreement means NOT TRUSTED. Criterion frozen in `reports/replay_criterion.json` (hash begins `983bd10d`).
- Nav success: final pose within 5 cm / 10 deg, held for 1 s, no fall, no hole incursion (`nav/checker.py`, independent of planner/localizer).
- Source: `CLAUDE_RESUME_2026-11-09_PAUSED.md` "Non-negotiable constraints"; CONTRACTS.md Phase H.

### 2.2 Definition changes from the original plan, and why

| # | Original plan | Frozen definition | Reason and evidence |
|---|---|---|---|
| 1 | Map inflation sigma = open-loop lateral std from Phase B (`lateral_sigma`, r = r_foot + 3 sigma) | Closed-loop follower cross-track sigma (RMS over the follower-gate runs of the selected walker); atlantis_walk sigma = 0.018955012692105925 m (open-loop value kept as provenance: 0.0619 m) | Open-loop 4 s drift (0.051-0.112 m) gave r_inflate 0.24-0.42 m and closed the 0.76 m doorways for 6 of 7 walkers (reports/REQUESTS.md, "map inflation sigma definition"). Defined in `nav/mapsigma.py`. This was a definition change reported to the user. |
| 2 | SPL path length p_i from raw 50 Hz trunk xy | Centred 30-step (0.6 s) moving average (MA30, endpoint-padded) of trunk xy; raw length recorded as `path_length_raw` | Audit A-01: gait sway (~13 mm RMS, 1.8 Hz) inflates raw length 1.6-2.4x, making SPL 0.8 unreachable (pilot SPL terms 0.35 raw vs 0.69 MA30). Window sweep: w30 removes ~85% of the excess; longer windows cut real corners. Raw value still reported. In v2 GT: nav_000 raw 3.479 m vs MA30 1.604 m. |
| 3 | Success judged on final frame only | Final pose must stay inside 5 cm / 10 deg for a **1 s goal dwell** (EpisodeRunner appends a settle tail; reports `goal_settled`) | Audit A-08: a trace passing through the goal at 0.3 m/s could count as success. |
| 4 | Shortest-path l_i = octile Dijkstra on the inflated map | **Any-angle footprint-only l_i**: radius-only inflation, 16-connected Dijkstra without corner cutting plus line-of-sight string pulling (`footprint_16_dijkstra_los_v1`) | Audit A-11: octile on the inflated map was not a lower bound (+6.8% median, +66% max). |
| 5 | Planner 0.2 s wall-clock budget | **Deterministic 30000-expansion budget** (2 s wall cap as a safety net); partial-plan following; warmed planner | Wall-clock budget made plans load-dependent (7-9% of repeated queries changed) and the first plan per GridMap timed out (cold complete 4/10 vs warm 10/10). 30000 expansions: 0/100 changed under load; idle wall p95 0.121-0.150 s (REQUESTS.md D1-D3; STATUS Phase D gate). |
| 6 | A04: hole check on trunk xy and geom centres | Hole check on collision-geom extents (near-floor collision AABBs overlap the hole mask) | A foot could overhang the hole by 23.9 mm unflagged. Approved geometry definition. |
| 7 | A09: checker fall = tilt >60 deg or absolute trunk z <0.05 m; Sim differed | One definition: tilt >60 deg or trunk height above the lowest foot centre <0.02 m (independent relative-foot fall) | The absolute-z rule called an upright pose on stair_1 "fallen". Approved fall definition. |
| 8 | Timeout fixed at 120 s | `max(120 s, 25 s/m x geodesic)` frozen in the seed file | Audit risk note: pilot pace 10-36 s/m. |
| 9 | Seeds checked with `is_lethal` only | Dry `Sim.reset` of every start/goal/pre-pose, free-object clearance, 1 s all-object stability; infeasible objects excluded | Audit A-02, A-03. v2 re-validation: 0 reset, 0 clearance, 0 stability rejections. |
| 10 | `step_over_height` default 0.02 (CONTRACTS.md) | **0.0 (conservatively unmeasured)**; MAP_VERSION 4 handles zero-height floor clipping | A valid 0 value had made the whole apartment lethal (REQUESTS.md H to C defect). No step capability was claimed. |

Other repaired audit items (A02, A03, A05-A07, A10, A12) are in `reports/audit_benchmark.md` ("Resolution update"). All 12 audit regression tests pass and no xfail markers remain.
Scoring was not changed to erase failures. Frozen failures stay counted (e.g. conservative-AABB hole incursions are distinguished in `reports/benchmark_diagnosis_v1.md` but not excluded).

---

## 3. Phase gates A-H

| Gate | Result | Exact numbers and sources |
|---|---|---|
| **A** Harness | **PASS** | Infer parity bit-identical (max obs diff 0.0; 2.2e-15 for pick), deterministic seeds, md5/rejected-policy/version guards, trace/video infrastructure. test_sim 18 passed + 1 expected xfail (`flat walk 0.2 m/s` standing-start deadzone). RTF 68x flat, 40x apartment (reports/sim_rtf.json). Training dynamics `cb8561f6_nominal_v2` (sim.py md5 `d07e522d` to `8b5c40b9`). Fidelity vs mjlab only "improved, partial, not certified" (sim_fidelity_v2.md). |
| **B** Walker selection | **PASS** | All 7 accepted walkers remeasured on training v2 (7 COMPLETE lines in `reports/measure_training_v3.log`). Every walker: 0/1960 grid falls. Selected `atlantis_walk` (`reports/walker_gate.json`): 232 safe commands, 0/1960 falls, max forward 0.2831 m/s, 256/256 usable primitives. Others: velocity_hard 229 / 0.2962; walk 201 / 0.2573 (192 prim.); atlantis_velstand_rough_wip 197 / 0.2572; velstand_flat 190 / 0.2234; velstand_rough 185 / 0.2093; velstand 200 / 0.2296. Idle check 0/8 falls each (Wilson upper 0.324). Caveat in the file: measured flat only; small zero-fall cells do not establish rare-fall safety. Rejected `velocity_hard_4250` excluded. |
| **C** Maps | **PASS** | 2 cm grid, footprint radius 0.083 m, conservative zero step-over, stair cliff, MAP_VERSION 4. Original validation (STATUS Phase C, before v2 sigma): 10,000 vertical rays 96.8% strict / 99.990% 1-cell-tolerant agreement; 0 missed obstacle or cliff points; horizontal rays 100.0%; stair hole 2000/2000 cliff+lethal; tests/test_map.py 6 passed. Frozen scene closure of 80 files: SHA256 `7d02c6dc...3048`. Map is one free component at atlantis_walk sigma. |
| **D** Planner | **PASS** | `tools/planner_gate` on real map + measured primitives, follower `c491455b` era: **100/100 complete**, 0 lethal swept points, plan p95 **0.1868 s**, wall p95 **0.1905 s**, heuristic admissibility 7/7 goals, cross-room 83/83, far pair and cold start 100%; idle/busy2 result changes 7%/4%. tests/test_planner_measured.py 5 passed. Archived `reports/follower_v2_c491455/`. Note: the benchmark's all-call plan-time p95 is 0.289 s (GT) and 0.465 s (odom) under an 8-process farm, i.e. above 0.2 s (see section 10). |
| **E** Follower | **PASS for the 3 eligible walkers; velstand_flat FAIL (withheld)** | Follower SHA256 `c491455b...4b37`; 15/15 unit tests. Canonical gates, training v2, seeds 0-4 x 4 courses, thresholds unchanged: **atlantis_walk 20/20** (0 falls, worst RMS 23.1105 mm, sigma 0.018955 m); **walk 20/20** (0 falls, worst 23.1967 mm, sigma 0.018073 m); **velocity_hard 20/20** (0 falls, worst 15.4669 mm, sigma 0.014616 m). **velstand_flat FAIL 16/20 reached/passed**, 0 falls, worst RMS **43.921 mm**, sigma 0.025420 m; failed straight seed 0 and s_bend seeds 0, 1, 4. Its comparison is withheld unless it passes a future gate. Held-out rotation scenarios (dev, not the gate): promoted 6/6 vs archived 99128a5 3/6, but one case took 63.7 s for 0.5 m displacement (follower_rotation_progress_promotion_v2.md). Earlier v2 history: atlantis_walk 19/20 (straight seed 4, 34.7 mm) was repaired. |
| **F** Localization | **Render v2: gate PASS (dev); benchmark FAILS** | **Rerun with fixed camera** (`reports/localization_gate_render_v2.json`, same 5 seeds x 300 steps): fiducial final XY 7.4-18.4 mm (before fix 49.1-134.2 mm in `localization_gate.json`), RMS 7.8-11.7 mm, 23-38 measurements per seed, bearing error median 0.01-0.6 deg (before 1.1-3.9 deg). v3 odometry benchmark 121/200: still FAILS 85%. Earlier dev gate (`reports/localization_gate_v2.json`): 5 seeds x 300 steps, 150 renders, 85 detections/85 updates; odom final XY 5.7-14.9 mm, odom+fiducial 1.9-14.0 mm. Did not generalize: full odometry benchmark 75/200 (37.5%). Post-hoc (`reports/localization_diagnosis_frozen_v2.json`): 62,359 native camera renders; terminal estimate-vs-GT error median 43.9 mm, p90 90.8 mm, p95 114.7 mm, max 167.7 mm over 200 nav episodes; successful (n=75) median 31.4 mm, p95 64.8 mm; failed (n=125) median 56.0 mm, p95 127.5 mm. Assumptions in section 9. |
| **G** Skills | **FAIL / unsupported** | Fail-closed skill guards and contract-derived kick positioning integrated. Isolated kick: right 227/245, mirrored-left 228/245, 0 falls each. Integrated benchmark kick 0/50 per mode. Pick unsupported (0/80 actual-box trials). Recovery unsupported on canonical morphology. See section 8. |
| **H** Benchmark | **FAIL (targets), tooling PASS, replay PASS** | Audit 12/12 fixed. Frozen seeds v1/v2/v3; full v2 runs in both modes and a full v3 odometry run: all targets fail except observed fall rate. Independent replay PASS 2/20 (section 5). Policy comparisons complete (section 6). Final suite 163 passed, 1 skipped, 1 xfailed (section 12). |

---

## 4. Benchmark results (frozen v2, selected `atlantis_walk`, 300 tasks: 200 nav / 50 kick / 50 pick)

Reports: `reports/bench_frozen_v2_atlantis_walk_gt_20261110_020423.json`, `reports/bench_frozen_v2_atlantis_walk_odom_fiducial_20261110_022933.json`. Seed file `data/seeds_v2.json`, canonical SHA256 `7fc18235...8248fed` (matches both reports). Gate `passed: false` in both; both record `reference_replay: pending`.

### 4.1 Navigation (n = 200 each)

| Metric | GT | odom+fiducial | Target (GT / odom) |
|---|---:|---:|---|
| Success | 187/200 = 93.5% | 75/200 = 37.5% | >=95% / >=85% |
| Wilson 95% CI (success) | 89.198-96.162% | 31.085-44.386% | |
| SPL | 0.666099 | 0.257415 | >=0.80 / >=0.70 |
| SPL 95% CI (bootstrap) | 0.63755-0.69065 | 0.21280-0.30545 | |
| Falls | 0/200 = 0% | 3/200 = 1.5% | <=2% |
| Fall Wilson 95% CI | 0-1.885% | 0.511-4.317% | |
| Stair-hole incursions | 5 (2.5%; Wilson 1.07-5.72%) | 10 (5.0%; Wilson 2.74-8.96%) | 0 |
| Time to goal, successes only: p50 / p90 / p95 / p99 (s) | 54.82 / 89.40 / 106.73 / 127.94 | 58.02 / 92.28 / 102.99 / 120.46 | |
| Plan-call time, all calls: p50 / p90 / p95 / p99 (s) | 0.1597 / 0.2492 / 0.2886 / 0.3892 | 0.3215 / 0.4389 / 0.4649 / 0.5827 | planner gate p95 <0.2 s |
| Mean MA30 path length of successes (m) | 5.04 | 4.82 | |
| Episodes with duration >=120 s | 7 | 9 | |
| Gate checks | success F, spl F, falls T, stair_hole F, safety_observed T, full_frozen_suite T | same | |

Notes: the time-to-goal percentiles in the handoff and report are over successful episodes (A-series audit fix). Time p99 above 120 s is possible because the cap is `max(120 s, 25 s/m x geodesic)`.

### 4.1b Odometry + fiducial v3 (fiducial render v2; current result)

Report `reports/bench_frozen_v3_atlantis_walk_odom_fiducial_20261110_070315.json`; seeds `data/seeds_v3.json`, canonical SHA256 `49fd00a09433c45f5c9d9f07d682b1617a0faffa9ab426504d0fbebf24393f39` (parent v2 `7fc18235...`; identical 300 tasks and master seed; 300/300 tasks re-validated, 0 rejections; `reports/fork_seeds_v3.log`). Only `nav/localization.py` changed (SHA256 `159f5db6...6d34`, recorded as `localization_sha256` in the seed provenance and enforced by `nav/bench.py`).

| Metric | v2 odom (pre-fix) | **v3 odom (fixed)** | Target |
|---|---:|---:|---|
| Nav success | 75/200 = 37.5% (31.1-44.4%) | **121/200 = 60.5% (53.6-67.0%)** | >=85% FAIL |
| SPL | 0.2574 (0.213-0.305) | **0.4188 (0.370-0.469)** | >=0.70 FAIL |
| Falls | 3 (1.5%) | **1 (0.5%; Wilson 0.09-2.78%)** (nav_147) | <=2% observed PASS |
| Stair-hole incursions | 10 | **8** (nav_002, 075, 085, 090, 095, 147, 159, 162) | 0 FAIL |
| Failure reasons | pos_tol 116 incl. combos | pos_tol 70, stair_hole 8, yaw_tol 3, goal_not_settled 1, fall 1 | |
| Stack reasons | follower_stuck 4, timeout 4, recovery_unsupported 3, plan_incomplete 5 | follower_stuck 4, timeout 5, recovery_unsupported 1, plan_incomplete 5 | |
| Time to goal p50/p95 (s) | 58.02 / 102.99 | 56.78 / 113.76 | |
| Plan-call p50/p95 (s) | 0.322 / 0.465 | 0.429 / 0.601 | gate 0.2 FAIL |
| Kick / pick | 0/50, 0/50 | 0/50, 0/50 | |

Paired over the same 200 nav tasks: both succeed 49, v3-only 72, v2-only 26; exact McNemar p = 3.7e-6. v3 odometry succeeds on 117 of the 187 GT-successful tasks. The 26 regressions show the closed-loop outcome is sensitive (chaotic) to small estimate changes; the net effect is large and significant. Artifacts: 300 NPZ, 179 failure JSON + 179 non-empty MP4, 0 export errors (`reports/traces|failures/frozen_v3_atlantis_walk_odom_fiducial/`).

GT under v3 seeds: GT mode does not use the camera. A 10-episode v3 GT run (`reports/v3_gt_parity10.log`) produced qpos traces bit-identical to v2 for 10/10 episodes, so the v2 GT result (187/200) stands as the current GT result; a full 200-episode v3 GT rerun was not performed.

### 4.2 v1 versus v2

| Run | Seeds SHA256 | Controller | Nav success | SPL | Falls | Holes | Kick / pick |
|---|---|---|---:|---:|---:|---:|---|
| v1 GT (`bench_frozen_v1_atlantis_walk_gt_20261110_012319.json`) | `65d6fdee...f0` | follower `99128a5` | 81/200 = 40.5% (33.94-47.42%) | 0.291196 (0.24217-0.34064) | 1 (0.5%; Wilson 0.09-2.78%) | 5 | 2/50, 0/50 |
| v2 GT | `7fc18235...8fed` | follower `c491455b` | 187/200 = 93.5% (89.20-96.16%) | 0.666099 | 0 (0-1.885%) | 5 | 0/50, 0/50 |
| v2 odom+fiducial | same | same | 75/200 = 37.5% | 0.257415 | 3 | 10 | 0/50, 0/50 |

- v1 nav failure reasons: pos_tol 76, yaw_tol 40, stair_hole 5, goal_not_settled 1, fall 1. v1 time p50/p95 55.08/108.74 s; plan p50/p95 0.166/0.294 s.
- v2 is a **controller-only fork** (same 300 tasks, same master seed, explicit provenance fork). The change that mattered was the rotation-progress repair, which targeted the v1 pure-yaw stalls (>90% of the final 30 s spent commanding in-place yaw for representative v1 failures). Because the fix was designed after inspecting v1 failures, v2 is not an unbiased fresh estimate of the final system; a new held-out seed set was not run.
- No v1 odometry benchmark was run, so there is no v1-versus-v2 odometry comparison.
- The 5 stair-hole incursions are unchanged from v1 (not necessarily the same episodes; no per-episode match was done here).

### 4.3 Failure reasons (nav)

GT (13 failures): pos_tol 7, stair_hole 5, yaw_tol 1. By stack reason: plan_incomplete 5, episode_timeout 2, follower_stuck 1, none (hole) 5.
- All 5 GT `plan_incomplete` nav failures (nav_063, nav_066, nav_103, nav_115, nav_137) failed at t = 0 with plan-log reason `start_lethal`: the start cell is inside the inflated lethal map, so the planner refused to plan. Final position errors 2.7-6.5 m (they never moved). This is a seed-versus-map inconsistency (dry reset passes, but the planner map treats the start as lethal), not a planner-quality failure. It is worth 2.5 points of GT success by itself.
- Remaining non-hole failures: one follower_stuck (115.5 s, 7 cm / 6 deg), one episode_timeout at yaw 160 deg (145.8 s, 1 cm position), one timeout at 7 cm / 48 deg (124.7 s).
- Stair-hole failures (5): durations 56-110 s, near-goal poses (1-4 cm), so the hole was entered near or at the goal (cause not diagnosed here; the geometry-extent definition A04 makes the check stricter than trunk-centre tests).

Odometry (125 failures): pos_tol alone 110, stair_hole alone 5, stair_hole+pos_tol 3, fall+stair_hole+pos_tol 2, fall+pos_tol 1, yaw_tol 2, goal_not_settled 2. Stack reasons: follower_stuck 4, episode_timeout 4, recovery_unsupported 3 (the 3 falls), plan_incomplete 5. 116 episodes carry `pos_tol` (including those combined with other reasons). The dominant failure is final position outside 5 cm, consistent with estimate-versus-GT terminal error (median 43.9 mm, failed-episode median 56.0 mm).

### 4.4 Per start-room and goal-room breakdown (`by_room_pair` in each report; 36 pairs)

| Start room | n | GT succ. | GT holes | Odom succ. | Odom holes |
|---|---:|---:|---:|---:|---:|
| bathroom | 34 | 31 | 0 | 16 | 4 |
| bedroom | 34 | 32 | 1 | 9 | 1 |
| corridor | 31 | 27 | 3 | 14 | 2 |
| kitchen | 35 | 33 | 0 | 8 | 1 |
| living | 30 | 30 | 0 | 15 | 1 |
| office | 36 | 34 | 1 | 13 | 1 |

| Goal room | n | GT succ. | GT holes | Odom succ. | Odom holes |
|---|---:|---:|---:|---:|---:|
| bathroom | 28 | 24 | 2 | 15 | 0 |
| bedroom | 29 | 27 | 1 | 8 | 2 |
| corridor | 55 | 51 | 1 | 19 | 7 |
| kitchen | 29 | 29 | 0 | 9 | 0 |
| living | 28 | 26 | 0 | 11 | 1 |
| office | 31 | 30 | 1 | 13 | 0 |

| Pair type | n | GT succ. | Odom succ. |
|---|---:|---:|---:|
| same-room | 68 | 66 (97.1%) | 26 (38.2%) |
| cross-room | 132 | 121 (91.7%) | 49 (37.1%) |

Worst GT pairs: office->bathroom 1/3 (SPL 0.254, 1 hole, time p50 111 s); bedroom->bathroom 2/3 (1 hole); corridor->living 2/3; kitchen->corridor 5/7; bathroom->bedroom 3/4; bathroom->living 3/4. Corridor is the hardest goal for odometry holes (7 of 10 incursions end in a corridor-goal task; the stair hole lies at x -0.40..0, y -1.40..-0.70, inside the corridor region of the map). Room-pair cell counts are 3-14 episodes, so individual pair rates are anecdotal. Odometry has several pairs at 0 successes (bedroom->bathroom 0/3, bedroom->living 0/3, kitchen->bedroom 0/3, kitchen->office 0/4, living->bedroom 0/3). Full per-pair SPL, fall, CI, time and plan-time are in the JSONs under `by_room_pair`.

### 4.5 Kick and pick (50 each, both modes)

- GT kick 0/50: skill_not_started 43, ball_moved_before_kick 49 (overlapping reasons), too_slow 7, wrong_direction 5; stack: follower_stuck 40, episode_timeout 2, plan_incomplete 1 (kick_036, `start_lethal`).
- Odometry kick 0/50: skill_not_started 49, ball_moved_before_kick 49, too_slow 1, wrong_direction 1; stack: follower_stuck 28, episode_timeout 20, plan_incomplete 1.
- Pick 0/50 in both modes: skill_not_started 50, not_lifted 50; stack skill_unsupported_pick 50. Blocked tasks are retained and failed, not removed.
- 0/50 gives a Wilson upper bound of 7.13%. Kick plan time p50/p95: GT 0.0355/0.0812 s, odom 0.0751/0.1336 s.
- Artifacts: GT all 300 NPZ + 113 failure JSON + 113 non-empty MP4; odometry all 300 NPZ + 225 failure JSON + 225 non-empty MP4; no export errors (handoff). Directories: `reports/traces/frozen_v2_atlantis_walk_{gt,odom_fiducial}/`, `reports/failures/frozen_v2_atlantis_walk_{gt,odom_fiducial}/`.

---

## 5. Independent mjlab replay

**Verdict: PASS. 2/20 episodes disagree (rate 0.100, threshold <=0.10; Wilson 95% CI 0.028-0.301).** Reports `reports/replay_frozen_v2_gt_final.json` / `.md`; criterion SHA256 `983bd10df3df90929037385a3cf0f622e55a4b4102551c02a44e1f61544c6beb` (unchanged).

- 20 GT traces selected by the frozen sampler; nav exact replays bit-identical for 20/20.
- mjlab reference: 20 worlds x 4981 steps (`reports/replay_frozen_v2_gt_resume2_mjlab/`, deterministic: identical to the first resume attempt).
- **nav_014 and nav_142: mjlab reference went non-finite** (from steps 2288 and 2146). On the finite prefix the reference trunk falls into the stair hole and through the world (z = -5.7 m by step 600 on nav_014) before the solver blows up. The frozen criterion does not define non-finite references, so `tools/replay_compare.py` now counts any such episode as a **disagreement** (it can never help the verdict). Diagnostic only: on the finite prefix both outcomes (fell, hole) lie inside the nav replicate set (all 7 nav replicates fall in both), i.e. 0/20 under a prefix adjudication.
- Remaining 18: 0 disagreements in fell, stair_hole, static_collision and final_position. Note mjlab fell in 15/18 open-loop replays, and goal success agreement is low (open-loop replay of closed-loop commands; not part of the criterion).
- QACC instability warnings appear in **nav-side** perturbed replicates of nav_107 and nav_117 (deterministic), not in the trace or the mjlab side.
- History: the first run (session 79386) was interrupted (SIGINT); two resumed runs crashed at scoring with `KeyError: 'final_pose'` because of the non-finite reference. Logs preserved: `replay_frozen_v2_gt{,_mjlab}.log`, `replay_frozen_v2_gt_resume{,2}{,_mjlab}.log`, `replay_frozen_v2_gt_final.log`.
- Caveat: the pass is at the threshold, and its CI upper bound is 30%. Trust is established under the frozen rule, not with margin.

Facts used (from the draft):
- Frozen criterion (`reports/replay_criterion.json`, hash begins `983bd10d`): per episode, fell / stair-hole / static-collision must lie in the set of {nav original + 7 nav replicates}; final xy within X = max(0.10 m, 2 x max nav-replicate endpoint spread); PASS iff disagreement <=10% over n = 20.
- The first run (session 79386, `reports/replay_frozen_v2_gt.log`, `replay_frozen_v2_gt_mjlab.log`) was interrupted with SIGINT (exit 130). Nav-side clouds had 20 entries but no output was written; no JSON/MD/verdict exists. The mjlab log contains QACC instability warnings and a KeyboardInterrupt.
- A second attempt with output prefix `replay_frozen_v2_gt_resume` appears in `reports/replay_frozen_v2_gt_resume.log` (log only; at the time of this draft it had started the mjlab replay and logged nav clouds for nav_003, nav_007, nav_014; no verdict was present).
- Pilot-only evidence under the older v1/v2 sims: v1 pilots FAIL 1/5, v2 pilots PASS 0/5 (reports/replay_pilot_check_v2). n = 5 and not the benchmark sample; the v2 nav replicate spread is larger (up to 1.3 m) so the pass partly reflects a looser position tolerance.
- Rule if disagreement >10%: benchmark NOT TRUSTED; investigate; any new physics requires a new dynamics version, new seed version and a rerun of B through H.

---

## 6. Policy comparison

`reports/policy_comparison_v2.json` / `.md`. Same v2 seeds, GT, training dynamics, nav-only, 200/200 matched episodes (episode-ID SHA256 identical); each walker uses its own passing follower gate and sigma (walk 0.018073, velocity_hard 0.014616). Two-sided pooled two-proportion z-tests, alpha 0.05. `velstand_flat` withheld (follower gate 16/20).

| Policy | Success (Wilson) | SPL (CI) | Falls | Holes | Time p50/p95 (s) | vs atlantis_walk |
|---|---|---|---|---|---|---|
| **atlantis_walk (selected)** | 187/200 = 93.5% (89.2-96.2) | 0.666 (0.638-0.691) | 0 | 5 | 54.82 / 106.73 | base |
| walk | 173/200 = 86.5% (81.1-90.6) | 0.604 (0.568-0.642) | 0 | 3 | 53.86 / 107.51 | success -7.0 pp, **p = 0.020**; falls/holes n.s. |
| velocity_hard | 181/200 = 90.5% (85.6-93.8) | 0.632 (0.599-0.664) | 8 (4.0%) | 9 | 44.58 / 81.76 | success -3.0 pp n.s.; **falls +4.0 pp, p = 0.004** |

All three fail the GT targets. Reports: `bench_frozen_v2_walk_nav_gt_20261110_060143.json` (27 failure JSON/MP4), `bench_frozen_v2_velocity_hard_nav_gt_20261110_061247.json` (19). The comparison farms ran via `tools/run_comparison_serial_video.py` (same `run_bench`/EpisodeRunner, 4 workers, serial video) to respect the 9-process cap. Known report quirk: the `frozen_provenance` block in comparison reports copies the seed file's policy/sigma (atlantis_walk), not the comparison policy's; the runtime config and traces carry the correct policy md5 and sigma.

Facts already known before resume: no comparison had been launched at pause. Eligible comparators are `walk` and `velocity_hard` (own passing envelope, map and sigma, same v2 seeds, nav-only, GT mode, `--comparison-policy --tasks nav --seeds data/seeds_v2.json`; the CLI default seeds are v1). `velstand_flat` is withheld (16/20 follower gate). Comparison goes through `tools/policy_compare.py`, which keeps strict seed/mode/count matching.

---

## 7. Shortfall analysis and causes

Full analysis: `reports/shortfall_analysis_v2.md` with data in `reports/shortfall_analysis_v2/` (trace analysis under frozen definitions; no scoring changes).

**GT success (187/200; 3 more needed for 95%).** The 13 failures fall into three classes, each large enough on its own: (1) 5 `start_lethal` refusals: starts free on the 2 cm map (0.140 m clearance) but snapped to a lattice node 14-23 mm away that lies inside the 0.1396 m inflation; (2) 5 stair-hole flags, all after reaching the goal pose (2 genuine ~9 mm right-foot incursions: nav_065, nav_152; 3 conservative-AABB only); (3) 3 goal limit cycles (nav_018, 120, 171): each in-place turn shifts the trunk ~38 mm against a 48 mm budget, so position and yaw repairs undo each other.

**GT SPL (0.666 vs 0.8): dominated by inefficiency on successes, not failures.** SPL loss 66.8 units = 13.0 from failures + 53.8 from successes; even 200/200 would give SPL 0.712. Path ratio median 1.363. In-place turning is 155 m of the 260 m excess path and 31.7% of time (15.4 turns/episode, 882 direction reversals). Planned paths are already 1.42x the shortest path with ~30 sharp kinks; the follower tracks path direction only, so each kink triggers a turn in place (r = 0.65). Removing pure-yaw path gives SPL 0.805 as an accounting bound, not a prediction.

**Stair holes.** Plans keep >=0.143 m from the hole but the trunk came within 0.043-0.061 m: a 5.5-7.4 sigma deviation against 3 sigma inflation; 84% of flagged GT frames are during in-place turns.

**Odometry v2.** 102/125 failures declared done on the estimate and were rejected by the referee (median 70 mm off); estimate error plateaus at 55-60 mm, above the 50 mm tolerance. The analysis found the fiducial-camera joint-mapping defect (camera posed with wrong joints, median 27 deg orientation error) which the 6 s gate missed. **This was fixed in this session (sections 4.1b, F gate)**, lifting odometry nav to 121/200. The remaining odometry gap still includes "done" being declared on a single estimate (fix A3 below, not implemented).

**Kick (0/50).** Ball disturbed before the kick in 49/50; first contact is a foot during the approach (median 6.7 s, ~5 cm from the pre-kick pose). The ball is not a planning obstacle (34/49 plans pass within 0.123 m of it), and the pre-kick pose puts the trunk 0.102 m from the ball, inside footprint + ball radius. The isolated sweeps (227/245, 228/245) placed the ball after the robot settled, so they never tested the approach.

### 7.1 Known causes from existing reports (not new analysis)

GT navigation (13 failures, 6.5%):
1. **start_lethal planning refusals: 5/200 (2.5%)**, all at t = 0 (plan_log). Starts that pass the dry Sim.reset but lie in the inflated lethal region. Alone this accounts for 2.5 of the 6.5 points of GT shortfall (this is derived from the report; it was not previously flagged in the handoff or STATUS).
2. **Stair-hole incursions: 5/200 (2.5%)**, all in episodes that otherwise reached or nearly reached the goal. Related to the geometry-extent hole definition (A04), cliff clearance and repeated position repair noted in follower_rotation_progress_promotion_v2.md ("cliff clearance, repeated position repair, path/body-heading compatibility").
3. **Efficiency**: SPL 0.666 against 0.8 with 93.5% success. Successful-episode time p50 54.8 s, p95 106.7 s; in the pilots, 10-36 s per metre. Residual in-place rotation and repeated position repair; the held-out case that took 63.7 s / 2.392 m for 0.5 m displacement. Maximum measured forward speed is only 0.283 m/s and standing-start walkers ignore |vx| <= 0.2 (xfail test), so a low SPL ceiling exists by construction of the walker.
4. Follower stuck 1, timeouts 2.

Odometry+fiducial navigation:
1. **Localization drift**: terminal error median 43.9 mm (failed 56.0 mm vs successful 31.4 mm) against a 5 cm goal tolerance, so about 110 of the 125 failures end only out of position tolerance. The short development gate (final XY 1.9-14.0 mm over 300 steps) did not generalize to 20-120 s apartment episodes.
2. Hole incursions double (10 vs 5) and 3 falls (all `recovery_unsupported`).
3. Plan time p50/p95 0.32/0.46 s versus 0.16/0.29 s in GT: the odometry run was slower (more replans, load), above the planner gate's 0.2 s.
4. Same start_lethal refusals (5) and the same planner/follower limits as GT.

Skills: see section 8 (necessary approach guard did not prove dynamic reachability; ball disturbed before kick in 49/50; follower stuck near the ball in 40/50 GT kick episodes; pick has no demonstrated lift capability; recovery models do not work on the canonical morphology because it lacks full ground collision, so settled trunk z is about -0.106 m).

Simulation fidelity: independent replay unresolved; residual engine differences (mujoco_warp drops penetrating foot-box contacts in ~6% of substeps; initiation of `[0,0,1]` after a stop under-reproduced: nav about 33% vs mjlab 4/6).

---

## 8. Skills

### 8.1 Pick (actual boxes): 0/80

(`reports/pick_recovery_remedy.md`, `pick_recovery_remedy_pick*.json`.) Actual frozen targets obj_0..obj_3: 70 mm boxes (half extents 35 mm), 50 g, compiled physical-field equality checked against `apartment.xml`. 80 persisted trials = 2 policies x 2 morphologies x 4 boxes x 5 seeds, all handoffs accepted, 0 robot falls.

| Morphology | Policy | >=20 mm peak lift | Final (frozen origin criterion) | Robot-object contact | Max lift |
|---|---|---:|---:|---:|---:|
| walk_backlash | pick | 0/20 | 0/20 | 0/20 | -0.285 mm |
| walk_backlash | atlantis_pick_flat | 0/20 | 0/20 | 0/20 | -0.285 mm |
| allcollisions_backlash | pick | 0/20 | 0/20 | 20/20 | 4.140 mm |
| allcollisions_backlash | atlantis_pick_flat | 0/20 | 0/20 | 20/20 | 9.922 mm |

Restoring collision contact does not produce a lift. The pinned GroundPick scene has no actual object or grasp training and a passive jaw; the payload is a sampled 10-40 g external force surrogate. In installed reference mjlab 1.3.0, the zero-weight `mouth_payload_force` hook is skipped (control confirmed), but the historical training manager is unverified, so no claim is made that the original run lacked payload forces. Earlier proxy-ball experiments (35 mm / 15 g ball, 0/10 lifted by 20 mm; max 11.7 mm) are consistent. Earlier failed metadata attempts (160 executed pick attempts in total) are disclosed in `pick_recovery_remedy_failed_metadata.log` and not counted as primary evidence. Samples are correlated (identical boxes) and cover one flat placement protocol, so this is failure of the sampled protocols and not a proof of impossibility.

### 8.2 Recovery: 210 trials

3 policies x 2 morphologies x (3 immediate + 4 settled states) x 5 seeds = 210. Success = tilt <15 deg, trunk-foot height >=90 mm held 0.5 s. Ball-free.

| Morphology | Policy | Immediate fallen (front/back) | Immediate sit | Settled fallen | 2 s atlantis_walk handoff falls |
|---|---|---:|---:|---:|---:|
| walk (canonical) | atlantis_standup_flat | 0/10 | 0/5 | 0/20 | n/a |
| walk | standup_rough | 0/10 | 0/5 | 0/20 | n/a |
| walk | velstand | 0/10 | 0/5 | 0/20 | n/a |
| allcollisions | atlantis_standup_flat | 0/10 | 5/5 | 0/20 | 0/5 |
| allcollisions | standup_rough | 0/10 | 5/5 | 0/20 | 0/5 |
| allcollisions | velstand | 10/10 | 5/5 | 20/20 | 0/35 |

- **Full-collision velstand: 15/15 immediate + 20/20 settled = 35/35**, all within 6 s, 0/35 falls in the 2 s `atlantis_walk` handoff. This is promising diagnostic evidence for an existing model on an alternate morphology. It is **NOT canonical support**: the frozen benchmark uses walk morphology; enabling full ground collision changes physics and invalidates the frozen envelopes, planner and benchmark provenance. Canonical walk-morphology recovery stays 0 for all three policies, with mean settled initial trunk z -105.8 mm (floor penetration, a collision-support confound) versus +40.0 mm on full collision.
- The two standup models recover seated states only (5/15 immediate each on full collision), not fallen states. The 210-trial local reset distribution is an approximation, not independent mjlab randomization.
- Recovery is marked unsupported and fails safely; in the odometry benchmark this produced 3 `recovery_unsupported` falls.

### 8.3 Kick: isolated envelope versus integrated 0/50

- Isolated, benchmark morphology, `atlantis_walk` handoff, apartment-matched ball (40 mm / 25 g; the older sweep used 35 mm / 15 g and gave 87/245 right, 92/245 left): 49 placement offsets x 5 seeds per side. **Right 227/245 (92.7%, Wilson 88.7-95.3%), mirrored-left 228/245 (93.1%, 89.2-95.6%), 0/245 falls each (upper 1.54%), nominal 5/5.** Largest all-success rectangle 36 cells, 25 mm half-width per axis. Seeds are shared across offsets, so the intervals are descriptive and interpolation between 10 mm grid points is unvalidated. The ball is placed after settling, so approach interaction is not modelled.
- **Integrated benchmark: 0/50 (GT) and 0/50 (odom).** The sampled 25 mm half-width placement window may be tighter than what the follower delivers (not measured here); the contract-derived approach guard was necessary but did not prove dynamic reachability. GT: follower_stuck 40, ball disturbed before kick 49, too_slow 7, wrong_direction 5. v1 had 2/50 with a looser approach, so the "tighter contract-derived" approach made kick worse in integration.
- Mirrored kick uses `MirroredPolicy`; kick traces cannot be replayed by the independent replay tool until mirror flags and switch steps are recorded (now recorded: `record_kick_mirror`).

---

## 9. Assumptions and gaps

- **Simulated fused orientation**: the IMU `framequat` is a simulated fused orientation with declared noise, not an independently validated inertial fusion stack. It anchors heading in odometry modes.
- **Virtual foot contacts**: binary foot-contact sensors come from simulated collision events (not GT pose). Real hardware contact sensing is unvalidated.
- **Surveyed object targets**: the kick/pick target position is the GT object xy supplied by EpisodeRunner (audit note: a GT leak into the stack, declared for odometry runs). No object detection is simulated.
- **Fiducial render residual**: the render model lacks the training backlash joints, so the fixed camera pose still has ~1.4 deg median (2.4 deg max) orientation error vs the training sim's own head camera.
- **Fiducial landmarks are known-map**: simulated camera, calibrated projection bias; the camera path needed approved native rendering. Camera model is simulator, not a real camera.
- **Unverified historical training manager**: the zero-weight payload hook is skipped in the installed mjlab 1.3.0; the manager used by the historical run is unverified.
- **Fidelity not certified**: sim_fidelity_v2.md says "improved, still partial agreement, not certified". mjlab itself is stochastic (replicate spread p50 0.24 m, max 1.12 m); mujoco_warp engine contact differences are not portable into nav.
- **Single-environment evidence**: all results are CPU MuJoCo at training dynamics; infer-mode (deployment) physics differs. Stale-tracking: envelopes, follower gates and contracts predating dynamics v2 carried no `dynamics_version` and were invalidated and re-measured.
- **Statistical caveats**: 0 falls in 200 has an upper bound of 1.88% (GT) and 3 falls has 4.32% (odometry); walker gate cells with 0/8 have upper 0.324; correlated seeds in sweeps overstate precision.
- **v1 to v2 fork**: v2 was designed after seeing v1 failures (same tasks), so its 93.5% is not a held-out estimate.
- **Seed-versus-map gap**: 5 nav and 1 kick starts are `start_lethal` for the planner though they pass the dry Sim.reset (lattice snapping into inflation; fix A1).
- **Step-over height is 0 (unmeasured)**; the robot is given no stair/step capability.
- **Time ceiling**: walkers' measured maximum speed is 0.283 m/s at best; speed tracking is 30-55% of command.
- **Flat-measured envelopes**: Phase B/E gates are on flat/box floors, not the apartment.
- Kick and pick success semantics in `nav/checker.py` are the project's own definitions (kick: ball speed/direction after recorded skill onset, rejecting >2 cm pre-kick ball movement; pick: >=20 mm final object-origin lift, <10 mm drift), not externally specified.
- Process note: nav is not a git repo; backups live in `reports/bak_2026-11-09_resume/` and (this session's changes) `reports/bak_2026-11-09_camfix/` with SHA256SUMS.

---

## 10. Next steps and future training specification

Done this session: (1) independent replay verdict PASS 2/20; (2) walk / velocity_hard comparisons; (3) shortfall analysis; (A2) fiducial camera fix, versioned and benchmarked as v3. The items in 10.1 are superseded where marked.

Fixes that need no retraining (from `shortfall_analysis_v2.md`; **none implemented except A2**; each needs new source hashes, its gates, a new seed fork (v4) and an H rerun; a geometry/physics change additionally needs a new dynamics version and B-H):

| ID | Fix | Targets | Requalification |
|---|---|---|---|
| A1 | Planner snaps start/goal to the nearest free lattice node, or seed validation checks the snapped node | 5 GT start_lethal (+1 kick) | D, v4, H |
| A2 | **DONE**: fiducial render qpos by joint name | odometry | F rerun, v3, H odom (done) |
| A3 | Require a converged localization fix (innovation/covariance gate, no GT) before declaring done | odom pos_tol failures | F/E, v4, H odom |
| A4 | Goal repair with lateral/forward safe commands, budget the ~38 mm per-turn shift, bound oscillation | 3 GT limit cycles, kick precision | E (3 walkers), D, v4, H |
| A5 | String-pull/smooth plans and heading-aware following (turn while walking with safe `fwd_turn`) | GT SPL (main lever), holes | D, E, v4, H; do not change MA30/geodesic scoring |
| A6 | Cliff-aware motion (no in-place turns near the hole) and high-quantile inflation near cliffs | holes in both modes | C, D, E, v4, H |
| A7 | Ball as a planning obstacle, staged straight approach, hand off on current ball pose, re-target | kick 0/50 | G with approach-inclusive sweep, v4, H |
| A8 | Recovery via full-collision velstand (35/35) | falls / recovery | new dynamics version, mjlab validation, full B-H; not usable in v2/v3 |

Future training (category b): B1 pick on the actual 70 mm / 50 g box (gate >=90% with Wilson lower >=80%); B2 recovery, only if A8's walking gates fail on full collision; B3 walk-to-ball-and-kick if A7 is insufficient; B4 a walker trained for low-drift turning if A4/A5 fall short of SPL 0.8. Details in `reports/shortfall_analysis_v2.md` section (b).

Suggested order: A1 + A4 make 190/200 GT plausible; A5 is the only lever large enough for GT SPL; A3 for odometry; A6 for zero holes. All are hypotheses until a v4 run passes.

### 10.1 Items already known from the handoff

1. (DONE: PASS 2/20, section 5) Establish the independent mjlab replay verdict (20 GT episodes) with a fresh output name; preserve `replay_frozen_v2_gt.log` and `replay_frozen_v2_gt_mjlab.log`. If >10% disagreement, NOT TRUSTED: investigate training-only with independent evidence, then new dynamics version, new seed version, rerun B-H.
2. (DONE, section 6) Run same-seed nav200 GT comparisons for `walk` and `velocity_hard` (each with own passing gate); `velstand_flat` remains withheld.
3. Investigate GT efficiency (SPL), goal and safety failures (hole incursions, `start_lethal` starts), localization drift and the integrated kick approach, without criterion changes. Qualify any proven fix with a versioned evaluation; do not mix source changes into a frozen run.
4. Planner timing: bring the in-benchmark plan-time distribution (p95 0.289 s GT / 0.465 s odom) in line with the 0.2 s gate or report that the gate was measured on a quiet machine.
5. Priority skill opportunity: **full-collision velstand recovery** (35/35, 0/35 handoff falls). Integration path: a new full-collision development dynamics/morphology version, independently checked against pinned mjlab, then full walker measurement, follower and planner gates, recovery/handoff trials, a new immutable seed version, then H. Never relabel alternate-geometry successes as v2.
6. (DONE) Fresh full test suite last (section 12).

### 10.2 Future training specification (from `pick_recovery_remedy.md`; no training authorized now)

- Pick: actual 50 g / 70 mm free boxes; realistic passive-jaw/contact mechanics or a mechanically realizable acquisition mechanism; object-state estimates or validated placement distributions; actual contact acquisition and retention; >=20 mm **final** object-origin lift; no robot fall, <10 mm drift; selected-walker handoff and dwell; separate release task. Apply payload as an event/control hook, not a zero-weight reward. Export a new policy/observation and contract version if interfaces change.
- Recovery: train with the same full physical support geometry as the walker, a broad settled-fall distribution without floor penetration, and explicit recover-to-walker transitions.
- Kick: needs a policy or approach controller that delivers the placement tolerance (+/-25 mm) or a kick that is robust to larger offsets, plus protection against pre-kick ball contact.
- Existing models failing is accepted by the user; do not force support, fake a grasp or relax criteria.

---

## 11. Provenance and artifact paths

### 11.1 Hashes (from the handoff, and cross-checked against the benchmark reports where noted)

| Item | Value |
|---|---|
| v1 seed canonical SHA256 | `65d6fdee13bc36aeabdd7a0f9dea70ce7d247feb7c5f8b64b596094b10ff69f0` (matches v1 report) |
| v2 seed canonical SHA256 (`data/seeds_v2.json`) | `7fc18235fce79ed717288c091cc7482d7a86d9eb4c64fd52e330915273248fed` (matches both v2 reports) |
| Follower SHA256 (v2 canonical) | `c491455b732c5e3ed7469a4e1c4e0ab8bf1d7940f8dde81b0f25ceb3400b4c37` |
| Earlier follower (v1 evaluation) | `99128a5eabef8744b4caf768c8b914d5f1df2f810f7c846eb93182ac8f68f04a` (preserved as `follower_candidate_dev_v1/follower_baseline_99128a5.py`) |
| FSM SHA256 | `404372217ab115e1d2e4e6fbccf0212430bc5340a589aba4f984f73c6b593851` |
| Checker SHA256 | `b9636a84698212cd31bfea682994b191410a8ab4ed3c7f36ab467051ff1bd60f` |
| Skills contract SHA256 | `a9949b38516c4beeef4c2914b55ba438de8af7cd595dc03c2d906661bcd70181` |
| Scene closure (80 files) SHA256 | `7d02c6dc0c55cc6fb5c90a938d68902682cf51172c7be04191c13ec8df3f7048` |
| Envelope SHA256 (atlantis_walk, from report) | `e7a76b64fbd27160ce502622e176898276a12b6ca4a391b0761982ba6d3108dc` |
| Primitives SHA256 (atlantis_walk, from report) | `7b9dde27942996a86265e7aa63f9e49ab5f8fa8dc8118d041190eb7b3d194c38` |
| Replay criterion | `reports/replay_criterion.json`, SHA256 `983bd10df3df90929037385a3cf0f622e55a4b4102551c02a44e1f61544c6beb` (unchanged) |
| v3 seed canonical SHA256 (`data/seeds_v3.json`) | `49fd00a09433c45f5c9d9f07d682b1617a0faffa9ab426504d0fbebf24393f39` (parent v2; matches v3 report) |
| Localization SHA256 (render v2, v3 runs) | `159f5db6e2530ed48291f3956a5c50e511ec3ee5bb7fc7b1137734718f566d34` (pre-fix `d6f4bbd7...981d` backed up) |
| Session source changes | `nav/bench.py` `0ee6e85e...7f4b` (localization guard); `tools/fork_seeds.py` `95ceb9bc...3e85` (records localization hash, `--revision`); `tools/replay_compare.py` `b0afeec1...0b89` (non-finite reference = disagreement) |
| Provenance re-check at resume | `reports/provenance_check_resume.md`: follower, FSM, checker, contract, seed v1/v2, criterion and 20/20 lock md5 PASS; scene closure verified by the v3 fork invariant |
| Policy md5 atlantis_walk / walk / velocity_hard / velstand_flat / velstand | `5e13c5035d1cca4d8cf694c998c86532` / `2b68446b8d9a2d70a709cbd06d8b3a09` / `3e5876f2e2461459494a764e308d5111` / `2ddce6c90f752d42e34bb9ee1dfd2373` / `b6da32045e35d07aa986fb6cf315bee9` |
| Policy md5 pick / atlantis_pick_flat / atlantis_standup_flat / standup_rough | `9852e1c75ca03d3cba52cf074d1603da` / `853813281c855f869456725aa8d8b845` / `dcf3e735b1b5350ca5468af7316e4783` / `192742d41c38d9bb5d1c36158eae91f5` (partial checkpoint) |
| Lock md5 (benchmark `lock_md5`) | `2c4e68ffafe156162e44b785f7b6ce50` |
| Dynamics | `cb8561f6_nominal_v2` (`nav.sim.TRAINING_DYNAMICS_VERSION`); sim.py md5 `8b5c40b9...` |
| Reference mjlab | 1.3.0; reward_manager.py SHA256 `8d1679cc4b0581dc0000493e0a661f3976c5f9c54f800bd825aee340a685f2a7` |
| Closed-loop sigma (atlantis_walk / walk / velocity_hard) | 0.018955012692105925 / 0.018073072532659426 / 0.014615623182093675 m |
| Frozen options | `path_smoothing_steps` 30, `goal_dwell_s` 1.0, `geodesic_method` footprint_16_dijkstra_los_v1, `step_over_height` 0.0 |

### 11.2 Artifact paths (all under `~/microduck/nav/`)

- Contracts and state: `CONTRACTS.md`, `reports/CLAUDE_RESUME_2026-11-09_PAUSED.md`, `reports/STATUS.md`, `reports/REQUESTS.md`, `policies.lock.json`.
- Benchmarks: `reports/bench_frozen_v2_atlantis_walk_gt_20261110_020423.json`, `reports/bench_frozen_v2_atlantis_walk_odom_fiducial_20261110_022933.json`, `reports/bench_frozen_v1_atlantis_walk_gt_20261110_012319.json`, logs `reports/bench_frozen_v2_atlantis_walk_{gt,odom_fiducial}.log`.
- Traces/failures: `reports/traces/frozen_v2_atlantis_walk_{gt,odom_fiducial}/`, `reports/failures/frozen_v2_atlantis_walk_{gt,odom_fiducial}/`.
- Seeds: `data/seeds_v1.json`, `data/seeds_v2.json` (immutable).
- Gates: `reports/walker_gate.json`, `reports/measure_training_v3.log`, `reports/follower_v2_c491455/`, `reports/follower_rotation_progress_promotion_v2.md`, `reports/follower_repair_v2_2026-11-09.md`, `reports/planner_gate_atlantis_walk*`, `reports/localization_gate_v2.json`, `reports/localization_diagnosis_frozen_v2.json`.
- Audit and fidelity: `reports/audit_benchmark.md`, `reports/sim_fidelity_v2.md/.json`, `reports/replay_criterion.json`, `reports/replay_pilot_check_v2.*`, `reports/benchmark_diagnosis_v1.md`.
- Skills: `reports/skills_diag.md`, `reports/pick_recovery_remedy.md`, `reports/pick_recovery_remedy_{pick,pick_fullcollision,recovery,source_audit}.json`, `reports/pick_recovery_remedy_failed_metadata.log`, `tools/pick_lift_remedy.py`, `tools/recovery_remedy.py`.
- Replay: `reports/replay_frozen_v2_gt_final.{json,md}` (verdict), `reports/replay_frozen_v2_gt_resume2_mjlab/` (reference data), interrupted/crashed logs `replay_frozen_v2_gt{,_mjlab}.log`, `replay_frozen_v2_gt_resume{,2}{,_mjlab}.log`.
- This session: `reports/shortfall_analysis_v2.md` (+ dir), `reports/policy_comparison_v2.{json,md}`, `reports/localization_gate_render_v2.json`, `reports/fork_seeds_v3.log`, `data/seeds_v3.json`, `reports/bench_frozen_v3_atlantis_walk_odom_fiducial_20261110_070315.json` (+ traces/failures), `reports/v3_gt_parity10.log`, `reports/pytest_baseline_resume.log`, `reports/pytest_final_2026-11-10.log`.
- Tests: `reports/pytest_final_2026-11-10.log` (final), `reports/integration_resume_tests.log` (historical).

---

## 12. Test suite

**Final full suite, run last after all source changes, on an idle machine: 163 passed, 1 skipped, 1 xfailed, 0 failed (76.9 s, exit 0)** (`reports/pytest_final_2026-11-10.log`). The xfail is the known `flat walk 0.2 m/s` standing-start deadzone. Baseline at resume, before changes: 161 passed, 1 skipped, 1 xfailed (`reports/pytest_baseline_resume.log`); the two new tests cover the fiducial name mapping (training: unbound fails closed, orientation error <5 deg; infer: identity map, preserving render-v1 parity).

Known history (not a current result): broad suite 140 passed / 1 skipped / 1 known walker-start xfail (`reports/integration_resume_tests.log`, 81.43 s) before the last controller/video changes; later scoped checks passed (follower 15, measured planner 5, FSM 11, seed/fork/bench 11, video 4, skill 2, audit 12/12). No fresh final full-suite result existed at the time of this draft. An earlier pre-resume configuration was 106 passed / 2 skipped / 13 xfailed.
