# v5 FINAL — integrated validation (Stage 5)

Date 2026-11-12 (runs 21:48–23:04 UTC). **All results are DEV** (dev seed fork, `--follower-reports-dir`), simulation only, `atlantis_walk`, training dynamics `cb8561f6_nominal_v2`, 3 workers, every sim run under SIM_LOCK via `reports/dev/v5/A6/validation/run_locked.py`.
Driver: `reports/dev/v5/final_20261112/run_step.sh` (one step per locked run; refuses to overwrite a finished step). All evidence under `reports/dev/v5/final_20261112/`.

## Combined tree tested (live, no loader)

| file | sha256 | stage |
|---|---|---|
| nav/localization.py | 454e2b93… | A3 (Stage 1), run as `--mode learned+fiducial --ablation 1a` |
| nav/follower.py | fe8c2f50… | A5 MPPI (Stage 2) |
| nav/planner.py | 5b4f101d… | A6 (Stage 3) on A1 |
| nav/mapping.py | 9c5625db… (MAP_VERSION 5) | A6 |
| nav/a6_footprint.py | e74fd85b… | A6 |

Each step records hashes in `<step>/live_sha256.txt`. Skills: A7 approach unchanged. Nothing was promoted or edited in nav/ during Stage 5.

## Validation ladder: all rungs pass

| rung | result | evidence |
|---|---|---|
| 1 focused tests (planner, map, follower ×5, localization) | 88 passed | `rung1_focused/pytest.log` |
| 2 E gate ×3 walkers | walk/atlantis_walk/velocity_hard 20/20 each, 0 falls, worst RMS 21.6/18.8/16.7 mm | `rung2_egate/` |
| 3 strict fork | 300/300 unchanged tasks (nav/kick/pick identical to seeds_v3) → `data/dev/seeds_v5dev_FINAL_20261112T215037Z.json` | `rung3_fork/` |
| 4 planner gate | PASS a–d: 100/100 complete, 0 lethal, plan p95 188 ms (wall 192 ms), heuristic 7/7 | `rung4_planner/gate/` |
| 5 nav_125 smoke | success, SPL 0.872 | `rung5_nav125/` |
| 6 exact fixed-60 GT | 60/60, SPL 0.7949, 0 falls/holes (matches A6 accept 0.7949) | `rung6_fixed60/` |
| 7 full suite + protected | 232 passed / 1 skipped / 1 xfailed; protected 3307/3307 unchanged | `suite/` |

## Full benchmarks (new DEV fork, 300 tasks; nav and kick+pick run as separate locked batches)

| metric | target | v2/v3 baseline (FINAL_REPORT) | **v5 integrated** | verdict |
|---|---|---|---|---|
| GT nav success | ≥95% | 187/200 (93.5%) | **199/200 (99.5%, CI 97.2–99.9)** | PASS |
| GT nav SPL | ≥0.80 | 0.666 | **0.7976 (CI 0.783–0.811)** | **FAIL (0.0024 short)** |
| GT falls | ≤2% | 0% | 0.5% (1, nav_167) | PASS |
| GT stair holes | 0 | 5 | **0** | PASS |
| Odom nav success | ≥85% | 121/200 (60.5%), odom+fiducial v3 | **179/200 (89.5%, CI 84.5–93.0)** | PASS |
| Odom nav SPL | ≥0.70 | 0.419 | **0.7152 (CI 0.680–0.751)** | PASS (CI lower bound below target) |
| Odom falls | ≤2% | 0.5% | 0% | PASS |
| Odom stair holes | 0 | 8 | **0** | PASS |
| GT kick | — | 0/50 (v2); A7 DEV 21/50 | 19/50 | below A7 accept |
| Odom kick | — | 0/50 | 9/50 | — |
| Pick (both modes) | — | 0/50 | 0/50 (`skill_unsupported_pick`) | blocked, as before |
| Planner p95 (nav) | gate | 0.289 s GT / 0.601 s odom | 0.168 s GT / 0.195 s odom | — |

The benchmark's own `gate.passed` is false for every run. It also requires `full_frozen_suite`, which a DEV fork cannot satisfy, and a pending reference replay. This table reports the numeric targets only.

### Failures
- **GT (1/200):** nav_167 fall → `recovery_unsupported`. This is a lost success (v2 succeeded). 13 v2 failures now succeed, including all 5 stair-hole episodes.
- **Odom (21/200):** 20 `pos_tol` + 1 `goal_not_settled` (nav_065). All 21 end 5.0–8.4 cm from the goal (limit 5 cm), so they are terminal localization errors, not navigation failures. Versus v3 odom: +72 gained, 14 lost (nav_032 065 068 076 098 100 111 114 120 128 146 175 183 187). nav_065 and nav_090 match the A6 rung-7 odom misses.
- **Kick GT 19/50 vs A7's 21/50.** A7 was measured with the A4 follower and MAP_VERSION 4. This is a small drop (CI 25.9–51.9% covers 42%) and was not investigated further. Main stack reasons: ball unstable 12, staging blocked 7, stage clearance 4, run-in obstacle 3.

## Video
Not made with `tools/run_comparison_serial_video.py`: it hardcodes `procs=4`, which breaks the 3-worker cap. Instead, `tools/render_showcase.py` rendered existing traces side by side (no new simulation):
- `final_20261112/video/before_after_nav_065.mp4`: v2 GT (stair hole) vs v5 GT (success).
- `final_20261112/video/before_after_nav_167.mp4`: v2 GT (success) vs v5 GT (the one new fall).

## Stage accept/reject

| stage | outcome | evidence |
|---|---|---|
| 1 A3 localization | ACCEPTED DEV; 1a carried forward. 1b anchors hurt (33/60), 1c no measurable gain; both unused | `A3/gates/RESULTS.md` |
| 2 A5 MPPI follower | ACCEPTED DEV, promoted fe8c2f50. Optional smoother REJECTED | `A5/RESULTS.md` |
| 3 A6 heading-aware footprint | ACCEPTED DEV, promoted (MAP_VERSION 5). nav_070/nav_158 now reachable | `A6/RESULTS.md` |
| 4 skills plan | DONE, document only, nothing trained | `SKILLS_TRAINING_PLAN.md` |
| 5 integrated validation | ladder all pass. Targets: odom success/SPL, falls and holes PASS; **GT SPL FAIL 0.7976 vs 0.80**; kick/pick below need | this file |

## Caveats
DEV fork only; not a frozen-suite certification. Single walker. Learned-odometry thresholds came from independent routes only. No mjlab reference replay was run for v5. Pick and recovery remain blocked (Stage 4 plan).
