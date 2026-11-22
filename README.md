# microduck-nav

Simulation-only navigation stack for a small bipedal robot (CPU MuJoCo + BAM): mapping, lattice planner, MPPI path follower, learned + fiducial odometry, skill FSM.

- `nav/` — runtime stack (planner, follower, mapping, localization, FSM)
- `tools/` — gates, benchmarks, rendering
- `tests/` — test suite
- `CONTRACTS.md` — interface spec; `policies.lock.json` — policy registry
- `reports/` — final reports

Large artifacts (seed sets, traces, models, videos under `data/` and `reports/`) are not included.
