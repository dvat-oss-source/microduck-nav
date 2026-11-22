"""Lateral-error sigma used to inflate the planning map (see reports/REQUESTS.md).

Plan: inflation radius = footprint radius + 3 sigma of the sideways tracking error.
Phase B's envelope ``lateral_sigma`` is OPEN-LOOP end-y spread after 4 s of straight
walking; with it, the apartment's doorways close for most walkers. The error the robot
actually exhibits while navigating is the closed-loop cross-track error of the path
follower, measured on the real Sim by tools/follower_sim_gate.py. That value is used
for the frozen map; the open-loop value stays in the envelope as a B measurement.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def planning_lateral_sigma(policy, env, reports_dir=ROOT / "reports"):
    """Return (sigma, provenance dict). Raises if no matching follower gate exists."""
    path = Path(reports_dir) / f"follower_sim_{policy}.json"
    if not path.exists():
        raise RuntimeError(f"no real-Sim follower gate for {policy}: run tools/follower_sim_gate.py "
                           f"--policy {policy} --out {path.relative_to(ROOT)}")
    gate = json.loads(path.read_text())
    follower_hash = hashlib.sha256((Path(__file__).parent / "follower.py").read_bytes()).hexdigest()
    if gate.get("follower_sha256") != follower_hash:
        raise RuntimeError(f"{path.name} was produced by a different follower; rerun its real-Sim gate")
    if (gate.get("policy_md5") != env["policy_md5"] or gate.get("dynamics") != env.get("dynamics")
            or gate.get("dynamics_version") != env.get("dynamics_version")):
        raise RuntimeError(f"{path.name} does not match the envelope (md5/dynamics)")
    runs = gate["runs"]
    # Pool squared cross-track error across runs (each run's RMS weighted equally).
    sigma = math.sqrt(sum(r["cross_track_rms"] ** 2 for r in runs) / len(runs))
    return sigma, {"source": "closed_loop_follower_gate", "file": path.name,
                   "follower_sha256": follower_hash,
                   "n_runs": len(runs), "sigma": sigma,
                   "open_loop_lateral_sigma": env["lateral_sigma"],
                   "follower_gate_passed": bool(gate.get("passed"))}
