"""Policy registry + lock file (Phase A).

`nav/policies.lock.json` pins every ONNX the nav stack may run: path (relative
to the microduck root), md5, kind, how the 13D command block is driven, the
observation slot layout and the ONNX metadata. `Sim` refuses to load a policy
whose md5 differs from its lock entry.

Regenerate the lock (only when a policy file is intentionally replaced):
    $PY -m nav.policies --write

Imports only stdlib + onnxruntime (lazily), so it runs on the Linux batch node.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

NAV_ROOT = Path(__file__).resolve().parents[1]          # ~/microduck/nav
DEFAULT_LOCK_PATH = NAV_ROOT / "policies.lock.json"


def microduck_root() -> Path:
    """~/microduck, overridable with $MICRODUCK_ROOT (e.g. on the batch node)."""
    env = os.environ.get("MICRODUCK_ROOT")
    return Path(env).expanduser().resolve() if env else NAV_ROOT.parent


# 61D actor obs shared by the whole policy family (AGENTS.md invariant).
OBS_SLOTS = {
    "base_ang_vel": [0, 3],        # imu_ang_vel gyro sensor (body frame)
    "projected_gravity": [3, 6],   # R(trunk_base xquat)^T @ (0,0,-1)
    "joint_pos": [6, 20],          # q - DEFAULT_POSE (14, ctrl order)
    "joint_vel": [20, 34],         # dq (14)
    "actions": [34, 48],           # last raw policy action (14)
    "twist": [48, 51],             # command: vx, vy, wz | posture flag | phase cos/sin
    "head_pose": [51, 55],         # neck_pitch, head_pitch, head_yaw, head_roll deltas (0 in nav)
    "body_pose": [55, 61],         # x, y, z, roll, pitch, yaw (0 in nav)
}

# command_mode: how Sim.step(command) fills the 13D command block.
#   twist          cmd[0:3] = (vx, vy, wz)                     (walk / velstand)
#   zero           cmd = 0, command arg ignored                (stand, kicks, roulade, standup)
#   posture_flag   cmd[0] = command[0] (1 = sit, 0 = stand)    (sitstand)
#   phase          cmd[0:2] = cos/sin(2 pi phase), phase advanced by Sim (ground pick)
UPSTREAM = "upstream/policies"
FA = "runs/finished_atlantis"
AF = "runs/atlantis_final"
REGISTRY: dict[str, dict] = {
    # --- Pollen stand-ins (HF pollen-robotics/microduck-policies) ---
    "walk": dict(path=f"{UPSTREAM}/alpha_walking.onnx", kind="walk", command_mode="twist",
                 source="pollen", note="run 441tzs6d@3750, Velocity2-Rough-Backlash; primary nav walker"),
    "velstand": dict(path=f"{UPSTREAM}/velstand.onnx", kind="walk", command_mode="twist",
                     source="pollen", note="walk + stand at zero cmd (v8 Hub)"),
    "stand": dict(path=f"{UPSTREAM}/alpha_stand.onnx", kind="stand", command_mode="zero",
                  source="pollen"),
    "sitstand": dict(path=f"{UPSTREAM}/alpha_sitstand.onnx", kind="sitstand", command_mode="posture_flag",
                     source="pollen", ramp_s=2.0),
    "pick": dict(path=f"{UPSTREAM}/alpha_ground_pick.onnx", kind="pick", command_mode="phase",
                 source="pollen", period_s=4.0, end_phase=0.7, duration_s=2.8),
    "kick_left": dict(path=f"{UPSTREAM}/ball_kick_left.onnx", kind="kick_left", command_mode="zero",
                      source="pollen", duration_s=3.0, manifest_duration_s=0.5, needs_ball=True),
    "kick_right": dict(path=f"{UPSTREAM}/ball_kick_right.onnx", kind="kick_right", command_mode="zero",
                       source="pollen", duration_s=3.0, manifest_duration_s=0.5, needs_ball=True),
    "roulade": dict(path=f"{UPSTREAM}/roulade.onnx", kind="roulade", command_mode="zero",
                    source="pollen", duration_s=2.0, manifest_duration_s=1.0),
    # --- user's finished atlantis runs ---
    "atlantis_walk": dict(path=f"{FA}/walking/policy.onnx", kind="walk", command_mode="twist",
                          source="user", note="model_9999"),
    "atlantis_sitstand_flat": dict(path=f"{FA}/sitstand_flat/policy.onnx", kind="sitstand",
                                   command_mode="posture_flag", source="user", note="model_18499"),
    "atlantis_sitstand_rough": dict(path=f"{FA}/sitstand_rough/policy.onnx", kind="sitstand",
                                    command_mode="posture_flag", source="user", note="model_17750"),
    "atlantis_standup_flat": dict(path=f"{FA}/standup_flat/policy.onnx", kind="standup", command_mode="zero",
                                  source="user", note="model_16000; starts from SIT keyframe (extra init='sit')"),
    "atlantis_pick_flat": dict(path=f"{FA}/groundpick_flat_15000/policy.onnx", kind="pick", command_mode="phase",
                               source="user", period_s=4.0, end_phase=0.7, duration_s=2.8, note="model_15000"),
    "atlantis_velstand_rough_wip": dict(
        path="runs/atlantis_wip/velstand_rough_ws/2026-11-07_17-04-10_atlantis4_velstand_rough_ws_3000.onnx",
        kind="walk", command_mode="twist", source="user", note="WIP iter 3000 (superseded by velstand_rough)"),
    # --- runs/atlantis_final (copied 2026-11-09) ---
    "velstand_flat": dict(path=f"{AF}/velstand_flat_ws_2999.onnx", kind="walk", command_mode="twist",
                          source="user", note="final, iter 2999"),
    "velstand_rough": dict(path=f"{AF}/velstand_rough_ws_4249.onnx", kind="walk", command_mode="twist",
                           source="user", note="final, iter 4249"),
    "kick_right_user": dict(path=f"{AF}/ballkick_right_15749.onnx", kind="kick_right", command_mode="zero",
                            source="user", duration_s=3.0, needs_ball=True, note="final, iter 15749"),
    "standup_rough": dict(path=f"{AF}/standup_rough_partial.onnx", kind="standup", command_mode="zero",
                          source="user", note="PARTIAL (iter <= 6750 of 15000)"),
    "velocity_hard": dict(path=f"{AF}/velocity_hard_partial_4750.onnx", kind="walk", command_mode="twist",
                          source="user", note="Velocity-HardRough (stairs <=4 cm, slopes ~17 deg) resumed w/ "
                          "curriculum reset, iter 4750; walks (terrain sweep ~ alpha_walking, slightly faster)"),
    "velocity_hard_4250": dict(path=f"{AF}/velocity_hard_4250.onnx", kind="walk", command_mode="twist",
                               source="user", status="rejected",
                               note="DEGENERATE: collapsed to standing still (track_linear_velocity 0.018 at "
                               "iter 4250, stalls on every terrain cell). Do not use."),
}


class PolicyLockError(RuntimeError):
    pass


def md5_file(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _onnx_info(path: Path) -> dict:
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = dict(s.get_modelmeta().custom_metadata_map)
    inp, out = s.get_inputs()[0], s.get_outputs()[0]

    def floats(key):
        v = meta.get(key)
        return [float(x) for x in v.split(",")] if v else None

    return {
        "input_name": inp.name, "obs_dim": int(inp.shape[-1]),
        "output_name": out.name, "action_dim": int(out.shape[-1]),
        "observation_names": meta.get("observation_names", "").split(","),
        "command_names": meta.get("command_names", "").split(","),
        "joint_names": meta.get("joint_names", "").split(","),
        "default_joint_pos": floats("default_joint_pos"),
        "joint_stiffness": floats("joint_stiffness"),
        "joint_damping": floats("joint_damping"),
        "action_scale": float(meta.get("action_scale", "1.0")),
        "run_path": meta.get("run_path"),
        "metadata_raw": meta,
    }


def build_lock() -> dict:
    root = microduck_root()
    lock = {"_schema": 1, "_root": "paths relative to $MICRODUCK_ROOT (default ~/microduck)",
            "_obs_slots": OBS_SLOTS}
    for key, spec in REGISTRY.items():
        p = root / spec["path"]
        if not p.exists():
            print(f"[policies] skip {key}: {p} missing")
            continue
        info = _onnx_info(p)
        entry = dict(spec)
        entry["md5"] = md5_file(p)
        entry["size_bytes"] = p.stat().st_size
        entry["obs_slots"] = OBS_SLOTS if info["obs_dim"] == 61 else None
        entry.update({k: v for k, v in info.items()})
        if info["obs_dim"] != 61:
            entry["warning"] = f"obs_dim {info['obs_dim']} != 61: Sim cannot run this policy"
        lock[key] = entry
    return lock


def write_lock(path: str | Path = DEFAULT_LOCK_PATH) -> dict:
    lock = build_lock()
    Path(path).write_text(json.dumps(lock, indent=1) + "\n")
    return lock


def load_lock(path: str | Path = DEFAULT_LOCK_PATH) -> dict:
    return json.loads(Path(path).read_text())


def resolve(entry: dict) -> Path:
    return microduck_root() / entry["path"]


def verified_entry(key: str, lock_path: str | Path = DEFAULT_LOCK_PATH, lock: dict | None = None,
                   allow_rejected: bool = False) -> dict:
    """Lock entry for `key`, after checking the file's md5. Raises PolicyLockError.

    Entries with "status": "rejected" are refused unless allow_rejected=True.
    """
    lock = lock if lock is not None else load_lock(lock_path)
    if key not in lock or key.startswith("_"):
        raise PolicyLockError(f"policy key {key!r} not in lock {lock_path}; have "
                              f"{[k for k in lock if not k.startswith('_')]}")
    entry = lock[key]
    if entry.get("status") == "rejected" and not allow_rejected:
        raise PolicyLockError(f"{key}: lock status 'rejected' ({entry.get('note', '')}); "
                              "pass allow_rejected=True to run it anyway")
    p = resolve(entry)
    if not p.exists():
        raise PolicyLockError(f"{key}: {p} missing")
    got = md5_file(p)
    if got != entry["md5"]:
        raise PolicyLockError(f"{key}: md5 mismatch for {p}: lock {entry['md5']} file {got}. "
                              "Refusing to run. Re-lock deliberately with `python -m nav.policies --write`.")
    if entry.get("obs_dim") != 61 or entry.get("action_dim") != 14:
        raise PolicyLockError(f"{key}: obs_dim {entry.get('obs_dim')} / action_dim {entry.get('action_dim')} "
                              "not the 61/14 layout Sim implements")
    return entry


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.write:
        lk = write_lock()
        print(f"wrote {DEFAULT_LOCK_PATH}: {[k for k in lk if not k.startswith('_')]}")
    if a.check or not a.write:
        lk = load_lock()
        for k in lk:
            if not k.startswith("_"):
                verified_entry(k, lock=lk, allow_rejected=True)
                print(f"ok {k:30s} {lk[k]['md5']} {lk[k]['kind']} {lk[k].get('status', '')}")
