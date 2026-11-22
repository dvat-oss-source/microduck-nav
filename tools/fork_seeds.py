"""Explicit provenance-only fork: preserve every frozen task; never resample or overwrite."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_tasks(doc, gridmap, validator, primitives=None):
    """Recheck unchanged tasks against the current map, resets and stable objects.  With
    primitives, also require that the planner can produce a start lattice node (A1)."""
    from nav.skills import kick_pre_pose, pick_pre_pose
    checked = {"nav": 0, "kick": 0, "pick": 0}
    start_nodes = {"nearest": 0, "repaired": 0}
    for kind in checked:
        for i, task in enumerate(doc[kind]):
            label = f"{kind}_{i:03d}"
            objects = copy.deepcopy(task["object_poses"])
            target = task.get("ball_body", task.get("target_body"))
            if target:
                objects[target] = task["object_pose"]
            validator.set_layout(objects)
            poses = [("start", task["start"], None)]
            if kind == "nav":
                poses.append(("goal", task["goal"], None))
            else:
                xy, direction = task["object_pose"][:2], task["direction"]
                pres = ([kick_pre_pose(xy, direction, side) for side in ("right", "left")]
                        if kind == "kick" else [pick_pre_pose(xy, direction)])
                poses.extend((f"pre_{j}", p, target) for j, p in enumerate(pres))
            for role, pose, exclude in poses:
                if gridmap.is_lethal(*pose[:2]) or not validator.pose_ok(
                        task["seed"], pose, objects, exclude=exclude):
                    raise RuntimeError(f"unchanged {label} {role} invalid; fork aborted without resampling")
            if primitives is not None:
                from nav.seed_validation import planner_start_check
                ok, info = planner_start_check(gridmap, primitives, task["start"])
                if not ok:
                    raise RuntimeError(f"unchanged {label} start has no planner start node ({info}); fork aborted")
                start_nodes[info["mode"]] += 1
            if not validator.stable(task["seed"], task["start"], objects):
                raise RuntimeError(f"unchanged {label} unstable objects; fork aborted without resampling")
            checked[kind] += 1
            if sum(checked.values()) % 25 == 0:
                print(f"validated {sum(checked.values())}/300 unchanged tasks", flush=True)
    out = {"dry_sim_reset": True, "objects_stable_1s": True, "unchanged_tasks_checked": checked,
           "rejections": dict(validator.rejections)}
    if primitives is not None:
        out["planner_start_nodes"] = start_nodes
    return out


def fork(source, out, version, scene, revision="controller revision; unchanged training dynamics and identical tasks", dev=False, follower_reports_dir=None):
    from nav.bench import load_seeds, seeds_digest
    from nav.episode import read_artifacts
    from nav.mapping import build_map
    from nav.mapsigma import planning_lateral_sigma
    from nav.provenance import scene_sha256
    from nav.seed_validation import SeedValidator
    source, out = Path(source), Path(out)
    dev_dir = (ROOT / "data" / "dev").resolve()
    in_dev_dir = out.resolve().is_relative_to(dev_dir)
    if dev != in_dev_dir:
        raise ValueError("--dev forks must be under data/dev/; other forks must be outside data/dev/")
    if follower_reports_dir and not dev:
        raise ValueError("custom follower gates require --dev")
    if out.exists() or source.resolve() == out.resolve():
        raise FileExistsError("fork destination must be new; frozen seed files cannot be overwritten")
    original_bytes = source.read_bytes()
    parent = load_seeds(str(source))
    if version == parent["version"]:
        raise ValueError("fork needs a new explicit version")
    p = parent["provenance"]
    policy = p["policy"]
    env, _ = read_artifacts(policy)
    invariant = {
        "policy_md5": env["policy_md5"], "dynamics": env["dynamics"],
        "dynamics_version": env["dynamics_version"], "step_over_height": env["step_over_height"],
        "envelope_sha256": digest(ROOT / "data" / f"envelope_{policy}.json"),
        "primitives_sha256": digest(ROOT / "data" / f"primitives_{policy}.json"),
        "skill_contracts_sha256": digest(ROOT / "data" / "skill_contracts.json"),
        "checker_sha256": digest(ROOT / "nav" / "checker.py"),
        "scene_sha256": scene_sha256(scene, "training"),
    }
    for key, value in invariant.items():
        if p.get(key) != value:
            raise RuntimeError(f"{key} changed; this helper permits controller revisions only")
    sigma, sigma_prov = (planning_lateral_sigma(policy, env, follower_reports_dir)
                         if follower_reports_dir else planning_lateral_sigma(policy, env))
    if not sigma_prov["follower_gate_passed"]:
        raise RuntimeError("cannot fork for a failed current follower gate")
    gm = build_map(scene, lateral_sigma=sigma, step_over_height=env["step_over_height"])
    validation = validate_tasks(parent, gm, SeedValidator(scene, policy, gm),
                                str(ROOT / "data" / f"primitives_{policy}.json"))
    doc = copy.deepcopy(parent)
    doc.update(version=version, parent_seed_sha256=parent["sha256"])
    doc["provenance"].update(lateral_sigma=sigma, lateral_sigma_provenance=sigma_prov,
                             seed_validation=validation,
                             fsm_sha256=digest(ROOT / "nav" / "fsm.py"),
                             localization_sha256=digest(ROOT / "nav" / "localization.py"),
                             skills_sha256=digest(ROOT / "nav" / "skills.py"),
                             episode_sha256=digest(ROOT / "nav" / "episode.py"),
                             planner_sha256=digest(ROOT / "nav" / "planner.py"),
                             mapping_sha256=digest(ROOT / "nav" / "mapping.py"),
                             mapsigma_sha256=digest(ROOT / "nav" / "mapsigma.py"),
                             revision=revision)
    if dev:
        doc["provenance"]["dev"] = True
    else:
        doc["provenance"].pop("dev", None)
    assert all(doc[k] == parent[k] for k in ("nav", "kick", "pick", "master_seed"))
    if source.read_bytes() != original_bytes:
        raise RuntimeError("source seed bytes changed during validation; fork aborted")
    doc["sha256"] = seeds_digest(doc)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(doc, handle, indent=1)
    print(json.dumps({"out": str(out), "sha256": doc["sha256"],
                      "parent_seed_sha256": parent["sha256"], "validation": validation}, indent=2))
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--dev", action="store_true", help="write a development seed fork under data/dev/")
    ap.add_argument("--follower-reports-dir", help="development: directory with matching passing follower gates")
    ap.add_argument("--revision", default="controller revision; unchanged training dynamics and identical tasks",
                    help="human-readable reason for this provenance-only fork (recorded in the seed file)")
    ap.add_argument("--scene", default=str(Path.home() / "microduck/upstream/microduck_rl/src/mjlab_microduck/robot/microduck/scene_apartment.xml"))
    args = ap.parse_args()
    fork(args.source, args.out, args.version, args.scene, args.revision, dev=args.dev, follower_reports_dir=args.follower_reports_dir)


if __name__ == "__main__":
    main()
