"""Freeze feasible tasks once, with a footprint-only any-angle path metric and dry Sim checks."""
import heapq
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
DEFAULT_SCENE = os.path.expanduser(
    "~/microduck/upstream/microduck_rl/src/mjlab_microduck/robot/microduck/scene_apartment.xml")
HOLE_BOX = (-0.40, -1.40, 0.0, -0.70)  # only used to *bias* sampling; checker derives its own
BALLS = ["ball_0", "ball_1", "ball_2"]
OBJS = ["obj_0", "obj_1", "obj_2", "obj_3"]
OBJ_XY = {"ball_0": (-3.0, 0.0), "ball_1": (2.4, 2.3), "ball_2": (-0.3, -1.4),
          "obj_0": (-2.2, -1.2), "obj_1": (-2.5, 1.4), "obj_2": (2.1, -2.4), "obj_3": (-0.3, 0.1)}


def _in_box(xy, box):
    return box[0] <= xy[0] <= box[2] and box[1] <= xy[1] <= box[3]


def _bbox(r):
    a = np.asarray(r, float)
    if a.ndim == 1 and len(a) == 4:
        return a[0], a[1], a[2], a[3]
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


def _supercover(a, b):
    """All grid squares touched by a segment between cell centres; corners include both sides."""
    y, x = a
    dy, dx = b[0] - y, b[1] - x
    sy, sx = int(np.sign(dy)), int(np.sign(dx))
    tx = 0.5 / abs(dx) if dx else math.inf
    ty = 0.5 / abs(dy) if dy else math.inf
    yield y, x
    while (y, x) != tuple(b):
        if abs(tx - ty) < 1e-12:
            yield y, x + sx
            yield y + sy, x
            y, x = y + sy, x + sx
            tx += 1.0 / abs(dx); ty += 1.0 / abs(dy)
        elif tx < ty:
            x += sx; tx += 1.0 / abs(dx)
        else:
            y += sy; ty += 1.0 / abs(dy)
        yield y, x


def _visible(gm, a, b):
    h, w = gm.lethal.shape
    return all(0 <= i < h and 0 <= j < w and not gm.lethal[i, j]
               for i, j in _supercover(a, b))


def _metric_graph(gm):
    cached = getattr(gm, "_geodesic_graph", None)
    if cached is not None:
        return cached
    free = ~np.asarray(gm.lethal)
    h, w = free.shape
    ids = np.arange(h * w).reshape(h, w)
    rows, cols, values = [], [], []
    neighbours = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
    neighbours += [(i, j) for i in (-2, -1, 1, 2) for j in (-2, -1, 1, 2) if abs(i) + abs(j) == 3]
    for di, dj in neighbours:
        lo_i, hi_i = max(0, -di), min(h, h - di)
        lo_j, hi_j = max(0, -dj), min(w, w - dj)
        ok = np.ones((hi_i - lo_i, hi_j - lo_j), bool)
        for oi, oj in _supercover((0, 0), (di, dj)):
            ok &= free[lo_i + oi:hi_i + oi, lo_j + oj:hi_j + oj]
        src = ids[lo_i:hi_i, lo_j:hi_j][ok]
        rows.append(src)
        cols.append(src + di * w + dj)
        values.append(np.full(len(src), math.hypot(di, dj) * gm.res))
    graph = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(h * w, h * w)).tocsr()
    gm._geodesic_graph = graph
    return graph


def geodesic_from(gm, src_xy, return_predecessors=False):
    shape = gm.lethal.shape
    i, j = gm.world_to_cell(*src_xy)
    if not (0 <= i < shape[0] and 0 <= j < shape[1]) or gm.lethal[i, j]:
        empty = np.full(shape, np.inf)
        return (empty, None) if return_predecessors else empty
    dist, pred = dijkstra(_metric_graph(gm), directed=True, indices=i * shape[1] + j,
                          return_predecessors=True)
    return (dist.reshape(shape), pred) if return_predecessors else dist.reshape(shape)


def metric_path_length(gm, start, goal, predecessors):
    """16-connected Dijkstra followed by greedy line-of-sight string pulling."""
    a, b = tuple(gm.world_to_cell(*start)), tuple(gm.world_to_cell(*goal))
    if _visible(gm, a, b):
        return math.dist(start, goal)
    width = gm.lethal.shape[1]
    node, source = b[0] * width + b[1], a[0] * width + a[1]
    path = [b]
    while node != source:
        node = int(predecessors[node])
        if node < 0:
            return math.inf
        path.append(divmod(node, width))
    path.reverse()
    points = [path[0]]
    k = 0
    while k < len(path) - 1:
        nxt = len(path) - 1
        while nxt > k + 1 and not _visible(gm, path[k], path[nxt]):
            nxt -= 1
        points.append(path[nxt]); k = nxt
    def world(cell):
        if hasattr(gm, "cell_to_world"):
            return tuple(gm.cell_to_world(*cell))
        return (start[0] + (cell[1] - a[1]) * gm.res,
                start[1] + (cell[0] - a[0]) * gm.res)
    xy = [tuple(start)] + [world(c) for c in points[1:-1]] + [tuple(goal)]
    return sum(math.dist(x, y) for x, y in zip(xy, xy[1:]))


def _sample_free(gm, rng, box, margin=0.0):
    x0, y0, x1, y1 = box
    for _ in range(500):
        x, y = rng.uniform(x0 + margin, x1 - margin), rng.uniform(y0 + margin, y1 - margin)
        if not gm.is_lethal(x, y):
            return float(x), float(y)
    return None


def generate(build_map, scene, out_path, master_seed=20250101, version="v1", provenance=None,
             policy="atlantis_walk", metric_map=None, validate_sim=None):
    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} exists; frozen seeds cannot be overwritten")
    gm = build_map(scene)
    metric_map = metric_map or gm
    rooms = {k: _bbox(v) for k, v in gm.rooms.items()}
    names = sorted(rooms)
    rng = np.random.default_rng(master_seed)
    nav = []
    if validate_sim is None:
        # In-memory synthetic-map tests have no scene. The production CLI always enables this.
        validate_sim = Path(scene).is_file()
    validator = None
    layout = {}
    if validate_sim:
        from nav.seed_validation import SeedValidator
        validator = SeedValidator(scene, policy, gm)
        for _ in range(200):
            layout = {}
            for name in BALLS + OBJS:
                for _ in range(500):
                    room = names[int(rng.integers(len(names)))]
                    xy = _sample_free(gm, rng, rooms[room], .2)
                    if xy and all(math.dist(xy, pose[:2]) > .4 for pose in layout.values()):
                        layout[name] = [*xy, 0., 1., 0., 0., 0.]
                        break
            if len(layout) != len(BALLS + OBJS):
                continue
            validator.set_layout(layout)
            probe = _sample_free(gm, rng, rooms[names[0]], .2)
            if probe and validator.stable(master_seed, [*probe, 0.], layout):
                break
        else:
            raise RuntimeError("could not construct a stable free-object scene layout")

    def add(cat, s, g, ds, predecessors):
        if not math.isfinite(float(ds[metric_map.world_to_cell(*g)])):
            return False
        d = metric_path_length(metric_map, s, g, predecessors)
        if not math.isfinite(d) or d < 0.5:
            return False
        seed = int(rng.integers(0, 2**31 - 1))
        sp = [s[0], s[1], float(rng.uniform(-math.pi, math.pi))]
        gp = [g[0], g[1], float(rng.uniform(-math.pi, math.pi))]
        if validator and not (validator.pose_ok(seed, sp) and validator.pose_ok(seed, gp)):
            return False
        nav.append({"type": "nav", "category": cat, "seed": seed, "start": sp, "goal": gp,
                    "pos_tol": 0.05, "yaw_tol_deg": 10.0, "geodesic": d,
                    "timeout_s": max(120., 25. * d), "object_poses": layout,
                    "room_pair": [next((r for r in names if _in_box(s, rooms[r])), "unknown"),
                                  next((r for r in names if _in_box(g, rooms[r])), "unknown")]})
        return True

    def pair(cat, room_a, room_b, near_hole=False):
        for _ in range(200):
            s = _sample_free(gm, rng, rooms[room_a], 0.15)
            if s is None:
                return False
            g = (_sample_free(gm, rng, HOLE_BOX_EXP, 0.0) if near_hole else _sample_free(gm, rng, rooms[room_b], 0.15))
            if g is None:
                continue
            ds, predecessors = geodesic_from(metric_map, s, return_predecessors=True)
            if add(cat, s, g, ds, predecessors):
                return True
        return False

    HOLE_BOX_EXP = (HOLE_BOX[0] - 0.5, HOLE_BOX[1] - 0.5, HOLE_BOX[2] + 0.5, HOLE_BOX[3] + 0.5)
    # 60 same-room, 90 cross-room (round-robin over ordered pairs, forces doorways), 25 near-hole, rest random
    for i in range(60):
        r = names[i % len(names)]; pair("same_room", r, r)
    pairs = [(a, b) for a in names for b in names if a != b]
    for i in range(90):
        a, b = pairs[i % len(pairs)]; pair("cross_room", a, b)
    for i in range(25):
        pair("near_hole", names[i % len(names)], None, near_hole=True)
    guard = 0
    while len(nav) < 200 and guard < 2000:
        guard += 1
        pair("random", names[int(rng.integers(len(names)))], names[int(rng.integers(len(names)))])
    if len(nav) < 200:
        raise SystemExit("could only generate %d/200 nav tasks (map too constrained?)" % len(nav))
    nav = nav[:200]

    def object_task(kind, body, k):
        from nav.skills import kick_pre_pose, pick_pre_pose
        for _ in range(1000):
            if validator:
                room = names[int(rng.integers(len(names)))]
                xy = _sample_free(gm, rng, rooms[room], .2)
                if xy is None:
                    continue
                ox, oy = xy
            else:
                ox, oy = OBJ_XY[body]
            direction = float(rng.uniform(-math.pi, math.pi))
            pres = ([kick_pre_pose((ox, oy), direction, side) for side in ("right", "left")]
                    if kind == "kick" else [pick_pre_pose((ox, oy), direction)])
            if validator and any(gm.is_lethal(*pre[:2]) for pre in pres):
                continue
            ang = rng.uniform(-math.pi, math.pi); radius = rng.uniform(.3, .6)
            start = [ox + radius * math.cos(ang), oy + radius * math.sin(ang),
                     math.atan2(-math.sin(ang), -math.cos(ang))]
            if gm.is_lethal(*start[:2]):
                continue
            seed = int(rng.integers(0, 2**31 - 1))
            pose = [ox, oy, 0., 1., 0., 0., 0.]
            objects = dict(layout); objects[body] = pose
            if validator:
                if any(math.dist(pose[:2], other[:2]) < .15 for name, other in objects.items() if name != body):
                    continue
                if any(not validator.pose_ok(seed, pre, objects, exclude=body) for pre in pres):
                    continue
                if not validator.stable(seed, start, objects):
                    continue
            t = {"type": kind, "seed": seed, "start": start, "direction": direction,
                 "object_pose": pose, "object_poses": layout, "timeout_s": 120.,
                 "object_validation": {"duration_s": 1., "max_displacement_m": .01,
                                       "passed": bool(validator)}}
            if kind == "kick":
                t.update(ball_body=body, min_speed=.3, angle_tol_deg=30.)
            else:
                t.update(target_body=body, max_drift=.01, min_lift=.02, upright_tol_deg=30.)
            return t
        raise RuntimeError(f"no feasible {kind} task for {body}")

    kick = [object_task("kick", BALLS[i % 3], i) for i in range(50)]
    pick = [object_task("pick", OBJS[i % 4], i) for i in range(50)]
    doc = {"version": version, "master_seed": master_seed, "scene": os.path.basename(scene),
           "nav": nav, "kick": kick, "pick": pick}
    if provenance is not None:
        doc["provenance"] = dict(provenance, seed_validation={"dry_sim_reset": bool(validator),
                                    "objects_stable_1s": bool(validator),
                                    "rejections": validator.rejections if validator else {}})
    from nav.bench import seeds_digest
    doc["sha256"] = seeds_digest(doc)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "x") as handle:
        json.dump(doc, handle, indent=1)
    return doc


def main():
    try:
        from nav.mapping import build_map
    except Exception as e:
        sys.exit("make_seeds: nav.mapping.build_map is not available yet (%r). "
                 "Phase C must land first; seeds not generated." % (e,))
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default=DEFAULT_SCENE)
    ap.add_argument("out", nargs="?", default=os.path.join(ROOT, "data", "seeds_v1.json"))
    ap.add_argument("--policy", required=True, help="measured walker used to inflate the frozen map")
    args = ap.parse_args()
    scene, out = args.scene, args.out
    if os.path.exists(out):
        sys.exit("%s exists; seeds are frozen. Delete manually to regenerate (bumps baseline)." % out)
    from nav.episode import read_artifacts
    env, _ = read_artifacts(args.policy)
    if env.get("dynamics") != "training":
        raise RuntimeError("frozen benchmark requires current training dynamics measurements")
    from nav.mapsigma import planning_lateral_sigma
    sigma, sigma_prov = planning_lateral_sigma(args.policy, env)
    if not sigma_prov["follower_gate_passed"]:
        raise RuntimeError("cannot freeze seeds for a walker that failed its follower gate")
    from nav.skills import load_contracts
    protocol = load_contracts().get("protocol", {})
    if (protocol.get("dynamics_version") != env["dynamics_version"] or
            protocol.get("stand_policy") != args.policy or
            protocol.get("stand_policy_md5") != env["policy_md5"]):
        raise RuntimeError("regenerate skill contracts for the selected walker before freezing seeds")
    envelope_path = os.path.join(ROOT, "data", f"envelope_{args.policy}.json")
    from nav.provenance import scene_sha256
    provenance = {"policy": args.policy, "policy_md5": env["policy_md5"],
                  "dynamics": env.get("dynamics", "infer"), "dynamics_version": env["dynamics_version"],
                  "envelope_sha256": hashlib.sha256(open(envelope_path, "rb").read()).hexdigest(),
                  "primitives_sha256": hashlib.sha256(Path(ROOT, "data", f"primitives_{args.policy}.json").read_bytes()).hexdigest(),
                  "skill_contracts_sha256": hashlib.sha256(Path(ROOT, "data", "skill_contracts.json").read_bytes()).hexdigest(),
                  "checker_sha256": hashlib.sha256(Path(ROOT, "nav", "checker.py").read_bytes()).hexdigest(),
                  "lateral_sigma": sigma, "lateral_sigma_provenance": sigma_prov,
                  "step_over_height": env["step_over_height"],
                  "scene_sha256": scene_sha256(scene, "training"),
                  "path_smoothing_steps": 30, "goal_dwell_s": 1., "record_kick_mirror": True,
                  "geodesic_method": "footprint_16_dijkstra_los_v1",
                  "timeout": "max(120 s, 25 s/m * geodesic)",
                  "geodesic": "footprint-radius-only map; 16-connected Dijkstra without corner cutting; line-of-sight string pulling"}
    def measured_map(scene):
        return build_map(scene, lateral_sigma=sigma, step_over_height=env["step_over_height"])
    metric = build_map(scene, lateral_sigma=0., step_over_height=env["step_over_height"])
    d = generate(measured_map, scene, out, provenance=provenance, policy=args.policy,
                 metric_map=metric, validate_sim=True)
    print("wrote", out, d["sha256"])


if __name__ == "__main__":
    main()
