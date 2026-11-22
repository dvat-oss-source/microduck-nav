"""A1: bounded start/goal snap repair (nav/planner.py, 'Endpoint snap repair')."""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nav import planner as P  # noqa: E402
from nav.seed_validation import planner_start_check  # noqa: E402
from nav.testing_fixtures import SyntheticGridMap, make_house_map, make_synthetic_primitives  # noqa: E402

PRIMS = make_synthetic_primitives()


def _open_map():
    H, W = 100, 120
    occ = np.zeros((H, W), bool)
    occ[0:3], occ[-3:], occ[:, 0:3], occ[:, -3:] = True, True, True, True
    return SyntheticGridMap(occ, np.zeros_like(occ), rooms={})


def _mark_lethal(gm, cells):
    for i, j in cells:
        gm.lethal[i, j] = True
        gm.cost[i, j] = np.inf


def _boundary_starts(gm, pl, n=6, seed=0):
    """Free true poses whose nearest lattice node is lethal (the A1 failure geometry)."""
    rng = np.random.default_rng(seed)
    out = []
    H, W = gm.lethal.shape
    for _ in range(200000):
        xy = pl.view.world(rng.uniform([0, 0], [H - 1, W - 1]))
        pose = (float(xy[0]), float(xy[1]), float(rng.uniform(-math.pi, math.pi)))
        s = pl.snap(pose)
        if s >= 0 and not pl.node_ok[s // pl.nh] and not pl.view.lethal_at_world(xy)[0]:
            out.append(pose)
            if len(out) >= n:
                break
    return out


def test_repair_radius_is_derived_from_lattice():
    pl = P.get_planner(make_house_map(), PRIMS)
    assert math.isclose(pl.repair_radius, math.sqrt(2) * pl.P.lattice_cells * pl.view.res)


def test_free_start_with_lethal_snapped_node_now_plans():
    gm = make_house_map()
    pl = P.get_planner(gm, PRIMS)
    starts = _boundary_starts(gm, pl)
    assert starts, "fixture must contain the A1 geometry"
    goal = (5.5, 3.0, None)
    for st in starts:
        r = P.plan(gm, PRIMS, st, goal, budget_s=5.0)
        assert r.start_snap["mode"] == "repaired"
        if not r.complete:                      # unreachable regions are fine; refusal must not be start_lethal
            assert r.reason != "start_lethal"
            continue
        assert np.allclose(r.poses[0][:2], st[:2])          # path starts at the true pose
        assert r.start_snap["connector_m"] <= pl.repair_radius + 1e-12
        assert pl.connector_free(st[:2], r.start_snap["node_xy"])
        for x, y, _ in r.poses:
            assert not gm.is_lethal(x, y)
        ok, info = planner_start_check(gm, PRIMS, st)
        assert ok and info["mode"] == "repaired"


def test_lethal_true_start_still_refused():
    H, W = 150, 200
    occ = np.zeros((H, W), bool)
    occ[0:3], occ[-3:], occ[:, 0:3], occ[:, -3:] = True, True, True, True
    occ[:, 98:102] = True
    gm = SyntheticGridMap(occ, np.zeros_like(occ), rooms={})
    r = P.plan(gm, PRIMS, (2.0, 1.5, 0.0), (0.8, 1.5, 0.0))
    assert not r.complete and r.reason == "start_lethal"
    assert r.start_snap["reason"] == "true_pose_lethal"
    ok, _ = planner_start_check(gm, PRIMS, (2.0, 1.5, 0.0))
    assert not ok


def test_connector_is_swept_checked():
    gm = _open_map()
    pl0 = P.get_planner(gm, PRIMS)
    # true pose in the middle of a lattice square; nearest node made lethal, a wall of lethal
    # fine cells cuts off the second-nearest node.
    f = np.array([50.0, 60.0]) + np.array([0.7, 0.3])        # continuous fine index
    xy = pl0.view.world(f)
    nodes = [(50, 60), (52, 60), (50, 62), (52, 62)]
    d = [np.linalg.norm(pl0.view.world(np.array(n, float)) - xy) for n in nodes]
    order = [nodes[k] for k in np.argsort(d)]
    gm2 = _open_map()
    _mark_lethal(gm2, [order[0]])
    blocked = order[1]
    # lethal wall on the fine cells between the true pose and the second node (not touching the pose cell)
    pts = np.linspace(f, np.array(blocked, float), 20)[1:]
    wall = {tuple(np.rint(p).astype(int)) for p in pts} - {tuple(np.rint(f).astype(int))}
    _mark_lethal(gm2, wall)
    pl = P.get_planner(gm2, PRIMS)
    pose = (float(xy[0]), float(xy[1]), 0.0)
    assert not pl.view.lethal_at_world(xy)[0]
    assert not pl.connector_free(xy, pl.view.world(np.array(blocked, float)))
    s, info = pl.repair_start(pose)
    assert s >= 0
    chosen = tuple(np.rint(pl.view.fidx(np.array(info["node_xy"]))).astype(int))
    assert chosen not in (order[0], blocked)
    assert pl.connector_free(xy, info["node_xy"])


def _goal_repair_case(pl0):
    """Goal free, every node within goal_tol lethal, and some node in (goal_tol, repair_radius]
    with a free connector to the goal."""
    for a in np.arange(0.0, 2.0, 0.1):
        for b in np.arange(0.0, 2.0, 0.1):
            f = np.array([50.0 + a, 60.0 + b])
            if all(int(v) % pl0.P.lattice_cells == 0 for v in np.rint(f)):
                continue
            gxy = pl0.view.world(f)
            dn = np.linalg.norm(pl0.node_xy - gxy, axis=1)
            if not np.any((dn > pl0.P.goal_tol) & (dn <= pl0.repair_radius)):
                continue
            gm2 = _open_map()
            _mark_lethal(gm2, [tuple(pl0.node_fidx[c]) for c in np.nonzero(dn <= pl0.P.goal_tol)[0]])
            pl = P.get_planner(gm2, PRIMS)
            ring = np.nonzero(pl.node_ok & (dn > pl0.P.goal_tol) & (dn <= pl0.repair_radius))[0]
            if not pl.view.lethal_at_world(gxy)[0] and any(pl.connector_free(pl.node_xy[c], gxy) for c in ring):
                return gm2, gxy
    return None, None


def test_goal_repair_ends_on_true_goal():
    gm = _open_map()
    pl0 = P.get_planner(gm, PRIMS)
    gm2, gxy = _goal_repair_case(pl0)
    assert gm2 is not None
    start = tuple(float(v) for v in pl0.view.world(np.array([30.0, 30.0]))) + (0.0,)
    goal = (float(gxy[0]), float(gxy[1]), None)
    r0 = P.plan(gm, PRIMS, start, goal, budget_s=5.0)
    assert r0.complete and r0.goal_snap["mode"] == "tolerance"     # unchanged when tol has a free node
    r = P.plan(gm2, PRIMS, start, goal, budget_s=5.0)
    assert r.goal_snap["mode"] == "repaired" and r.goal_snap["n_goal_nodes"] >= 1
    assert r.complete, r.reason
    assert np.allclose(r.poses[-1][:2], gxy)
    assert r.goal_snap["connector_m"] <= P.get_planner(gm2, PRIMS).repair_radius + 1e-12
    for x, y, _ in r.poses:
        assert not gm2.is_lethal(x, y)
