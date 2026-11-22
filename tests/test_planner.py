"""Phase D planner tests.  Runs on synthetic fixtures always, and additionally on the real
nav/data/map_*.npz (Phase C) and nav/data/primitives_*.json (Phase B) when present."""
import copy
import glob
import json
import math
import os
import platform
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nav import planner as P  # noqa: E402
from nav.testing_fixtures import (SyntheticGridMap, make_house_map, make_synthetic_primitives,  # noqa: E402
                                  sample_free_poses)

DATA = os.path.join(ROOT, "data")
REPORT = os.path.join(ROOT, "reports", "planner_timing.json")
TOL = 1e-6


def _real_map():
    files = sorted(glob.glob(os.path.join(DATA, "map_*.npz")), key=os.path.getmtime)
    if not files:
        return None
    try:
        from nav.mapping import GridMap
        return GridMap.load(files[-1]), os.path.basename(files[-1])
    except Exception:  # noqa: BLE001 - mapping.py not importable yet
        return None


def _real_prims():
    files = sorted(glob.glob(os.path.join(DATA, "primitives_*.json")), key=os.path.getmtime)
    walk = [f for f in files if f.endswith("primitives_walk.json")]
    f = (walk or files[-1:] or [None])[0]
    return None if f is None else (f, os.path.basename(f))   # passed as a path: loaded unchanged


_FIX_MAP = None
_FIX_PRIMS = None


def _fixture_map():
    global _FIX_MAP
    if _FIX_MAP is None:
        _FIX_MAP = make_house_map()
    return _FIX_MAP


def _fixture_prims():
    global _FIX_PRIMS
    if _FIX_PRIMS is None:
        _FIX_PRIMS = make_synthetic_primitives()
    return _FIX_PRIMS


def _cases():
    cases = [("fixture_map+fixture_prims", "fix", "fix", 100)]
    rm, rp = _real_map(), _real_prims()
    if rm is not None:
        cases.append((f"{rm[1]}+fixture_prims", "real", "fix", 40))
    if rp is not None:
        cases.append((f"fixture_map+{rp[1]}", "fix", "real", 40))
        if rm is not None:
            cases.append((f"{rm[1]}+{rp[1]}", "real", "real", 40))
    return cases


CASES = _cases()


def _load_case(mkind, pkind):
    gm = _fixture_map() if mkind == "fix" else _real_map()[0]
    pr = _fixture_prims() if pkind == "fix" else _real_prims()[0]
    return gm, pr


def _queries(gm, n, seed=12345):
    rng = np.random.default_rng(seed)
    starts = sample_free_poses(gm, n, rng)
    goals = sample_free_poses(gm, n, rng)
    goals = [(g[0], g[1], None) if k % 5 == 4 else g for k, g in enumerate(goals)]
    return list(zip(starts, goals))


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def runs(request):
    name, mkind, pkind, n = request.param
    gm, pr = _load_case(mkind, pkind)
    pl = P.get_planner(gm, pr)                       # one-time per-map precompute (cached)
    out = []
    for s, g in _queries(gm, n):
        t = time.perf_counter()
        r = P.plan(gm, pr, s, g, budget_s=0.2, eps0=3.0)
        wall = time.perf_counter() - t
        ref = P.dijkstra_reference(gm, pr, s, g)
        out.append(dict(start=s, goal=g, res=r, wall=wall, ref=ref))
    return dict(name=name, gm=gm, prims=pr, planner=pl, runs=out, fixture=(mkind == "fix" and pkind == "fix"))


# --------------------------------------------------------------------------- (a) optimality bound
def test_a_bounded_suboptimality_and_goal_reached(runs):
    pl = runs["planner"]
    n_complete = n_unbounded = 0
    for q in runs["runs"]:
        r, ref, g = q["res"], q["ref"], q["goal"]
        if not math.isfinite(ref):
            assert not r.complete, "planner claims a path the brute-force search cannot find"
            continue
        if runs["fixture"]:
            assert r.complete, f"no path found within budget for {q['start']} -> {g} ({r.reason})"
        if not r.complete:
            continue
        n_complete += 1
        assert r.cost >= ref - TOL, "ARA* cheaper than the exhaustive optimum: edge tables disagree"
        if math.isfinite(r.eps_achieved):
            assert 1.0 <= r.eps_achieved <= 3.0 + 1e-9
            assert r.cost <= r.eps_achieved * ref + TOL, (r.cost, r.eps_achieved, ref)
        else:
            n_unbounded += 1                             # budget hit inside iteration 1: no bound claimed
        # goal condition: lattice end state within tol and in the goal heading bin
        ex, ey, eth = r.end_state_pose
        assert math.hypot(ex - g[0], ey - g[1]) <= pl.P.goal_tol + 1e-9
        if g[2] is not None:
            assert pl.heading_bin(eth) == pl.heading_bin(g[2])
        assert np.allclose(r.poses[-1], r.end_state_pose)
        assert len(r.primitive_ids) >= 1
    if runs["fixture"]:
        assert n_unbounded == 0


# --------------------------------------------------------------------------- (b) swept footprint
def test_b_swept_points_never_lethal(runs):
    gm = runs["gm"]
    n_pts = 0
    for q in runs["runs"]:
        poses = q["res"].poses
        for x, y, _ in poses:
            assert not gm.is_lethal(x, y), f"pose ({x:.3f},{y:.3f}) is lethal"
        n_pts += len(poses)
        if len(poses) > 1:
            xy = np.array([p[:2] for p in poses])
            assert np.linalg.norm(np.diff(xy, axis=0), axis=1).max() <= 0.05  # dense (~2 cm) path
    assert n_pts > 0


# --------------------------------------------------------------------------- (c) timing
def test_c_p95_planning_time(runs):
    rs = [q["res"] for q in runs["runs"]]
    t = np.array([r.plan_time_s for r in rs])
    wall = np.array([q["wall"] for q in runs["runs"]])
    first = np.array([r.first_solution_time_s for r in rs if r.first_solution_time_s is not None])
    eps = np.array([r.eps_achieved for r in rs if r.complete])
    ratio = np.array([q["res"].cost / q["ref"] for q in runs["runs"]
                      if q["res"].complete and math.isfinite(q["ref"]) and q["ref"] > 0])
    pl = runs["planner"]
    entry = {
        "n_queries": len(rs),
        "budget_s": 0.2,
        "eps0": 3.0,
        "plan_time_s": {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
                        "max": float(t.max())},
        "wall_time_s": {"p50": float(np.percentile(wall, 50)), "p95": float(np.percentile(wall, 95)),
                        "max": float(wall.max())},
        "first_solution_s": {"p50": float(np.percentile(first, 50)), "p95": float(np.percentile(first, 95)),
                             "max": float(first.max())} if len(first) else None,
        "complete_frac": float(np.mean([r.complete for r in rs])),
        "eps_achieved": {"mean": float(eps.mean()), "max": float(eps.max()),
                         "frac_optimal(eps=1)": float(np.mean(eps <= 1 + 1e-9))} if len(eps) else None,
        "cost_over_optimal": {"mean": float(ratio.mean()), "max": float(ratio.max())} if len(ratio) else None,
        "expansions_p50": float(np.median([r.expansions for r in rs])),
        "per_map_precompute_s": pl.precompute_s,
        "lattice": {"cells": [pl.Hc, pl.Wc], "states": pl.N, "res_m": pl.view.res * pl.P.lattice_cells,
                    "n_headings": pl.nh, "prims_per_heading": pl.Pmax, "v_E": pl.v_E, "v_C": pl.v_C},
        "map_shape": list(pl.view.lethal.shape),
        "machine": f"{platform.machine()} {platform.processor()} py{platform.python_version()}",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(REPORT) as f:
            rep = json.load(f)
    except (OSError, ValueError):
        rep = {}
    rep[runs["name"]] = entry
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(rep, f, indent=2)
    assert entry["plan_time_s"]["p95"] < 0.2, entry
    assert entry["wall_time_s"]["p95"] < 0.2, entry


# --------------------------------------------------------------------------- (d) heuristic
@pytest.mark.parametrize("case", CASES[:2], ids=[c[0] for c in CASES[:2]])
def test_d_heuristic_admissible_and_consistent(case):
    _, mkind, pkind, _ = case
    gm, pr = _load_case(mkind, pkind)
    pl = P.get_planner(gm, pr)
    rng = np.random.default_rng(7)
    goals = sample_free_poses(gm, 6, rng)
    goals[-1] = (goals[-1][0], goals[-1][1], None)
    for goal in goals:
        h = pl.heuristic(goal)
        ctg = P.cost_to_go_reference(gm, pr, goal)        # exact, all states (reverse Dijkstra)
        fin = np.isfinite(ctg)
        assert fin.sum() > 1000
        assert np.all(h[fin] <= ctg[fin] + 1e-9), float((h[fin] - ctg[fin]).max())
        # pruning by h = inf is exact: never prunes a state that can reach the goal
        assert not np.any(np.isinf(h) & fin)
        # random states vs the forward brute-force reference (dijkstra_reference)
        idx = rng.choice(np.nonzero(fin)[0], 5, replace=False)
        for s in idx:
            sp_ = pl.state_pose(int(s))
            ref = P.dijkstra_reference(gm, pr, sp_, goal)
            assert abs(ref - ctg[s]) < 1e-6
            assert h[s] <= ref + 1e-9
        # consistency h(s) <= c(s,s') + h(s') on random valid edges
        S = rng.choice(np.nonzero(fin)[0], 3000)
        for s in S.tolist():
            for kk, d in enumerate(pl.deltas[s % pl.nh]):
                c = pl.tab[s, kk]
                if np.isfinite(c):
                    assert h[s] <= c + h[s + d] + 1e-9
        # heuristic is informative (not trivially 0): median h / ctg
        ok = fin & (ctg > 0)
        assert np.median(h[ok] / ctg[ok]) > 0.5


# --------------------------------------------------------------------------- extras
def test_partial_path_on_timeout():
    gm, pr = _fixture_map(), _fixture_prims()
    P.get_planner(gm, pr)
    r = P.plan(gm, pr, (0.6, 0.6, 0.0), (6.3, 5.4, math.pi / 2), max_expansions=15)
    assert not r.complete and r.reason == "timeout_partial"
    pl = P.get_planner(gm, pr)
    _, _, h, _ = pl.goal_spec((6.3, 5.4, math.pi / 2))
    s0 = pl.snap((0.6, 0.6, 0.0))
    se = pl.snap(r.end_state_pose)
    assert h[se // pl.nh] < h[s0 // pl.nh]         # partial path makes progress (Baritone bestSoFar)
    for x, y, _ in r.poses:
        assert not gm.is_lethal(x, y)


def test_unreachable_and_lethal_goals():
    H, W = 150, 200
    occ = np.zeros((H, W), bool)
    occ[0:3], occ[-3:], occ[:, 0:3], occ[:, -3:] = True, True, True, True
    occ[:, 98:102] = True                                 # sealed wall: two disconnected halves
    gm = SyntheticGridMap(occ, np.zeros_like(occ), rooms={})
    pr = _fixture_prims()
    r = P.plan(gm, pr, (0.8, 1.5, 0.0), (3.2, 1.5, 0.0))
    assert not r.complete and r.reason == "goal_unreachable"
    assert not math.isfinite(P.dijkstra_reference(gm, pr, (0.8, 1.5, 0.0), (3.2, 1.5, 0.0)))
    r = P.plan(gm, pr, (0.8, 1.5, 0.0), (2.0, 1.5, 0.0))   # goal inside the wall
    assert not r.complete and r.reason == "goal_lethal"
    r = P.plan(gm, pr, (2.0, 1.5, 0.0), (0.8, 1.5, 0.0))
    assert not r.complete and r.reason == "start_lethal"


def test_any_heading_goal_and_json_path(tmp_path):
    gm, pr = _fixture_map(), _fixture_prims()
    path = tmp_path / "primitives_test.json"
    path.write_text(json.dumps(pr))
    s, g = (0.6, 0.6, 0.0), (2.5, 2.0, None)
    # Equivalence check, not a timing check: an unconstrained budget keeps the
    # result deterministic when the machine is loaded (ARA* is anytime).
    a = P.plan(gm, pr, s, g, budget_s=30.0)
    b = P.plan(gm, str(path), s, g, budget_s=30.0)        # file path, loaded unchanged
    assert a.complete and b.complete
    assert a.eps_achieved == 1.0 and b.eps_achieved == 1.0
    assert abs(a.cost - b.cost) < 1e-9
    assert abs(a.cost - P.dijkstra_reference(gm, str(path), s, g)) < 1e-9


def test_missing_headings_are_synthesized_by_rotation():
    gm = _fixture_map()
    pr = _fixture_prims()
    only0 = {"policy_md5": "x", "n_headings": 16,
             "primitives": [p for p in pr["primitives"] if p["start_heading_idx"] == 0]}
    pl = P.get_planner(gm, only0)
    assert len(pl.synthesized_headings) == 15
    r = P.plan(gm, only0, (0.6, 0.6, 0.0), (2.5, 2.0, math.pi / 2))
    assert r.complete


def test_plan_segment_chains_to_goal():
    gm, pr = _fixture_map(), _fixture_prims()
    s, g = (0.6, 0.6, 0.0), (6.3, 5.4, math.pi / 2)
    pl = P.get_planner(gm, pr)
    for _ in range(20):
        r = P.plan_segment(gm, pr, s, g, horizon_m=1.5)
        assert r.complete, r.reason
        for x, y, _ in r.poses:
            assert not gm.is_lethal(x, y)
        if r.final_segment:
            ex, ey, eth = r.end_state_pose
            assert math.hypot(ex - g[0], ey - g[1]) <= pl.P.goal_tol + 1e-9
            assert pl.heading_bin(eth) == pl.heading_bin(g[2])
            return
        assert r.subgoal is not None
        ex, ey, _ = r.end_state_pose
        assert math.hypot(ex - r.subgoal[0], ey - r.subgoal[1]) <= pl.P.goal_tol + 1e-9
        s = r.end_state_pose
    pytest.fail("segments did not reach the goal")


def test_should_replan():
    gm, pr = _fixture_map(), _fixture_prims()
    r = P.plan(gm, pr, (0.6, 0.6, 0.0), (2.6, 0.6, 0.0))
    assert r.complete
    mid = r.poses[len(r.poses) // 2]
    assert not P.should_replan(mid, r, 0.5, gm)
    assert P.should_replan((mid[0], mid[1] + 0.2, mid[2]), r, 0.5, gm)          # > 0.15 m off path
    assert P.replan_reason((mid[0], mid[1] + 0.2, mid[2]), r, 0.5, gm) == "off_path"
    assert P.replan_reason(mid, r, 2.01, gm) == "periodic"
    blocked = copy.copy(gm)
    blocked.lethal = gm.lethal.copy()
    late = r.poses[3 * len(r.poses) // 4]
    i, j = gm.world_to_cell(late[0], late[1])
    blocked.lethal[i - 3:i + 4, j - 3:j + 4] = True                              # new obstacle ahead
    assert P.replan_reason(mid, r, 0.5, blocked) == "blocked"
    assert P.replan_reason(r.poses[-1], r, 0.5, blocked) is None                 # obstacle now behind
    assert P.should_replan(mid, [], 0.0, gm)
