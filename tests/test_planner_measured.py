"""Phase D on the REAL map + REAL measured primitives of the selected navigator (fast subset of
tools/planner_gate.py).  Deterministic: uses an expansion limit / large wall budget so results do not
depend on CPU load; timing itself is gated only by tools/planner_gate.py."""
import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nav import planner as P  # noqa: E402
from tools import planner_gate as G  # noqa: E402

POLICY = json.load(open(os.path.join(ROOT, "reports", "walker_gate.json")))["selected"]
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", f"primitives_{POLICY}.json")),
                                reason="measured primitives missing")


@pytest.fixture(scope="module")
def setup():
    env, prims, _, gm, sigma, _ = G.load_setup(
        POLICY, os.environ.get('MICRODUCK_TEST_FOLLOWER_REPORTS_DIR'))
    pl = P.get_planner(gm, prims)
    return dict(env=env, prims=prims, gm=gm, pl=pl, lateral_sigma=sigma,
                min_clear=gm.params["r_inflate"] + 0.04,
                by_id={p["id"]: p for p in prims["primitives"]})


def test_map_matches_plan(setup):
    from scipy import ndimage
    gm = setup["gm"]
    # The qualified follower's measured sigma changes with the controller;
    # the footprint and the three-sigma safety rule do not.
    assert gm.params["robot_radius"] == gm.params["robot_footprint_radius"]
    assert gm.params["lateral_sigma"] == setup["lateral_sigma"]
    expected_radius = gm.params["robot_footprint_radius"] + 3.0 * setup["lateral_sigma"]
    assert abs(gm.params["r_inflate"] - expected_radius) < 0.005
    assert ndimage.label(~gm.lethal)[1] == 1
    assert len(gm.rooms) == 6
    assert not setup["pl"].synthesized_headings


def test_cross_room_bound_goal_swept(setup):
    gm, prims, pl = setup["gm"], setup["prims"], setup["pl"]
    rng = np.random.default_rng(3)
    rooms = list(gm.rooms)
    pairs = [(rooms[k % 6], rooms[(k + 1 + k // 6) % 6]) for k in range(8)]
    for k, (ra, rb) in enumerate(pairs):
        s = G.sample_in_room(gm, ra, rng, setup["min_clear"]) + (float(rng.uniform(-3, 3)),)
        th = float(rng.uniform(-3, 3)) if k % 4 else None
        g = G.sample_in_room(gm, rb, rng, setup["min_clear"]) + (th,)
        r = P.plan(gm, prims, s, g, budget_s=G.DET_BUDGET, max_expansions=40000)
        ref = P.dijkstra_reference(gm, prims, s, g)
        assert math.isfinite(ref) and r.complete, (ra, rb, r.reason)
        assert r.cost >= ref - 1e-6
        if math.isfinite(r.eps_achieved):
            assert r.cost <= r.eps_achieved * ref + 1e-6
        ok, info = G.goal_check(r, g, s, pl.P.goal_tol, pl.bin_w)
        assert ok, info
        sw = G.swept_check(gm, r.poses)
        assert sw["n_lethal"] == 0 and sw["max_step"] <= 0.05, sw
        pc = G.prim_check(pl, r, setup["by_id"])
        assert pc["unknown"] == 0 and pc["poses_consumed"]
        assert pc["max_dxy"] <= 0.02 * math.sqrt(2) + 1e-6 and pc["max_dth"] <= pl.bin_w / 2 + 1e-6, pc


def test_deterministic_with_expansion_limit(setup):
    gm, prims = setup["gm"], setup["prims"]
    s, g = (-3.0, 2.0, 0.0), (3.0, -2.0, math.pi / 2)
    a = P.plan(gm, prims, s, g, budget_s=G.DET_BUDGET, max_expansions=30000)
    b = P.plan(gm, prims, s, g, budget_s=G.DET_BUDGET, max_expansions=30000)
    assert G.same_result(a, b) and a.expansions == b.expansions


def test_heuristic_admissible_real(setup):
    gm, prims, pl = setup["gm"], setup["prims"], setup["pl"]
    rng = np.random.default_rng(11)
    goals = [G.sample_in_room(gm, "bathroom", rng, setup["min_clear"]) + (0.5,),
             G.sample_in_room(gm, "kitchen", rng, setup["min_clear"]) + (None,)]
    res = G.heuristic_check(gm, prims, pl, goals, rng, n_random=4)
    for x in res:
        assert x["ok"], x
        assert x["median_h_over_ctg"] > 0.5


def test_partial_plan_makes_progress(setup):
    gm, prims = setup["gm"], setup["prims"]
    geo = G.Geodesic(gm)
    s, g = (-3.0, 2.0, 0.0), (3.0, -2.0, 0.0)
    r = P.plan(gm, prims, s, g, budget_s=G.DET_BUDGET, max_expansions=30)
    assert not r.complete and r.reason == "timeout_partial"
    d = geo.from_point(*g[:2])
    assert d[geo.cell(*r.poses[-1][:2])] < d[geo.cell(*s[:2])]
    assert G.swept_check(gm, r.poses)["n_lethal"] == 0
