"""Phase C verification: raster vs MuJoCo ray geometry (10k vertical + 10k horizontal random rays)."""
import numpy as np
import mujoco
import pytest

from nav.mapping import (geom_world_points, DEFAULT_SCENE, GEOMGROUP_MASK, GridMap, build_map, masked_ray_model)

STEP = 0.02


def test_zero_step_capability_preserves_floor_and_cliff():
    gm = build_map(step_over_height=0.0)
    assert not gm.is_lethal(2.0, 0.0)
    assert gm.is_lethal(-0.2, -1.0)
    assert gm.cliff[gm.world_to_cell(-0.2, -1.0)]


@pytest.fixture(scope="module")
def env():
    gm = build_map(step_over_height=STEP)
    m, d = masked_ray_model(DEFAULT_SCENE)
    _H[0] = gm.params['robot_height']
    return gm, m, d


_ZR = {}


def _zrange(m, d, g):
    if g not in _ZR:
        p = geom_world_points(m, d, g)
        _ZR[g] = (p[:, 2].min(), p[:, 2].max()) if p is not None else (-1e9, 0.0)
    return _ZR[g]


def _down(m, d, x, y):
    """3-class truth from a vertical ray started at the top of the robot band: 0 free floor, 1 obstacle
    (hit geom overlaps the band [STEP, H] in z -- works from inside geoms and ignores overhangs such as
    desk tops above the robot), 2 cliff/void (no floor surface with z >= -STEP)."""
    H = _H[0]
    z = H - 1e-3
    gid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, d, np.array([x, y, z]), np.array([0, 0, -1.0]), GEOMGROUP_MASK, 1, -1, gid)
    if dist >= 0:
        zlo, zhi = _zrange(m, d, int(gid[0]))
        if zhi > STEP and zlo < H:
            return 1
    # upward ray from the bottom of the band (resolves coplanar ties when starting inside a geom)
    up = mujoco.mj_ray(m, d, np.array([x, y, STEP + 1e-3]), np.array([0, 0, 1.0]), GEOMGROUP_MASK, 1, -1, gid)
    if up >= 0:
        zlo, zhi = _zrange(m, d, int(gid[0]))
        if zhi > STEP and zlo < H:
            return 1
    if dist < 0:
        return 2
    top = z - dist
    return 0 if top >= -STEP else 2


_H = [0.0]


def _truth_grid(m, d, xs, ys):
    return np.array([_down(m, d, x, y) for x, y in zip(xs, ys)])


def test_vertical_agreement_and_no_missed_cliff(env):
    gm, m, d = env
    rng = np.random.default_rng(0)
    x0, y0, x1, y1 = gm.bounds
    n = 10000
    xs, ys = rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)
    t = _truth_grid(m, d, xs, ys)
    i, j = gm.world_to_cell(xs, ys)
    pred_occ, pred_cliff = gm.occ[i, j], gm.cliff[i, j]
    pred_leth_raw = pred_occ | pred_cliff
    truth_leth = t != 0
    # strict (point vs containing cell)
    strict = (pred_leth_raw == truth_leth).mean()
    # soundness: truth lethal but raster free (must be zero)
    missed = truth_leth & ~pred_leth_raw
    # tolerant: a raster-lethal cell with free truth is OK if truth within one cell (8-neighbours at +-res) is lethal
    fp = np.nonzero(pred_leth_raw & ~truth_leth)[0]
    bad_fp = 0
    for k in fp:
        near = [_down(m, d, xs[k] + a * gm.res, ys[k] + b * gm.res) != 0
                for a in (-1, 0, 1) for b in (-1, 0, 1)]
        bad_fp += not any(near)
    tol = 1.0 - (missed.sum() + bad_fp) / n
    cliff_truth = t == 2
    # truth-cliff points not inside an obstacle cell must be flagged cliff
    missed_cliff = (cliff_truth & ~pred_cliff & ~pred_occ).sum()
    # truth-cliff points must be lethal in the inflated map
    leth = np.array([gm.is_lethal(x, y) for x, y in zip(xs, ys)])
    missed_cliff_lethal = (cliff_truth & ~leth).sum()
    print(f"\nVERTICAL n={n}: strict agreement={strict:.5f}  tolerant(1 cell)={tol:.5f}  "
          f"missed-lethal={missed.sum()}  missed-cliff={missed_cliff}  missed-cliff-in-inflated={missed_cliff_lethal}  "
          f"truth cliff pts={cliff_truth.sum()} occ pts={(t == 1).sum()}")
    assert missed.sum() == 0
    assert missed_cliff == 0 and missed_cliff_lethal == 0
    assert tol >= 0.999


def test_cliff_hole_and_stairs(env):
    gm, m, d = env
    # stair hole x in [-0.4,0], y in [-1.4,-0.7]; every point inside must be lethal & cliff
    rng = np.random.default_rng(1)
    xs, ys = rng.uniform(-0.39, -0.01, 2000), rng.uniform(-1.39, -0.71, 2000)
    assert all(gm.is_lethal(x, y) for x, y in zip(xs, ys))
    i, j = gm.world_to_cell(xs, ys)
    assert gm.cliff[i, j].all()
    # just beyond the hole edge plus robot radius must be non-lethal somewhere (corridor remains passable)
    assert not gm.is_lethal(-0.2, -2.2) and not gm.is_lethal(-0.2, 0.2)


def test_horizontal_rays(env):
    gm, m, d = env
    rng = np.random.default_rng(2)
    H = gm.params["robot_height"]
    n = 10000
    gid = np.zeros(1, np.int32)
    done = viol = hits = 0
    over = []
    while done < n:
        x = rng.uniform(-3.9, 3.9)
        y = rng.uniform(-2.9, 2.9)
        if _down(m, d, x, y) != 0:
            continue  # start from free floor points only (not inside geometry)
        z = rng.uniform(STEP + 1e-3, H - 1e-3)
        th = rng.uniform(0, 2 * np.pi)
        v = np.array([np.cos(th), np.sin(th), 0.0])
        dist = mujoco.mj_ray(m, d, np.array([x, y, z]), v, GEOMGROUP_MASK, 1, -1, gid)
        done += 1
        if dist < 0:
            continue
        hits += 1
        # hit point nudged 0.2*res past the surface must fall in an occupied cell (conservative raster)
        px, py = x + (dist + 0.2 * gm.res) * v[0], y + (dist + 0.2 * gm.res) * v[1]
        i, j = gm.world_to_cell(px, py)
        ok = gm.occ[i, j]
        # points just before the surface too
        px2, py2 = x + (dist - 1e-4) * v[0], y + (dist - 1e-4) * v[1]
        i2, j2 = gm.world_to_cell(px2, py2)
        ok = gm.occ[i2, j2]  # cell holding the surface point must be occupied
        viol += not ok
    agree = 1 - viol / max(hits, 1)
    print(f"\nHORIZONTAL n={n} rays, hits={hits}: hit-surface-cell occupied agreement={agree:.5f} (violations={viol})")
    assert agree >= 0.999


def test_robot_and_free_geoms_excluded(env):
    gm, m, d = env
    # ball_0 at (-3,0), obj_0 at (-2.2,-1.2): movable, must not be obstacles
    for x, y in ((-3.0, 0.0), (-2.2, -1.2), (-0.3, 0.1)):
        assert not gm.occ[gm.world_to_cell(x, y)]
    # robot start (0,0) free
    assert not gm.is_lethal(0.0, 0.0)


def test_cache_roundtrip(env, tmp_path):
    gm, _, _ = env
    p = tmp_path / "m.npz"
    gm.save(p)
    g2 = GridMap.load(p)
    assert (g2.occ == gm.occ).all() and g2.rooms == gm.rooms and len(g2.doorways) == len(gm.doorways)
    assert np.array_equal(np.isinf(g2.cost), np.isinf(gm.cost))


def test_rooms_and_doorways(env):
    gm, _, _ = env
    assert set(gm.rooms) == {"kitchen", "living", "corridor", "bedroom", "office", "bathroom"}
    assert len(gm.doorways) == 6
    for dw in gm.doorways:
        assert not gm.is_lethal(*dw["center"]), dw
    for r in gm.rooms:
        assert len(gm.sample_free(r, 5)) == 5
