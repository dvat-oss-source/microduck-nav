"""Phase D gate on the REAL map + REAL measured primitives (one command).

    cd ~/microduck/nav && $PY -m tools.planner_gate [--policy atlantis_walk] [--n 100] [--no-load]

Gate (plan text, not relaxed):
  (a) 100 random start/goal queries over all 6 rooms incl. cross-room: ARA* cost <= eps_achieved x
      dijkstra_reference cost; complete plans reach the goal (checked against the *requested* goal).
  (b) zero swept points of returned paths in lethal cells: path poses (densified to res/4) are
      re-rasterized with gm.world_to_cell against gm.lethal (no planner internals).
  (c) p95 planning time < 200 ms single-threaded (plan_time_s and caller wall time); CPU load reported.
  (d) heuristic admissible: h <= exact cost-to-go (reverse Dijkstra) on all states, and h <=
      dijkstra_reference on random states, for one goal per room (+1 any-heading goal).
Extra (reported, not part of the pass/fail verdict):
  (e) load sensitivity of the 0.2 s wall-clock budget (idle repeat and 2 busy-loop processes) and a
      deterministic max_expansions calibration (budget_s large, so results do not depend on load).
  (f) cross-room completion within budget; partial plans make progress (independent 8-conn geodesic).
  (g) farthest room pair kitchen <-> bathroom: completion and time.
Outputs reports/planner_gate_<policy>.json and .md.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.csgraph import dijkstra as cs_dijkstra  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nav import planner as P  # noqa: E402
from nav.episode import read_artifacts  # noqa: E402
from nav.mapping import build_map  # noqa: E402
from nav.mapsigma import planning_lateral_sigma  # noqa: E402

TOL = 1e-6
BUDGET = 0.2
FORCED_PARTIAL_EXP = 30   # (f): small limit that forces partial plans on cross-room queries
DET_BUDGET = 60.0          # deterministic mode: wall budget never binds, max_expansions does


# ----------------------------------------------------------------------------- setup
def load_setup(policy, follower_reports_dir=None):
    env, prims = read_artifacts(policy)
    env_path = ROOT / "data" / f"envelope_{policy}.json"
    prims_path = ROOT / "data" / f"primitives_{policy}.json"
    reports = Path(follower_reports_dir) if follower_reports_dir is not None else ROOT / "reports"
    follower_path = reports / f"follower_sim_{policy}.json"
    follower_bytes = follower_path.read_bytes()
    follower = json.loads(follower_bytes)
    if (follower.get("policy_md5") != env["policy_md5"]
            or follower.get("dynamics") != env.get("dynamics")
            or follower.get("dynamics_version") != env.get("dynamics_version")):
        raise RuntimeError(f"{follower_path.name} does not match the measured artifacts")
    if not follower.get("passed"):
        raise RuntimeError(f"{follower_path.name} did not pass; refusing to certify its map inflation")
    artifact_digest = hashlib.sha256(env_path.read_bytes() + b"\0" + prims_path.read_bytes()).hexdigest()
    follower_digest = hashlib.sha256(follower_bytes).hexdigest()
    sigma, prov = planning_lateral_sigma(policy, env, reports_dir=reports)
    gm = build_map(lateral_sigma=sigma, step_over_height=env["step_over_height"])
    prov = dict(prov, artifact_sha256=artifact_digest, follower_gate_sha256=follower_digest)
    return env, prims, prims_path, gm, sigma, prov


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def cpu_snapshot():
    snap = {"loadavg": list(os.getloadavg()), "ncpu": os.cpu_count()}
    try:
        out = subprocess.run(["ps", "-Ao", "pcpu=,comm="], capture_output=True, text=True, timeout=5).stdout
        busy = []
        for line in out.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and float(parts[0]) >= 50.0:
                busy.append([float(parts[0]), os.path.basename(parts[1])[:40]])
        snap["procs_ge_50pct"] = sorted(busy, reverse=True)
    except Exception as e:  # noqa: BLE001
        snap["procs_error"] = str(e)
    return snap


class Geodesic:
    """Independent 8-connected geodesic distance on non-lethal cells (not the planner's D2)."""

    def __init__(self, gm):
        free = ~np.asarray(gm.lethal, bool)
        H, W = free.shape
        self.H, self.W, self.res = H, W, gm.res
        idx = np.arange(H * W).reshape(H, W)
        rows, cols, w = [], [], []
        for di, dj, c in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))):
            i0, i1 = max(0, -di), H - max(0, di)
            j0, j1 = max(0, -dj), W - max(0, dj)
            m = free[i0:i1, j0:j1] & free[i0 + di:i1 + di, j0 + dj:j1 + dj]
            rows.append(idx[i0:i1, j0:j1][m]); cols.append(idx[i0 + di:i1 + di, j0 + dj:j1 + dj][m])
            w.append(np.full(int(m.sum()), c * gm.res))
        r, c, d = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
        self.G = sp.csr_matrix((d, (r, c)), shape=(H * W, H * W))
        self.gm = gm

    def cell(self, x, y):
        i, j = self.gm.world_to_cell(x, y)
        return int(i) * self.W + int(j)

    def from_point(self, x, y):
        return cs_dijkstra(self.G, directed=False, indices=self.cell(x, y))


def sample_in_room(gm, room, rng, min_clear, x_range=None):
    x0, y0, x1, y1 = gm.rooms[room]
    H, W = gm.lethal.shape
    ii, jj = np.mgrid[0:H, 0:W]
    X, Y = gm.cell_to_world(ii, jj)
    ok = (~gm.lethal) & (gm.clearance >= min_clear) & (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
    if x_range is not None:
        ok &= (X >= x_range[0]) & (X <= x_range[1])
    k = int(rng.integers(ok.sum()))
    return float(X[ok][k]), float(Y[ok][k])


def make_queries(gm, n, rng, min_clear):
    rooms = list(gm.rooms)
    pairs = [(a, b) for a in rooms for b in rooms]               # all 36 ordered pairs first
    while len(pairs) < n:
        pairs.append((rooms[int(rng.integers(len(rooms)))], rooms[int(rng.integers(len(rooms)))]))
    pairs = pairs[:n]
    qs = []
    for k, (ra, rb) in enumerate(pairs):
        s = sample_in_room(gm, ra, rng, min_clear) + (float(rng.uniform(-math.pi, math.pi)),)
        g = sample_in_room(gm, rb, rng, min_clear) + (float(rng.uniform(-math.pi, math.pi)),)
        if k % 5 == 4:
            g = (g[0], g[1], None)                               # 20% any-heading goals
        qs.append(dict(k=k, start=s, goal=g, rooms=[ra, rb], cross=ra != rb))
    return qs


# ----------------------------------------------------------------------------- independent checks
def swept_check(gm, poses):
    """Re-rasterize path poses (densified to res/4) against gm.lethal. Returns dict."""
    if not poses:
        return dict(n_points=0, n_lethal=0, max_step=0.0, min_clearance=None)
    xy = np.array([p[:2] for p in poses], float)
    pts = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        L = float(np.linalg.norm(b - a))
        m = max(1, int(math.ceil(L / (gm.res / 4))))
        t = np.arange(1, m + 1)[:, None] / m
        pts.append(a + t * (b - a))
    pts = np.concatenate(pts)
    H, W = gm.lethal.shape
    i, j = gm.world_to_cell(pts[:, 0], pts[:, 1])
    i, j = np.asarray(i, int), np.asarray(j, int)
    inside = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    leth = np.ones(len(pts), bool)
    leth[inside] = gm.lethal[i[inside], j[inside]]
    clr = np.zeros(len(pts), np.float32)
    clr[inside] = gm.clearance[i[inside], j[inside]]
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.zeros(1)
    return dict(n_points=int(len(pts)), n_lethal=int(leth.sum()), max_step=float(steps.max()),
                min_clearance=float(clr.min()))


def goal_check(r, goal, start, tol, bin_w):
    """Complete plans: last pose within tol of requested goal xy and inside the goal heading bin;
    first pose at the snapped start."""
    ex, ey, eth = r.poses[-1]
    d = math.hypot(ex - goal[0], ey - goal[1])
    dth = 0.0 if goal[2] is None else abs(wrap(eth - goal[2]))
    sx, sy, sth = r.poses[0]
    ds = math.hypot(sx - start[0], sy - start[1])
    dsth = abs(wrap(sth - start[2]))
    ok = d <= tol + 1e-9 and dth <= bin_w / 2 + 1e-9 and ds <= 0.02 * math.sqrt(2) + 1e-6 and dsth <= bin_w / 2 + 1e-9
    return ok, dict(goal_dist=d, goal_dtheta=dth, start_dist=ds, start_dtheta=dsth)


def prim_check(pl, r, prim_by_id):
    """Each returned segment's displacement vs the measured primitive's end_mean (snap <= half lattice
    cell per axis, dtheta within half bin). Uses pl.prims only to split poses into segments."""
    if not r.primitive_ids:
        return dict(n=0, max_dxy=0.0, max_dth=0.0, unknown=0)
    by_id = {p.id: p for ps in pl.prims for p in ps}
    i0, mdxy, mdth, unknown = 0, 0.0, 0.0, 0
    for pid in r.primitive_ids:
        if pid not in prim_by_id:
            unknown += 1
            continue
        n = len(by_id[pid].samples)
        a, b = r.poses[i0], r.poses[i0 + n]
        em = prim_by_id[pid]["end_mean"]
        mdxy = max(mdxy, math.hypot(b[0] - a[0] - em[0], b[1] - a[1] - em[1]))
        mdth = max(mdth, abs(wrap(b[2] - a[2] - em[2])))
        i0 += n
    return dict(n=len(r.primitive_ids), max_dxy=mdxy, max_dth=mdth, unknown=unknown,
                poses_consumed=i0 + 1 == len(r.poses))


def pct(a, q):
    return float(np.percentile(np.asarray(a, float), q)) if len(a) else None


def tstats(a):
    a = np.asarray(a, float)
    return dict(p50=pct(a, 50), p95=pct(a, 95), max=float(a.max()) if len(a) else None, n=int(len(a)))


# ----------------------------------------------------------------------------- runs
def run_queries(gm, prims, pl, qs, budget_s=BUDGET, max_expansions=None):
    out = []
    for q in qs:
        pl._goal_cache.clear()                    # every query pays its own D2 Dijkstra
        c0, t0 = time.process_time(), time.perf_counter()
        r = P.plan(gm, prims, q["start"], q["goal"], budget_s=budget_s, max_expansions=max_expansions)
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        out.append(dict(res=r, wall=wall, cpu=cpu))
    return out


def summarize_run(runs):
    rs = [x["res"] for x in runs]
    return dict(complete_frac=float(np.mean([r.complete for r in rs])),
                plan_time_s=tstats([r.plan_time_s for r in rs]), wall_time_s=tstats([x["wall"] for x in runs]),
                cpu_over_wall=float(sum(x["cpu"] for x in runs) / max(1e-9, sum(x["wall"] for x in runs))),
                expansions=tstats([r.expansions for r in rs]),
                converged_eps1_frac=float(np.mean([r.complete and r.eps_achieved <= 1 + 1e-9 for r in rs])),
                eps_inf_frac=float(np.mean([r.complete and not math.isfinite(r.eps_achieved) for r in rs])))


def same_result(a, b):
    if a.complete != b.complete:
        return False
    if not (math.isfinite(a.cost) and math.isfinite(b.cost)):
        return a.cost == b.cost
    return abs(a.cost - b.cost) <= 1e-9 * max(1.0, abs(a.cost)) and a.primitive_ids == b.primitive_ids


def diff_frac(ra, rb):
    n = len(ra)
    flag = sum(a["res"].complete != b["res"].complete for a, b in zip(ra, rb))
    cost = sum(a["res"].complete and b["res"].complete and abs(a["res"].cost - b["res"].cost) > 1e-9
               for a, b in zip(ra, rb))
    anyd = sum(not same_result(a["res"], b["res"]) for a, b in zip(ra, rb))
    return dict(n=n, complete_flag_changed=flag / n, cost_changed=cost / n, any_result_changed=anyd / n)


class BusyLoad:
    def __init__(self, k):
        self.k, self.p = k, []

    def __enter__(self):
        self.p = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(self.k)]
        time.sleep(1.0)
        return self

    def __exit__(self, *a):
        for p in self.p:
            p.kill()
        for p in self.p:
            p.wait()


def heuristic_check(gm, prims, pl, goals, rng, n_random=8):
    out = []
    for goal in goals:
        h = pl.heuristic(goal)
        ctg = P.cost_to_go_reference(gm, prims, goal)
        fin = np.isfinite(ctg)
        viol = float(np.max(h[fin] - ctg[fin])) if fin.any() else None
        pruned_reachable = int(np.sum(np.isinf(h) & fin))
        idx = rng.choice(np.nonzero(fin)[0], n_random, replace=False)
        rnd = []
        for s in idx.tolist():
            sp_ = pl.state_pose(int(s))
            ref = P.dijkstra_reference(gm, prims, sp_, goal)
            rnd.append(dict(h=float(h[s]), ref=float(ref), ok=bool(h[s] <= ref + 1e-9)))
        ok = fin & (ctg > 0)
        out.append(dict(goal=goal, n_finite_states=int(fin.sum()), max_h_minus_ctg=viol,
                        pruned_reachable=pruned_reachable, median_h_over_ctg=float(np.median(h[ok] / ctg[ok])),
                        random_states=rnd,
                        ok=bool(viol is not None and viol <= 1e-9 and pruned_reachable == 0 and all(x["ok"] for x in rnd))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="atlantis_walk")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--no-load", action="store_true", help="skip the busy-loop load-sensitivity run")
    ap.add_argument("--det-grid", default="20000,30000,40000,50000,60000,80000")
    ap.add_argument("--n-far", type=int, default=20)
    ap.add_argument("--follower-reports-dir", type=Path,
                    help="development: directory containing current passing follower gates")
    ap.add_argument("--reports-dir", type=Path,
                    help="development: directory for the planner gate report")
    args = ap.parse_args(argv)
    if args.follower_reports_dir is not None and args.reports_dir is None:
        ap.error("--follower-reports-dir requires --reports-dir to preserve canonical reports")
    if (args.follower_reports_dir is not None
            and args.reports_dir.resolve() == (ROOT / "reports").resolve()):
        ap.error("development follower gates require output outside the canonical reports directory")
    T0 = time.time()
    rng = np.random.default_rng(args.seed)
    env, prims, prims_path, gm, sigma, prov = load_setup(args.policy, args.follower_reports_dir)
    prim_by_id = {p["id"]: p for p in prims["primitives"]}
    pl = P.get_planner(gm, prims)
    tol, bin_w = pl.P.goal_tol, pl.bin_w
    min_clear = gm.params["r_inflate"] + 0.04       # snapped lattice node (<= 2.9 cm away) stays free
    geo = Geodesic(gm)
    rep = dict(policy=args.policy, policy_md5=env["policy_md5"], dynamics=env.get("dynamics"),
               dynamics_version=env.get("dynamics_version"),
               artifact_sha256=prov["artifact_sha256"], follower_gate_sha256=prov["follower_gate_sha256"],
               primitives_file=prims_path.name, n_primitives=len(prims["primitives"]),
               map=dict(shape=list(gm.lethal.shape), res=gm.res, r_inflate=gm.params["r_inflate"],
                        lateral_sigma=sigma, lateral_sigma_provenance=prov,
                        step_over_height=env["step_over_height"], rooms=list(gm.rooms)),
               lattice=dict(states=pl.N, prims_per_heading=pl.Pmax, synthesized_headings=list(pl.synthesized_headings),
                            v_E=pl.v_E, v_C=pl.v_C, precompute_s=pl.precompute_s),
               machine=f"{platform.machine()} {platform.processor()} py{platform.python_version()}",
               date=time.strftime("%Y-%m-%d %H:%M:%S"), seed=args.seed, budget_s=BUDGET, eps0=3.0)

    # ---------------------------------------------------------------- main 100-query run (idle)
    qs = make_queries(gm, args.n, rng, min_clear)
    cpu_before = cpu_snapshot()
    idle = run_queries(gm, prims, pl, qs)
    cpu_after = cpu_snapshot()
    refs = []
    for q in qs:
        t = time.perf_counter()
        refs.append(P.dijkstra_reference(gm, prims, q["start"], q["goal"]))
        q["ref_time"] = time.perf_counter() - t
    rows, a_fail, b_fail, unbounded, n_complete = [], [], [], 0, 0
    for q, x, ref in zip(qs, idle, refs):
        r = x["res"]
        sw = swept_check(gm, r.poses)
        row = dict(k=q["k"], rooms=q["rooms"], cross=q["cross"], start=q["start"], goal=q["goal"],
                   complete=r.complete, reason=r.reason, cost=r.cost, ref=ref, eps=r.eps_achieved,
                   ratio=(r.cost / ref) if r.complete and math.isfinite(ref) and ref > 0 else None,
                   plan_time=r.plan_time_s, wall=x["wall"], expansions=r.expansions,
                   first_solution=r.first_solution_time_s, swept=sw)
        if not math.isfinite(ref):
            if r.complete:
                a_fail.append((q["k"], "complete but reference infinite"))
        elif r.reason == "goal_unreachable" or r.reason in ("start_lethal", "goal_lethal"):
            a_fail.append((q["k"], f"reference finite but planner says {r.reason}"))
        if r.complete:
            n_complete += 1
            if r.cost < ref - TOL:
                a_fail.append((q["k"], "cheaper than brute force"))
            if math.isfinite(r.eps_achieved):
                if not r.cost <= r.eps_achieved * ref + TOL:
                    a_fail.append((q["k"], f"bound violated {r.cost} > {r.eps_achieved}*{ref}"))
            else:
                unbounded += 1
            okg, gd = goal_check(r, q["goal"], q["start"], tol, bin_w)
            row["goal_check"] = gd
            if not okg:
                a_fail.append((q["k"], f"goal not reached {gd}"))
            pc = prim_check(pl, r, prim_by_id)
            row["prim_check"] = pc
        if sw["n_lethal"] > 0 or sw["max_step"] > 0.05:
            b_fail.append((q["k"], sw))
        rows.append(row)

    comp = [r for r in rows if r["complete"]]
    ratios = [r["ratio"] for r in comp if r["ratio"] is not None]
    eps_f = [r["eps"] for r in comp if math.isfinite(r["eps"])]
    pcs = [r["prim_check"] for r in comp if r["prim_check"]["n"]]
    rep["a_bound"] = dict(n=len(rows), n_ref_finite=int(sum(math.isfinite(r) for r in refs)), n_complete=n_complete,
                          complete_frac=n_complete / len(rows), n_unbounded_eps_inf=unbounded,
                          eps_achieved=dict(mean=float(np.mean(eps_f)), max=float(np.max(eps_f)),
                                            frac_eps1=float(np.mean(np.asarray(eps_f) <= 1 + 1e-9))) if eps_f else None,
                          cost_over_ref=dict(mean=float(np.mean(ratios)), max=float(np.max(ratios))) if ratios else None,
                          prim_end_vs_measured=dict(max_dxy_m=max(p["max_dxy"] for p in pcs),
                                                    max_dtheta_rad=max(p["max_dth"] for p in pcs),
                                                    unknown_ids=sum(p["unknown"] for p in pcs),
                                                    all_poses_consumed=all(p["poses_consumed"] for p in pcs)) if pcs else None,
                          failures=a_fail, passed=not a_fail,
                          ref_time_s=tstats([q["ref_time"] for q in qs]))
    rep["b_swept"] = dict(n_paths=sum(1 for r in rows if r["swept"]["n_points"]),
                          n_points=sum(r["swept"]["n_points"] for r in rows),
                          n_lethal=sum(r["swept"]["n_lethal"] for r in rows),
                          max_pose_step_m=max(r["swept"]["max_step"] for r in rows),
                          min_clearance_m=min(r["swept"]["min_clearance"] for r in rows if r["swept"]["min_clearance"] is not None),
                          robot_radius_m=gm.params["robot_radius"], r_inflate_m=gm.params["r_inflate"],
                          failures=b_fail, passed=not b_fail)
    sm = summarize_run(idle)
    rep["c_timing"] = dict(sm, cpu_before=cpu_before, cpu_after=cpu_after,
                           first_solution_s=tstats([r["first_solution"] for r in rows if r["first_solution"] is not None]),
                           passed=bool(sm["plan_time_s"]["p95"] < BUDGET and sm["wall_time_s"]["p95"] < BUDGET))
    print(f"[a] complete {n_complete}/{len(rows)} fails {len(a_fail)}  [b] lethal {rep['b_swept']['n_lethal']}  "
          f"[c] plan p95 {sm['plan_time_s']['p95']:.4f} wall p95 {sm['wall_time_s']['p95']:.4f} load {cpu_before['loadavg'][0]:.2f}",
          flush=True)

    # ---------------------------------------------------------------- (d) heuristic
    goals = []
    for k, room in enumerate(gm.rooms):
        g = sample_in_room(gm, room, rng, min_clear) + (float(rng.uniform(-math.pi, math.pi)),)
        goals.append(g)
    goals.append(goals[0][:2] + (None,))
    hc = heuristic_check(gm, prims, pl, goals, rng)
    rep["d_heuristic"] = dict(goals=hc, passed=all(x["ok"] for x in hc))
    print(f"[d] heuristic ok {sum(x['ok'] for x in hc)}/{len(hc)}", flush=True)

    # ---------------------------------------------------------------- (f) cross-room + partial progress
    def progress(q, r):
        d = geo.from_point(q["goal"][0], q["goal"][1])
        ds, de = d[geo.cell(*q["start"][:2])], d[geo.cell(*r.poses[-1][:2])] if r.poses else math.inf
        return float(ds), float(de)

    def fsec(rows_sub, runs_sub, qs_sub):
        n = len(rows_sub)
        partial = []
        for q, x in zip(qs_sub, runs_sub):
            r = x["res"]
            if not r.complete and r.reason == "timeout_partial":
                ds, de = progress(q, r)
                partial.append(dict(k=q["k"], geo_start=ds, geo_end=de, progress_m=ds - de, made_progress=bool(de < ds)))
        return dict(n=n, complete=sum(r.complete for r in (x["res"] for x in runs_sub)),
                    complete_frac=float(np.mean([x["res"].complete for x in runs_sub])) if n else None,
                    partial=len(partial), partial_all_progress=all(p["made_progress"] for p in partial),
                    partial_detail=partial)

    cross_i = [i for i, q in enumerate(qs) if q["cross"]]
    same_i = [i for i, q in enumerate(qs) if not q["cross"]]
    rep["f_cross_room"] = dict(
        cross=fsec([rows[i] for i in cross_i], [idle[i] for i in cross_i], [qs[i] for i in cross_i]),
        same_room=fsec([rows[i] for i in same_i], [idle[i] for i in same_i], [qs[i] for i in same_i]),
        cross_wall_p95=pct([idle[i]["wall"] for i in cross_i], 95))
    # forced-partial progress check (small expansion limit) on all cross-room queries
    forced = run_queries(gm, prims, pl, [qs[i] for i in cross_i], budget_s=DET_BUDGET, max_expansions=FORCED_PARTIAL_EXP)
    fp = []
    for i, x in zip(cross_i, forced):
        r = x["res"]
        if not r.complete:
            ds, de = progress(qs[i], r)
            sw = swept_check(gm, r.poses)
            fp.append(dict(k=qs[i]["k"], progress_m=ds - de, made_progress=bool(de < ds), n_lethal=sw["n_lethal"]))
    rep["f_cross_room"]["forced_partial"] = dict(
        n_partial=len(fp), frac_progress=float(np.mean([p["made_progress"] for p in fp])) if fp else None,
        progress_m=tstats([p["progress_m"] for p in fp]) if fp else None,
        lethal_points=sum(p["n_lethal"] for p in fp),
        no_progress=[p for p in fp if not p["made_progress"]])
    print(f"[f] cross complete {rep['f_cross_room']['cross']['complete']}/{len(cross_i)}", flush=True)

    # ---------------------------------------------------------------- (g) farthest pair kitchen <-> bathroom
    farq = []
    for k in range(args.n_far):
        a = sample_in_room(gm, "kitchen", rng, min_clear, x_range=(-4.0, -2.5))
        b = sample_in_room(gm, "bathroom", rng, min_clear, x_range=(2.5, 4.0))
        th = (float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(-math.pi, math.pi)))
        s, g = (a + (th[0],), b + (th[1],)) if k % 2 == 0 else (b + (th[0],), a + (th[1],))
        farq.append(dict(k=1000 + k, start=s, goal=g, rooms=["kitchen", "bathroom"][::(1 if k % 2 == 0 else -1)],
                         cross=True))
    far = run_queries(gm, prims, pl, farq)
    far_rows = []
    for q, x in zip(farq, far):
        r = x["res"]
        ref = P.dijkstra_reference(gm, prims, q["start"], q["goal"])
        sw = swept_check(gm, r.poses)
        ok_b = (not r.complete) or (math.isfinite(r.eps_achieved) and r.cost <= r.eps_achieved * ref + TOL) \
            or not math.isfinite(r.eps_achieved)
        okg = goal_check(r, q["goal"], q["start"], tol, bin_w)[0] if r.complete else None
        ds, de = progress(q, r)
        far_rows.append(dict(k=q["k"], dir="->".join(q["rooms"]), complete=r.complete, reason=r.reason,
                             wall=x["wall"], plan_time=r.plan_time_s, eps=r.eps_achieved, cost=r.cost, ref=ref,
                             bound_ok=ok_b, goal_ok=okg, lethal=sw["n_lethal"], geo_m=ds, progress_m=ds - de))
    rep["g_far_pair"] = dict(n=len(far_rows), complete_frac=float(np.mean([r["complete"] for r in far_rows])),
                             wall_time_s=tstats([r["wall"] for r in far_rows]),
                             geodesic_m=tstats([r["geo_m"] for r in far_rows]),
                             eps_finite_frac=float(np.mean([math.isfinite(r["eps"]) for r in far_rows if r["complete"]]))
                             if any(r["complete"] for r in far_rows) else None,
                             all_bound_ok=all(r["bound_ok"] for r in far_rows),
                             all_goal_ok=all(r["goal_ok"] for r in far_rows if r["complete"]),
                             lethal_points=sum(r["lethal"] for r in far_rows), rows=far_rows)
    print(f"[g] far complete {rep['g_far_pair']['complete_frac']:.2f} wall p95 {rep['g_far_pair']['wall_time_s']['p95']:.3f}",
          flush=True)

    # ---------------------------------------------------------------- (h) cold start (as episode.py/fsm.py call it)
    # episode.py builds a fresh GridMap object per episode; get_planner caches by id(), so the first
    # plan() of every episode pays the per-map precompute inside its 0.2 s budget.
    cold_q = [dict(k=-1, start=(2.07, -2.01, math.pi), goal=(-3.35, -1.23, 0.0), rooms=["bathroom", "living"])] + \
        [qs[i] for i in cross_i[:9]]
    cold = []
    for q in cold_q:
        gm_c = build_map(lateral_sigma=sigma, step_over_height=env["step_over_height"])
        pr_c = dict(prims)
        t = time.perf_counter()
        r = P.plan(gm_c, pr_c, q["start"], q["goal"])
        w = time.perf_counter() - t
        pc = P.get_planner(gm_c, pr_c).precompute_s
        rw = P.plan(gm_c, pr_c, q["start"], q["goal"])
        cold.append(dict(k=q["k"], rooms=q["rooms"], cold_complete=r.complete, cold_reason=r.reason,
                         cold_expansions=r.expansions, cold_wall=w, precompute_s=pc, warm_complete=rw.complete))
    P._CACHE.clear()
    pl = P.get_planner(gm, prims)
    rep["h_cold_start"] = dict(n=len(cold), cold_complete_frac=float(np.mean([c["cold_complete"] for c in cold])),
                               warm_complete_frac=float(np.mean([c["warm_complete"] for c in cold])),
                               precompute_s=tstats([c["precompute_s"] for c in cold]), rows=cold)
    print(f"[h] cold-start complete {rep['h_cold_start']['cold_complete_frac']:.2f} "
          f"warm {rep['h_cold_start']['warm_complete_frac']:.2f}", flush=True)

    # ---------------------------------------------------------------- (e) load sensitivity of the wall budget
    allq = qs
    idle2 = run_queries(gm, prims, pl, allq)
    e = dict(idle_repeat=dict(diff_frac(idle, idle2), summary=summarize_run(idle2), cpu=cpu_snapshot()))
    if not args.no_load:
        with BusyLoad(2):
            cpu_l = cpu_snapshot()
            loaded = run_queries(gm, prims, pl, allq)
        e["busy2"] = dict(diff_frac(idle, loaded), summary=summarize_run(loaded), cpu=cpu_l)
    print(f"[e] idle-repeat changed {e['idle_repeat']['any_result_changed']:.2f}"
          + (f", busy2 changed {e['busy2']['any_result_changed']:.2f}" if "busy2" in e else ""), flush=True)

    # deterministic max_expansions calibration (wall budget non-binding)
    grid = [int(v) for v in args.det_grid.split(",")]
    det = []
    for N in grid:
        rr = run_queries(gm, prims, pl, allq, budget_s=DET_BUDGET, max_expansions=N)
        s = summarize_run(rr)
        bound_ok = all((not x["res"].complete) or (not math.isfinite(x["res"].eps_achieved))
                       or x["res"].cost <= x["res"].eps_achieved * ref + TOL for x, ref in zip(rr, refs))
        det.append(dict(max_expansions=N, complete_frac=s["complete_frac"], wall_time_s=s["wall_time_s"],
                        eps_inf_frac=float(np.mean([x["res"].complete and not math.isfinite(x["res"].eps_achieved)
                                                    for x in rr])),
                        cost_over_ref_max=max((x["res"].cost / ref for x, ref in zip(rr, refs)
                                               if x["res"].complete and ref > 0), default=None),
                        bound_ok=bound_ok, cpu=cpu_snapshot()["loadavg"], _runs=rr))
        print(f"[e] det N={N}: complete {s['complete_frac']:.2f} wall p95 {s['wall_time_s']['p95']:.4f}", flush=True)
    ok_n = [d for d in det if d["wall_time_s"]["p95"] < BUDGET]
    chosen = max(ok_n, key=lambda d: d["max_expansions"]) if ok_n else min(det, key=lambda d: d["max_expansions"])
    # recommended value with headroom: largest N whose idle wall max < 0.2 s and p95 < 0.15 s
    safe = [d for d in det if d["wall_time_s"]["p95"] < 0.15 and d["wall_time_s"]["max"] < BUDGET]
    rec = max(safe, key=lambda d: d["max_expansions"]) if safe else chosen
    N = rec["max_expansions"]
    rep_a = run_queries(gm, prims, pl, allq, budget_s=DET_BUDGET, max_expansions=N)
    det_check = dict(max_expansions=N, idle_repeat=diff_frac(rec["_runs"], rep_a))
    if not args.no_load:
        with BusyLoad(2):
            rep_l = run_queries(gm, prims, pl, allq, budget_s=DET_BUDGET, max_expansions=N)
        det_check["busy2"] = dict(diff_frac(rec["_runs"], rep_l), wall_time_s=summarize_run(rep_l)["wall_time_s"])
    # what changes vs the wall-clock budget run
    det_check["vs_wallclock_idle"] = diff_frac(idle, rec["_runs"])
    cross_det = [rec["_runs"][i]["res"].complete for i in cross_i]
    det_check["cross_complete_frac"] = float(np.mean(cross_det))
    farN = run_queries(gm, prims, pl, farq, budget_s=DET_BUDGET, max_expansions=N)
    det_check["far_pair"] = dict(complete_frac=float(np.mean([x["res"].complete for x in farN])),
                                 wall_time_s=tstats([x["wall"] for x in farN]))
    for d in det:
        d.pop("_runs")
    e["deterministic_sweep"] = det
    e["largest_N_idle_p95_lt_200ms"] = chosen["max_expansions"]
    e["recommended"] = det_check
    rep["e_load"] = e

    rep["verdict"] = dict(a=rep["a_bound"]["passed"], b=rep["b_swept"]["passed"], c=rep["c_timing"]["passed"],
                          d=rep["d_heuristic"]["passed"])
    rep["passed"] = all(rep["verdict"].values())
    rep["queries"] = rows
    rep["runtime_s"] = time.time() - T0
    reports = args.reports_dir if args.reports_dir is not None else ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"planner_gate_{args.policy}.json"
    out.write_text(json.dumps(rep, indent=1, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o)))
    (reports / f"planner_gate_{args.policy}.md").write_text(render_md(rep))
    print(f"PASSED={rep['passed']} {rep['verdict']} -> {out} ({rep['runtime_s']:.0f} s)")
    return rep


def render_md(rep):
    a, b, c, d, e, f, g = (rep[k] for k in ("a_bound", "b_swept", "c_timing", "d_heuristic", "e_load",
                                            "f_cross_room", "g_far_pair"))
    L = [f"# Phase D planner gate — {rep['policy']} (real map + measured primitives)", "",
         f"{rep['date']}, {rep['machine']}, seed {rep['seed']}, budget {rep['budget_s']} s, eps0 {rep['eps0']}. "
         f"Map {rep['map']['shape']} @ {rep['map']['res']} m, r_inflate {rep['map']['r_inflate']:.4f} m "
         f"(sigma {rep['map']['lateral_sigma']:.4f} closed-loop). {rep['n_primitives']} primitives "
         f"({rep['lattice']['prims_per_heading']}/heading, synthesized headings {rep['lattice']['synthesized_headings']}), "
         f"{rep['lattice']['states']} lattice states, precompute {rep['lattice']['precompute_s']:.3f} s.", "",
         f"**Verdict: {'PASS' if rep['passed'] else 'FAIL'}** {rep['verdict']}", "",
         "## (a) bound + goal reached",
         f"- {a['n']} queries, reference finite {a['n_ref_finite']}, complete {a['n_complete']} ({a['complete_frac']:.2f}), "
         f"eps=inf (no bound proven) {a['n_unbounded_eps_inf']}.",
         f"- eps_achieved {a['eps_achieved']}; cost/ref {a['cost_over_ref']}.",
         f"- segment end vs measured end_mean: {a['prim_end_vs_measured']}.",
         f"- failures: {a['failures'] or 'none'}", "",
         "## (b) swept points (independent re-rasterization against gm.lethal)",
         f"- {b['n_paths']} paths, {b['n_points']} points (densified res/4): lethal {b['n_lethal']}; max pose step "
         f"{b['max_pose_step_m']:.4f} m; min clearance on path {b['min_clearance_m']:.4f} m (r_inflate {b['r_inflate_m']:.4f}, "
         f"robot radius {b['robot_radius_m']:.4f}).", "",
         "## (c) timing (single-threaded; ARA* is pure Python, scipy Dijkstra single-thread)",
         f"- plan_time_s {c['plan_time_s']}", f"- wall_time_s {c['wall_time_s']}",
         f"- first solution {c['first_solution_s']}; cpu/wall {c['cpu_over_wall']:.3f}",
         f"- CPU load before {c['cpu_before']['loadavg']} after {c['cpu_after']['loadavg']} on {c['cpu_before']['ncpu']} cores; "
         f"other procs >= 50% CPU: {c['cpu_before'].get('procs_ge_50pct')}", "",
         "## (d) heuristic admissibility",
         *[f"- goal {tuple(None if v is None else round(v, 3) for v in x['goal'])}: finite states {x['n_finite_states']}, "
           f"max(h - ctg) {x['max_h_minus_ctg']:.3g}, pruned-reachable {x['pruned_reachable']}, median h/ctg "
           f"{x['median_h_over_ctg']:.3f}, random states ok {sum(r['ok'] for r in x['random_states'])}/{len(x['random_states'])}"
           for x in d["goals"]], "",
         "## (e) load sensitivity of the 0.2 s wall-clock budget",
         f"- idle repeat: {({k: v for k, v in e['idle_repeat'].items() if k not in ('summary', 'cpu')})}, "
         f"load {e['idle_repeat']['cpu']['loadavg']}, wall p95 {e['idle_repeat']['summary']['wall_time_s']['p95']:.4f}"]
    if "busy2" in e:
        L.append(f"- 2 busy loops: {({k: v for k, v in e['busy2'].items() if k not in ('summary', 'cpu')})}, "
                 f"load {e['busy2']['cpu']['loadavg']}, complete {e['busy2']['summary']['complete_frac']:.2f}, "
                 f"wall p95 {e['busy2']['summary']['wall_time_s']['p95']:.4f}")
    L += ["- deterministic max_expansions sweep (budget_s=60, wall budget non-binding):", "",
          "| max_expansions | complete | eps=inf frac | cost/ref max | wall p50 | wall p95 | wall max |", "|---|---|---|---|---|---|---|"]
    for x in e["deterministic_sweep"]:
        w = x["wall_time_s"]
        L.append(f"| {x['max_expansions']} | {x['complete_frac']:.2f} | {x['eps_inf_frac']:.2f} | "
                 f"{x['cost_over_ref_max']:.3f} | {w['p50']:.4f} | {w['p95']:.4f} | {w['max']:.4f} |")
    r = e["recommended"]
    L += ["", f"- largest N with idle wall p95 < 200 ms: {e['largest_N_idle_p95_lt_200ms']}; recommended (p95 < 150 ms, "
          f"max < 200 ms): **{r['max_expansions']}**. Repeat idle {r['idle_repeat']}; "
          + (f"under 2 busy loops {r['busy2']}; " if "busy2" in r else "")
          + f"vs wall-clock idle run {r['vs_wallclock_idle']}; cross-room complete {r['cross_complete_frac']:.2f}; "
          f"far pair {r['far_pair']}.", "",
          "## (f) cross-room",
          f"- cross-room: {({k: v for k, v in f['cross'].items() if k != 'partial_detail'})}; wall p95 {f['cross_wall_p95']:.4f}",
          f"- same-room: {({k: v for k, v in f['same_room'].items() if k != 'partial_detail'})}",
          f"- forced partial ({FORCED_PARTIAL_EXP} expansions): {f['forced_partial']}", "",
          "## (g) kitchen <-> bathroom (far halves)",
          f"- n {g['n']}, complete {g['complete_frac']:.2f}, wall {g['wall_time_s']}, geodesic {g['geodesic_m']}, "
          f"eps finite among complete {g['eps_finite_frac']}, bound ok {g['all_bound_ok']}, goal ok {g['all_goal_ok']}, "
          f"lethal {g['lethal_points']}", "",
          "## (h) cold start (fresh GridMap object per episode, as episode.py does)",
          f"- first plan() per map: complete {rep['h_cold_start']['cold_complete_frac']:.2f}, second (warm) "
          f"{rep['h_cold_start']['warm_complete_frac']:.2f}; precompute {rep['h_cold_start']['precompute_s']} charged to the "
          f"0.2 s budget. Rows: " + "; ".join(f"{c['rooms']} cold={c['cold_complete']}/{c['cold_reason'] or 'ok'} "
                                             f"exp={c['cold_expansions']}" for c in rep['h_cold_start']['rows']), ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
