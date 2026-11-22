"""Phase D: state-lattice ARA* planner for Microduck (simulation only).

Public API (nav/CONTRACTS.md, Phase D)
--------------------------------------
    plan(gridmap, primitives, start, goal, budget_s=0.2, eps0=3.0, params=None) -> PlanResult
    plan_segment(gridmap, primitives, start, goal, horizon_m=2.5, budget_s=0.2, eps0=3.0) -> PlanResult
    dijkstra_reference(gridmap, primitives, start, goal, params=None) -> float
    cost_to_go_reference(gridmap, primitives, goal, params=None) -> np.ndarray (per lattice state)
    should_replan(pose_est, path, t_since_plan, gridmap, ...) -> bool   (replan_reason() gives why)
    get_planner(gridmap, primitives, params=None) -> LatticePlanner     (cached; exposes heuristic())

``primitives`` is the Phase B JSON dict, a path to it, or the bare list of primitive dicts.
``gridmap`` is anything with the Phase C GridMap attributes (res, lethal, cost, cell_to_world,
world_to_cell, is_lethal).  The planner probes cell_to_world/world_to_cell to learn the
row/column <-> x/y convention and whether cell_to_world returns centers or corners, so it does
not depend on how mapping.py lays out its arrays.

Lattice (Pivtoraiko & Kelly)
----------------------------
State = (lattice cell, heading bin of n_headings=16), packed into one int ``s = cell*16 + h``.
Lattice nodes sit on the centers of every ``lattice_cells``-th map cell (default 2 -> 4 cm on a
2 cm map).  Why 4 cm: (a) it is below the 5 cm goal tolerance, so a 5 cm goal ball always
contains lattice nodes (~19 of them); (b) endpoint snapping error is <= 2 cm per axis, well
inside the follower's tracking error and the primitive end covariance; (c) it cuts the state
count 4x vs 2 cm (26k cells x 16 headings on a 7x6 m map) which is what makes ARA* converge
inside 0.2 s in pure Python.  Collision checks are NOT coarsened: they are done on the 2 cm map.

Edges: every primitive whose start_heading_idx == h.  Endpoint = start + end_mean, with the xy
part rounded to the lattice and dtheta rounded to a heading bin.  The snap correction (<= 1/2
lattice cell per axis) is blended linearly along the swept polyline, so the geometry that is
collision-checked, the geometry that is returned in PlanResult.poses and the lattice node
sequence are all the same curve.  Because nodes always sit on a map-cell center, the 2 cm cell
offsets touched by a primitive are translation invariant and precomputed once per primitive
("per-heading edge cell offsets").  With those offsets the validity + clearance cost of every
(cell, heading, primitive) edge is computed for the whole map with numpy strided slices
(~0.1 s per 7x6 m map, cached keyed on a hash of lethal/cost; like Baritone's cached chunks).

Edge cost (seconds, like Baritone ActionCosts which are in ticks):
    c(e) = duration * (1 + w_clear * cbar_e + w_back*[backward] + w_fall*fall_rate) + w_turn * |dtheta_snapped|
cbar_e = mean map cost along the swept points, taken as the arc-length-weighted mean over the
edge's cell chain path Pi_e (see below): cbar_e = sum_i len_i * cmin_i / sum_i len_i, where cmin_i is
the min map cost of the cells step i touches (turn-in-place: cost of the cell).  Every term is
>= 0, so c(e) >= duration * (1 + w_clear * cbar_e) >= duration.
An edge is dropped if any swept point (densified to <= res/2 spacing, plus the snapped endpoint)
is lethal / off-map, or if the center or any of >= 8 boundary points (one per ~res of perimeter,
<= 32) of the sigma_k=3 ellipse of end_cov[:2,:2] around the endpoint is lethal.  Primitives with
fall_rate > max_fall_rate are not used.

Heuristic and its admissibility (consistency) proof
---------------------------------------------------
    h(s) = max(h_E(s), h_D(s))
    h_E(s) = max(0, |p(s) - p_goal| - tol) / v_E,   v_E = max_e |p(B)-p(A)| / duration_e
    h_D(s) = D2(cell(s)) / v_C,                     v_C = max_e len(Pi_e) / duration_e
D2 = Dijkstra over NON-lethal 2 cm cells, 16-connected (8 neighbours + knight moves (1,2)/(2,1)
that also require the two cells the segment crosses to be non-lethal), move weight
len_move * (1 + w_clear * min cost of the cells the move touches), multi-source from every
non-lethal cell whose center is within tol of the goal ("8-connected Dijkstra on the inflated
grid" of the spec, upgraded: the plain 8-connected metric over-estimates straight 22.5 deg motion
by 8%, which would force a 20% weaker scaling; knight moves cut that to ~2%, and the clearance
weight makes h track the w_clear term).
Pi_e = shortest 16-connected path from cell(A) to cell(B) that uses only cells of edge e's swept
chain (computed once per primitive; chain = cells of the densified swept points).

Claim: h is consistent, h(A) <= c(A,B) + h(B) for every edge, and h = 0 on goal states, hence
admissible (h(s) <= true cost-to-go) and ARA*'s eps-suboptimality bound holds.
 * h_E: |p(A)-g| <= |p(A)-p(B)| + |p(B)-g| and x -> max(0, x - tol) is 1-Lipschitz and
   monotone, so h_E(A) - h_E(B) <= |p(A)-p(B)|/v_E <= duration_e <= c(A,B).  Goal states have
   |p - g| <= tol so h_E = 0.  (Snapping is to a fixed lattice and nodes are lattice points, so
   p(B)-p(A) is the same for every start node of a primitive: v_E is exact.)
 * h_D: densified points are <= res/2 apart, so consecutive points lie in identical or
   8-adjacent cells: the chain is 8-connected, starts in cell(A) and (because the snap correction
   is blended in) ends exactly in cell(B), so Pi_e exists.  A valid edge has every chain cell
   non-lethal, so Pi_e is a path in the D2 graph with D2-weight
   W(Pi_e) = sum_i len_i (1 + w_clear cmin_i) = len(Pi_e) (1 + w_clear cbar_e), hence
   D2(A) <= W(Pi_e) + D2(B) (triangle inequality).  With len(Pi_e) <= v_C duration_e:
   W(Pi_e)/v_C <= duration_e (1 + w_clear cbar_e) <= c(A,B), so h_D(A) <= c(A,B) + h_D(B).
   Goal lattice nodes are cell centers within tol, i.e. D2 seeds, so h_D = 0 there.
   If D2(cell) = inf the cell is disconnected from the goal in the D2 graph, and since every
   valid lattice path is a D2-graph path, the state cannot reach the goal: pruning it is exact.
 * The max of consistent heuristics that vanish on the goal is consistent.
For plan_segment's intermediate goal the final-goal D2 is reused shifted:
h_D'(s) = max(0, D2(s) - max_{seed of subgoal} D2) / v_C; D2(A) <= d(A, sub) + D2(sub) gives the
same proof without a second Dijkstra.  tests/test_planner.py checks h <= exact cost-to-go
(reverse Dijkstra on the full lattice) on all states for several goals, h <= dijkstra_reference
on random states, and h(s) <= c + h(s') on random edges.

Endpoint snap repair (fix A1)
-----------------------------
A true start/goal can be free on the 2 cm map while its nearest 4 cm lattice node is lethal
(the node is up to sqrt(2)/2 * spacing = 2.83 cm away and the inflation boundary passes between).
Then, and only then, the planner searches the lattice nodes within SNAP_REPAIR_SPACINGS * spacing
(= sqrt(2) * 4 cm = 5.66 cm: the smallest radius that always contains all four corners of the
lattice square holding the point) for the nearest non-lethal node joined to the true pose by a
straight connector whose points (densified to <= res/2, both ends included) are all non-lethal
on the 2 cm lethal map.  The lethal map is the footprint-inflated map, so this is the same
swept-footprint check the primitives get.  The start connector keeps the true heading bin
(heading snapping does not change lethality; turning is left to lattice primitives) and the node
must have at least one usable edge at that heading.  A lethal true pose is still refused
(start_lethal / goal_lethal: fail closed).  The start connector is prepended to PlanResult.poses.
For the goal, only when no non-lethal node lies inside goal_tol, nodes within the repair radius
whose connector to the true goal is free become goal states; the heuristic then uses the repair
radius as its tolerance (h stays 0 on every goal state, so the consistency proof is unchanged),
and the connector is appended so the path ends on the true goal.  PlanResult.start_snap /
goal_snap log snap distance, mode and reason.

Search: ARA* (Likhachev, Gordon, Thrun 2003)
--------------------------------------------
eps0 = 3, then 2.5, 2, 1.5, 1.25, 1.1, 1.0 (skipping values above the already proven eps').
Each iteration reuses g-values; states improved while CLOSED go to INCONS and are merged into
OPEN for the next iteration (no search from scratch).  After each completed iteration
eps' = min(eps, g(goal) / min_{OPEN u INCONS}(g + h)) is a proven suboptimality bound and is
returned as eps_achieved.  If the budget expires inside the very first iteration after a goal
state was generated, the path is returned complete=True but with eps_achieved = inf (no bound).
If no goal was reached, the best partial path is returned (state minimizing h among generated
states, Baritone ``bestSoFar``), complete=False.  The budget includes the per-query D2 Dijkstra
(~10 ms) and a reserve for path extraction.

Baritone ideas adopted (from memory of AStarPathFinder / PathExecutor / ActionCosts /
CalculationContext; no network)
 * AStarPathFinder: binary-heap open set over integer state keys (Baritone packs x,y,z into a
   long ``BetterBlockPos.longHash``; we pack cell*16+heading); time-bounded search that, on
   timeout, returns the best partial path toward the goal ("bestSoFar", Baritone picks the node
   with best h + g/coefficient that is far enough from the start - we use pure min-h as asked);
   PathExecutor then walks it and plans the next segment.
 * Moves enum: a fixed, small set of moves with precomputed offsets evaluated per node; here
   the measured primitives play that role with precomputed per-heading cell offsets.
 * ActionCosts: costs are physical time (ticks there, seconds here) with additive penalties,
   and an "infinite" cost for impossible moves (we drop the edge).
 * CalculationContext: a snapshot of everything the search reads (map arrays, edge tables,
   cost params) built before the search, so the hot loop never calls back into the world/map
   object.  We cache it per map hash (Baritone caches chunks).
 * PathExecutor: replan when the bot is too far off the path, when the path becomes blocked
   (movement cost turns infinite), and plan ahead in segments (plan_segment / should_replan).
Not adopted: Baritone's inflated heuristic (COST_HEURISTIC ~3.56, no bound).  ARA* gives the
same "fast first answer" behaviour with a proven, shrinking bound.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as _cs_dijkstra

INF = float("inf")
SQRT2 = math.sqrt(2.0)
DEFAULT_EPS_SCHEDULE = (3.0, 2.5, 2.0, 1.5, 1.25, 1.1, 1.0)


# ----------------------------------------------------------------------------- data types
SNAP_REPAIR_SPACINGS = math.sqrt(2.0)   # repair radius in lattice spacings (see module docstring)

@dataclass(frozen=True)
class PlannerParams:
    lattice_cells: int = 2          # lattice spacing in map cells (2 -> 4 cm on a 2 cm map)
    goal_tol: float = 0.05          # m
    w_clear: float = 2.0            # multiplies mean map cost (0..1) along the swept points
    w_turn: float = 0.3             # s per rad of snapped heading change
    w_back: float = 1.0             # extra fraction of duration for backward motion
    w_fall: float = 20.0            # extra fraction of duration per unit fall_rate
    max_fall_rate: float = 0.05     # primitives above this are not used
    sigma_k: float = 3.0            # end-covariance ellipse check (k-sigma)
    reserve_s: float = 0.015        # part of budget kept for bound computation + path extraction


@dataclass
class PlanResult:
    poses: list                      # [(x, y, theta)], ~2 cm / ~0.15 rad spacing
    primitive_ids: list
    cost: float                      # s, sum of lattice edge costs
    eps_achieved: float              # proven suboptimality bound (inf if none proven)
    complete: bool                   # path ends in the goal region
    plan_time_s: float
    reason: str = ""                 # "", "start_lethal", "goal_unreachable", "timeout_partial", ...
    expansions: int = 0
    iterations: list = field(default_factory=list)   # [{eps, eps_prime, cost, t}]
    first_solution_time_s: float | None = None
    start_state_pose: tuple | None = None
    end_state_pose: tuple | None = None
    subgoal: tuple | None = None     # plan_segment: intermediate goal used (None = final goal)
    final_segment: bool = True
    start_snap: dict | None = None   # A1: {"mode": "nearest"|"repaired", "snap_m", "connector_m", "reason"}
    goal_snap: dict | None = None    # A1: {"mode": "tolerance"|"repaired", ...}
    start_connector_n: int = 0       # A1: poses prepended before the first lattice pose
    goal_connector_n: int = 0        # A1: poses appended after the last lattice pose

    @property
    def ok(self):
        return self.complete


@dataclass
class _Prim:
    id: str
    hd: int
    hd_end: int
    dbins: int
    off: tuple                      # (di, dj) fine-cell offset of the endpoint (multiple of k)
    check: np.ndarray               # (n, 2) int unique fine-cell offsets that must be non-lethal
    cost_steps: list                # [(len_cells, (q,2) cells touched)] along the 16-conn chain path
    chain_len: float                # m, length of that path
    disp: float                     # m, |p(B) - p(A)|
    duration: float
    base_cost: float
    clear_coef: float               # duration * w_clear
    samples: np.ndarray             # (q, 3) world-relative dx, dy, dtheta for output poses
    covariance_offsets: np.ndarray # endpoint 3-sigma covariance boundary, relative to snapped end


# ----------------------------------------------------------------------------- map adapter
class _MapView:
    """Snapshot of a GridMap: arrays + affine fine-index <-> world map (convention-agnostic)."""

    def __init__(self, gm):
        self.gm = gm
        self.lethal = np.ascontiguousarray(np.asarray(gm.lethal, bool))
        cost = np.asarray(gm.cost, np.float64)
        self.cost = np.where(np.isfinite(cost), np.clip(cost, 0.0, None), 0.0)
        self.H, self.W = self.lethal.shape
        self.res = float(gm.res)
        c00 = np.array(gm.cell_to_world(0, 0), float)
        u = np.array(gm.cell_to_world(1, 0), float) - c00
        v = np.array(gm.cell_to_world(0, 1), float) - c00
        probe = c00 - 0.3 * (u + v)
        ij = tuple(int(a) for a in gm.world_to_cell(float(probe[0]), float(probe[1])))
        self.center00 = c00 if ij == (0, 0) else c00 + 0.5 * (u + v)
        self.A = np.column_stack([u, v])
        self.Ainv = np.linalg.inv(self.A)

    def fidx(self, xy):
        """world (..., 2) -> continuous fine index (..., 2); integers are cell centers."""
        return (np.asarray(xy, float) - self.center00) @ self.Ainv.T

    def world(self, ij):
        return self.center00 + np.asarray(ij, float) @ self.A.T

    def lethal_at_world(self, xy):
        ij = np.rint(self.fidx(np.atleast_2d(xy))).astype(np.int64)
        i, j = ij[:, 0], ij[:, 1]
        inside = (i >= 0) & (i < self.H) & (j >= 0) & (j < self.W)
        out = np.ones(len(ij), bool)
        out[inside] = self.lethal[i[inside], j[inside]]
        return out

    def fingerprint(self):
        h = hashlib.blake2b(digest_size=16)
        h.update(self.lethal.tobytes())
        h.update(self.cost.astype(np.float32).tobytes())
        h.update(np.r_[self.center00, self.A.ravel(), self.res].tobytes())
        h.update(str(getattr(self.gm, "footprint_fingerprint", "")).encode())
        fp = getattr(self.gm, "footprint", None)
        if fp is not None and all(hasattr(fp, a) for a in
                ("cliff", "hull", "origin", "res", "cross_track_q_m", "turn_swept_radius_m")):
            h.update(np.asarray(fp.cliff, bool).tobytes())
            h.update(np.asarray(fp.hull, float).tobytes())
            h.update(np.r_[fp.origin, fp.res, fp.cross_track_q_m,
                           fp.turn_swept_radius_m].tobytes())
        return h.hexdigest()


# ----------------------------------------------------------------------------- primitives
def load_primitives(primitives):
    """Accept the JSON dict, a path, or a bare list; return (n_headings, list_of_dicts, meta)."""
    if isinstance(primitives, (str, os.PathLike)):
        with open(primitives) as f:
            primitives = json.load(f)
    if isinstance(primitives, dict):
        nh = int(primitives.get("n_headings", 16))
        plist = list(primitives["primitives"])
        meta = {k: v for k, v in primitives.items() if k != "primitives"}
    else:
        plist = list(primitives)
        nh = 16
        meta = {}
    return nh, plist, meta


def _rotate_prim(p, k_from, k_to, nh):
    """Rotate a world-aligned primitive from heading k_from to k_to (fills missing headings)."""
    a = (k_to - k_from) * 2 * math.pi / nh
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    q = dict(p)
    em = np.array(p["end_mean"], float)
    q["end_mean"] = [*(R @ em[:2]).tolist(), float(em[2])]
    cov = np.array(p["end_cov"], float)
    R3 = np.eye(3)
    R3[:2, :2] = R
    q["end_cov"] = (R3 @ cov @ R3.T).tolist()
    q["swept"] = [(R @ np.asarray(pt[:2], float)).tolist() for pt in p.get("swept", [])]
    q["start_heading_idx"] = k_to
    q["id"] = f"{p['id']}@rot{k_to}"
    return q


# 16-connected moves: (di, dj, length in cells, cells the segment crosses that must also be free)
MOVES16 = [(di, dj, math.hypot(di, dj), ()) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
for _a, _b in ((1, 2), (2, 1)):
    for _si in (-1, 1):
        for _sj in (-1, 1):
            _di, _dj = _a * _si, _b * _sj
            _cross = ((0, _dj // 2), (_di, _dj // 2)) if abs(_dj) == 2 else ((_di // 2, 0), (_di // 2, _dj))
            MOVES16.append((_di, _dj, math.sqrt(5.0), _cross))


def _induced_d16(cells, src, dst):
    """Shortest 16-connected path from src to dst using only `cells` (and only knight moves whose
    two crossed cells are in `cells`).  Returns (length_in_cells, steps) with
    steps = [(step_len_cells, (cells touched by the step: from, to, crossed...)), ...].
    Every such path is a path of the D2 graph whenever all `cells` are non-lethal, which is what
    the consistency proof needs."""
    S = {tuple(int(v) for v in c) for c in cells}
    dist = {src: 0.0}
    back = {}
    pq = [(0.0, src)]
    while pq:
        d, c = heapq.heappop(pq)
        if c == dst:
            steps = []
            while c != src:
                pc, w, cross = back[c]
                steps.append((w, (pc, c) + tuple((pc[0] + a, pc[1] + b) for a, b in cross)))
                c = pc
            return d, steps[::-1]
        if d > dist[c]:
            continue
        for di, dj, w, cross in MOVES16:
            n = (c[0] + di, c[1] + dj)
            if n not in S or any((c[0] + a, c[1] + b) not in S for a, b in cross):
                continue
            nd = d + w
            if nd < dist.get(n, INF):
                dist[n] = nd
                back[n] = (c, w, cross)
                heapq.heappush(pq, (nd, n))
    raise AssertionError("chain does not connect start to end")


def _process_prims(view: _MapView, primitives, P: PlannerParams):
    nh, plist, meta = load_primitives(primitives)
    by_h = {}
    for p in plist:
        by_h.setdefault(int(p["start_heading_idx"]) % nh, []).append(p)
    synthesized = []
    if not by_h:
        raise ValueError("no primitives")
    for k in range(nh):
        if k not in by_h:
            src = min(by_h, key=lambda q: min((k - q) % nh, (q - k) % nh))
            by_h[k] = [_rotate_prim(p, src, k, nh) for p in by_h[src]]
            synthesized.append(k)
    res, kc = view.res, P.lattice_cells
    bin_w = 2 * math.pi / nh
    out = [[] for _ in range(nh)]
    for k in range(nh):
        for p in by_h[k]:
            dur = float(p["duration"])
            fall = float(p.get("fall_rate", 0.0) or 0.0)
            if not dur > 0 or fall > P.max_fall_rate:
                continue
            out[k].append(_build_prim(view, p, k, nh, bin_w, res, kc, dur, fall, P))
    return nh, out, synthesized, meta


def _build_prim(view, p, k, nh, bin_w, res, kc, dur, fall, P):
    em = np.asarray(p["end_mean"], float)
    end_xy = em[:2]
    dbins = int(round(float(em[2]) / bin_w))
    hd_end = (k + dbins) % nh
    # snap endpoint to lattice (fine offsets multiple of kc); nodes sit on cell centers
    e_f = (end_xy) @ view.Ainv.T
    off = (np.rint(e_f / kc) * kc).astype(int)
    snapped_xy = off @ view.A.T
    corr = snapped_xy - end_xy
    # polyline start -> swept -> end, blend correction by arclength fraction
    pts = [np.zeros(2)]
    for q in p.get("swept", []) or []:
        pts.append(np.asarray(q[:2], float))
    pts.append(end_xy.copy())
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    frac = cum / cum[-1] if cum[-1] > 1e-9 else np.linspace(0.0, 1.0, len(pts))
    q_pts = pts + frac[:, None] * corr[None, :]
    # densify to spacing <= res/2
    dense, dfrac = [q_pts[0]], [0.0]
    for a in range(len(q_pts) - 1):
        d = np.linalg.norm(q_pts[a + 1] - q_pts[a])
        n = max(1, int(math.ceil(d / (0.5 * res))))
        for t in range(1, n + 1):
            dense.append(q_pts[a] + (q_pts[a + 1] - q_pts[a]) * t / n)
            dfrac.append(frac[a] + (frac[a + 1] - frac[a]) * t / n)
    dense = np.array(dense)
    dfrac = np.array(dfrac)
    cells = np.rint(dense @ view.Ainv.T).astype(int)
    assert tuple(cells[-1]) == tuple(off), "snapped endpoint must end in the lattice cell"
    # chain (dedupe consecutive): an 8-connected cell path start -> end
    keep = np.r_[True, np.any(np.diff(cells, axis=0) != 0, axis=1)]
    chain = cells[keep]
    steps = np.abs(np.diff(chain, axis=0))
    assert steps.size == 0 or steps.max() <= 1, "chain must be 8-connected"
    chain_cells, csteps = _induced_d16(chain, (0, 0), tuple(off))
    chain_len = res * chain_cells
    cost_steps = [(w, np.array(cl, int)) for w, cl in csteps]
    # 3-sigma ellipse of end_cov (position) around the snapped endpoint
    cov = np.asarray(p["end_cov"], float)[:2, :2]
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    radii = P.sigma_k * np.sqrt(np.clip(vals, 0.0, None))
    n_b = int(min(32, max(8, math.ceil(2 * math.pi * float(radii.max()) / res))))
    phis = np.linspace(0, 2 * math.pi, n_b, endpoint=False)
    ell = snapped_xy[None, :] + (np.stack([np.cos(phis), np.sin(phis)], 1) * radii[None, :]) @ vecs.T
    ell_cells = np.rint(ell @ view.Ainv.T).astype(int)
    check = np.unique(np.vstack([chain, ell_cells, off[None, :]]), axis=0)
    # output pose samples (relative), theta interpolated linearly in fraction
    dth = dbins * bin_w
    samples = []
    last = np.zeros(2)
    last_t = 0.0
    for idx in range(1, len(dense)):
        th = dfrac[idx] * dth
        if (np.linalg.norm(dense[idx] - last) >= 0.02 - 1e-9 or abs(th - last_t) >= 0.15
                or idx == len(dense) - 1):
            samples.append((dense[idx][0], dense[idx][1], th))
            last, last_t = dense[idx], th
    if not samples or samples[-1][2] != dth:
        samples.append((snapped_xy[0], snapped_xy[1], dth))
    heading = np.array([math.cos(k * bin_w), math.sin(k * bin_w)])
    backward = float(snapped_xy @ heading) < -1e-6 or float(end_xy @ heading) < -1e-6
    base = dur * (1.0 + (P.w_back if backward else 0.0) + P.w_fall * fall) + P.w_turn * abs(dth)
    return _Prim(id=str(p["id"]), hd=k, hd_end=hd_end, dbins=dbins, off=(int(off[0]), int(off[1])),
                 check=check, cost_steps=cost_steps, chain_len=chain_len, disp=float(np.linalg.norm(snapped_xy)), duration=dur,
                 base_cost=base, clear_coef=dur * P.w_clear, samples=np.array(samples, float),
                 covariance_offsets=np.asarray(ell - snapped_xy, float))


# ----------------------------------------------------------------------------- planner
class LatticePlanner:
    """Per (map, primitives, params) precomputed lattice ('CalculationContext')."""

    def __init__(self, gridmap, primitives, params: PlannerParams | None = None):
        t0 = time.perf_counter()
        self.P = P = params or PlannerParams()
        self.view = view = _MapView(gridmap)
        self.fingerprint = view.fingerprint()
        self.nh, self.prims, self.synthesized_headings, self.meta = _process_prims(view, primitives, P)
        nh, k = self.nh, P.lattice_cells
        H, W = view.H, view.W
        self.Hc, self.Wc = Hc, Wc = -(-H // k), -(-W // k)
        self.n_cells = Hc * Wc
        self.N = self.n_cells * nh
        self.Pmax = max(len(ps) for ps in self.prims)
        self.bin_w = 2 * math.pi / nh
        # lattice node positions (fine idx and world)
        a, b = np.meshgrid(np.arange(Hc), np.arange(Wc), indexing="ij")
        self.node_fidx = np.stack([a.ravel() * k, b.ravel() * k], 1)
        self.node_xy = view.world(self.node_fidx)
        inside = (self.node_fidx[:, 0] < H) & (self.node_fidx[:, 1] < W)
        ok = np.zeros(self.n_cells, bool)
        ok[inside] = ~view.lethal[self.node_fidx[inside, 0], self.node_fidx[inside, 1]]
        self.node_ok = ok
        allp = [p for ps in self.prims for p in ps]
        self.v_E = max((p.disp / p.duration for p in allp), default=0.0)
        self.v_C = max((p.chain_len / p.duration for p in allp), default=0.0)
        self._build_tables()
        self._build_d2_graph()
        self._goal_cache = {}
        # goal_spec's is_goal depends on the footprint (pose_safe at goal headings); the cache is
        # cleared whenever the attached footprint object changes (the reference is kept, so the
        # identity cannot be reused by a later object).
        self._goal_fp = None
        # exact per-primitive-edge verdicts (footprint-only; blacklist-independent), reused across
        # replan rounds and queries; cleared with the goal cache when the footprint changes
        self._exact_cache = {}
        self._prefilter = None       # (cliff_dist_m or None if no cliff, footprint extent m) when usable
        self._footprint_table = None
        self._cheap_tab = self.tab.copy()
        self._sync_footprint_caches()
        self._full_graph = None
        self.precompute_s = time.perf_counter() - t0

    # ---- edge tables
    def _build_tables(self):
        view, P, k = self.view, self.P, self.P.lattice_cells
        H, W, Hc, Wc, nh = view.H, view.W, self.Hc, self.Wc, self.nh
        allp = [p for ps in self.prims for p in ps]
        M = int(np.abs(np.vstack([p.check for p in allp])).max()) + 2 * k
        lp = np.ones((H + 2 * M, W + 2 * M), bool)
        lp[M:M + H, M:M + W] = view.lethal
        cp = np.zeros((H + 2 * M, W + 2 * M), np.float64)
        cp[M:M + H, M:M + W] = view.cost
        Lph = {(pi, pj): np.ascontiguousarray(lp[pi::k, pj::k]) for pi in range(k) for pj in range(k)}
        Cph = {(pi, pj): np.ascontiguousarray(cp[pi::k, pj::k]) for pi in range(k) for pj in range(k)}

        def view_of(ph, di, dj):
            r, c = M + di, M + dj
            return ph[(r % k, c % k)][r // k: r // k + Hc, c // k: c // k + Wc]

        tab = np.full((Hc, Wc, nh, self.Pmax), INF)
        self.deltas = []
        for hd in range(nh):
            dl = []
            for kk, p in enumerate(self.prims[hd]):
                bad = np.zeros((Hc, Wc), bool)
                for di, dj in p.check:
                    bad |= view_of(Lph, int(di), int(dj))
                if p.cost_steps:     # length-weighted mean of per-step min cost along the chain
                    csum = np.zeros((Hc, Wc))
                    wsum = 0.0
                    for w, cl in p.cost_steps:
                        m = view_of(Cph, int(cl[0, 0]), int(cl[0, 1]))
                        for di, dj in cl[1:]:
                            m = np.minimum(m, view_of(Cph, int(di), int(dj)))
                        csum += w * m
                        wsum += w
                    mean = csum / wsum
                else:                # turn in place: cost of the cell
                    mean = view_of(Cph, 0, 0)
                val = p.base_cost + p.clear_coef * mean
                val[bad] = INF
                tab[:, :, hd, kk] = val
                dcell = (p.off[0] // k) * Wc + (p.off[1] // k)
                dl.append(dcell * nh + (p.hd_end - hd))
            self.deltas.append(dl)
        tab[~self.node_ok.reshape(Hc, Wc)] = INF
        self.tab = tab.reshape(self.N, self.Pmax)

    # ---- 2D Dijkstra graph over fine non-lethal cells (16-connected, see MOVES16)
    def _build_d2_graph(self):
        view = self.view
        H, W, res = view.H, view.W, view.res
        Fp = np.zeros((H + 4, W + 4), bool)
        Fp[2:H + 2, 2:W + 2] = ~view.lethal
        Cp = np.zeros((H + 4, W + 4))
        Cp[2:H + 2, 2:W + 2] = view.cost
        idx = np.arange(H * W).reshape(H, W)
        rows, cols, ws = [], [], []
        wc = self.P.w_clear

        def F(arr, di, dj):
            return arr[2 + di:2 + di + H, 2 + dj:2 + dj + W]

        for di, dj, w, cross in MOVES16:
            if (di, dj) < (0, 0):
                continue  # undirected: one direction per pair
            m = F(Fp, 0, 0) & F(Fp, di, dj)
            cmin = np.minimum(F(Cp, 0, 0), F(Cp, di, dj))
            for a, b in cross:
                m = m & F(Fp, a, b)
                cmin = np.minimum(cmin, F(Cp, a, b))
            ii, jj = np.nonzero(m)
            rows.append(idx[ii, jj])
            cols.append(idx[ii + di, jj + dj])
            ws.append(w * res * (1.0 + wc * cmin[ii, jj]))
        r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(ws)
        self.d2_graph = sp.csr_matrix((np.r_[w, w], (np.r_[r, c], np.r_[c, r])), shape=(H * W, H * W))

    # ---- goal handling
    def heading_bin(self, theta):
        return int(round(theta / self.bin_w)) % self.nh

    def _goal_seeds(self, gxy, tol=None):
        view = self.view
        tol = self.P.goal_tol if tol is None else float(tol)
        f = view.fidx(gxy)
        r = int(math.ceil(tol / view.res)) + 2
        ii, jj = np.meshgrid(np.arange(int(round(f[0])) - r, int(round(f[0])) + r + 1),
                             np.arange(int(round(f[1])) - r, int(round(f[1])) + r + 1), indexing="ij")
        ij = np.stack([ii.ravel(), jj.ravel()], 1)
        ij = ij[(ij[:, 0] >= 0) & (ij[:, 0] < view.H) & (ij[:, 1] >= 0) & (ij[:, 1] < view.W)]
        d = np.linalg.norm(view.world(ij) - gxy, axis=1)
        ij = ij[(d <= tol) & ~view.lethal[ij[:, 0], ij[:, 1]]]
        return ij[:, 0] * view.W + ij[:, 1]

    def d2_from(self, gxy, tol=None):
        """Clearance-weighted 16-connected 2D Dijkstra distance over fine cells from the tol-ball
        around gxy (see module docstring).  Cached per goal position (and tol if given)."""
        key = ("d2", round(float(gxy[0]), 6), round(float(gxy[1]), 6)) + (() if tol is None else (round(float(tol), 9),))
        hit = self._goal_cache.get(key)
        if hit is not None:
            return hit
        seeds = self._goal_seeds(np.asarray(gxy, float), tol)
        if len(seeds) == 0:
            d2 = np.full(self.view.H * self.view.W, INF)
        else:
            d2 = _cs_dijkstra(self.d2_graph, directed=False, indices=seeds, min_only=True)
        self._cache_put(key, (d2, seeds))
        return d2, seeds

    def _cache_put(self, key, val):
        if len(self._goal_cache) > 16:
            self._goal_cache.pop(next(iter(self._goal_cache)))
        self._goal_cache[key] = val

    def goal_spec(self, goal, d2_goal=None, repair=False):
        """-> (is_goal bytearray over states, h list over lattice cells, h numpy over cells).

        goal = (x, y, theta|None).  If d2_goal (an (x,y) final goal) is given, h_D is the shifted
        final-goal D2 (plan_segment intermediate goal), see module docstring.
        """
        self._sync_footprint_caches()
        gx, gy = float(goal[0]), float(goal[1])
        gth = goal[2] if len(goal) > 2 else None
        key = ("spec", round(gx, 6), round(gy, 6), None if gth is None else self.heading_bin(gth),
               None if d2_goal is None else (round(float(d2_goal[0]), 6), round(float(d2_goal[1]), 6))) \
            + (("repair",) if repair else ())
        tol = self.repair_radius if repair else self.P.goal_tol
        hit = self._goal_cache.get(key)
        if hit is not None:
            return hit
        gxy = np.array([gx, gy])
        inside = (self.node_fidx[:, 0] < self.view.H) & (self.node_fidx[:, 1] < self.view.W)
        lin = self.node_fidx[inside, 0] * self.view.W + self.node_fidx[inside, 1]
        if d2_goal is None:
            d2, seeds = self.d2_from(gxy, tol if repair else None)
            dD = np.full(self.n_cells, INF)
            dD[inside] = d2[lin]
        else:
            d2, _ = self.d2_from(np.asarray(d2_goal, float))
            seeds = self._goal_seeds(gxy)
            dD = np.full(self.n_cells, INF)
            if len(seeds):
                shift = float(np.max(d2[seeds]))
                dD[inside] = np.maximum(0.0, d2[lin] - shift)
        dE = np.maximum(0.0, np.linalg.norm(self.node_xy - gxy, axis=1) - tol)
        h = np.maximum(dE / self.v_E if self.v_E > 0 else 0.0, dD / self.v_C if self.v_C > 0 else 0.0)
        h[~self.node_ok] = INF
        if len(seeds) == 0:
            h[:] = INF
        goal_cells = np.nonzero(self.node_ok & (np.linalg.norm(self.node_xy - gxy, axis=1) <= tol))[0]
        if repair:
            goal_cells = np.array([c for c in goal_cells.tolist()
                                   if self.connector_free(self.node_xy[c], gxy)], dtype=np.int64)
        is_goal = bytearray(self.N)
        hds = range(self.nh) if gth is None else [self.heading_bin(gth)]
        footprint = getattr(self.view.gm, "footprint", None)
        for c in goal_cells.tolist():
            for hd in hds:
                if footprint is not None:
                    gx, gy = gxy
                    nx, ny = self.node_xy[c]
                    ntheta = _wrap(hd * self.bin_w)
                    gtheta = ntheta if gth is None else float(gth)
                    if not footprint.pose_safe((float(nx), float(ny), ntheta)):
                        continue
                    if repair:
                        pts = self.connector_points((nx, ny), gxy)
                        route = []
                        for q, (x, y) in enumerate(pts):
                            a = q / max(len(pts) - 1, 1)
                            route.append((float(x), float(y), _wrap(ntheta + a * _wrap(gtheta-ntheta))))
                        if not self._footprint_swept_safe(
                                route, heading_deltas=self._route_heading_deltas(route)):
                            continue
                is_goal[c * self.nh + hd] = 1
        h[goal_cells] = 0.0
        spec = (is_goal, h.tolist(), h, len(goal_cells))
        self._cache_put(key, spec)
        return spec

    def heuristic(self, goal):
        """h over all states (numpy, len N) for tests."""
        _, _, h, _ = self.goal_spec(goal)
        return np.repeat(h, self.nh)

    # ---- state helpers
    @property
    def repair_radius(self):
        return SNAP_REPAIR_SPACINGS * self.P.lattice_cells * self.view.res

    def connector_points(self, a_xy, b_xy):
        a, b = np.asarray(a_xy, float), np.asarray(b_xy, float)
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / (0.5 * self.view.res))))
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        return a + t * (b - a)

    def connector_free(self, a_xy, b_xy):
        """Straight segment a->b: every point (<= res/2 apart, both ends included) non-lethal on
        the 2 cm footprint-inflated lethal map."""
        return not bool(self.view.lethal_at_world(self.connector_points(a_xy, b_xy)).any())

    def repair_start(self, pose):
        """-> (state, info).  Nearest non-lethal lattice node within repair_radius joined to the
        true pose by a free connector and with at least one usable edge at the true heading bin.
        state = -1 if the true pose is lethal or no candidate exists (fail closed)."""
        xy = np.array(pose[:2], float)
        info = {"mode": "repaired", "reason": "snapped_node_lethal"}
        if bool(self.view.lethal_at_world(xy)[0]):
            info["reason"] = "true_pose_lethal"
            return -1, info
        cand = np.nonzero(self.node_ok & (np.linalg.norm(self.node_xy - xy, axis=1) <= self.repair_radius))[0]
        hd = self.heading_bin(float(pose[2]))
        d = np.linalg.norm(self.node_xy[cand] - xy, axis=1)
        for c in cand[np.lexsort((cand, d))].tolist():
            s = c * self.nh + hd
            if not np.isfinite(self.tab[s]).any():
                continue
            conn = self.connector_points(xy, self.node_xy[c])
            target_pose = self.state_pose(s)
            route, node_pose = self._connector_route(pose, conn, target_pose[2])
            conn_poses = route + [node_pose]   # exactly the poses plan() will output
            if (self.connector_free(xy, self.node_xy[c])
                    and self._footprint_pose_safe(target_pose)
                    and self._footprint_swept_safe(
                        conn_poses, heading_deltas=self._route_heading_deltas(conn_poses))):
                info.update(connector_m=float(np.linalg.norm(self.node_xy[c] - xy)),
                            node_xy=[float(v) for v in self.node_xy[c]])
                return s, info
        info["reason"] = "no_free_node_in_radius"
        return -1, info

    def _sync_footprint_caches(self):
        """Drop footprint-dependent caches when the attached footprint object changes."""
        fp_now = getattr(self.view.gm, "footprint", None)
        if fp_now is not self._goal_fp:
            self._goal_cache = {}
            self._exact_cache = {}
            self._goal_fp = fp_now
            self._prefilter = None
            if fp_now is not None and all(hasattr(fp_now, a) for a in
                                          ("hull", "cliff", "origin", "res", "turn_swept_radius_m",
                                           "cross_track_q_m")):
                from scipy.ndimage import distance_transform_edt
                cliff = np.asarray(fp_now.cliff, bool)
                cliff_dist = (distance_transform_edt(~cliff) * float(fp_now.res)) if cliff.any() else None
                extent = (max(float(np.max(np.linalg.norm(fp_now.hull, axis=1))),
                              float(fp_now.turn_swept_radius_m)) + float(fp_now.cross_track_q_m))
                self._prefilter = (cliff_dist, extent)
            self.tab = self._cheap_tab.copy()
            self._footprint_table = None
            self._full_graph = None
            self._build_footprint_table(fp_now)

    def _build_footprint_table(self, fp):
        """Conservative closed-square cliff tests at the lattice's fixed sub-cell phase.

        Only the unmodified geometry Footprint supports this optimization. Other checkers
        retain the lazy exact/blacklist path. Setup runs before the planning deadline.
        """
        if fp is None:
            return
        from nav.a6_footprint import Footprint
        if type(fp) is not Footprint:
            return
        # The cliff geometry uses world x/y squares. Fall back on other affine grids.
        res = float(fp.res)
        if (res != self.view.res or fp.cliff.shape != (self.view.H, self.view.W)
                or not np.allclose(self.view.A, [[0, res], [res, 0]], rtol=0, atol=1e-12)):
            return
        xy0 = self.node_xy[0]
        base = np.floor((xy0 - fp.origin[:2]) / res).astype(int)[::-1]
        phase = xy0 - (fp.origin[:2] + (base[::-1] + 0.5) * res)
        phase_error = float(np.abs(self.node_xy - (xy0 + self.node_fidx[:, ::-1] * res)).max())
        # The phase is retained, including non-centred nodes; no cell alignment assumption.
        self._footprint_offsets = {}
        for hd, ps in enumerate(self.prims):
            for kk, p in enumerate(ps):
                offsets = self._swept_cell_offsets(fp, p, phase, phase_error)
                self._footprint_offsets[hd, kk] = offsets
        margin = int(max(np.abs(v + base).max() for v in self._footprint_offsets.values())) + self.P.lattice_cells
        H, W = fp.cliff.shape
        cliff = np.ones((H + 2 * margin, W + 2 * margin), bool)
        cliff[margin:margin + H, margin:margin + W] = fp.cliff
        safe = np.ones((self.Hc, self.Wc, self.nh, self.Pmax), bool)
        k = self.P.lattice_cells
        for (hd, kk), offsets in self._footprint_offsets.items():
            bad = np.zeros((self.Hc, self.Wc), bool)
            for di, dj in offsets:
                i, j = margin + base + (di, dj)
                bad |= cliff[i:i + k * self.Hc:k, j:j + k * self.Wc:k]
            safe[:, :, hd, kk] = ~bad
        self._footprint_table = safe.reshape(self.N, self.Pmax)
        self.tab[~self._footprint_table] = INF

    def _swept_cell_offsets(self, fp, p, phase, phase_error=0.0):
        """Union conservative closed-square rasters of local sweep envelopes.

        Endpoint hulls plus circular-arc sagitta enclose every interpolated polygon,
        independently of ceil/roundoff in the exact checker's subdivision count. Add
        its maximum local sagitta too. Turn disks use exact circle/square distances.
        """
        from nav.a6_footprint import convex_hull, _minkowski_square
        th0 = p.hd * self.bin_w
        poses = [(float(dx), float(dy), _wrap(th0 + dth)) for dx, dy, dth in p.samples]
        deltas = self._route_heading_deltas(poses)
        ex, ey, eth = poses[-1]
        ring = [(float(ex + dx), float(ey + dy), eth) for dx, dy in p.covariance_offsets]
        ring_delta = deltas[-1] if deltas else 0.0
        routes = [(poses, deltas)]
        if ring:
            routes.append((ring + ring[:1], [ring_delta] * (len(ring) + 1)))
        rh = float(np.linalg.norm(fp.hull, axis=1).max())
        q = float(fp.cross_track_q_m)
        exact_pad = (rh + q) * (1 - math.cos(0.05 / 2))
        numerical = phase_error + 1e-9
        cells = []

        def candidates(lo, hi):
            lo = np.floor(lo / fp.res - 0.5).astype(int)
            hi = np.ceil(hi / fp.res + 0.5).astype(int)
            jj, ii = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
            offsets = np.column_stack((ii.ravel(), jj.ravel()))
            return offsets, offsets[:, ::-1] * fp.res

        def raster(poly, pad):
            poly = _minkowski_square(poly, pad + numerical) + phase
            offsets, centers = candidates(poly.min(axis=0), poly.max(axis=0))
            edges = np.roll(poly, -1, axis=0) - poly
            axes = np.vstack((np.eye(2), np.column_stack((-edges[:, 1], edges[:, 0]))))
            pp, cc = poly @ axes.T, centers @ axes.T
            radius = 0.5 * fp.res * np.abs(axes).sum(axis=1)
            touches = np.all((cc + radius >= pp.min(axis=0) - 1e-12)
                             & (cc - radius <= pp.max(axis=0) + 1e-12), axis=1)
            cells.append(offsets[touches])

        for route, ds in routes:
            if len(route) == 1:
                raster(fp._pose_polygon(route[0], ds[0]), 0.0)
            for a, b, da, db in zip(route[:-1], route[1:], ds[:-1], ds[1:]):
                a, b = np.asarray(a), np.asarray(b)
                yaw, delta = _wrap(b[2] - a[2]), _wrap(db - da)
                normal = yaw + delta
                # Partition the actual unwrapped normal rotation too. Each arc is
                # <= 0.05 rad, so its endpoint-chord sagitta bound applies.
                n = max(1, int(math.ceil(max(abs(yaw), abs(normal)) / 0.05)))
                pad = rh * (1 - math.cos(abs(yaw) / n / 2))
                pad += q * (1 - math.cos(abs(normal) / n / 2)) + exact_pad
                previous = None
                for step in range(n + 1):
                    u = step / n
                    pose = (a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), a[2] + u * yaw)
                    poly = fp._pose_polygon(pose, da + u * delta)
                    if previous is not None:
                        raster(convex_hull(np.vstack((previous, poly))), pad)
                    previous = poly
                dist = float(np.linalg.norm(b[:2] - a[:2]))
                if dist <= 1e-12 and abs(yaw) > 1e-10:
                    center = a[:2] + phase
                    radius = fp.turn_swept_radius_m + q + dist + math.sqrt(2) * numerical
                    offsets, centers = candidates(center - radius, center + radius)
                    delta_xy = np.maximum(np.abs(centers - center) - 0.5 * fp.res, 0)
                    cells.append(offsets[np.sum(delta_xy * delta_xy, axis=1) <= radius * radius])
        return np.unique(np.vstack(cells), axis=0)

    def _edge_far_from_cliffs(self, pts):
        """Conservative skip for the exact edge check. True only if no cliff square can touch the
        swept footprint of an edge whose sample and covariance-ring points are `pts`. False means
        "unknown, run the exact check" (including when no prefilter is available).

        Every swept point lies within h + R + sagitta of the bbox centre c (h = half bbox diagonal,
        R = footprint extent about the trunk point, sagitta <= res/4). The centre's cell-centre
        distance d is within 0.71 res of c and a cliff square lies within 0.71 res of its centre.
        So if d > h + R + 2.5 res, no cliff cell can be touched."""
        if self._prefilter is None:
            return False
        cliff_dist, extent = self._prefilter
        if cliff_dist is None:
            return True                      # no cliff cells at all: nothing to touch
        fp = self.view.gm.footprint
        xy = np.asarray(pts, float)[:, :2]
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        c = 0.5 * (lo + hi)
        h = 0.5 * float(np.linalg.norm(hi - lo))
        j = int(math.floor((c[0] - float(fp.origin[0])) / fp.res))
        i = int(math.floor((c[1] - float(fp.origin[1])) / fp.res))
        if not (0 <= i < cliff_dist.shape[0] and 0 <= j < cliff_dist.shape[1]):
            return False
        return bool(cliff_dist[i, j] > h + extent + 2.5 * float(fp.res))

    @staticmethod
    def _connector_route(true_pose, conn_xy, th_node):
        """A1 connector poses, shared by the exact check and the output route so they cannot
        disagree. conn_xy[0] is the true start, conn_xy[-1] the snapped node (heading th_node).
        Heading interpolates from the true start heading over the connector; the true start pose
        itself keeps its heading. Returns (connector poses without the node, node pose)."""
        n = max(len(conn_xy) - 1, 1)
        th0 = float(true_pose[2])
        dth = _wrap(float(th_node) - th0)
        route = [(float(x), float(y), _wrap(th0 + (k / n) * dth))
                 for k, (x, y) in enumerate(conn_xy[:-1])]
        return route, (float(conn_xy[-1][0]), float(conn_xy[-1][1]), float(th_node))

    def _footprint_pose_safe(self, pose):
        fp = getattr(self.view.gm, "footprint", None)
        return True if fp is None else bool(fp.pose_safe(pose))

    def _footprint_swept_safe(self, poses, heading_deltas=None):
        fp = getattr(self.view.gm, "footprint", None)
        return True if fp is None else bool(fp.swept_safe(poses, heading_deltas=heading_deltas))

    @staticmethod
    def _route_heading_deltas(poses):
        deltas = []
        for q, pose in enumerate(poses):
            if len(poses) < 2:
                deltas.append(0.0)
                continue
            a, b = max(0, q - 1), min(len(poses) - 1, q + 1)
            dx, dy = poses[b][0] - poses[a][0], poses[b][1] - poses[a][1]
            deltas.append(0.0 if dx * dx + dy * dy < 1e-12
                          else _wrap(math.atan2(dy, dx) - pose[2]))
        return deltas

    def _primitive_edge_safe(self, s, kk):
        """Exact verdict for one primitive edge from state s: measured sample sweep plus the
        3-sigma endpoint covariance ring (both oriented by the local travel tangent)."""
        if self._footprint_table is not None:
            return bool(self._footprint_table[s, kk])
        return self._primitive_edge_safe_exact(s, kk)

    def _primitive_edge_safe_exact(self, s, kk):
        """Uncached geometry reference, also used by soundness tests."""
        c, hd = divmod(s, self.nh)
        p = self.prims[hd][kk]
        x0, y0 = self.node_xy[c]
        th0 = hd * self.bin_w
        poses = [(float(x0 + dx), float(y0 + dy), _wrap(th0 + dth))
                 for dx, dy, dth in p.samples]
        deltas = self._route_heading_deltas(poses)
        # Preserve the original 3-sigma endpoint covariance set while applying
        # the orientation-aware footprint against raw cliff cells.
        endx, endy, endth = poses[-1]
        ring = [(float(endx + dx), float(endy + dy), endth) for dx, dy in p.covariance_offsets]
        if self._edge_far_from_cliffs(poses + ring):
            return True                      # provably far from every cliff cell: exact check skipped
        if not self._footprint_swept_safe(poses, heading_deltas=deltas):
            return False
        ring_delta = deltas[-1] if deltas else 0.0
        if ring and (not all(self._footprint_pose_safe_delta(q, ring_delta) for q in ring)
                     or not self._footprint_swept_safe(
                         ring + ring[:1], heading_deltas=[ring_delta] * (len(ring) + 1))):
            return False
        return True

    def _footprint_pose_safe_delta(self, pose, heading_delta):
        return bool(self.view.gm.footprint.pose_safe(pose, heading_delta=heading_delta))

    def _exact_bad_edges(self, states, ks, deadline=None):
        """Exact verdicts for the provisional path's primitive edges only (cheap cell-table checks
        are untouched). Returns the set of unsafe (s, kk), or None if the deadline passes before
        every unknown edge is verified (the caller must then fail closed)."""
        self._sync_footprint_caches()
        if self._footprint_table is not None:
            # Every verdict was proved at setup time; like a cache hit it needs no
            # geometry work after ARA* has used its refinement deadline.
            return {(int(s), int(kk)) for s, kk in zip(states[:-1], ks)
                    if not self._footprint_table[s, kk]}
        perf = time.perf_counter
        bad = set()
        for s, kk in zip(states[:-1], ks):
            key = (int(s), int(kk))
            ok = self._exact_cache.get(key)
            if ok is None:
                if deadline is not None and perf() > deadline:
                    return None
                ok = self._primitive_edge_safe(*key)
                self._exact_cache[key] = ok
            if not ok:
                bad.add(key)
        return bad

    def snap(self, pose):
        f = self.view.fidx(np.array(pose[:2], float)) / self.P.lattice_cells
        a, b = int(round(f[0])), int(round(f[1]))
        if not (0 <= a < self.Hc and 0 <= b < self.Wc):
            return -1
        return (a * self.Wc + b) * self.nh + self.heading_bin(float(pose[2]))

    def state_pose(self, s):
        c, hd = divmod(s, self.nh)
        x, y = self.node_xy[c]
        return (float(x), float(y), _wrap(hd * self.bin_w))

    # ---- ARA*
    def plan(self, start, goal, budget_s=0.2, eps0=3.0, t_start=None, _d2_goal=None, max_expansions=None):
        t0 = time.perf_counter() if t_start is None else t_start
        deadline = t0 + budget_s - self.P.reserve_s
        s0 = self.snap(start)
        sxy = np.array(start[:2], float)
        start_snap = {"mode": "nearest", "snap_m": None if s0 < 0 else
                      float(np.linalg.norm(self.node_xy[s0 // self.nh] - sxy))}
        start_conn = None
        goal_snap = {"mode": "tolerance"}
        goal_conn = None

        def fail(reason):
            return PlanResult([], [], INF, INF, False, time.perf_counter() - t0, reason=reason,
                              start_state_pose=None if s0 < 0 else self.state_pose(s0),
                              start_snap=start_snap, goal_snap=goal_snap)

        if not self._footprint_pose_safe(start):
            return fail("start_footprint_lethal")
        if len(goal) > 2 and goal[2] is not None:
            if not self._footprint_pose_safe(goal):
                return fail("goal_footprint_lethal")
        elif getattr(self.view.gm, "footprint", None) is not None:
            if not any(self._footprint_pose_safe((goal[0], goal[1], h))
                       for h in np.linspace(-math.pi, math.pi, self.nh, endpoint=False)):
                return fail("goal_footprint_lethal")

        footprint_on = getattr(self.view.gm, "footprint", None) is not None
        # Footprint triggers: the snapped state pose is unsafe, or the direct snap jump from the true
        # start to the node (never output, the path starts at the node) sweeps a cliff. Either one
        # goes through the A1 repair, which exact-checks its own connector.
        if s0 >= 0 and (not self.node_ok[s0 // self.nh]
                        or (footprint_on and (not self._footprint_pose_safe(self.state_pose(s0))
                                              or not self._snap_jump_safe(start, s0)))):
            s_rep, info = self.repair_start(start)
            start_snap.update(info)
            if s_rep >= 0:
                s0 = s_rep
                start_conn = self.connector_points(sxy, self.node_xy[s0 // self.nh])
            elif footprint_on and self.node_ok[s0 // self.nh]:
                return fail("start_lethal")   # footprint-only trigger that A1 repair cannot satisfy
        if s0 < 0 or not self.node_ok[s0 // self.nh]:
            return fail("start_lethal")
        is_goal, hl, _, n_goal = self.goal_spec(goal, d2_goal=_d2_goal)
        if n_goal == 0 and _d2_goal is None:
            gxy = np.array(goal[:2], float)
            if not bool(self.view.lethal_at_world(gxy)[0]):
                is_goal, hl, _, n_goal = self.goal_spec(goal, repair=True)
                goal_snap = {"mode": "repaired", "reason": "no_free_node_in_tol", "n_goal_nodes": int(n_goal)}
                goal_conn = gxy
            else:
                goal_snap = {"mode": "tolerance", "reason": "true_pose_lethal"}
        if n_goal == 0:
            return fail("goal_lethal")
        if is_goal[s0]:
            ps = self.state_pose(s0)
            res = PlanResult([ps], [], 0.0, 1.0, True, time.perf_counter() - t0,
                             start_state_pose=ps, end_state_pose=ps)
        elif hl[s0 // self.nh] == INF:
            return fail("goal_unreachable")
        elif getattr(self.view.gm, "footprint", None) is None:
            out = self._arastar(s0, is_goal, hl, eps0, deadline, t0,
                                max_expansions if max_expansions is not None else 1 << 62)
            res = self._result(s0, out, t0)
        else:
            blocked_edges = set()
            max_exp = max_expansions if max_expansions is not None else 1 << 62
            while time.perf_counter() < deadline:
                out = self._arastar(s0, is_goal, hl, eps0, deadline, t0, max_exp,
                                    blocked_edges=blocked_edges)
                res = self._result(s0, out, t0)
                if not res.complete:
                    # A1 selects best_s only on a strict heuristic improvement.
                    # Timeout exhausts the original time/expansion budget, but a
                    # partial may still be delivered if every edge is verified.
                    # Table/cache hits remain usable after the search deadline;
                    # unknown generic-checker edges must fit inside that deadline.
                    verified = False
                    if out["timed_out"] and out["best_s"] != s0:
                        states, ks = self._extract(s0, out["best_s"], out["par"])
                        verified = self._exact_bad_edges(states, ks, deadline) == set()
                    if not verified:
                        res.poses, res.primitive_ids = [], []
                        if out["timed_out"]:
                            res.reason = "timeout_no_exact_path"
                    break
                states, ks = self._extract(s0, out["goal_s"], out["par"])
                bad = self._exact_bad_edges(states, ks, deadline)
                if bad is None:
                    res.poses, res.primitive_ids = [], []
                    res.complete = False
                    res.reason = "timeout_exact_replan"
                    break
                if not bad:
                    break
                blocked_edges.update(bad)
                if time.perf_counter() >= deadline:
                    res.poses, res.primitive_ids = [], []
                    res.complete = False
                    res.reason = "timeout_exact_replan"
                    break
            else:
                res = PlanResult([], [], INF, INF, False, time.perf_counter() - t0,
                                 reason="timeout_exact_replan")
        res.start_snap, res.goal_snap = start_snap, goal_snap
        if start_conn is not None and res.poses:
            route, _ = self._connector_route(start, start_conn, res.poses[0][2])
            res.poses = route + list(res.poses)
            res.start_connector_n = len(start_conn) - 1
        if goal_conn is not None and res.complete and res.poses:
            ex, ey, eth = res.poses[-1]
            pts = self.connector_points((ex, ey), goal_conn)[1:]
            goal_snap["connector_m"] = float(np.linalg.norm(goal_conn - np.array([ex, ey])))
            gth = float(goal[2]) if len(goal) > 2 and goal[2] is not None else eth
            route = []
            for k, (x, y) in enumerate(pts):
                a = (k + 1) / max(len(pts), 1)
                route.append((float(x), float(y), _wrap(eth + a * _wrap(gth-eth))))
            res.poses = list(res.poses) + route
            res.goal_connector_n = len(pts)
        # No whole-path re-validation here: every output segment is covered by an exact check
        # before it can be returned (start jump or repaired start connector above; each primitive
        # edge by _exact_bad_edges in the final round; goal connector inside goal_spec(repair=True)).
        return res

    def _snap_jump_safe(self, start, s0):
        """Exact check of the direct true-start -> snapped-node jump (non-repair case)."""
        node_xy = self.node_xy[s0 // self.nh]
        ps0 = self.state_pose(s0)
        route, node_pose = self._connector_route(
            start, np.vstack([np.asarray(start[:2], float), node_xy]), ps0[2])
        poses = route + [node_pose]
        return self._footprint_swept_safe(poses, heading_deltas=self._route_heading_deltas(poses))

    def _arastar(self, s0, is_goal, hl, eps0, deadline, t0, max_exp=1 << 62,
                 blocked_edges=None):
        NH, N = self.nh, self.N
        tab, deltas = self.tab, self.deltas
        perf = time.perf_counter
        heappush, heappop = heapq.heappush, heapq.heappop
        PK = self.Pmax
        g = [INF] * N
        par = {}
        g[s0] = 0.0
        h0 = hl[s0 // NH]
        best_h, best_s = h0, s0
        eps = float(eps0)
        heap = [(eps * h0, s0)]
        closed = bytearray(N)
        incons = []
        g_goal, goal_s = INF, -1
        expansions = 0
        iters = []
        eps_proven = INF
        first_t = None
        timed_out = False
        schedule = [e for e in DEFAULT_EPS_SCHEDULE if e < eps]
        while True:
            # ---------------- ImprovePath(eps)
            while heap:
                f, s = heap[0]
                if f >= g_goal:
                    break
                heappop(heap)
                if closed[s]:
                    continue
                closed[s] = 1
                expansions += 1
                if (not (expansions & 127) and perf() > deadline) or expansions > max_exp:
                    timed_out = True
                    break
                gs = g[s]
                dl = deltas[s % NH]
                sPK = s * PK
                for kk, c in enumerate(tab[s].tolist()):
                    if c == INF:
                        continue
                    if blocked_edges is not None and (s, kk) in blocked_edges:
                        continue
                    s2 = s + dl[kk]
                    g2 = gs + c
                    if g2 < g[s2]:
                        g[s2] = g2
                        par[s2] = sPK + kk
                        if is_goal[s2]:
                            if g2 < g_goal:
                                g_goal, goal_s = g2, s2
                                if first_t is None:
                                    first_t = perf() - t0
                            continue
                        if closed[s2]:
                            incons.append(s2)
                            continue
                        h2 = hl[s2 // NH]
                        if h2 == INF:
                            continue
                        heappush(heap, (g2 + eps * h2, s2))
                        if h2 < best_h:
                            best_h, best_s = h2, s2
            if timed_out:
                break
            if goal_s < 0:
                break  # OPEN exhausted (or all remaining keys infinite): no path
            # ---------------- iteration complete: proven bound
            lb = min((g[s] + hl[s // NH] for _, s in heap if not closed[s]), default=INF)
            lb = min(lb, min((g[s] + hl[s // NH] for s in incons), default=INF))
            eps_p = 1.0 if lb == INF else max(1.0, min(eps, g_goal / lb)) if lb > 0 else eps
            eps_proven = min(eps_proven, eps_p)
            iters.append({"eps": eps, "eps_prime": eps_p, "cost": g_goal, "t": perf() - t0,
                          "expansions": expansions})
            if eps_p <= 1.0 + 1e-9:
                break
            schedule = [e for e in schedule if e < eps_p - 1e-9]
            # stop if the deadline would pass while rebuilding OPEN (~0.4 us per entry)
            if not schedule or perf() + 4e-7 * (len(heap) + len(incons)) > deadline:
                break
            eps = schedule.pop(0)
            open_states = {s for _, s in heap if not closed[s]}
            open_states.update(incons)
            heap = [(g[s] + eps * hl[s // NH], s) for s in open_states]
            heapq.heapify(heap)
            closed = bytearray(N)
            incons = []
        return dict(g=g, par=par, goal_s=goal_s, g_goal=g_goal, best_s=best_s, expansions=expansions,
                    iters=iters, eps_proven=eps_proven, first_t=first_t, timed_out=timed_out)

    def _extract(self, s0, s_end, par):
        states, ks = [s_end], []
        s = s_end
        while s != s0:
            ps, kk = divmod(par[s], self.Pmax)
            states.append(ps)
            ks.append(kk)
            s = ps
        states.reverse()
        ks.reverse()
        return states, ks

    def path_from_states(self, states, ks):
        poses = [self.state_pose(states[0])]
        ids, cost = [], 0.0
        for s, kk in zip(states[:-1], ks):
            c, hd = divmod(s, self.nh)
            p = self.prims[hd][kk]
            x0, y0 = self.node_xy[c]
            th0 = hd * self.bin_w
            for dx, dy, dth in p.samples:
                poses.append((float(x0 + dx), float(y0 + dy), _wrap(th0 + dth)))
            ids.append(p.id)
            cost += float(self.tab[s, kk])
        return poses, ids, cost

    def _result(self, s0, out, t0):
        complete = out["goal_s"] >= 0
        s_end = out["goal_s"] if complete else out["best_s"]
        reason = ""
        if not complete:
            reason = "timeout_partial" if out["timed_out"] else "goal_unreachable"
        if s_end == s0 or s_end < 0:
            states, ks = [s0], []
        else:
            states, ks = self._extract(s0, s_end, out["par"])
        poses, ids, cost = self.path_from_states(states, ks)
        eps_a = out["eps_proven"] if complete else INF
        return PlanResult(poses, ids, cost, eps_a, complete,
                          time.perf_counter() - t0, reason=reason, expansions=out["expansions"],
                          iterations=out["iters"], first_solution_time_s=out["first_t"],
                          start_state_pose=self.state_pose(s0),
                          end_state_pose=self.state_pose(states[-1]))

    # ---- reference (brute force) ----------------------------------------------------
    def full_graph(self):
        if self._full_graph is None:
            nh = self.nh
            rows, cols, data = [], [], []
            S = np.arange(self.N)
            hd = S % nh
            for h in range(nh):
                sel = S[hd == h]
                for kk, dlt in enumerate(self.deltas[h]):
                    c = self.tab[sel, kk]
                    m = np.isfinite(c)
                    rows.append(sel[m])
                    cols.append(sel[m] + dlt)
                    data.append(c[m])
            r, c, d = np.concatenate(rows), np.concatenate(cols), np.concatenate(data)
            o = np.lexsort((d, c, r))          # keep the cheapest of parallel edges
            r, c, d = r[o], c[o], d[o]
            first = np.r_[True, (r[1:] != r[:-1]) | (c[1:] != c[:-1])]
            self._full_graph = sp.csr_matrix((d[first], (r[first], c[first])), shape=(self.N, self.N))
        return self._full_graph

    def reference_cost(self, start, goal):
        # The reference graph must start at the same A1-repaired node as plan().
        # Comparing against the original snap changes the problem and can report
        # a valid repaired path as "cheaper than brute force".
        self._sync_footprint_caches()
        if not self._footprint_pose_safe(start):
            return INF
        s0 = self.snap(start)
        footprint_on = getattr(self.view.gm, "footprint", None) is not None
        if s0 >= 0 and (not self.node_ok[s0 // self.nh]
                        or (footprint_on and (not self._footprint_pose_safe(self.state_pose(s0))
                                              or not self._snap_jump_safe(start, s0)))):
            s_rep, _ = self.repair_start(start)
            if s_rep >= 0:
                s0 = s_rep
            elif footprint_on and self.node_ok[s0 // self.nh]:
                return INF
        if s0 < 0 or not self.node_ok[s0 // self.nh]:
            return INF
        is_goal, _, _, _ = self.goal_spec(goal)
        if is_goal[s0]:
            return 0.0
        dist = _cs_dijkstra(self.full_graph(), directed=True, indices=s0)
        gs = np.frombuffer(bytes(is_goal), np.uint8).astype(bool)
        return float(dist[gs].min()) if gs.any() else INF

    def reference_cost_to_go(self, goal):
        is_goal, _, _, _ = self.goal_spec(goal)
        seeds = np.nonzero(np.frombuffer(bytes(is_goal), np.uint8))[0]
        if len(seeds) == 0:
            return np.full(self.N, INF)
        return _cs_dijkstra(self.full_graph().T.tocsr(), directed=True, indices=seeds, min_only=True)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------- module API
_CACHE = {}


def get_planner(gridmap, primitives, params: PlannerParams | None = None) -> LatticePlanner:
    """Cached LatticePlanner; rebuilt when the map's lethal/cost arrays change."""
    params = params or PlannerParams()
    pkey = (os.path.abspath(primitives), os.path.getmtime(primitives)) \
        if isinstance(primitives, (str, os.PathLike)) else id(primitives)
    key = (id(gridmap), pkey, params)
    ent = _CACHE.get(key)
    fp = _MapView(gridmap).fingerprint() if ent is not None else None
    if ent is not None and ent[0] is gridmap and ent[1] is primitives and ent[2].fingerprint == fp:
        ent[2]._sync_footprint_caches()  # attachment changes are setup, before plan() starts its timer
        return ent[2]
    pl = LatticePlanner(gridmap, primitives, params)
    if len(_CACHE) > 4:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (gridmap, primitives, pl)
    return pl


def plan(gridmap, primitives, start, goal, budget_s=0.2, eps0=3.0, params=None,
         max_expansions=None) -> PlanResult:
    """ARA* on the state lattice.  goal = (x, y, theta) or (x, y, None) for any heading.
    max_expansions: optional node limit (Baritone-style), mainly for deterministic tests."""
    pl = get_planner(gridmap, primitives, params)
    t0 = time.perf_counter()          # per-map precompute is one-time setup, not planning time (D1)
    return pl.plan(start, goal, budget_s=budget_s, eps0=eps0, t_start=t0, max_expansions=max_expansions)


def plan_segment(gridmap, primitives, start, goal, horizon_m=2.5, budget_s=0.2, eps0=3.0,
                 params=None) -> PlanResult:
    """Segmented planning (Baritone plan-ahead): if the goal is farther than horizon_m along the
    2D Dijkstra path, plan to an intermediate goal (any heading) on that path at ~horizon_m,
    nudged toward low map cost.  result.subgoal / result.final_segment say which was used."""
    pl = get_planner(gridmap, primitives, params)
    t0 = time.perf_counter()          # per-map precompute is one-time setup, not planning time (D1)
    gxy = np.array(goal[:2], float)
    d2, _ = pl.d2_from(gxy)
    view = pl.view
    f = np.rint(view.fidx(np.array(start[:2], float))).astype(int)
    i, j = int(f[0]), int(f[1])
    H, W = view.H, view.W
    if not (0 <= i < H and 0 <= j < W) or not np.isfinite(d2[i * W + j]) or d2[i * W + j] <= horizon_m:
        r = pl.plan(start, goal, budget_s=budget_s, eps0=eps0, t_start=t0)
        r.final_segment = True
        return r
    # descend the D2 field for horizon_m of path length
    walked = 0.0
    while walked < horizon_m:
        best = None
        for di, dj, wc, _ in MOVES16:
            a, b = i + di, j + dj
            if 0 <= a < H and 0 <= b < W:
                w = view.res * wc
                v = d2[a * W + b] + w
                if best is None or v < best[0]:
                    best = (v, a, b, w)
        if best is None or not np.isfinite(best[0]) or d2[best[1] * W + best[2]] >= d2[i * W + j]:
            break
        _, i, j, w = best
        walked += w
    # nudge to the lowest-cost cell within 0.25 m that does not move away from the goal
    r = int(round(0.25 / view.res))
    a0, a1, b0, b1 = max(0, i - r), min(H, i + r + 1), max(0, j - r), min(W, j + r + 1)
    sub_d2 = d2.reshape(H, W)[a0:a1, b0:b1]
    sub_c = np.where(view.lethal[a0:a1, b0:b1], INF, view.cost[a0:a1, b0:b1])
    sub_c = np.where(sub_d2 <= d2[i * W + j] + 0.05, sub_c, INF)
    ai, bj = np.unravel_index(int(np.argmin(sub_c)), sub_c.shape)
    sxy = view.world(np.array([a0 + ai, b0 + bj]))
    sub = (float(sxy[0]), float(sxy[1]), None)
    res = pl.plan(start, sub, budget_s=budget_s, eps0=eps0, t_start=t0, _d2_goal=gxy)
    res.subgoal = sub
    res.final_segment = False
    return res


def dijkstra_reference(gridmap, primitives, start, goal, params=None) -> float:
    """Brute-force uniform-cost search (eps=1, no budget) over the identical lattice graph
    (scipy C Dijkstra from the start state, min over goal states).  Independent of the ARA*
    code path except for the shared edge table."""
    return get_planner(gridmap, primitives, params).reference_cost(start, goal)


def cost_to_go_reference(gridmap, primitives, goal, params=None) -> np.ndarray:
    """Exact cost-to-go of every lattice state (reverse Dijkstra from the goal states)."""
    return get_planner(gridmap, primitives, params).reference_cost_to_go(goal)


# ----------------------------------------------------------------------------- replanning
def _path_poses(path):
    if isinstance(path, PlanResult):
        return path.poses
    return list(path)


def replan_reason(pose_est, path, t_since_plan, gridmap, off_path_m=0.15, period_s=2.0,
                  lookahead_m=None):
    """None if no replan is needed, else one of "no_path", "off_path", "blocked", "periodic"."""
    poses = _path_poses(path)
    if not poses:
        return "no_path"
    xy = np.array([p[:2] for p in poses], float)
    q = np.array(pose_est[:2], float)
    if len(xy) == 1:
        d_all = np.linalg.norm(xy - q, axis=1)
        k0, dmin = 0, float(d_all[0])
    else:
        a, b = xy[:-1], xy[1:]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-12)
        t = np.clip(((q - a) * ab).sum(1) / L2, 0.0, 1.0)
        d = np.linalg.norm(a + t[:, None] * ab - q, axis=1)
        k0 = int(np.argmin(d))
        dmin = float(d[k0])
    if dmin > off_path_m:
        return "off_path"
    ahead = xy[k0:]
    if lookahead_m is not None:
        seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ahead, axis=0), axis=1))]
        ahead = ahead[seg <= lookahead_m]
    for x, y in ahead:
        if gridmap.is_lethal(float(x), float(y)):
            return "blocked"
    if t_since_plan >= period_s:
        return "periodic"
    return None


def should_replan(pose_est, path, t_since_plan, gridmap, off_path_m=0.15, period_s=2.0) -> bool:
    """True if > off_path_m from the path, any remaining path pose is lethal, or every period_s."""
    return replan_reason(pose_est, path, t_since_plan, gridmap, off_path_m, period_s) is not None
