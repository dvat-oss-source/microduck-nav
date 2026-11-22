# P2 candidate c3 (v6, 2026-11-13): live A5 MPPI follower fe8c2f50 + c1 path smoothing (candidate1, 0c7bf44c)
# + c2 final-approach tail (candidate2, f2c9bb66). Order in set_path: smooth the planned poses, then tail.
# Built by reports/dev/v6/P2/c3/build_combo.py. No constants beyond those of c1 and c2.
# Stage 2 MPPI rescue candidate; isolated report artifact, not promoted.
"""Deterministic CPU MPPI path follower, retaining the accepted A4 goal state machine.

The measured bilinear command map, embedded latency/lag fits, slow-start model,
gait-period pose filtering and online constant-disturbance estimates predict
12 control ticks. MPPI optimizes continuous yaw sequences with path, heading,
clearance, reversal and command-change costs; forward speed follows the A4
measured gait schedule. Fixed antithetic perturbations remove tick-to-tick
sampling imbalance. Terminal heading limits retain A4 lateral-bias recovery.

RPPFollower preserves the public 10 Hz interface and A4 stop/align/repair/
settled-turn logic. Dynamics are embedded so a source-only validation snapshot
uses the same fit. DYNAMICS_PATH identifies provenance; it is not a runtime
dependency. See DIAGNOSIS.md beside this candidate for development evidence.
"""
from __future__ import annotations
import json, math, os
import numpy as np

DEFAULT_ENVELOPE = {"vx": [-0.15, 0.3], "vy": [-0.15, 0.15], "wz": [-1.0, 1.0]}
def _resolve_data_dir():
    """Resolve measured resources independently of candidate/deployment depth."""
    configured = os.environ.get("NAV_DATA_DIR")
    if configured:
        directory = os.path.realpath(os.path.expanduser(configured))
        if not os.path.isdir(directory):
            raise RuntimeError(f"NAV_DATA_DIR is not a directory: {directory}")
        return directory
    directory = os.path.dirname(os.path.realpath(__file__))
    while True:
        if (os.path.isfile(os.path.join(directory, "CONTRACTS.md"))
                and os.path.isdir(os.path.join(directory, "data"))
                and os.path.isdir(os.path.join(directory, "nav"))):
            return os.path.realpath(os.path.join(directory, "data"))
        parent = os.path.dirname(directory)
        if parent == directory:
            raise RuntimeError("Cannot locate navigation project data from follower module; "
                               "set NAV_DATA_DIR to the measured data directory")
        directory = parent


DATA_DIR = _resolve_data_dir()
AXES = ("vx", "vy", "wz")


def load_envelope(key="walk"):
    p = os.path.join(DATA_DIR, f"envelope_{key}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return dict(DEFAULT_ENVELOPE)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# v5 A5 candidate 2 MPPI constants (set a priori from cost-scale reasoning, see DIAGNOSIS.md;
# not tuned on benchmark episodes or E-gate seeds)
MPPI_SEED = 20261111            # fixed antithetic exploration reused across ticks
MPPI_SAMPLES = 64               # random perturbations of the warm-started nominal sequence
MPPI_HORIZON = 12               # steps of dt (1.2 s: walk forward delay 0.375 s + response)
MPPI_LAMBDA = 0.05              # softmax temperature (cost units)
MPPI_KNOTS = 4                 # perturbations are linear between this many knots (smooth sequences)
MPPI_SIGMA_VX = 0.0           # vx is not sampled (A4 speed schedule via the measured map)
MPPI_SIGMA_WZ = 0.3            # exploration std, wz command (rad/s)
MPPI_WZ_LEVELS = 9             # deterministic constant-wz samples across the gait-on wz range
MPPI_SWAY_FILTER = True        # average the pose over whole gait periods, then advance with the model
MPPI_SWAY_MIN_TICKS = 4        # averaging window: whole gait periods spanning >= this many ticks
MPPI_PROX = 0.20              # clearance penalty starts below this distance (m)
MPPI_END_SMOOTH_FADE = False    # flip/delta-wz penalties scale with A4's end_fade factor (0 at the end)
MPPI_PSI_END_SCALE = 1.0       # heading-reference cap at the path end = this x A4 psi_end
MPPI_SIGN_MODE = False          # softmax mean taken inside the lower-free-energy wz-sign mode
MPPI_DOB_WINDOW_S = 2.0       # yaw-rate disturbance observer window (s); 0 disables
MPPI_DOB_MAX = 0.3            # |yaw-rate disturbance| clip (rad/s)
MPPI_DOB_LATERAL = True        # also estimate body-frame sideways drift and use it in the rollouts
MPPI_DOB_VY_MAX = 0.03          # |lateral drift| clip (m/s)
MPPI_W = dict(e=1000.0, psi=40.0, psi_end=10.0, progress=2.0, speed=20.0, clearance=20.0, flip=5.0, dvx=1.0, dwz=8.0, wcap=10.0)
# Fitted dynamics are EMBEDDED (candidate 1 read dynamics_fit.json next to its own file; the ladder runs a
# snapshot copy in ladder/snapshot/ with no json beside it, so candidate 1's E gate silently used the envelope
# FOL fallback).  Source: reports/dev/v5/A5/candidate2/dynamics_fit.json (copy of reports/dev/v5/A5/dynamics_fit.json) (sha256 4415a3820a68d2a5ff549f038114260c5bf0460c6d0c93c049c7a149ffe08b68), "model" block per policy plus the steering
# calibration gait period.  The SHA256 of this file therefore covers the dynamics actually used.
DYNAMICS_SOURCE = 'reports/dev/v5/A5/candidate2/dynamics_fit.json (copy of reports/dev/v5/A5/dynamics_fit.json)'
DYNAMICS_SOURCE_SHA256 = '4415a3820a68d2a5ff549f038114260c5bf0460c6d0c93c049c7a149ffe08b68'
MPPI_DYNAMICS = {'atlantis_velstand_rough_wip': {'delay_vx': 0.03359742053441708,
                                 'delay_wz': 0.008405028747167058,
                                 'dt': 0.1,
                                 'gait_on_vx': 0.2,
                                 'gait_period_s': None,
                                 'heading_deadband_rad': None,
                                 'slow_start_gain_ratio': 1.145892154242775,
                                 'slow_start_s': 0.16646090684498724,
                                 'slow_start_vx_abs_max': 0.2,
                                 'tau_vx': 0.13286348631057016,
                                 'tau_wz': 0.08072663347097601,
                                 'w_turn_max_measured': 1.935823937886471},
 'atlantis_walk': {'delay_vx': 0.05801522569630623,
                   'delay_wz': 0.0,
                   'dt': 0.1,
                   'gait_on_vx': 0.2,
                   'gait_period_s': 0.5,
                   'heading_deadband_rad': 0.057091344945478725,
                   'slow_start_gain_ratio': 1.0140928387070058,
                   'slow_start_s': 0.20747464871566743,
                   'slow_start_vx_abs_max': 0.2,
                   'tau_vx': 0.1494594230193612,
                   'tau_wz': 0.2571809893713911,
                   'w_turn_max_measured': 1.6276710913581267},
 'velocity_hard': {'delay_vx': 0.15379943860223833,
                   'delay_wz': 0.0,
                   'dt': 0.1,
                   'gait_on_vx': 0.3,
                   'gait_period_s': 0.25,
                   'heading_deadband_rad': 0.055756394967515596,
                   'slow_start_gain_ratio': 0.7300924364544418,
                   'slow_start_s': 0.23098643941009028,
                   'slow_start_vx_abs_max': 0.2,
                   'tau_vx': 0.07718700080785196,
                   'tau_wz': 0.1565449500521511,
                   'w_turn_max_measured': 1.3801730278884694},
 'velstand': {'delay_vx': 2.5849394142282115e-25,
              'delay_wz': 3.0681323082346533e-21,
              'dt': 0.1,
              'gait_on_vx': 0.2,
              'gait_period_s': None,
              'heading_deadband_rad': None,
              'slow_start_gain_ratio': 1.0786351378203007,
              'slow_start_s': 0.18882098183900958,
              'slow_start_vx_abs_max': 0.2,
              'tau_vx': 0.18882098183900958,
              'tau_wz': 0.1706365988453773,
              'w_turn_max_measured': 1.9683453420734147},
 'velstand_flat': {'delay_vx': 0.018296552542483153,
                   'delay_wz': 1.7011717825572303e-21,
                   'dt': 0.1,
                   'gait_on_vx': 0.2,
                   'gait_period_s': None,
                   'heading_deadband_rad': None,
                   'slow_start_gain_ratio': 1.1433536285707686,
                   'slow_start_s': 0.33142810208854645,
                   'slow_start_vx_abs_max': 0.2,
                   'tau_vx': 0.3131315495460633,
                   'tau_wz': 0.294217654175528,
                   'w_turn_max_measured': 2.414442780978237},
 'velstand_rough': {'delay_vx': 0.006840873416945997,
                    'delay_wz': 1.7698760925532293e-14,
                    'dt': 0.1,
                    'gait_on_vx': 0.2,
                    'gait_period_s': None,
                    'heading_deadband_rad': None,
                    'slow_start_gain_ratio': 0.9346643641467949,
                    'slow_start_s': 0.41370317350274594,
                    'slow_start_vx_abs_max': 0.2,
                    'tau_vx': 0.40686230008579993,
                    'tau_wz': 0.135463406801415,
                    'w_turn_max_measured': 1.770294463033681},
 'walk': {'delay_vx': 0.3748876380607845,
          'delay_wz': 0.021753171669441573,
          'dt': 0.1,
          'gait_on_vx': 0.3,
          'gait_period_s': 0.5,
          'heading_deadband_rad': 0.053136057103545055,
          'slow_start_gain_ratio': 0.9845919792801419,
          'slow_start_s': 0.4631401944311118,
          'slow_start_vx_abs_max': 0.2,
          'tau_vx': 0.0882525563703273,
          'tau_wz': 0.11181782146584208,
          'w_turn_max_measured': 1.0842741889169265}}
DYNAMICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamics_fit.json")            # kept for interface parity with candidate 1; not read


def load_dynamics(policy_key=None):
    """Embedded fitted MPPI dynamics for a policy key; None if the policy has no fit (envelope FOL fallback)."""
    if policy_key is None or policy_key not in MPPI_DYNAMICS:
        return None
    return dict(MPPI_DYNAMICS[policy_key])


class Envelope:
    def __init__(self, env=None):
        if env is None:
            env = DEFAULT_ENVELOPE
        elif isinstance(env, str):
            env = load_envelope(env) if not env.endswith(".json") else json.load(open(env))
        self.raw = env
        self.lo = np.array([env[a][0] for a in AXES], float)
        self.hi = np.array([env[a][1] for a in AXES], float)
        self.half = np.maximum((self.hi - self.lo) / 2, 1e-6)
        sc = env.get("safe_cmds")
        self.safe = None
        if sc is not None and len(sc) > 0:
            s = np.vstack([np.asarray(sc, float).reshape(-1, 3), np.zeros((1, 3))])
            self.safe = s
            sn = s / self.half
            if len(sn) > 1:
                d = np.linalg.norm(sn[:, None] - sn[None], axis=2)
                np.fill_diagonal(d, np.inf)
                self.radius = float(np.median(d.min(1)))
            else:
                self.radius = 0.0
        fol = env.get("fol") or {}
        d = lambda a, k, dflt: float(fol.get(a, {}).get(k, dflt))
        self.tau = np.array([d(a, "tau", 0.0) for a in AXES])
        self.delay = np.array([d(a, "delay", 0.0) for a in AXES])
        self.gain = np.array([max(d(a, "gain", 1.0), 0.3) for a in AXES])
        self.has_fol = bool(fol)

    def clamp(self, cmd):
        c = np.clip(np.asarray(cmd, float), self.lo, self.hi)
        if self.safe is None:
            return c
        sn = self.safe / self.half
        def dist(x):
            return np.linalg.norm(sn - x / self.half, axis=1).min()
        if dist(c) <= self.radius + 1e-9:
            return c
        for s in np.linspace(0.95, 0.0, 20):
            if dist(c * s) <= self.radius + 1e-9:
                return c * s
        return self.safe[np.argmin(np.linalg.norm(sn - c / self.half, axis=1))].copy()


# ---------------------------------------------------------------------------
class CommandModel:
    """Measured (vx_cmd, wz_cmd) -> steady (v, w) map and its inverse.

    Built from envelope["measurement"]["rows"] (vy_cmd == 0 slice).  A cell
    "moves" when its achieved forward speed is a useful fraction of the command
    and consistent across seeds.  Without rows a linear map from the FOL gains is
    synthesised (no deadzones), which is what the kinsim fixture assumes.
    """

    def __init__(self, env: Envelope, rot_target=0.6):
        raw = env.raw
        rows = ((raw.get("measurement") or {}).get("rows")) or []
        rows = [r for r in rows if abs(r["command"][1]) < 1e-9 and r.get("falls", 0) == 0]
        self.measured = len(rows) > 0
        if self.measured:
            vxs = sorted({round(r["command"][0], 6) for r in rows})
            wzs = sorted({round(r["command"][2], 6) for r in rows})
            V = np.full((len(vxs), len(wzs)), np.nan); W = V.copy()
            Vstd = np.zeros_like(V); Wstd = np.zeros_like(V); Wend = np.zeros_like(V)
            secs = float(raw["measurement"].get("seconds_per_command", 4.0))
            for r in rows:
                i = vxs.index(round(r["command"][0], 6)); j = wzs.index(round(r["command"][2], 6))
                V[i, j], W[i, j] = r["mean_velocity"][0], r["mean_velocity"][2]
                es = r.get("end_std", [0, 0, 0]); em = r.get("end_mean", [0, 0, 0])
                Vstd[i, j], Wstd[i, j], Wend[i, j] = es[0], es[2], em[2]
            if np.isnan(V).any():          # incomplete grid: fill holes with nearest wz in the row
                for i in range(len(vxs)):
                    ok = ~np.isnan(V[i])
                    if ok.any():
                        V[i] = np.interp(wzs, np.array(wzs)[ok], V[i][ok]); W[i] = np.interp(wzs, np.array(wzs)[ok], W[i][ok])
                    else:
                        V[i] = 0.0; W[i] = 0.0
            self.vxs, self.wzs, self.V, self.W = np.array(vxs), np.array(wzs), V, W
            # forward "moves": >= 25% of command, >= 2 cm/s, endpoint spread < 50% of distance
            dist = np.abs(V) * secs
            self.moves = (np.sign(self.vxs)[:, None] * V >= np.maximum(0.25 * np.abs(self.vxs)[:, None], 0.02)) \
                & (Vstd < 0.5 * np.maximum(dist, 1e-6))
            # rotation consistency for the pure-rotation column
            self.rot_ok = (np.abs(Wend) > 1e-6) & (Wstd <= 0.25 * np.abs(np.where(np.abs(Wend) > 0, Wend, W * secs))) \
                & (np.abs(W) >= np.maximum(0.3 * np.abs(self.wzs)[None, :], 0.2))
        else:
            g = env.gain
            self.vxs = np.unique(np.round(np.linspace(0.0, env.hi[0], 7), 6))
            if env.lo[0] < 0:
                self.vxs = np.unique(np.concatenate([[env.lo[0]], self.vxs]))
            self.wzs = np.unique(np.round(np.linspace(env.lo[2], env.hi[2], 9), 6))
            self.V = g[0] * self.vxs[:, None] + 0 * self.wzs[None, :]
            self.W = g[2] * self.wzs[None, :] + 0 * self.vxs[:, None]
            self.moves = np.abs(self.vxs)[:, None] * np.ones_like(self.W) > 1e-9
            self.rot_ok = np.abs(self.W) > 1e-9
        self._derive(env, rot_target)

    def _derive(self, env, rot_target):
        pos = [i for i, vx in enumerate(self.vxs) if vx > 1e-9]
        # gait-on level: lowest vx whose row and all higher rows move for every wz
        self.i_min = None
        for i in pos:
            if all(self.moves[k].all() for k in pos if k >= i):
                self.i_min = i; break
        if self.i_min is None:                       # nothing reliable: use the best-moving row
            self.i_min = pos[int(np.argmax([self.moves[k].sum() for k in pos]))] if pos else int(np.argmax(self.vxs))
        self.vx_min = float(self.vxs[self.i_min])
        self.vx_max = float(self.vxs[-1])
        self.vx_start = float(self.vxs[min(self.i_min + 1, len(self.vxs) - 1)]) if self.measured else self.vx_min
        # fine grid of the bilinear forward map over the gait-on region
        fvx = np.linspace(self.vx_min, self.vx_max, max(2, int(round((self.vx_max - self.vx_min) / 0.01)) + 1))
        fwz = np.linspace(self.wzs[0], self.wzs[-1], max(2, int(round((self.wzs[-1] - self.wzs[0]) / 0.02)) + 1))
        GX, GZ = np.meshgrid(fvx, fwz, indexing="ij")
        self.gx, self.gz = GX.ravel(), GZ.ravel()
        self.gv, self.gw = self.forward(self.gx, self.gz)
        self.w_turn_max = float(min(np.max(self.gw), -np.min(self.gw)))
        # turn in place: per sign, smallest |wz| that rotates consistently from standstill
        i0 = int(np.argmin(np.abs(self.vxs)))
        self.rot = {}
        for sgn in (1, -1):
            js = [j for j in range(len(self.wzs)) if self.wzs[j] * sgn > 1e-9]
            js.sort(key=lambda j: abs(self.wzs[j]))
            good = [j for j in js if self.rot_ok[i0, j] and self.W[i0, j] * sgn > 0]
            if not self.measured:          # linear: command giving the target rate
                wz = float(np.clip(sgn * rot_target / env.gain[2], env.lo[2], env.hi[2]))
                self.rot[sgn] = (wz, abs(wz) * env.gain[2])
            elif good:
                j = good[0]; self.rot[sgn] = (float(self.wzs[j]), abs(float(self.W[i0, j])))
            else:
                j = js[-1] if js else int(np.argmax(np.abs(self.wzs))); self.rot[sgn] = (float(self.wzs[j]), max(abs(float(self.W[i0, j])), 0.05))
        # stopping distance (forward) per unit speed
        sd = env.raw.get("stop_dist") or {}
        if sd.get("mean") is not None and sd.get("command_vx"):
            vref = float(self.forward(float(sd["command_vx"]), 0.0)[0])
            self.stop_per_v = float(sd["mean"]) / max(vref, 0.02)
            self.stop_spread = float(sd.get("max", sd["mean"]) - sd["mean"])
        elif env.has_fol:
            self.stop_per_v = float(env.delay[0] + env.tau[0])
            self.stop_spread = 0.0
        else:                                   # nothing measured: assume a 0.3 s lag
            self.stop_per_v = 0.3
            self.stop_spread = 0.0
        # backward creep for small overshoots: negative level that moves straight back
        j0 = int(np.argmin(np.abs(self.wzs)))
        negs = [i for i, vx in enumerate(self.vxs) if vx < -1e-9 and self.moves[i, j0]]
        self.vx_back = float(self.vxs[max(negs)]) if negs else None
        self.v_back = float(self.V[max(negs), j0]) if negs else 0.0

    def forward(self, vx, wz):
        vx = np.clip(np.asarray(vx, float), self.vxs[0], self.vxs[-1])
        wz = np.clip(np.asarray(wz, float), self.wzs[0], self.wzs[-1])
        i = np.clip(np.searchsorted(self.vxs, vx) - 1, 0, len(self.vxs) - 2)
        j = np.clip(np.searchsorted(self.wzs, wz) - 1, 0, len(self.wzs) - 2)
        a = (vx - self.vxs[i]) / (self.vxs[i + 1] - self.vxs[i])
        b = (wz - self.wzs[j]) / (self.wzs[j + 1] - self.wzs[j])
        def bil(M):
            return (1 - a) * (1 - b) * M[i, j] + a * (1 - b) * M[i + 1, j] + (1 - a) * b * M[i, j + 1] + a * b * M[i + 1, j + 1]
        return bil(self.V), bil(self.W)

    def inverse(self, v_d, w_d, vx_lo=None, sv=0.06, sw=0.03):
        """(vx_cmd, wz_cmd, v_pred, w_pred) best achieving (v_d, w_d) inside the gait-on region."""
        m = np.ones_like(self.gx, bool) if vx_lo is None else self.gx >= vx_lo - 1e-9
        cost = ((self.gv[m] - v_d) / sv) ** 2 + ((self.gw[m] - w_d) / sw) ** 2 + 1e-4 * self.gz[m] ** 2
        k = int(np.argmin(cost))
        return float(self.gx[m][k]), float(self.gz[m][k]), float(self.gv[m][k]), float(self.gw[m][k])

    def v_at(self, vx_cmd):
        return float(self.forward(vx_cmd, 0.0)[0])

    def summary(self):
        return {"measured": self.measured, "vx_min": self.vx_min, "vx_start": self.vx_start, "vx_max": self.vx_max,
                "v_min": self.v_at(self.vx_min), "v_max": self.v_at(self.vx_max), "w_turn_max": self.w_turn_max,
                "rotate": {str(k): v for k, v in self.rot.items()}, "stop_per_v": self.stop_per_v}


# ---------------------------------------------------------------------------
# ---- P2 candidate (a): planner-consistent path smoothing before tracking (from candidate1) ----
SMOOTH_ENABLED = True

ALPHA = 0.5          # Laplacian relaxation weight (standard, chosen a priori)

MAX_SWEEPS = 400     # convergence cap

TOL_MOVE = 1e-5      # stop when the max move (m) falls below this

def _clear(gm, xy):
    if _full_map(gm):
        return np.asarray(gm._lookup(gm.clearance, xy[:, 0], xy[:, 1], np.float32(0.0)), float)
    return _cell_clear(gm, xy)

def _lethal(gm, xy):
    if _full_map(gm):
        return np.asarray(gm.is_lethal(xy[:, 0], xy[:, 1]), bool).reshape(-1)
    return _cell_clear(gm, xy) < R_INFLATE_FALLBACK

def _tangent(P):
    d = np.gradient(P, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])

def _relax(gm, P0, floor):
    P = P0.copy()
    for _ in range(MAX_SWEEPS):
        cand = P.copy()
        cand[1:-1] = P[1:-1] + ALPHA * (0.5 * (P[:-2] + P[2:]) - P[1:-1])
        ok = (~_lethal(gm, cand[1:-1])) & (_clear(gm, cand[1:-1]) >= floor - 1e-12)
        new = P.copy()
        new[1:-1][ok] = cand[1:-1][ok]
        move = np.abs(new - P).max()
        P = new
        if move < TOL_MOVE:
            break
    return P

def _resample(P, spacing):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    n = max(2, int(round(L / spacing)) + 1)
    s = np.linspace(0, L, n)
    return np.column_stack([np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])])

def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def smooth_run(gm, run, rc):
    """run: (n, 3) moving run. Returns (poses (m, 3), changed flag); original run if any check fails."""
    if len(run) < 3:
        return run, False
    # (1) reverse steps: the planner drives some segments backward (step opposite to heading). Tangent headings
    #     cannot express that, so such runs keep their original poses.
    steps = run[1:, :2] - run[:-1, :2]
    fwd = np.cos(run[:-1, 2]) * steps[:, 0] + np.sin(run[:-1, 2]) * steps[:, 1]
    if np.any(fwd < -1e-12):
        return run, False
    # (2) sharpness limit: smoothed headings may not turn faster per step than the sharpest raw step of this run.
    raw_dth = np.abs(np.array([_wrap(b - a) for a, b in zip(run[:-1, 2], run[1:, 2])]))
    sharp_limit = float(raw_dth.max()) if len(raw_dth) else 0.0
    P0 = run[:, :2]
    floor_val = min(float(np.min(_clear(gm, P0))), rc)
    P = _relax(gm, P0, floor_val)
    mean_sp = float(np.linalg.norm(np.diff(P0, axis=0), axis=1).sum() / (len(P0) - 1))
    if mean_sp <= 0:
        return run, False
    Q = _resample(P, mean_sp)
    th = _tangent(Q)
    ok = (~_lethal(gm, Q)) & (_clear(gm, Q) >= floor_val - 1e-12)
    if not ok.all():
        return run, False
    interior = np.array([_wrap(b - a) for a, b in zip(th[1:-2], th[2:-1])]) if len(th) > 4 else np.zeros(0)
    if interior.size and np.abs(interior).max() > sharp_limit + 1e-9:
        return run, False
    th[0], th[-1] = float(run[0, 2]), float(run[-1, 2])
    # full planner predicate (heading-aware cliff footprint) on every interior pose; endpoints are original poses
    for k in range(1, len(Q) - 1):
        if _map_lethal_pose(gm, float(Q[k, 0]), float(Q[k, 1]), float(th[k])):
            return run, False
    return np.column_stack([Q, th]), True

def smooth_poses(gm, poses):
    A = np.asarray([[p[0], p[1], p[2] if len(p) > 2 else 0.0] for p in poses], float)
    rc = _map_rc(gm)
    n = len(A)
    stats = dict(runs=0, smoothed=0, fallback=0)
    if n < 3:
        return A, stats
    moving = np.linalg.norm(np.diff(A[:, :2], axis=0), axis=1) > 1e-9
    out = []
    k = 0
    while k < n:
        if k < n - 1 and moving[k]:
            j = k
            while j < n - 1 and moving[j]:
                j += 1
            run = A[k:j + 1]                      # poses k..j; every step between them moves
            stats['runs'] += 1
            sm, changed = smooth_run(gm, run, rc)
            stats['smoothed' if changed else 'fallback'] += 1
            out.extend(np.asarray(sm, float).tolist())
            k = j + 1
        else:
            out.append(A[k].tolist())
            k += 1
    return np.asarray(out, float), stats


# ---- P2 candidate (c2): single curvature-bounded final-approach tail (cubic Hermite) ----
# Only change vs nav/follower.py (fe8c2f50): set_path() keeps the planned poses before the pose whose remaining arc
# to the goal is L_app, then replaces the rest with ONE cubic Hermite curve that starts on the planned tangent and
# arrives at the goal pose with the goal heading. Goal pose, goal heading and tolerances are unchanged.
# Constants (no benchmark data): r_turn = larger of the two finest measured pure-yaw turning radii of the envelope
# (the A4 repair model's yaw_rows, measured on the robot); kappa_max = 1/r_turn; L_app = pi * r_turn (a half turn at
# the tightest measured radius is always fitted). Tangent speeds are a 7x7 grid search (the "optimization"): minimal
# bending energy subject to curvature <= kappa_max, planner pose predicate, clearance floor.
# Fail closed: any violated constraint keeps the original planned poses.
TAIL_ENABLED = True
TAIL_SCALES = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.3)   # tangent speed as a fraction of the chord
TAIL_N = 200                                            # curve samples for the constraint checks
TAIL_STEP = 0.02                                        # emitted pose spacing (m), the planner lattice pitch


# ---- P2 map-agnostic predicates (Option A) ----
# Real gridmaps keep the heading-aware is_pose_lethal / _lookup path. WallClearance-like maps (nav/courses.py) use
# clearance >= r_inflate. R_INFLATE_FALLBACK = live value in reports/dev/v5/final_20261112/rung4_planner/gate/
# planner_gate_atlantis_walk.json (/map/r_inflate) = nav/mapping.py:504 robot_radius + 3*lateral_sigma. Not tuned.
R_INFLATE_FALLBACK = 0.1342555965538314


def _full_map(gm):
    return isinstance(getattr(gm, "params", None), dict) and "r_inflate" in gm.params \
        and hasattr(gm, "is_pose_lethal") and hasattr(gm, "_lookup")


def _map_rc(gm):
    return float(gm.params["r_inflate"]) if _full_map(gm) else R_INFLATE_FALLBACK


def _cell_clear(gm, xy):
    H, W = gm.clearance.shape
    i = np.clip((np.asarray(xy[:, 0], float) - gm.x0) / gm.res, 0, H - 1).astype(int)
    j = np.clip((np.asarray(xy[:, 1], float) - gm.y0) / gm.res, 0, W - 1).astype(int)
    return np.asarray(gm.clearance[i, j], float)


def _map_lethal_pose(gm, x, y, th):
    if _full_map(gm):
        return bool(gm.is_pose_lethal(x, y, th))
    return bool(_cell_clear(gm, np.array([[x, y]], float))[0] < R_INFLATE_FALLBACK)


def _tail_clear(gm, xy):
    if _full_map(gm):
        return np.asarray(gm._lookup(gm.clearance, xy[:, 0], xy[:, 1], np.float32(0.0)), float)
    return _cell_clear(gm, xy)


def _tail_coef(P0, T0, P1, T1, a, b):
    """Cubic Hermite p(u) = c0 + c1 u + c2 u^2 + c3 u^3 from P0 (tangent T0, speed a) to P1 (tangent T1, speed b)."""
    return P0, a * T0, 3.0 * (P1 - P0) - 2.0 * a * T0 - b * T1, 2.0 * (P0 - P1) + a * T0 + b * T1


def _tail_at(coef, u):
    c0, c1, c2, c3 = coef
    u = np.asarray(u, float)[:, None]
    pos = c0 + c1 * u + c2 * u ** 2 + c3 * u ** 3
    vel = c1 + 2 * c2 * u + 3 * c3 * u ** 2
    acc = 2 * c2 + 6 * c3 * u
    return pos, vel, acc


def _tail_eval(coef, T0, kappa_max):
    """Feasibility of the whole curve (curvature bound, no cusp). Traversed forward, heading = tangent, so
    the robot always moves along its own heading; a backward pass is an explicit reverse check in tail_poses."""
    u = np.linspace(0.0, 1.0, TAIL_N)
    pos, vel, acc = _tail_at(coef, u)
    sp = np.hypot(vel[:, 0], vel[:, 1])
    if sp.min() < 1e-6:                                         # cusp: heading undefined
        return None
    kappa = (vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]) / sp ** 3
    if np.abs(kappa).max() > kappa_max:
        return None
    energy = float(np.sum(kappa ** 2 * sp) / TAIL_N)            # ~ integral of kappa^2 ds
    return u, pos, vel, sp, kappa, energy


def tail_poses(gm, A, kappa_max, L_app, rc):
    """A: (n, 3) planned poses. Returns (poses (m, 3), stats). Original poses on any violation (fail closed)."""
    n = len(A)
    seg = np.linalg.norm(np.diff(A[:, :2], axis=0), axis=1)
    rem = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])     # remaining arc from pose k to the goal
    ks = np.nonzero(rem >= L_app)[0]
    if ks.size == 0:
        return A, dict(status="short_path")
    k = int(ks.max())
    if k >= n - 2:
        return A, dict(status="no_room")
    tail = A[k:]
    step = np.diff(tail[:, :2], axis=0)
    hd = tail[:-1, 2]
    if np.any(np.cos(hd) * step[:, 0] + np.sin(hd) * step[:, 1] < -1e-12):
        return A, dict(status="reverse_in_tail")                  # planner drives backward here: keep it
    mv = np.nonzero(np.linalg.norm(step, axis=1) > 1e-9)[0]
    if mv.size == 0:
        return A, dict(status="degenerate")
    T0 = tail[mv[0] + 1, :2] - tail[mv[0], :2]
    T0 = T0 / np.linalg.norm(T0)
    P0 = tail[0, :2].copy()
    P1 = tail[-1, :2].copy()
    gth = float(tail[-1, 2])
    T1 = np.array([math.cos(gth), math.sin(gth)])
    D = float(np.linalg.norm(P1 - P0))
    if D < 1e-6:
        return A, dict(status="degenerate")
    floor = min(float(np.min(_tail_clear(gm, tail[:, :2]))), rc)  # the original tail's own clearance, capped at r_inflate
    best = None
    for sa in TAIL_SCALES:
        for sb in TAIL_SCALES:
            coef = _tail_coef(P0, T0, P1, T1, sa * D, sb * D)
            ev = _tail_eval(coef, T0, kappa_max)
            if ev is None:
                continue
            u, pos, vel, sp, kappa, E = ev
            xy = pos[1:-1]
            th = np.arctan2(vel[1:-1, 1], vel[1:-1, 0])
            if any(_map_lethal_pose(gm, float(a), float(b), float(c))
                   for a, b, c in zip(xy[:, 0], xy[:, 1], th)):
                continue
            if np.any(_tail_clear(gm, xy) < floor - 1e-12):
                continue
            if best is None or E < best[0]:
                best = (E, sa, sb, coef, ev)
    if best is None:
        return A, dict(status="no_feasible", kappa_max=kappa_max, L_app=L_app, D=D, floor=floor)
    E, sa, sb, coef, (u, pos, vel, sp, kappa, _) = best
    s = np.concatenate([[0.0], np.cumsum(0.5 * (sp[1:] + sp[:-1]) * (u[1] - u[0]))])
    L_tail = float(s[-1])
    m_int = max(1, int(round(L_tail / TAIL_STEP)) - 1)
    uu = np.interp(np.linspace(0.0, L_tail, m_int + 2)[1:-1], s, u)
    p_i, v_i, _ = _tail_at(coef, uu)
    new = [[float(P0[0]), float(P0[1]), math.atan2(float(T0[1]), float(T0[0]))]]
    new += [[float(p_i[q, 0]), float(p_i[q, 1]), math.atan2(float(v_i[q, 1]), float(v_i[q, 0]))] for q in range(len(uu))]
    new.append([float(P1[0]), float(P1[1]), gth])
    out = np.vstack([A[:k], np.asarray(new, float)])
    return out, dict(status="tail", k=k, n_old=n, n_new=len(out), E=E, scale_start=sa, scale_end=sb,
                     kappa_max=kappa_max, max_kappa=float(np.abs(kappa).max()), L_app=L_app, L_tail=L_tail,
                     D=D, floor=floor)


def tail_apply(follower, poses):
    """Repair-model turning radius of THIS envelope -> kappa_max and L_app. Returns (A, stats)."""
    A = np.asarray([[p[0], p[1], p[2] if len(p) > 2 else 0.0] for p in poses], float)
    if getattr(follower, "_rotation_levels", None) is None:
        follower._rotation_levels = {1: [], -1: []}          # _repair_model reads it; empty = fine level only
    rm = follower._repair_model()
    if rm is None:
        return A, dict(status="no_repair_model")
    r_turn = max(float(rm["yaw_rows"][round(float(rm["fine"][sg]), 6)]["r"]) for sg in (1, -1))
    if not r_turn > 0:
        return A, dict(status="bad_radius")
    rc = _map_rc(follower.gridmap)
    out, st = tail_poses(follower.gridmap, A, 1.0 / r_turn, math.pi * r_turn, rc)
    st["r_turn"] = r_turn
    return out, st


class RPPFollower:
    def __init__(self, envelope=None, gridmap=None, dt=0.1, cruise=0.16,
                 omega=1.3, zeta=0.9, k_int=1.0, ie_max=0.1, predict=0.1, sway_window=0.1,
                 end_fade=(0.5, 0.2), psi_far=math.radians(30), psi_end=math.radians(3), final_slow=0.3,
                 rotate_thresh=math.radians(60), rotate_exit=math.radians(15),
                 tol_pos=0.05, tol_yaw=math.radians(10), pos_margin=0.002, yaw_margin=math.radians(2.0),
                 prox_dist=0.20, prox_min=0.5, turn_frac=0.6,
                 stuck_time=4.0, stuck_dist=0.03, settle_max=2.0, anticip0=0.15, max_fixes=25):
        self.env = Envelope(envelope)
        self.model = CommandModel(self.env)
        self.gridmap = gridmap
        self.__dict__.update(dict(dt=dt, omega=omega, zeta=zeta, k_int=k_int, ie_max=ie_max, predict=predict, sway_window=sway_window, end_fade=end_fade, psi_far=psi_far, psi_end=psi_end, final_slow=final_slow,
            rotate_thresh=rotate_thresh, rotate_exit=rotate_exit, tol_pos=tol_pos, tol_yaw=tol_yaw,
            pos_margin=pos_margin, yaw_margin=yaw_margin, prox_dist=prox_dist, prox_min=prox_min,
            turn_frac=turn_frac, stuck_time=stuck_time, stuck_dist=stuck_dist, settle_max=settle_max,
            anticip0=anticip0, max_fixes=max_fixes))
        m = self.model
        self.cruise = float(np.clip(cruise, m.v_at(m.vx_min), m.v_at(m.vx_max)))
        self._mppi_setup()
        self.path = None
        self.info = {}

    # ------------------------------------------------------------------ path
    def set_path(self, poses):
        poses = list(poses)
        if SMOOTH_ENABLED and self.gridmap is not None and len(poses) >= 3:
            poses, self.smooth_stats = smooth_poses(self.gridmap, poses)
            poses = [tuple(float(v) for v in p) for p in poses]
        if TAIL_ENABLED and self.gridmap is not None and len(poses) >= 3:
            A, self.tail_stats = tail_apply(self, poses)
            poses = [tuple(float(v) for v in p) for p in A]
        P = np.asarray([[p[0], p[1], p[2] if len(p) > 2 else 0.0] for p in poses], float)
        self.path = P
        self.goal = P[-1].copy()
        xy = P[:, :2]
        if len(P) >= 2:
            # drop duplicate points; tangent of the last real segment extends the path
            keep = np.concatenate([[True], np.hypot(*np.diff(xy, axis=0).T) > 1e-9])
            xy = xy[keep]
        if len(xy) >= 2:
            d = xy[-1] - xy[-2]; th_end = math.atan2(d[1], d[0])
        else:
            th_end = float(P[-1, 2])
        ext = xy[-1] + np.outer(np.arange(1, 26) * 0.02, [math.cos(th_end), math.sin(th_end)])
        X = np.vstack([xy, ext])
        seg = X[1:] - X[:-1]
        self.X, self.seglen = X, np.hypot(*seg.T)
        self.segdir = np.arctan2(seg[:, 1], seg[:, 0])
        self.s = np.concatenate([[0.0], np.cumsum(self.seglen)])
        self.s_end = float(self.s[len(xy) - 1])
        # curvature (smoothed over ~6 cm) at segment starts
        hd = np.unwrap(self.segdir)
        k = np.gradient(hd, self.s[:-1]) if len(hd) > 2 else np.zeros_like(hd)
        w = 3
        self.kappa = np.convolve(np.pad(k, (w, w), mode="edge"), np.ones(2 * w + 1) / (2 * w + 1), mode="valid") if len(k) else k
        self._set_heading_reference(P, all(len(p) > 2 for p in poses))
        self.th_end = th_end
        self.k = 0
        self.state = "track"
        self.t = 0.0
        self.hist = []            # (t, x, y, th_unwrapped)
        self.prog = []
        self.ie = 0.0
        self.eh = []
        self.terminal_errors = []
        self.last_cmd = np.zeros(3)
        self._mppi_reset()
        self.moving_since = None
        self.state_t = 0.0
        self.fixes = 0
        self.anticip = {1: self.anticip0, -1: self.anticip0}
        self.rot_stop = None
        self.min_ticks = {1: 1, -1: 1}
        self.stuck = False
        self._rotation_floor = {1: 0.0, -1: 0.0}
        self._rotation_attempt = None
        self._rotation_error = None
        self.escalation_events = []
        self._rotation_levels = {1: [], -1: []}
        if self.env.safe is not None and self.model.measured:
            for sgn in (1, -1):
                self._rotation_levels[sgn] = sorted({float(cmd[2]) for cmd in self.env.safe
                    if abs(cmd[0]) < 1e-9 and abs(cmd[1]) < 1e-9 and sgn * cmd[2] > 1e-9
                    and np.all(cmd >= self.env.lo) and np.all(cmd <= self.env.hi)}, key=abs)
        self.rotation_window = 2.0 * max(self.dt, float(self.env.delay[0] + self.env.tau[0]),
                                          float(self.env.delay[2] + self.env.tau[2]))

        # A4 final-repair state
        self.rm = self._repair_model()
        self.best = None              # (score, x, y, th, t) at settled checks
        self.repair_log = []
        self._axis_hist = {}          # axis -> (direction, err_before, sigma)
        self._coarse_level = None
        self._tr = None
        self._banned = set()          # translation commands that did not move from a settled stance
        self._last_action = None

    # ------------------------------------------------------------------ A4 model
    def _quantum_s(self):
        """Shortest measured primitive duration for this policy (Phase B), else the vx
        response horizon 2*(delay+tau)."""
        md5 = self.env.raw.get("policy_md5")
        if md5:
            try:
                for fn in sorted(os.listdir(DATA_DIR)):
                    if fn.startswith("primitives_") and fn.endswith(".json"):
                        with open(os.path.join(DATA_DIR, fn)) as f:
                            doc = json.load(f)
                        if doc.get("policy_md5") == md5 and doc.get("primitives"):
                            return float(min(q["duration"] for q in doc["primitives"])), fn, doc
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return float(max(self.dt, 2.0 * (self.env.delay[0] + self.env.tau[0]))), None, None


    @staticmethod
    def _prim_body_stats(doc, tq):
        """Single-command primitives of duration tq -> {cmd: (body-frame mean xy, per-seed RMS xy)}
        averaged over start headings (end_mean is world-aligned, start-relative)."""
        nh = int(doc.get("n_headings", 16))
        acc = {}
        for q in doc["primitives"]:
            seq = q.get("cmd_sequence") or []
            if len(seq) != 1 or abs(float(q["duration"]) - tq) > 1e-9:
                continue
            c = tuple(round(float(v), 6) for v in seq[0][:3])
            h = 2 * math.pi * int(q["start_heading_idx"]) / nh
            em = np.asarray(q["end_mean"], float)
            cov = np.asarray(q["end_cov"], float)
            b = np.array([math.cos(h) * em[0] + math.sin(h) * em[1], -math.sin(h) * em[0] + math.cos(h) * em[1]])
            rms = math.sqrt(float(b @ b) + float(cov[0, 0] + cov[1, 1]))
            acc.setdefault(c, []).append((b, rms, float(em[2])))
        return {c: (np.mean([a[0] for a in v], 0), float(np.mean([a[1] for a in v])), float(np.mean([a[2] for a in v])),
                    float(np.sqrt(np.mean([a[1] ** 2 - float(a[0] @ a[0]) for a in v]))))
                for c, v in acc.items()}


    def _repair_model(self):
        """Constants of the A4 final repair, all from envelope measurements.  None when the
        envelope has no measured rows/safe set (kinsim fixture): legacy repair only."""
        rows = ((self.env.raw.get("measurement") or {}).get("rows")) or []
        if not rows or self.env.safe is None or not self.model.measured:
            return None
        secs = float(self.env.raw["measurement"].get("seconds_per_command", 4.0))
        tq, tq_src, pdoc = self._quantum_s()
        pst = self._prim_body_stats(pdoc, tq) if pdoc is not None else {}
        pst2 = self._prim_body_stats(pdoc, 2 * tq) if pdoc is not None else {}
        tp = self.tol_pos - self.pos_margin
        ty = self.tol_yaw - self.yaw_margin
        safe = {tuple(np.round(c, 6)) for c in self.env.safe.tolist()}
        U = []
        for r in rows:
            c = tuple(round(float(v), 6) for v in r["command"])
            if abs(c[2]) > 1e-9 or (abs(c[0]) < 1e-9 and abs(c[1]) < 1e-9) or c not in safe:
                continue
            if r.get("falls", 0) or not r.get("progress", True):
                continue
            mv = np.asarray(r["mean_velocity"], float)
            em, es = np.asarray(r["end_mean"], float), np.asarray(r["end_std"], float)
            sp = float(np.hypot(mv[0], mv[1]))
            if sp < 0.02 or np.hypot(es[0], es[1]) > 0.25 * np.hypot(em[0], em[1]):
                continue                      # same consistency / speed floor as CommandModel
            if abs(mv[2]) * tq > ty / 2:
                continue                      # yaw drift of one quantum must stay inside half the yaw budget
            u = mv[:2] / sp
            step, sigma = sp * tq, float(np.hypot(es[0], es[1])) * tq / secs
            if c in pst:                      # measured primitive: first duration whose step >= 2 seed-sd
                step, sigma = max(float(pst[c][0] @ u), 0.0), pst[c][3]
                if step < 2 * sigma:
                    step, sigma = ((max(float(pst2[c][0] @ u), 0.0), pst2[c][3]) if c in pst2 else (0.0, sigma))
                    if step < 2 * sigma:
                        step = 0.0            # no controllable step measured: not usable
            U.append(dict(cmd=np.array(c, float), u=u, v=sp, step=step, sigma=sigma))
        # pure-yaw rows: turning circle radius r_L and endpoint spread per safe level
        yaw_rows = {}
        for r in rows:
            c = r["command"]
            if abs(c[0]) < 1e-9 and abs(c[1]) < 1e-9 and abs(c[2]) > 1e-9 and not r.get("falls", 0):
                mv, es = np.asarray(r["mean_velocity"], float), np.asarray(r["end_std"], float)
                if abs(mv[2]) >= 0.3 * abs(c[2]) and mv[2] * c[2] > 0:      # level actually rotates
                    yaw_rows[round(float(c[2]), 6)] = dict(r=float(np.hypot(mv[0], mv[1]) / abs(mv[2])),
                                                           d0=float(np.hypot(es[0], es[1])) * tq / secs)
        fine = {sgn: round(float(self.model.rot[sgn][0]), 6) for sgn in (1, -1)}
        coarse = {}
        for sgn in (1, -1):
            lv = [L for L in self._rotation_levels[sgn] if round(L, 6) in yaw_rows]
            coarse[sgn] = min(lv, key=lambda L: (yaw_rows[round(L, 6)]["r"], abs(L))) if lv else fine[sgn]
        if not all(round(fine[s_], 6) in yaw_rows and round(coarse[s_], 6) in yaw_rows for s_ in (1, -1)):
            return None
        d0 = max(v["d0"] for v in yaw_rows.values())
        turn_prims = [(c, v) for c, v in pst.items() if abs(c[0]) < 1e-9 and abs(c[1]) < 1e-9 and round(c[2], 6) in yaw_rows]
        if turn_prims:                        # per-turn shift not explained by the turning circle
            d0 = max(max(0.0, v[1] - 2 * yaw_rows[round(c[2], 6)]["r"] * abs(math.sin(v[2] / 2))) for c, v in turn_prims)
        U = [c for c in U if c["step"] > 0]
        s_min = min((c["step"] for c in U), default=float("inf"))   # smallest measured quantum
        rho = s_min / 2 if U else tp
        theta_c = {}
        for sgn in (1, -1):
            rL = yaw_rows[fine[sgn]]["r"]
            room = tp - rho - d0
            theta_c[sgn] = (2 * math.asin(min(1.0, room / (2 * rL))) if room > 0 and rL > 0 else 0.0)
        Dpi = max(d0 + 2 * yaw_rows[round(coarse[s_], 6)]["r"] for s_ in (1, -1))
        n_max = int(math.ceil((tp + Dpi) / s_min)) + 2 if U else 2
        return dict(tq=tq, tq_src=tq_src, U=U, yaw_rows=yaw_rows, fine=fine, coarse=coarse, d0=d0,
                    s_min=s_min, rho=rho, theta_c=theta_c, n_max=n_max, tp=tp, ty=ty)


    def _D(self, theta, level):
        y = self.rm["yaw_rows"][round(float(level), 6)]
        return self.rm["d0"] + 2.0 * y["r"] * abs(math.sin(theta / 2.0))


    def _reversal_ok(self, axis, direction, err_now):
        last = self._axis_hist.get(axis)
        if last is None or float(np.dot(direction, last[0])) > 0:
            return True
        return err_now <= last[1] - last[2]


    def _plan_repair(self, x, y, th, dgoal, eyaw):
        """-> ("rotate", level, sign, theta) | ("translate", U entry, err, k) | ("legacy",)."""
        rm = self.rm
        tp, ty = self.tol_pos - self.pos_margin, self.tol_yaw - self.yaw_margin
        c, s_ = math.cos(th), math.sin(th)
        e_b = np.array([c * (self.goal[0] - x) + s_ * (self.goal[1] - y),
                        -s_ * (self.goal[0] - x) + c * (self.goal[1] - y)])
        theta = abs(eyaw)
        sgn = 1 if eyaw > 0 else -1
        if theta > ty and theta > rm["theta_c"][sgn]:
            if not self._reversal_ok("yaw", np.array([sgn]), theta):
                return ("legacy",)                 # guard: no back-to-back yaw reversal without progress
            return ("rotate", rm["coarse"][sgn], sgn, theta)
        d_fine = self._D(theta, rm["fine"][sgn]) if theta > ty else 0.0
        if theta > ty and dgoal + d_fine <= tp:
            if not self._reversal_ok("yaw", np.array([sgn]), theta):
                return ("legacy",)                 # guard: no back-to-back yaw reversal without progress
            return ("rotate", rm["fine"][sgn], sgn, theta)
        best = None
        if dgoal > max(tp - d_fine, rm["rho"]) and rm["U"]:
            secs = float(self.env.raw["measurement"].get("seconds_per_command", 4.0))
            k_max = max(1, int(secs // rm["tq"]))           # rows were measured for `secs`
            for cand in rm["U"]:
                if tuple(cand["cmd"].tolist()) in self._banned or not self._reversal_ok("translate", cand["u"], dgoal):
                    continue
                for k in range(1, k_max + 1):
                    # predicted residual after k measured quanta (open loop)
                    res = float(np.linalg.norm(e_b - k * cand["step"] * cand["u"]))
                    if res > dgoal - cand["sigma"]:
                        continue
                    key = (res, k)
                    if best is None or key < best[0]:
                        best = (key, cand, k)
        if theta > ty and (best is None or best[0][0] > tp - d_fine):
            # translation cannot make room for the fine turn: turn now, translate afterwards
            if not self._reversal_ok("yaw", np.array([sgn]), theta):
                return ("legacy",)                 # guard: no back-to-back yaw reversal without progress
            return ("rotate", rm["fine"][sgn], sgn, theta)
        if best is not None:
            return ("translate", best[1], dgoal, best[2])
        return ("legacy",)


    def _translate(self, x, y, th):
        """A4: hold one exact measured-safe translation command (no turning) for the planned
        number of measured quanta (open loop: the quanta were measured from standstill, a
        shorter closed-loop burst does not start the gait), or until the remaining error
        along its direction is gone, or the error grows."""
        tr = self._tr
        cand = tr["cand"]
        c, s_ = math.cos(th), math.sin(th)
        e_b = np.array([c * (self.goal[0] - x) + s_ * (self.goal[1] - y),
                        -s_ * (self.goal[0] - x) + c * (self.goal[1] - y)])
        proj = float(np.dot(e_b, cand["u"]))
        el = self.t - self.state_t
        ticks = int(round(el / self.dt))
        if ticks >= tr["ticks"] or (el > 1e-9 and proj <= 0.0) \
                or (el >= self.rm["tq"] and float(np.hypot(*e_b)) > tr["e0"] + cand["sigma"]):
            self._tr = None
            self._set_state("settle")
            self.info = {"stage": "translate_stop", "proj": proj}
            return self._out(np.zeros(3), "tracking")
        self.info = {"stage": "translate", "proj": proj, "cmd": cand["cmd"].tolist()}
        return self._out(cand["cmd"].copy(), "tracking")


    # ------------------------------------------------------------------ helpers
    def _set_heading_reference(self, poses, has_heading):
        """Separate body heading from the quantized XY sampling of a primitive.

        Monotone Hermite interpolation makes heading and its arc-length
        derivative continuous within each moving run. Repeated XY poses split
        runs: an explicit stopped turn must not become translational curvature.
        Requested start/goal yaw is handled by alignment/final repair, rather
        than bent into the first/last moving segment.
        """
        self.heading_coeff = None
        if not has_heading:
            return
        lengths = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
        moving = np.flatnonzero(lengths > 1e-9)
        coeff = np.zeros((4, len(self.seglen)))
        end_heading = float(poses[-1, 2])
        offset = 0
        for run in np.split(moving, np.flatnonzero(np.diff(moving) != 1) + 1):
            if not len(run):
                continue
            first, last = int(run[0]), int(run[-1] + 1)
            h = np.unwrap(poses[first:last + 1, 2]).copy()
            ds = lengths[run]
            # Endpoint yaw may encode a requested stationary turn. Infer the
            # incoming/outgoing moving heading from interior path samples.
            if len(h) == 3 and first == 0 and last == len(poses) - 1:
                h[:] = h[1]
            elif len(h) > 2:
                if first == 0:
                    h[0] = h[1] - (h[2] - h[1]) * ds[0] / ds[1]
                if last == len(poses) - 1:
                    h[-1] = h[-2] + (h[-2] - h[-3]) * ds[-1] / ds[-2]
            elif first == 0 and last == len(poses) - 1:
                h[:] = self.segdir[offset]
            elif first == 0:
                h[0] = h[1]
            elif last == len(poses) - 1:
                h[-1] = h[-2]
            secant = np.diff(h) / ds
            slope = np.r_[secant[0], np.zeros(len(h) - 2), secant[-1]]
            for j in range(1, len(h) - 1):
                a, b = secant[j - 1:j + 1]
                if a * b > 0:
                    # Weighted harmonic slopes preserve monotonicity without
                    # overshooting between nonuniformly spaced heading samples.
                    w1, w2 = 2 * ds[j] + ds[j - 1], ds[j] + 2 * ds[j - 1]
                    slope[j] = (w1 + w2) / (w1 / a + w2 / b)
            n = len(run)
            coeff[:, offset:offset + n] = np.array([
                h[:-1], slope[:-1],
                (3 * secant - 2 * slope[:-1] - slope[1:]) / ds,
                (slope[:-1] + slope[1:] - 2 * secant) / ds ** 2])
            offset += n
            end_heading = float(h[-1])
        coeff[0, offset:] = end_heading
        self.heading_coeff = coeff

    def _heading_reference(self, s):
        j = self._seg_at(s)
        if self.heading_coeff is None:
            return float(self.segdir[j]), float(self.kappa[j])
        a, b, c, d = self.heading_coeff[:, j]
        u = float(np.clip(s - self.s[j], 0.0, self.seglen[j]))
        return float(a + u * (b + u * (c + u * d))), float(b + u * (2 * c + 3 * u * d))

    def _project(self, x, y):
        lo = max(0, self.k - 5); hi = min(len(self.seglen), self.k + 80)
        A = self.X[lo:hi]; d = self.X[lo + 1:hi + 1] - A
        L2 = np.maximum((d ** 2).sum(1), 1e-12)
        t = np.clip(((np.array([x, y]) - A) * d).sum(1) / L2, 0, 1)
        q = A + t[:, None] * d
        dist = np.hypot(q[:, 0] - x, q[:, 1] - y)
        j = int(np.argmin(dist)); kk = lo + j
        self.k = kk
        th = self.segdir[kk]
        e = -math.sin(th) * (x - q[j, 0]) + math.cos(th) * (y - q[j, 1])     # left of path > 0
        return float(self.s[kk] + t[j] * self.seglen[kk]), e, float(th)

    def _seg_at(self, s):
        return int(np.clip(np.searchsorted(self.s, s) - 1, 0, len(self.seglen) - 1))

    def _vel(self, window=0.4):
        """Smoothed (v_forward, w) from pose history over ~a gait cycle."""
        if len(self.hist) < 2:
            return 0.0, 0.0
        t1, x1, y1, h1 = self.hist[-1]
        old = [q for q in self.hist if q[0] <= t1 - window + 1e-9]
        t0, x0, y0, h0 = old[-1] if old else self.hist[0]
        T = max(t1 - t0, 1e-6)
        hm = 0.5 * (h0 + h1)
        return ((x1 - x0) * math.cos(hm) + (y1 - y0) * math.sin(hm)) / T, (h1 - h0) / T

    def _still(self, window=0.3, dpos=0.004, dyaw=0.02):
        t1, x1, y1, h1 = self.hist[-1]
        old = [q for q in self.hist if q[0] <= t1 - window + 1e-9]
        if not old or t1 - self.state_t < window - 1e-9:
            return False
        q = [r for r in self.hist if r[0] >= t1 - window - 1e-9]
        xs = np.array(q)
        return (np.hypot(xs[:, 1] - x1, xs[:, 2] - y1).max() < dpos and np.abs(xs[:, 3] - h1).max() < dyaw)

    def _set_state(self, s):
        self.state = s; self.state_t = self.t

    def _out(self, cmd, status):
        cmd = self.env.clamp(np.asarray(cmd, float))
        cmd = self._monitor_rotation_progress(cmd)
        self.last_cmd = cmd
        self._mppi_applied(cmd)
        if self.info.get("stage") != "track":
            self._mppi_nom = None
        self.t += self.dt
        return cmd, status

    def _rotate_cmd(self, err):
        self._rotation_error = float(err)
        sgn = 1 if err > 0 else -1
        wz = self.model.rot[sgn][0]
        if self.state == "rotate_fix" and self._coarse_level is not None and self._coarse_level * sgn > 0:
            wz = self._coarse_level            # A4: level chosen by _plan_repair (exact safe level)
        if abs(self._rotation_floor[sgn]) > abs(wz):
            wz = self._rotation_floor[sgn]
        return np.array([0.0, 0.0, wz])

    def _monitor_rotation_progress(self, cmd):
        """Escalate a stalled turn only to an exact measured safe pure-yaw level."""
        pure_yaw = abs(cmd[0]) < 1e-9 and abs(cmd[1]) < 1e-9 and abs(cmd[2]) > 1e-9
        if not pure_yaw or not self.hist:
            self._rotation_attempt = None
            self._rotation_error = None
            return cmd
        requested = self._rotation_error
        if (self.state == "rotate_fix" and requested is not None
                and abs(requested) <= self.tol_yaw - self.yaw_margin):
            self._rotation_attempt = None
            return cmd
        sgn = 1 if cmd[2] > 0 else -1
        now, _, _, heading = self.hist[-1]
        previous = self._rotation_attempt
        if previous is None or previous['sign'] != sgn or abs(previous['command'] - cmd[2]) > 1e-9:
            self._rotation_attempt = {'sign': sgn, 't': now, 'heading': heading, 'command': float(cmd[2]), 'requested_error': requested}
            return cmd
        elapsed = now - previous['t']
        if elapsed < self.rotation_window:
            return cmd
        achieved = sgn * (heading - previous['heading'])
        predicted_rate = abs(float(self.model.forward(0.0, cmd[2])[1]))
        expected = predicted_rate * elapsed
        if previous['requested_error'] is not None:
            # A small requested turn cannot deliver a full-rate response for an
            # entire long fitted horizon. Compare with the actual angle requested.
            expected = min(expected, abs(previous['requested_error']))
        # Observe two measured response horizons before requiring a quarter of
        # the measured yaw response. Latch the stronger level until a new path.
        if achieved < 0.25 * expected:
            stronger = [level for level in self._rotation_levels[sgn] if abs(level) > abs(cmd[2]) + 1e-9]
            if stronger:
                level = stronger[0]
                self._rotation_floor[sgn] = level
                self.escalation_events.append({'t': now, 'state': self.state, 'from_wz': float(cmd[2]),
                    'to_wz': level, 'elapsed_s': elapsed, 'observed_yaw_rad': achieved,
                    'predicted_yaw_rad': expected, 'window_s': self.rotation_window})
                cmd = np.array([0.0, 0.0, level])
        self._rotation_attempt = {'sign': sgn, 't': now, 'heading': heading, 'command': float(cmd[2]), 'requested_error': requested}
        return cmd

    # ------------------------------------------------------------------ v5 A5 MPPI (candidate 2)
    def _mppi_setup(self):
        env = self.env
        key = env.raw.get("policy_key")
        fit = load_dynamics(key) if key else None
        gait = None
        if fit is not None:
            dly_v, tau_v = fit["delay_vx"], fit["tau_vx"]
            dly_w, tau_w = fit["delay_wz"], fit["tau_wz"]
            self._slow_s, self._slow_band = float(fit["slow_start_s"]), float(fit["slow_start_vx_abs_max"])
            gait = fit.get("gait_period_s")
            self._dyn_source = "dynamics_fit:" + str(key)
        else:
            dly_v, tau_v = float(env.delay[0]), float(env.tau[0])
            dly_w, tau_w = float(env.delay[2]), float(env.tau[2])
            self._slow_s, self._slow_band = 0.0, 0.0
            self._dyn_source = "envelope_fol"
        self._dv = int(round(dly_v / self.dt))
        self._dw = int(round(dly_w / self.dt))
        self._av = 1.0 - math.exp(-self.dt / max(tau_v, 1e-6))
        self._aw = 1.0 - math.exp(-self.dt / max(tau_w, 1e-6))
        # sway filter window: whole gait periods spanning >= MPPI_SWAY_MIN_TICKS ticks, odd tick count so the
        # window centre is a control tick; the averaged pose is advanced over the centre lag with the model.
        self._nf = 1
        if MPPI_SWAY_FILTER and gait:
            per = float(gait)
            n = max(1, int(math.ceil(MPPI_SWAY_MIN_TICKS * self.dt / per - 1e-9)))
            nf = int(round(n * per / self.dt))
            self._nf = nf + 1 if nf % 2 == 0 else nf
        self._lag = (self._nf - 1) // 2
        self._Lb = max(self._dv, self._dw) + self._lag + 1
        self._mppi_reset()

    def _mppi_reset(self):
        self._mppi_nom = None
        self._applied = [np.zeros(3) for _ in range(self._Lb)]
        self._wm = 0.0                 # disturbance observer: model yaw rate of the applied commands
        self._dob = []                 # (t, gait-averaged heading, model yaw rate) per tracking tick
        self._wbias = 0.0
        self._vybias = 0.0            # body-frame lateral drift velocity (m/s, left > 0)

    def _mppi_observe(self):
        """Offset-free MPC: estimate a constant yaw-rate disturbance (measured map bias, seed-dependent gait
        asymmetry) as the gait-averaged heading change minus the model-predicted change over MPPI_DOB_WINDOW_S
        of tracking.  Replaces the role of the A4 cascade's lateral integrator."""
        if not MPPI_DOB_WINDOW_S or len(self.hist) < self._nf:
            return
        m = self.model
        vx_in = float(self._applied[-1 - self._dv][0]); wz_in = float(self._applied[-1 - self._dw][2])
        w_ss = float(m.forward(vx_in, wz_in)[1])
        self._wm += self._aw * (w_ss - self._wm)
        qa = np.asarray(self.hist[-self._nf:], float)
        th_avg = float(qa[:, 3].mean())
        if self._dob and self.t - self._dob[-1][0] > 1.5 * self.dt:
            self._dob = []             # tracking was interrupted (align turn): restart the window, keep the estimate
        self._dob.append((self.t, th_avg, self._wm, float(qa[:, 1].mean()), float(qa[:, 2].mean())))
        n = int(round(MPPI_DOB_WINDOW_S / self.dt))
        if len(self._dob) > n + self._lag:
            self._dob = self._dob[-(n + self._lag + 1):]
            # heading averages are centred lag ticks back: compare with the model over the same interval
            seg = self._dob[-(n + self._lag + 1):len(self._dob) - self._lag]
            dth = self._dob[-1][1] - self._dob[-1 - n][1]
            pred = sum(q[2] for q in seg[1:]) * self.dt
            self._wbias = float(np.clip((dth - pred) / (n * self.dt), -MPPI_DOB_MAX, MPPI_DOB_MAX))
            if MPPI_DOB_LATERAL:
                # sideways creep at vy = 0 (walk/atlantis_walk ~ +4 mm/s): gait-averaged displacement over the
                # window, lateral component in the mean-heading frame (zero for a constant-rate arc)
                a, b = self._dob[-1 - n], self._dob[-1]
                h = 0.5 * (a[1] + b[1])
                lat = -math.sin(h) * (b[3] - a[3]) + math.cos(h) * (b[4] - a[4])
                self._vybias = float(np.clip(lat / (n * self.dt), -MPPI_DOB_VY_MAX, MPPI_DOB_VY_MAX))

    def _mppi_applied(self, cmd):
        self._applied.append(np.array(cmd, float))
        del self._applied[:-self._Lb]

    def _mppi_clearance(self, px, py):
        """Vectorized clearance lookup for production and analytic map conventions.

        Unknown, nonfinite, lethal and out-of-bounds cells have zero clearance.
        Production GridMap is y-first with ``origin``; analytic WallClearance
        is x-first with ``x0/y0``. Clipping indices only makes indexing safe:
        the separate validity mask prevents an outside pose from borrowing an
        edge or wrapping to the opposite side of the map.
        """
        gm = self.gridmap
        if gm is None:
            return None
        px, py = np.broadcast_arrays(np.asarray(px, float), np.asarray(py, float))
        out = np.zeros(px.shape, float)
        try:
            clearance = np.asarray(gm.clearance)
            if clearance.ndim != 2 or 0 in clearance.shape:
                return out
            h, w = clearance.shape
            finite = np.isfinite(px) & np.isfinite(py)
            if hasattr(gm, "origin") or (hasattr(gm, "x0") and hasattr(gm, "y0")):
                res = float(gm.res)
                if not math.isfinite(res) or res <= 0:
                    return out
                if hasattr(gm, "origin"):
                    x0, y0 = (float(v) for v in gm.origin)
                    fi, fj = (py - y0) / res, (px - x0) / res
                else:
                    x0, y0 = float(gm.x0), float(gm.y0)
                    fi, fj = (px - x0) / res, (py - y0) / res
                valid = finite & np.isfinite(fi) & np.isfinite(fj) & (fi >= 0) & (fi < h) & (fj >= 0) & (fj < w)
                i = np.floor(np.clip(np.where(valid, fi, 0), 0, h - 1)).astype(np.intp)
                j = np.floor(np.clip(np.where(valid, fj, 0), 0, w - 1)).astype(np.intp)
                values = np.asarray(clearance[i, j], float)
                valid &= np.isfinite(values)
                if hasattr(gm, "lethal"):
                    lethal = np.asarray(gm.lethal)
                    if lethal.shape != clearance.shape:
                        return out
                    valid &= ~lethal[i, j].astype(bool)
                return np.where(valid, np.maximum(values, 0.0), 0.0)
            # Compatibility for a map exposing only the contracted scalar
            # world_to_cell method. Never infer safety when lookup fails.
            for q in np.ndindex(px.shape):
                if not finite[q]:
                    continue
                try:
                    i, j = gm.world_to_cell(float(px[q]), float(py[q]))
                    if 0 <= i < h and 0 <= j < w and not (hasattr(gm, "lethal") and gm.lethal[i, j]):
                        value = float(clearance[i, j])
                        if math.isfinite(value):
                            out[q] = max(value, 0.0)
                except (AttributeError, TypeError, ValueError, IndexError, OverflowError):
                    continue
        except (AttributeError, TypeError, ValueError, IndexError, OverflowError):
            return out
        return out

    def _mppi_start_pose(self, x, y, th):
        """Gait-period average of the measured pose (sway filter); returns (x, y, th, lag ticks)."""
        if self._nf <= 1 or len(self.hist) < self._nf:
            return x, y, th, 0
        q = np.asarray(self.hist[-self._nf:], float)
        return float(q[:, 1].mean()), float(q[:, 2].mean()), float(q[:, 3].mean()), self._lag

    def _mppi_vx(self, wz, v_d, vx_lo):
        """A4 speed schedule through the measured map: for each wz, the gait-on vx whose predicted forward
        speed is closest to v_d (the A4 inverse restricted to the chosen wz)."""
        m = self.model
        G = np.linspace(vx_lo, m.vx_max, max(2, int(round((m.vx_max - vx_lo) / 0.005)) + 1))
        wz = np.asarray(wz, float)
        V = np.asarray(m.forward(np.broadcast_to(G, wz.shape + G.shape), wz[..., None])[0], float)
        return G[np.argmin(np.abs(V - v_d) + 1e-6 * (G - vx_lo), axis=-1)]

    def _mppi_plan(self, x, y, th, v0, w0, v_d, vx_lo, tick):
        """Deterministic MPPI over continuous wz sequences inside the gait-on region; vx of every step follows
        the unchanged A4 speed schedule (v_d) through the measured map.  Returns (cmd3, diagnostics)."""
        m, W = self.model, MPPI_W
        H = MPPI_HORIZON
        self._mppi_observe()
        wb, vyb = self._wbias, self._vybias
        rng = np.random.default_rng(MPPI_SEED)
        lo = np.array([vx_lo, float(m.wzs[0])])
        hi = np.array([m.vx_max, float(m.wzs[-1])])
        last = self._applied[-1]
        if self._mppi_nom is None:
            s_now = self._project(x, y)[0]
            kff = self._heading_reference(s_now)[1]
            vx0, wz0, _, _ = m.inverse(v_d, v_d * kff, vx_lo=vx_lo)
            nom = np.tile([vx0, wz0], (H, 1))
        else:
            nom = np.vstack([self._mppi_nom[1:], self._mppi_nom[-1:]])
        nom = np.clip(nom, lo, hi)
        sig = np.array([MPPI_SIGMA_VX, MPPI_SIGMA_WZ])
        half_knots = rng.standard_normal((MPPI_SAMPLES // 2, MPPI_KNOTS, 2)) * sig
        knots = np.concatenate([half_knots, -half_knots], axis=0)
        tk = np.linspace(0.0, H - 1.0, MPPI_KNOTS)
        wk = np.clip(1.0 - np.abs(np.arange(H)[:, None] - tk[None]) / max(tk[1] - tk[0], 1e-9), 0.0, 1.0)
        noise = np.einsum("hk,nkc->nhc", wk, knots)      # piecewise-linear (smooth) perturbations
        U = [nom[None], np.tile(np.clip(last[[0, 2]], lo, hi), (1, H, 1))]
        if MPPI_WZ_LEVELS > 1:                  # deterministic coverage of the turn range (sharp curves)
            lv = np.linspace(lo[1], hi[1], MPPI_WZ_LEVELS)
            C = np.repeat(nom[None], MPPI_WZ_LEVELS, 0)
            C[:, :, 1] = lv[:, None]
            U.append(C)
        U.append(nom[None] + noise)
        U = np.clip(np.concatenate(U, 0), lo, hi)
        # vx is not sampled: A4 speed schedule through the measured map, evaluated at the nominal wz of each
        # step (the measured forward speed depends only weakly on wz); the output vx uses the final wz.
        U[:, :, 0] = self._mppi_vx(nom[:, 1], v_d, vx_lo)[None, :]
        N = U.shape[0]

        # command timeline: past applied commands (index < 0) followed by the sampled sequence
        xs, ys, ths, lag = self._mppi_start_pose(x, y, th)
        Lb = self._Lb
        past = np.array(self._applied, float)[:, [0, 2]]
        Ext = np.concatenate([np.broadcast_to(past[None], (N, Lb, 2)), U], axis=1)
        idx = np.arange(N)
        px = np.full(N, xs); py = np.full(N, ys); pth = np.full(N, ths)
        v = np.full(N, float(v0)); w = np.full(N, float(w0))
        held = np.zeros(N)
        cost = np.zeros(N)
        lo_w = max(0, self.k - 5); hi_w = min(len(self.seglen), self.k + 80)
        Ap = self.X[lo_w:hi_w]; D = self.X[lo_w + 1:hi_w + 1] - Ap
        L2 = np.maximum((D ** 2).sum(1), 1e-12)
        seg_gl = np.arange(lo_w, hi_w)
        w_cap = 0.9 * m.w_turn_max
        s_end = float(self.s_end)
        a0, a1 = self.end_fade
        s_prev = None
        prev = np.broadcast_to(last[[0, 2]], (N, 2))
        v_ref = max(float(v0), m.v_at(m.vx_min), 0.05)
        k_e, k_psi = (self.omega / v_ref) ** 2, 2 * self.zeta * self.omega
        for j in range(-lag, H):
            vx_in = Ext[:, Lb + j - self._dv, 0]
            wz_in = Ext[:, Lb + j - self._dw, 1]
            v_ss, w_ss = m.forward(vx_in, wz_in)
            v_ss = np.asarray(v_ss, float); w_ss = np.asarray(w_ss, float)
            if self._slow_s > 0.0:      # slow start: no forward speed from standstill in the low band
                slow = (vx_in > 1e-9) & (vx_in <= self._slow_band + 1e-9) & (np.abs(v) < 0.02)
                held = np.where(slow, held + self.dt, 0.0)
                v_ss = np.where(slow & (held < self._slow_s), 0.0, v_ss)
            v = v + self._av * (v_ss - v)
            w = w + self._aw * (w_ss + wb - w)
            mid = pth + 0.5 * w * self.dt
            px = px + (v * np.cos(mid) - vyb * np.sin(mid)) * self.dt
            py = py + (v * np.sin(mid) + vyb * np.cos(mid)) * self.dt
            pth = pth + w * self.dt
            if j < 0:
                continue                 # pre-roll from the filtered pose to "now" with applied commands
            P = np.stack([px, py], 1)
            tt = np.clip(((P[:, None, :] - Ap[None]) * D[None]).sum(-1) / L2[None], 0.0, 1.0)
            q = Ap[None] + tt[..., None] * D[None]
            dist = np.hypot(q[..., 0] - px[:, None], q[..., 1] - py[:, None])
            jj = np.argmin(dist, axis=1)
            e = dist[idx, jj]
            gl = seg_gl[jj]
            d_ = D[jj]
            side = np.sign(d_[:, 0] * (py - Ap[jj, 1]) - d_[:, 1] * (px - Ap[jj, 0]))   # left of path > 0
            s_pt = self.s[gl] + tt[idx, jj] * self.seglen[gl]
            if self.heading_coeff is None:
                head = self.segdir[gl]
            else:
                a_, b_, c_, d_ = self.heading_coeff[:, gl]
                u_ = np.clip(s_pt - self.s[gl], 0.0, self.seglen[gl])
                head = a_ + u_ * (b_ + u_ * (c_ + u_ * d_))
            fade = np.clip((s_end - s_pt - a1) / max(a0 - a1, 1e-6), 0.0, 1.0)
            # heading reference toward the path (the A4 cascade's lateral law with its own omega/zeta and
            # end_fade cap) so that steering back to the path is not penalised as heading error
            psi_max = MPPI_PSI_END_SCALE * self.psi_end + (self.psi_far - MPPI_PSI_END_SCALE * self.psi_end) * fade
            terminal_distance = np.maximum(s_end - s_pt, max(self.tol_pos, v_ref * (self.env.delay[0] + self.env.tau[0] + m.stop_per_v)))
            lateral_budget = max(0.0, (self.tol_pos - self.pos_margin) / math.sqrt(2.0) - m.stop_spread)
            correction = np.where(abs(self._terminal_bias) > lateral_budget, np.arctan2(e, terminal_distance), 0.0)
            psi_max = np.maximum(psi_max, np.minimum(self.psi_far, correction))
            psi_ref = np.clip(-k_e * v_ref * side * e / k_psi, -psi_max, psi_max)
            psi = (pth - head - psi_ref + math.pi) % (2 * math.pi) - math.pi
            s_cl = np.minimum(s_pt, s_end)
            cost += W["e"] * e ** 2 + W["psi"] * (1.0 + W["psi_end"] * (1.0 - fade)) * psi ** 2
            if s_prev is not None:
                cost -= W["progress"] * (s_cl - s_prev)
            s_prev = s_cl
            cost += W["speed"] * (v - v_d) ** 2
            cl = self._mppi_clearance(px, py)
            if cl is not None:
                cost += W["clearance"] * np.maximum(MPPI_PROX - cl, 0.0) ** 2 / MPPI_PROX ** 2
            cost += W["wcap"] * np.maximum(np.abs(w) - w_cap, 0.0) ** 2
            # command terms: explicit wz sign-flip penalty (steering-metrics definition) and delta-command
            u = U[:, j]
            smooth = fade if MPPI_END_SMOOTH_FADE else 1.0     # travel: smooth; last end_fade metres: accurate
            cost += smooth * W["flip"] * (u[:, 1] * prev[:, 1] < 0)
            cost += W["dvx"] * ((u[:, 0] - prev[:, 0]) / self.env.half[0]) ** 2
            cost += smooth * W["dwz"] * ((u[:, 1] - prev[:, 1]) / self.env.half[2]) ** 2
            prev = u
        beta = float(cost.min())
        wts = np.exp(-(cost - beta) / MPPI_LAMBDA)
        mode = 0
        if MPPI_SIGN_MODE:
            # two-mode MPPI over the sign of the first wz: pick the mode with the lower free energy
            # F = -lambda log sum exp(-cost/lambda) (each sample's cost already contains the flip penalty), then
            # average only inside that mode, so the softmax mean cannot hover across zero between ticks
            pos = U[:, 0, 1] >= 0.0
            fp, fn = float(wts[pos].sum()), float(wts[~pos].sum())
            mode = 1 if fp >= fn else -1
            wts = np.where(pos if mode > 0 else ~pos, wts, 0.0)
        wts /= wts.sum()
        nom_new = np.clip(np.tensordot(wts, U, axes=(0, 0)), lo, hi)
        nom_new[:, 0] = self._mppi_vx(nom_new[:, 1], v_d, vx_lo)   # A4 speed schedule at the chosen wz
        self._mppi_nom = nom_new
        cmd = np.array([nom_new[0, 0], 0.0, nom_new[0, 1]])
        info = {"cost_min": beta, "ess": float(1.0 / np.sum(wts ** 2)), "n": int(N), "lag": int(lag), "mode": mode, "wbias": wb, "vybias": vyb,
                "dyn": self._dyn_source, "v_d": float(v_d)}
        return cmd, info

    # ------------------------------------------------------------------ main
    def compute(self, pose2_est, gt_vel_or_None=None):
        if self.path is None or len(self.path) == 0:
            return np.zeros(3), "done"
        if self.stuck:
            return np.zeros(3), "stuck"
        x, y, th = (float(v) for v in pose2_est[:3])
        if self.hist:
            th = self.hist[-1][3] + wrap(th - self.hist[-1][3])
        self.hist.append((self.t, x, y, th))
        self.hist = self.hist[-40:]
        gx, gy, gth = self.goal
        dgoal = math.hypot(gx - x, gy - y)
        eyaw = wrap(gth - th)
        v_est, w_est = self._vel()

        if self.state == "done":
            return self._out(np.zeros(3), "done")

        # stuck: no translation/rotation progress while trying to move along the path
        if self.state in ("track", "align"):
            self.prog.append((self.t, x, y, th))
            old = [q for q in self.prog if q[0] <= self.t - self.stuck_time + 1e-9]
            if old:
                q = old[-1]
                if math.hypot(x - q[1], y - q[2]) < self.stuck_dist and abs(th - q[3]) < 0.1:
                    self.stuck = True; self.info = {"reason": "no progress"}
                    return np.zeros(3), "stuck"
                self.prog = [q2 for q2 in self.prog if q2[0] >= q[0]]
        else:
            self.prog = []

        if self.state in ("track", "align"):
            return self._track(x, y, th, v_est, w_est)
        if self.state == "settle":
            if self._still() or self.t - self.state_t >= self.settle_max:
                if self.rot_stop is not None:        # learn from the settled outcome of a rotation burst
                    sgn, h_stop, w_stop, h0, err0, used_wz = self.rot_stop
                    over = (th - h_stop) * sgn
                    achieved = (th - h0) * sgn
                    if achieved < 0.3 * abs(err0):
                        # the yaw relaxed back (posture twist, no step): require longer bursts
                        self.min_ticks[sgn] = min(2 * self.min_ticks[sgn], 16)
                        current = max((used_wz, self._rotation_floor[sgn]), key=abs)
                        stronger = [level for level in self._rotation_levels[sgn] if abs(level) > abs(current) + 1e-9]
                        if stronger:
                            level = stronger[0]
                            self._rotation_floor[sgn] = level
                            self.escalation_events.append({'t': self.t, 'state': self.state,
                                'reason': 'settled_rotation_relaxed', 'from_wz': float(current),
                                'to_wz': level, 'observed_yaw_rad': achieved,
                                'requested_yaw_rad': abs(err0)})
                    elif w_stop > 0.2:                # real rotation: learn the post-stop overshoot
                        a = float(np.clip(over / w_stop, 0.0, 0.6))
                        self.anticip[sgn] = 0.5 * self.anticip[sgn] + 0.5 * a
                    self.rot_stop = None
                return self._check(x, y, th, dgoal, eyaw)
            return self._out(np.zeros(3), "tracking")
        if self.state == "rotate_fix":
            return self._rotate_fix(th, eyaw, w_est)
        if self.state == "translate":
            return self._translate(x, y, th)
        if self.state == "approach":
            return self._approach(x, y, th, dgoal, v_est)
        return self._out(np.zeros(3), "tracking")

    # ------------------------------------------------------------------ stages
    def _track(self, x, y, th, v_est, w_est):
        m = self.model
        s, e, th_p = self._project(x, y)
        th_p, _ = self._heading_reference(s)
        psi = wrap(th - th_p)
        cg, sg = math.cos(self.th_end), math.sin(self.th_end)
        d_along = (self.goal[0] - x) * cg + (self.goal[1] - y) * sg
        moving = v_est > 0.25 * m.v_at(m.vx_min)

        # stop where the measured stopping distance ends at the goal
        v_now = max(v_est, 0.0) if moving else 0.0
        stop_d = m.stop_per_v * v_now + 0.5 * self.dt * v_now
        if self.s_end - s < 0.5 and (d_along <= stop_d or s >= self.s_end + 0.02):
            self._set_state("settle")
            self.info = {"stage": "stop", "d_along": d_along, "stop_d": stop_d}
            return self._out(np.zeros(3), "tracking")

        # large misalignment: turn in place toward the path
        L = 0.15
        j = self._seg_at(s + L)
        dx, dy = self.X[j + 1, 0] - x, self.X[j + 1, 1] - y
        alpha = wrap(math.atan2(dy, dx) - th) if math.hypot(dx, dy) > 0.05 else psi
        if self.state == "track" and abs(alpha) > self.rotate_thresh:
            self._set_state("align")
        if self.state == "align":
            if abs(alpha) < self.rotate_exit or (alpha * w_est > 0 and abs(alpha) <= abs(w_est) * self.anticip[1 if alpha > 0 else -1]):
                self._set_state("track")
            else:
                self.info = {"stage": "align", "alpha": alpha}
                return self._out(self._rotate_cmd(alpha), "rotate")

        # speed schedule: cruise, limited by curvature ahead and proximity
        ks = [self._heading_reference(s + d)[1] for d in np.linspace(0, 0.3, 7)]
        kmax = max(abs(k) for k in ks)
        v_d = self.cruise
        if self.s_end - s < self.final_slow:      # slowest gait-on speed: smallest stop spread
            v_d = m.v_at(m.vx_min)
        if kmax > 1e-6:
            v_d = min(v_d, self.turn_frac * m.w_turn_max / kmax)
        if self.gridmap is not None:
            try:
                i, jj = self.gridmap.world_to_cell(x, y)
                cl = float(self.gridmap.clearance[i, jj])
                if cl < self.prox_dist:
                    v_d *= max(self.prox_min, cl / self.prox_dist)
            except Exception:
                pass

        # v5 A5 candidate 2: MPPI replaces the A4 steering cascade + inverse-model choice. The stop, align,
        # speed schedule (above) and gait kick (below) are the unchanged A4 logic.
        v_ctrl = max(min(v_d, m.v_at(m.vx_max)), m.v_at(m.vx_min))
        vx_lo = m.vx_min
        if not moving:
            if self.moving_since is None:
                self.moving_since = self.t
            if self.t - self.moving_since >= 0.4:    # gait not started: kick with the next level
                vx_lo = m.vx_start
        else:
            self.moving_since = None
        v0 = max(v_est, 0.0) if moving else 0.0
        bias_window = max(self.dt, self.env.delay[0] + 2.0 * self.env.tau[0])
        self.terminal_errors.append((self.t, e))
        self.terminal_errors = [(t, err) for t, err in self.terminal_errors if t >= self.t - bias_window]
        self._terminal_bias = float(np.mean([err for _, err in self.terminal_errors]))
        cmd, mi = self._mppi_plan(x, y, th, v0, w_est, v_ctrl, vx_lo, int(round(self.t / self.dt)))
        self.info = {"stage": "track", "s": s, "e": e, "psi": psi, "v_d": v_ctrl, "mppi": mi}
        return self._out([float(cmd[0]), 0.0, float(cmd[2])], "tracking")

    def _check(self, x, y, th, dgoal, eyaw):
        tp = self.tol_pos - self.pos_margin
        ty = self.tol_yaw - self.yaw_margin
        self.info = {"stage": "check", "dgoal": dgoal, "eyaw": eyaw}
        if dgoal <= tp and abs(eyaw) <= ty:
            self._set_state("done")
            return self._out(np.zeros(3), "done")
        score = max(dgoal / tp, abs(eyaw) / ty)
        if self.best is None or score < self.best[0]:
            self.best = (score, x, y, th, self.t)
        self.fixes += 1
        bound = self.max_fixes
        if self.fixes > bound:
            self.stuck = True; self.info = {"reason": "final fixes exhausted", "dgoal": dgoal, "eyaw": eyaw,
                                            "bound": bound, "best": self.best}
            return np.zeros(3), "stuck"
        if self.rm is not None:
            prev = self._last_action
            if prev is not None and prev[0] == "translate" and dgoal > prev[2] - prev[1]["sigma"]:
                # this exact command did not reduce the error from a settled stance: do not reuse it
                self._banned.add(tuple(prev[1]["cmd"].tolist()))
            self._last_action = None
            act = self._plan_repair(x, y, th, dgoal, eyaw)
            self.repair_log.append({"t": self.t, "dgoal": dgoal, "eyaw": eyaw, "act": act[0]})
            if act[0] == "rotate":
                _, level, sgn, theta = act
                self._axis_hist = {"yaw": (np.array([sgn]), theta, math.radians(1.0))}   # back-to-back only
                self._coarse_level = level
                self._set_state("rotate_fix")
                return self._rotate_fix(th, eyaw, 0.0)
            if act[0] == "translate":
                _, cand, e0, k = act
                self._axis_hist = {"translate": (cand["u"], e0, cand["sigma"])}          # back-to-back only
                self._last_action = ("translate", cand, e0)
                self._tr = dict(cand=cand, e0=e0, ticks=int(round(k * self.rm["tq"] / self.dt)))
                self._set_state("translate")
                return self._translate(x, y, th)
        if dgoal > tp:
            self._set_state("approach")
            return self._approach(x, y, th, dgoal, 0.0)
        self._set_state("rotate_fix")
        return self._rotate_fix(th, eyaw, 0.0)

    def _rotate_fix(self, th, eyaw, w_est, target=None):
        err = eyaw if target is None else target
        if self.t - self.state_t < 1e-9:
            self._rot_dir = 1 if err > 0 else -1
            self._rot0 = (th, err)
        sgn = self._rot_dir
        w_now = self._vel(0.2)[1]
        ticks = int(round((self.t - self.state_t) / self.dt))
        # stop when the remaining angle is covered by the learned post-stop overshoot
        # (not before the learned minimum burst that produces a real step-rotation)
        if (ticks >= self.min_ticks[sgn] and err * sgn <= max(w_now * sgn, 0.0) * self.anticip[sgn]) \
                or err * sgn < -math.radians(2) or self.t - self.state_t > 6.0:
            self.rot_stop = (sgn, th, abs(w_now), self._rot0[0], self._rot0[1], float(self.last_cmd[2]))
            self._coarse_level = None
            self._set_state("settle")
            self.info = {"stage": "rotate_stop", "err": err}
            return self._out(np.zeros(3), "rotate")
        self.info = {"stage": "rotate_fix", "err": err, "w": w_now}
        return self._out(self._rotate_cmd(err), "rotate")

    def _approach(self, x, y, th, dgoal, v_est):
        """Position fix after a stop: short straight walk (forward, or backward when the
        envelope has a straight-backward level and the goal is behind), else face the
        goal first.  Every walk ends with the measured stopping distance."""
        m = self.model
        c, s_ = math.cos(th), math.sin(th)
        ex = c * (self.goal[0] - x) + s_ * (self.goal[1] - y)
        ey = -s_ * (self.goal[0] - x) + c * (self.goal[1] - y)
        bearing = math.atan2(ey, ex)
        w_est = self._vel(0.2)[1]
        ph = self._appr if self.t - self.state_t > 1e-9 else None
        if ph is None:
            if abs(ey) <= 0.6 * self.tol_pos and ex > 0:
                ph = {"phase": "fwd"}
            elif abs(ey) <= 0.6 * self.tol_pos and m.vx_back is not None:
                ph = {"phase": "back"}
            else:
                ph = {"phase": "face"}
            self._appr = ph
        if ph["phase"] == "face":
            # absolute facing target latched once: at a few cm the bearing is dominated by
            # the trunk sway of stepping in place and would run away if recomputed
            if "target" not in ph:
                ph["target"] = th + bearing
                ph["dir"] = 1 if bearing > 0 else -1
                ph["t0"] = self.t
            a = ph["target"] - th
            sgn = ph["dir"]
            ticks = int(round((self.t - ph["t0"]) / self.dt))
            if (ticks >= self.min_ticks[sgn] and a * sgn <= max(w_est * sgn, 0.0) * self.anticip[sgn]) \
                    or a * sgn < 0 or self.t - ph["t0"] > 6.0:
                ph.update(phase="pause", t=self.t)
                return self._out(np.zeros(3), "rotate")
            return self._out(self._rotate_cmd(a), "rotate")
        if ph["phase"] == "pause":
            if self._still() or self.t - ph["t"] > self.settle_max:
                ph.pop("target", None)
                ph["faces"] = ph.get("faces", 0) + 1
                # each facing turn shifts the trunk by a few cm (stance change), so at
                # short range re-facing can chase its own tail: at most two turns
                if abs(bearing) > math.radians(60) and ph["faces"] < 2:
                    ph["phase"] = "face"
                elif abs(bearing) <= math.radians(60):
                    ph["phase"] = "fwd"
                else:
                    self._appr = None; self._set_state("settle")
            return self._out(np.zeros(3), "tracking")
        v_now = self._vel()[0]
        if ph["phase"] == "back":
            if -ex <= m.stop_per_v * max(-v_now, 0.0) + 0.5 * self.dt * max(-v_now, 0.0) or self.t - self.state_t > 4.0:
                self._appr = None; self._set_state("settle")
                return self._out(np.zeros(3), "tracking")
            return self._out([m.vx_back, 0.0, 0.0], "tracking")
        # forward
        if ex <= m.stop_per_v * max(v_now, 0.0) + 0.5 * self.dt * max(v_now, 0.0) or abs(bearing) > math.radians(75) \
                or self.t - self.state_t > 6.0:
            self._appr = None; self._set_state("settle")
            return self._out(np.zeros(3), "tracking")
        vx, wz, _, _ = m.inverse(m.v_at(m.vx_min), float(np.clip(1.5 * bearing, -m.w_turn_max, m.w_turn_max)),
                                 vx_lo=m.vx_min if v_now > 0.02 else m.vx_start)
        return self._out([vx, 0.0, wz], "tracking")
