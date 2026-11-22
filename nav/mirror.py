"""Phase G -- left/right mirroring of Microduck policies (mirrored left-kick).

Pollen ships ball_kick_left.onnx AND ball_kick_right.onnx.  The user's own
trained policies may only exist for one foot, so this module wraps a policy
(e.g. the right-kick) in a bilateral mirror transform to obtain the other foot:

    act(obs) = mirror_action( policy( mirror_obs(obs) ) )

The tables below are copied VERBATIM from
    upstream/microduck_rl/src/mjlab_microduck/tasks/symmetry.py
(_JOINT_PERM, _JOINT_SIGN, _OBS_PERM, _OBS_SIGN).  symmetry.py itself imports
torch/tensordict/mjlab at module level, so it is not imported here; the test
suite asserts (via ast) that these copies equal the upstream source.

Kick-policy observation layout (61-D, "unified 61D actor layout"); identical to
the walking layout, so symmetry.py covers every index:
    [0:3]   base_ang_vel      IMU gyro (roll, pitch, yaw)
    [3:6]   projected_gravity body frame
    [6:20]  joint_pos_rel     14 joints, q - DEFAULT_POSE
    [20:34] joint_vel         14 joints (raw qvel, rad/s)
    [34:48] last_action       14 joints (raw policy output)
    [48:61] command (13-D)    twist(3) head(4) body(6); ALL ZERO for kicks
Joint order: left leg 0-4 (yaw, roll, pitch, knee, ankle), neck_pitch, head_pitch,
head_yaw, head_roll (5-8), right leg 9-13.

The kick policies are BLIND to the ball (upstream microduck_ball_kick_env_cfg:
"The policy is BLIND to the ball (no ball obs in the actor)").  So no
ball-relative slot needs mirroring; the only left/right asymmetry is the ball
SPAWN position in the trunk yaw frame: (+0.09, -0.042) m for the right foot and
(+0.09, +0.042) m for the left (BALL_OFFSET_X / BALL_OFFSET_ABS_Y, +-0.015 m
uniform noise per axis in training).  See `kick_ball_target`.

Action: motor_target = DEFAULT_POSE + action * 1.0.  Because DEFAULT_POSE is
itself mirror-symmetric under (perm, sign) -- asserted in the tests -- mirroring
the raw action with the same perm/sign mirrors the joint targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Tables copied verbatim from upstream symmetry.py (checked by tests/test_mirror.py)
# ---------------------------------------------------------------------------
_JOINT_PERM: list[int] = [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]

_JOINT_SIGN: list[float] = [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1]

_OBS_PERM: list[int] = (
    [0, 1, 2]                           # base_ang_vel (indices unchanged)
    + [3, 4, 5]                         # projected_gravity
    + [6 + j for j in _JOINT_PERM]     # joint_pos
    + [20 + j for j in _JOINT_PERM]    # joint_vel
    + [34 + j for j in _JOINT_PERM]    # last_action
    + [48, 49, 50]                      # twist command
    + [51, 52, 53, 54]                  # head command
    + [55, 56, 57, 58, 59, 60]          # body command
)

_OBS_SIGN: list[float] = (
    [-1.0, 1.0, -1.0]   # base_ang_vel: negate roll, yaw
    + [1.0, -1.0, 1.0]  # projected_gravity: negate gy
    + _JOINT_SIGN       # joint_pos
    + _JOINT_SIGN       # joint_vel
    + _JOINT_SIGN       # last_action
    + [1.0, -1.0, -1.0] # twist: negate lin_vel_y, ang_vel_z
    + [1.0, 1.0, -1.0, -1.0]  # head: negate head_yaw, head_roll
    + [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]  # body: negate y, roll, yaw
)

OBS_DIM = 61
ACT_DIM = 14

_OBS_PERM_NP = np.asarray(_OBS_PERM, dtype=np.int64)
_OBS_SIGN_NP = np.asarray(_OBS_SIGN, dtype=np.float32)
_ACT_PERM_NP = np.asarray(_JOINT_PERM, dtype=np.int64)
_ACT_SIGN_NP = np.asarray(_JOINT_SIGN, dtype=np.float32)

# Ball spawn in the robot yaw frame (infer_policy.py / ball-kick env cfg).
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
BALL_RADIUS = 0.035
BALL_POS_NOISE_XY = 0.015


def kick_ball_target(foot: str) -> tuple[float, float]:
    """Nominal ball centre (x, y) in the trunk yaw frame for a kick with `foot`.

    right -> (0.09, -0.042), left -> (0.09, +0.042).  Mirroring maps one into the
    other by negating y, exactly like the obs transform negates lateral slots.
    """
    if foot not in ("left", "right"):
        raise ValueError(foot)
    return (BALL_OFFSET_X, -BALL_OFFSET_ABS_Y if foot == "right" else BALL_OFFSET_ABS_Y)


def mirror_obs(obs: np.ndarray) -> np.ndarray:
    """Bilateral mirror of a 61-D actor observation (shape (..., 61)). Involution."""
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape[-1] != OBS_DIM:
        raise ValueError(f"expected last dim {OBS_DIM}, got {obs.shape}")
    return obs[..., _OBS_PERM_NP] * _OBS_SIGN_NP


def mirror_action(action: np.ndarray) -> np.ndarray:
    """Bilateral mirror of a 14-D raw policy action (shape (..., 14)). Involution."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != ACT_DIM:
        raise ValueError(f"expected last dim {ACT_DIM}, got {action.shape}")
    return action[..., _ACT_PERM_NP] * _ACT_SIGN_NP


class MirroredPolicy:
    """ONNX policy wrapped in a left/right mirror: act = M_a(pi(M_o(obs))).

    Parameters
    ----------
    onnx_path : path of the policy to mirror (e.g. ball_kick_right.onnx to get a
        left kick).  Alternatively pass an existing onnxruntime session via
        `session=`.
    """

    def __init__(self, onnx_path: str | Path | None = None, session=None):
        if session is None:
            if onnx_path is None:
                raise ValueError("need onnx_path or session")
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            session = ort.InferenceSession(str(onnx_path), sess_options=so,
                                           providers=["CPUExecutionProvider"])
        self.session = session
        self.in_name = session.get_inputs()[0].name
        self.out_name = session.get_outputs()[0].name
        self.obs_dim = int(session.get_inputs()[0].shape[-1])
        self.act_dim = int(session.get_outputs()[0].shape[-1])
        if self.obs_dim != OBS_DIM or self.act_dim != ACT_DIM:
            raise ValueError(
                f"ONNX dims ({self.obs_dim}->{self.act_dim}) != mirror tables "
                f"({OBS_DIM}->{ACT_DIM})")

    def raw(self, obs: np.ndarray) -> np.ndarray:
        """Underlying (un-mirrored) policy: obs (61,) or (B,61) -> action."""
        x = np.asarray(obs, dtype=np.float32)
        single = x.ndim == 1
        rows = x.reshape(-1, OBS_DIM)
        # The exported graphs have a fixed batch dimension of 1: run row by row.
        out = np.concatenate([
            self.session.run([self.out_name], {self.in_name: r[None]})[0] for r in rows
        ], axis=0).astype(np.float32)
        return out[0] if single else out

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Mirrored action for a (61,) or (B,61) observation in the TARGET side's frame."""
        return mirror_action(self.raw(mirror_obs(obs)))

    __call__ = act


def kick_sweep(sim_factory, policy_key, offsets, seeds):
    """Closed-loop ball-offset sweep of a kick policy (STUB -- implemented in a later wave).

    Intended behaviour (to be run once nav.sim exists):

    For every ball offset (dx, dy) in `offsets` (metres, added to the nominal ball
    spawn for the policy's foot -- i.e. relative to (0.09, -/+0.042) in the trunk
    yaw frame; typically a +-3 cm grid such as {-0.03,-0.015,0,0.015,0.03}^2) and
    every seed in `seeds`:
        sim = sim_factory()   # fresh nav.sim.Sim on the ball scene, policy_key kick
        sim.reset(seed, extra={"ball_offset": (dx, dy)})
        run the kick policy (`policy_key`, e.g. "kick_left" = Pollen left,
        "kick_left_mirrored" = MirroredPolicy(ball_kick_right)) for ~2 s,
        record: ball displacement / peak speed in the trunk yaw frame at reset,
        ball direction error, robot fell (tilt > 60 deg), robot still standing.
    A trial is a SUCCESS iff the robot did not fall AND the ball ends >= 0.3 m
    from its start within +-30 deg of the trunk's forward axis.

    Returns a dict:
        {"policy_key": str,
         "offsets": list of (dx, dy),
         "n_seeds": int,
         "success": ndarray (n_offsets,) success rate over seeds,
         "fall_rate": ndarray (n_offsets,),
         "ball_dx_mean": ndarray (n_offsets,), "ball_dy_mean": ndarray (n_offsets,),
         "overall_success": float}
    so that mirrored-right vs Pollen-left success rates can be compared offset by
    offset (expect comparable overall rate and mirror-symmetric offset maps).
    """
    raise NotImplementedError("kick_sweep is implemented in a later wave using nav.sim")
