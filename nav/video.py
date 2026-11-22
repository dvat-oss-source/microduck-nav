"""Render a Sim episode trace (list of qpos, Sim.trace) to mp4 offscreen.

    from nav.video import render_trace, save_trace
    save_trace(sim, "ep.npz")                       # always works (qpos + provenance)
    render_trace(sim.model, sim.trace, "ep.mp4")    # mujoco.Renderer + imageio-ffmpeg

If offscreen GL is unavailable (e.g. a headless Linux node without EGL/OSMesa:
set MUJOCO_GL=egl or osmesa), render_trace falls back to writing
<out>.qpos.npz and returns that path instead. Rendering uses its own MjData, so
it never disturbs the Sim's state.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

POLICY_HZ = 50


def save_trace(sim, path, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"provenance": sim.provenance(), "hz": POLICY_HZ, **(extra or {})}
    np.savez_compressed(path, qpos=np.asarray(sim.trace), meta=json.dumps(meta, default=float))
    return path


def load_trace(path):
    z = np.load(path, allow_pickle=False)
    return z["qpos"], json.loads(str(z["meta"]))


def render_trace(model: mujoco.MjModel, trace, out_path, fps: int = 25, width: int = 640, height: int = 480,
                 camera: str = "track", distance: float = 0.9, azimuth: float = 135.0, elevation: float = -25.0,
                 overlay=None) -> Path:
    """Render qpos frames (recorded at 50 Hz) to an mp4 at `fps` (frames subsampled).

    camera: "track" (follows trunk_base) or "top" (fixed bird's-eye over the trunk start).
    overlay: optional callable(frame_idx, rgb ndarray) -> rgb ndarray for annotations.
    Returns the written path (mp4, or .qpos.npz fallback).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace = np.asarray(trace)
    stride = max(1, int(round(POLICY_HZ / fps)))
    try:
        import imageio.v2 as imageio

        width = min(width, model.vis.global_.offwidth)
        height = min(height, model.vis.global_.offheight)
        renderer = mujoco.Renderer(model, height=height, width=width)
        data = mujoco.MjData(model)
        cam = mujoco.MjvCamera()
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        if camera == "track":
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = trunk
            cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [trace[0][0], trace[0][1], 0.0]
            cam.distance, cam.azimuth, cam.elevation = max(distance, 3.0), 90.0, -89.0
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=7,
                                    macro_block_size=8)
        try:
            for i in range(0, len(trace), stride):
                data.qpos[:] = trace[i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                rgb = renderer.render()
                if overlay is not None:
                    rgb = overlay(i, rgb)
                writer.append_data(rgb)
        finally:
            writer.close()
            renderer.close()
        return out_path
    except Exception as e:  # offscreen GL / ffmpeg unavailable -> keep the data
        fb = out_path.with_suffix(".qpos.npz")
        np.savez_compressed(fb, qpos=trace, error=str(e))
        return fb
