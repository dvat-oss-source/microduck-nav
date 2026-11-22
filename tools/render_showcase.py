"""Render tracking and top-down trails at 4x, optionally comparing two traces."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav import checker


class Replay:
    def __init__(self, trace_dir, episode, width=640, height=540):
        with np.load(Path(trace_dir) / (episode + '.npz'), allow_pickle=False) as z:
            self.meta = json.loads(str(z['metadata']))
            self.task = json.loads(str(z['task']))
            self.q, self.t = z['qpos'].copy(), z['t'].copy()
        self.model = checker.load_model(self.meta['model_path'])
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        self.tracking = mujoco.Renderer(self.model, height, width)
        self.top = mujoco.Renderer(self.model, height, width)
        trunk = self.model.body('trunk_base').id
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = trunk
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.1, 135., -28.
        self.overhead = mujoco.MjvCamera()
        self.overhead.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.overhead.azimuth, self.overhead.elevation = 90., -89.9
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        qadr = checker._free_qadr(self.model, trunk)
        self.xy = self.q[:, qadr:qadr + 2]

    def bounds(self, low, high):
        self.overhead.lookat[:] = [*(low + high) / 2, 0.]
        self.overhead.distance = max(high - low) * 1.35 + 1.5

    @staticmethod
    def add(scene, kind, size, position, color):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.asarray(size, float),
                           np.asarray(position, float), np.eye(3).ravel(), np.asarray(color, np.float32))
        scene.ngeom += 1

    def frame(self, elapsed, label):
        i = min(len(self.t) - 1, max(0, int(np.searchsorted(self.t, elapsed, side='right') - 1)))
        self.data.qpos[:] = self.q[i]
        mujoco.mj_forward(self.model, self.data)
        self.tracking.update_scene(self.data, camera=self.camera)
        tracking = self.tracking.render()
        self.top.update_scene(self.data, camera=self.overhead, scene_option=self.options)
        scene = self.top.scene
        for pose, color in ((self.task['start'], [.15, .75, .25, 1]),
                            (self.task['goal'], [.9, .2, .2, 1])):
            self.add(scene, mujoco.mjtGeom.mjGEOM_CYLINDER, [.09, .01, 0],
                     [pose[0], pose[1], .02], color)
            self.add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.035, 0, 0],
                     [pose[0] + .15 * np.cos(pose[2]), pose[1] + .15 * np.sin(pose[2]), .04], color)
        for j in range(0, i, 25):
            self.add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.025, 0, 0],
                     [*self.xy[j], .03], [.1, .45, .95, 1])
        self.add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.06, 0, 0],
                 [*self.xy[i], .08], [1, .8, 0, 1])
        pair = np.concatenate([tracking, np.full((self.height, 6, 3), 255, np.uint8),
                               self.top.render()], axis=1)
        banner = Image.new('RGB', (pair.shape[1], 36), '#101923')
        ImageDraw.Draw(banner).text((12, 10), f'{label} | simulated t={self.t[i]:.2f}s | 4x speed | tracking + top-down',
                                   fill='white')
        return np.concatenate([np.asarray(banner), pair], axis=0)

    def close(self):
        self.tracking.close()
        self.top.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode')
    parser.add_argument('out', type=Path)
    parser.add_argument('--trace-dir', default='reports/traces/frozen_v2_atlantis_walk_gt')
    parser.add_argument('--compare-trace-dir', help='render this second trace beside the first')
    parser.add_argument('--label', default='BEFORE - frozen v2 GT')
    parser.add_argument('--compare-label', default='AFTER - A5 candidate')
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f'refusing to overwrite existing video: {args.out}')
    replays = []
    try:
        replays.append(Replay(args.trace_dir, args.episode))
        if args.compare_trace_dir:
            replays.append(Replay(args.compare_trace_dir, args.episode))
        points = np.vstack([p for r in replays for p in
                            (r.xy, np.asarray([r.task['start'][:2], r.task['goal'][:2]]))])
        for replay in replays:
            replay.bounds(points.min(0), points.max(0))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        labels = [args.label, args.compare_label]
        end = max(r.t[-1] for r in replays)
        with imageio.get_writer(args.out, fps=25, codec='libx264', quality=8,
                                macro_block_size=8, ffmpeg_params=['-threads', '1']) as writer:
            for elapsed in np.arange(0., end + .08, .16):
                frames = [r.frame(elapsed, label) for r, label in zip(replays, labels)]
                frame = np.concatenate(frames, axis=1)
                writer.append_data(frame)
            for _ in range(50):
                writer.append_data(frame)
        print(args.out, f'{end:.2f} simulated seconds, 4x speed')
    finally:
        for replay in replays:
            replay.close()


if __name__ == '__main__':
    main()
