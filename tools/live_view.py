"""Real-time (1x) playback of a recorded benchmark episode in the interactive MuJoCo viewer.
Space: pause/resume (viewer). Trail = path walked; green = start, red = goal. Loops until the window is closed."""
import argparse, sys, time, json, numpy as np, mujoco, mujoco.viewer
sys.path.insert(0, ".")
from nav import checker
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('episode', nargs='?', default='nav_125')
parser.add_argument('--trace-dir', default='reports/traces/frozen_v2_atlantis_walk_gt')
args = parser.parse_args()
ep = args.episode
z = np.load(f"{args.trace_dir}/{ep}.npz", allow_pickle=False)
meta = json.loads(str(z["metadata"])); task = json.loads(str(z["task"])); q = z["qpos"]; t = z["t"]
m = checker.load_model(meta["model_path"]); d = mujoco.MjData(m)
trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
xy = q[:, :2]

def add(scn, typ, size, pos, rgba):
    if scn.ngeom >= scn.maxgeom: return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], typ, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1

with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; v.cam.trackbodyid = trunk
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.4, 135.0, -30.0
    while v.is_running():
        t0 = time.perf_counter()
        for i in range(len(q)):
            if not v.is_running(): break
            target = t0 + (t[i] - t[0])
            while time.perf_counter() < target: time.sleep(0.001)
            with v.lock():
                d.qpos[:] = q[i]; mujoco.mj_forward(m, d); d.time = t[i]
                scn = v.user_scn; scn.ngeom = 0
                for p, c in ((task["start"], [0.15, 0.75, 0.25, 1]), (task["goal"], [0.9, 0.2, 0.2, 1])):
                    add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.09, 0.005, 0], [p[0], p[1], 0.006], c)
                for j in range(0, i, 25):
                    add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0, 0], [xy[j, 0], xy[j, 1], 0.02], [0.1, 0.45, 0.95, 1])
            v.sync()
        time.sleep(2.0)   # hold at goal, then loop
