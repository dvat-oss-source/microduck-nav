import numpy as np
from nav.localization import Localizer, FiducialDetector, FID_PALETTE, FID_CODES
from nav.sim import Sim


def test_odom_uses_sensors_not_eval_truth():
    s = Sim('flat'); gt = s.reset(1, start_pose=(0, 0, 0))
    loc = Localizer(s, 'odom', sensor_noise=False); loc.reset(gt.pose2)
    for _ in range(100):
        gt, _ = s.step(np.array([.3, 0, 0]))
        est = loc.update(s, gt_for_eval_only=object())
    assert np.all(np.isfinite(est))
    assert np.linalg.norm(est[:2] - gt.pose2[:2]) < .15
    assert abs(est[0]) > .01


def test_gt_mode_returns_explicit_ground_truth():
    s = Sim('flat'); gt = s.reset(1, start_pose=(0, 0, 0))
    loc = Localizer(s, 'gt'); loc.reset(gt.pose2)
    assert np.allclose(loc.update(s, gt), gt.pose2)


def test_detector_reads_color_pair_pixels():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[30:45, 40:55] = np.array(FID_PALETTE['R']) * 255
    image[45:60, 40:55] = np.array(FID_PALETTE['G']) * 255
    dets = FiducialDetector().detect(image)
    assert len(dets) == 1
    assert dets[0].id == FID_CODES.index(('R', 'G'))


def test_contact_odometry_holds_translation_when_airborne():
    from nav.localization import PyOdometry
    corners = [np.array([[0., 0., 0.]]), np.array([[0., 0., 0.]])]
    od = PyOdometry(corners)
    feet = [(np.array([0., 0., -1.]), np.eye(3))] * 2
    od.update(feet, np.eye(3), (True, False))
    before = od.position.copy()
    moved = [(np.array([1., 0., -1.]), np.eye(3))] * 2
    od.update(moved, np.eye(3), (False, False))
    assert np.array_equal(od.position, before)


def test_virtual_contact_sensor_matches_collision_events():
    s = Sim('flat', dynamics='training'); s.reset(4, start_pose=(0, 0, 0))
    loc = Localizer(s, 'odom', sensor_noise=False)
    seen = False
    for _ in range(60):
        gt, _ = s.step(np.zeros(3))
        frame = loc.reader.read(s.data)
        assert frame.foot_contacts == (gt.foot_contacts['left'], gt.foot_contacts['right'])
        seen |= any(frame.foot_contacts)
    assert seen


def test_fiducial_range_uses_pinhole_vertical_projection():
    from types import SimpleNamespace
    from nav.localization import Detection, FID_PATCH
    loc = object.__new__(Localizer)
    angle = .35
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    loc.kin = SimpleNamespace(camera_in_trunk=lambda q: (np.zeros(3), rotation))
    loc.camera = SimpleNamespace(f=300., W=320, H=240, rot_theta=0.)
    from nav.localization import LocalizerParams
    loc.p = LocalizerParams(); loc._R_tr = np.eye(3)
    # Project an independently constructed 16 cm vertical segment at known range.
    center = np.array([.22, .08, -2.]); vertical = rotation.T @ np.array([0., 0., 1.])
    top, bottom = center + FID_PATCH * vertical, center - FID_PATCH * vertical
    proj_v = lambda p: 120. - 300. * p[1] / -p[2]
    h = abs(proj_v(top) - proj_v(bottom))
    u, v = 160 + 300 * center[0] / -center[2], proj_v(center)
    det = Detection(0, u, v, h, 10.)
    out = loc._apply_camera_model([det], SimpleNamespace(q=np.zeros(14)))[0]
    expected = np.linalg.norm((rotation @ center)[:2])
    assert abs(out.range_h - expected) < 1e-9


def test_fiducial_camera_maps_qpos_by_joint_name():
    """Render v2: training sim (backlash joints interleaved) must drive the render model by joint name, not prefix."""
    import math
    import pytest
    import mujoco
    from nav.sim import Sim
    from nav.localization import FiducialCamera, FIDUCIAL_RENDER_VERSION
    assert FIDUCIAL_RENDER_VERSION == 2
    sim = Sim('apartment', dynamics='training')
    sim.reset(0, start_pose=(2., 0., 0.3))
    cam = FiducialCamera(sim.scene_xml)
    try:
        with pytest.raises(RuntimeError):
            cam.render(sim.data)                      # fail closed when unbound
        cam.bind(sim.model)
        for _ in range(60):
            sim.step(np.array([.25, 0., .3]))
        cam._set_qpos(sim.data)
        sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, 'head_camera')
        R1 = cam.data.cam_xmat[cam.cam_id].reshape(3, 3)
        R2 = sim.data.cam_xmat[sid].reshape(3, 3)
        ang = math.degrees(math.acos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))
        assert ang < 5.0                              # prefix copy gave ~27 deg median; residual = unmodelled backlash
        assert np.linalg.norm(cam.data.cam_xpos[cam.cam_id] - sim.data.cam_xpos[sid]) < 0.01
    finally:
        cam.close()


def test_fiducial_camera_infer_layout_unchanged():
    """Infer dynamics share the render layout: name map must equal the identity prefix map (render v1 parity)."""
    from nav.sim import Sim
    from nav.localization import FiducialCamera
    sim = Sim('apartment', dynamics='infer')
    cam = FiducialCamera(sim.scene_xml)
    try:
        cam.bind(sim.model)
        r, s = cam._qmap
        assert np.array_equal(np.sort(r), np.arange(cam.model.nq)) and np.array_equal(r, s)
    finally:
        cam.close()
