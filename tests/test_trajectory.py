"""Synthetic checks for the floor trajectory: known floor, known path, recover both."""

import cv2
import numpy as np
import pytest

import reconstruct_trajectory as rt
from replay.pose import load_camera, load_plate, object_points, solve
from replay.trajectory import (FloorTracker, fit_floor, measurement_cov, ray_to_floor, robot_path,
                               rts_smooth, CVKalmanFloor)

K, DIST = load_camera()
WIDTH, HEIGHT = load_plate("small")
FPS = 50.0
CAM_HEIGHT = 620.0
PITCH = np.radians(10.0)


def floor_to_camera(xy):
    """Floor (X right, Y forward, plate plane at Z = 0) to camera coordinates, camera CAM_HEIGHT above."""
    xy = np.atleast_2d(xy)
    p = np.column_stack([xy, np.zeros(len(xy))]) - np.array([0.0, 0.0, CAM_HEIGHT])
    right = np.array([1.0, 0.0, 0.0])
    forward = np.array([0.0, np.cos(PITCH), -np.sin(PITCH)])
    down = np.cross(forward, right)
    return p @ np.vstack([right, down, forward]).T


def project_centre(xy):
    px, _ = cv2.projectPoints(floor_to_camera(xy), np.zeros(3), np.zeros(3), K, DIST)
    return px.reshape(-1, 2)


def plane_points(n, rng, noise=20.0, outliers=0.15):
    xy = np.column_stack([rng.uniform(-800, 800, n), rng.uniform(2500, 5500, n)])
    cam = floor_to_camera(xy)
    up = np.array([0.0, -np.cos(PITCH), -np.sin(PITCH)])     # the floor normal, in camera coordinates
    cam = cam + rng.normal(0, noise, (n, 1)) * up
    bad = rng.random(n) < outliers
    cam[bad] *= rng.uniform(0.5, 1.5, (bad.sum(), 1))       # wrong depth along the ray, like a mis-paired plate
    return cam, bad


# ----------------------------------------------------------------------------- the floor plane

def test_floor_plane_recovers_height_and_pitch_despite_outliers():
    cam, bad = plane_points(600, np.random.default_rng(0))
    floor = fit_floor(cam)

    assert floor.camera_height_mm == pytest.approx(CAM_HEIGHT, abs=10)
    assert floor.pitch_deg == pytest.approx(10.0, abs=0.3)
    assert floor.residual_mm == pytest.approx(20.0, rel=0.25)
    assert floor.inliers[bad].mean() < 0.3                  # most outliers are rejected


def test_floor_frame_round_trip():
    cam, _ = plane_points(50, np.random.default_rng(1), noise=0.0, outliers=0.0)
    floor = fit_floor(cam)
    fl = floor.to_floor(cam)

    assert np.abs(fl[:, 2]).max() < 1e-6                     # on the plane
    assert np.allclose(floor.to_camera(fl), cam, atol=1e-6)


def test_fit_floor_needs_three_points():
    with pytest.raises(ValueError):
        fit_floor(np.zeros((2, 3)))


# ----------------------------------------------------------------------------- the ray measurement

def test_ray_meets_the_plane_where_the_point_was_projected_from():
    cam, _ = plane_points(40, np.random.default_rng(2), noise=0.0, outliers=0.0)
    floor = fit_floor(cam)
    truth = np.array([[-500.0, 2600.0], [0.0, 3500.0], [700.0, 5200.0]])
    pts, valid = ray_to_floor(project_centre(truth), K, DIST, floor)

    assert valid.all()
    assert np.allclose(floor.to_floor(pts)[:, :2], truth, atol=0.5)


def test_ray_above_the_horizon_is_invalid():
    cam, _ = plane_points(40, np.random.default_rng(3), noise=0.0, outliers=0.0)
    floor = fit_floor(cam)
    _, valid = ray_to_floor([[640.0, 0.0]], K, DIST, floor)   # top of the image, camera pitched only 10 degrees
    assert not valid[0]


def test_measurement_noise_is_long_along_the_line_of_sight_and_grows_with_range():
    cam, _ = plane_points(40, np.random.default_rng(4), noise=0.0, outliers=0.0)
    floor = fit_floor(cam)
    fx = K[0, 0]
    near = measurement_cov([0.0, 2000.0], floor, fx, 2.0, 30.0)
    far = measurement_cov([0.0, 5000.0], floor, fx, 2.0, 30.0)

    # straight ahead the line of sight is the Y axis
    assert near[1, 1] > 20 * near[0, 0]
    assert far[1, 1] > near[1, 1] and far[0, 0] > near[0, 0]
    # off to the side, the long axis turns with the line of sight
    side = measurement_cov([3000.0, 3000.0], floor, fx, 2.0, 30.0)
    w, v = np.linalg.eigh(side)
    assert abs(abs(v[:, 1] @ np.array([1.0, 1.0]) / np.sqrt(2)) - 1) < 1e-6


# ----------------------------------------------------------------------------- filter and smoother

def cv_path(n, rng):
    t = np.arange(n) / FPS
    truth = np.column_stack([600 * np.sin(t), 3500 + 800 * np.sin(0.5 * t)])
    return truth, truth + rng.normal(0, 60, truth.shape)


def test_rts_smoother_beats_the_forward_filter():
    truth, meas = cv_path(300, np.random.default_rng(5))
    R = np.eye(2) * 60.0 ** 2
    kf = CVKalmanFloor(meas[0], R, accel_std_mm_s2=2000.0)
    xf, Pf, xp, Pp, Fs = [kf.x.copy()], [kf.P.copy()], [None], [None], [None]
    for z in meas[1:]:
        Fs.append(kf.predict(1 / FPS))
        xp.append(kf.x.copy())
        Pp.append(kf.P.copy())
        kf.update(z, R)
        xf.append(kf.x.copy())
        Pf.append(kf.P.copy())
    xs, _ = rts_smooth(xf, Pf, xp, Pp, Fs)
    err_f = np.linalg.norm(np.array(xf)[50:, :2] - truth[50:], axis=1).mean()
    err_s = np.linalg.norm(np.array(xs)[50:, :2] - truth[50:], axis=1).mean()

    assert err_s < 0.8 * err_f
    assert err_s < 30.0                                      # well under the 60 mm measurement noise


def test_tracker_ignores_a_distractor_and_ends_a_track_after_a_long_gap():
    truth, meas = cv_path(200, np.random.default_rng(6))
    R = np.eye(2) * 60.0 ** 2
    tracker = FloorTracker(FPS, max_gap=5, accel_std_mm_s2=5000.0)
    for f in range(200):
        if 100 <= f < 110:                                   # a 10-frame gap, longer than max_gap
            tracker.step(f, [])
            continue
        cands = [(meas[f], R)]
        if f not in (0, 110):                                # a track only starts from a lone candidate
            cands.append((meas[f] + np.array([900.0, -700.0]), R))   # a second plate-like candidate nearby
        tracker.step(f, cands)
    tracks = tracker.finish()

    assert len(tracks) == 2
    first = tracks[0]
    used = np.array([z for z in first.z if z is not None])
    assert np.allclose(used, meas[:len(used)])               # always the right candidate
    assert first.frames[-1] == 99                            # trimmed to the last measurement


def test_robot_path_smooths_out_a_spinning_plate():
    rng = np.random.default_rng(7)
    n, radius = 400, 200.0
    t = np.arange(n) / FPS
    centre = np.column_stack([400 * np.sin(0.4 * t), 3800 + 600 * np.sin(0.3 * t)])
    spin = 2 * np.pi * 2.5 * t                               # 2.5 turns a second
    phase = (spin % (np.pi / 2)) - np.pi / 4                 # the plate most facing the camera, of four
    to_cam = -centre / np.linalg.norm(centre, axis=1, keepdims=True)
    c, s = np.cos(phase), np.sin(phase)
    facing = np.column_stack([c * to_cam[:, 0] - s * to_cam[:, 1], s * to_cam[:, 0] + c * to_cam[:, 1]])
    plates = centre + radius * facing + rng.normal(0, 10, (n, 2))

    tracker = FloorTracker(FPS, accel_std_mm_s2=200000.0)
    for f in range(n):
        tracker.step(f, [(plates[f], np.eye(2) * 10.0 ** 2)])
    (track,) = tracker.finish()
    frames, path = robot_path(track, FPS, accel_std_mm_s2=5000.0, spin_std_mm=150.0)

    err = np.linalg.norm(path[:, :2] - centre[frames], axis=1)
    assert np.median(err) < radius                           # within a chassis radius of the true centre
    plate_jitter = np.diff(np.array(track.x_smooth)[:, :2], axis=0).std()
    path_jitter = np.diff(path[:, :2], axis=0).std()
    assert path_jitter < 0.2 * plate_jitter


# ----------------------------------------------------------------------------- end to end

def test_end_to_end_recovers_a_path_from_noisy_plate_corners():
    rng = np.random.default_rng(8)
    n = 250
    t = np.arange(n) / FPS
    truth = np.column_stack([500 * np.sin(0.8 * t), 3000 + 1500 * (1 - np.cos(0.5 * t))])
    obj = object_points(WIDTH, HEIGHT)
    rows = []
    for f, xy in enumerate(truth):
        tvec = floor_to_camera(xy)[0]
        corners, _ = cv2.projectPoints(obj, np.zeros(3), tvec, K, DIST)
        corners = corners.reshape(-1, 2) + rng.normal(0, 0.7, (4, 2))
        r = solve(corners, K, DIST, WIDTH, HEIGHT)
        centre = corners.mean(axis=0)
        x, y, z = r["tvec_mm"]
        rows.append({"frame": f, "cx": centre[0], "cy": centre[1], "x_mm": x, "y_mm": y, "z_mm": z,
                     "reprojection_px": r["reprojection_px"], "width_px": r["width_px"]})

    floor, fl, consistent, tracks, check = rt.reconstruct(
        rows, np.arange(n) / FPS, K, DIST, pixel_std=1.0, height_std=10.0, accel=5000.0, max_gap=12, consistency_k=3.0)

    assert floor.pitch_deg == pytest.approx(10.0, abs=1.0)
    assert floor.camera_height_mm == pytest.approx(CAM_HEIGHT, rel=0.1)
    assert len(tracks) == 1
    xs = np.array(tracks[0].x_smooth)[:, :2]
    err = np.linalg.norm(xs - truth[tracks[0].frames], axis=1)
    assert np.median(err) < 40.0
