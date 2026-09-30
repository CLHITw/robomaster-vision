"""Synthetic checks for the dash-camera tooling: give it an answer and see if it finds it.

The calibration in dashcam_metrology/ has no external reference to be graded
against, so the code itself is graded here instead: intrinsics are invented,
images or corner points are generated from them, and the tools have to recover
the numbers they were built from.
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(HERE, "dashcam_metrology")
sys.path.insert(0, TOOLS)

from measure import load_camera, marker_object_points, solve_marker  # noqa: E402

CONFIG = os.path.join(TOOLS, "config", "camera_dashcam.json")
SIDE_MM = 100.0


@pytest.fixture(scope="module")
def camera():
    return load_camera(CONFIG)


def project(objp, rvec, tvec, K, D):
    pts, _ = cv2.projectPoints(objp, np.array(rvec, float), np.array(tvec, float), K, D)
    return pts.reshape(-1, 2)


# --------------------------------------------------------------- the config


def test_config_is_a_plausible_calibration():
    with open(CONFIG, encoding="utf-8") as f:
        c = json.load(f)
    w, h = c["image_size"]
    K = np.array(c["camera_matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    assert c["model"] == "pinhole", "replay/pose.py solves with pinhole intrinsics"
    assert abs(fx - fy) / fx < 0.02, "square pixels: fx and fy should agree"
    # a principal point far from the centre usually means a crop, not a lens
    assert abs(cx - w / 2) / w < 0.10
    assert abs(cy - h / 2) / h < 0.10
    assert c["rms_reprojection_px"] < 1.0
    assert c["square_size_mm"] != 25.0 or "measured" in c.get("note", ""), (
        "the nominal square size is a common transcription error; "
        "the measured one belongs here"
    )


# --------------------------------------------------------------- pose


@pytest.mark.parametrize("distance_mm", [400.0, 1000.0, 2500.0])
def test_recovers_a_head_on_marker(camera, distance_mm):
    K, D, _ = camera
    objp = marker_object_points(SIDE_MM)
    corners = project(objp, [0.0, 0.0, 0.0], [0.0, 0.0, distance_mm], K, D)

    dist, depth, err, _, _ = solve_marker(corners, SIDE_MM, K, D)

    assert depth == pytest.approx(distance_mm, rel=1e-4)
    assert dist == pytest.approx(distance_mm, rel=1e-4)
    assert err < 0.01


def test_recovers_a_tilted_marker(camera):
    K, D, _ = camera
    objp = marker_object_points(SIDE_MM)
    rvec, tvec = [0.25, -0.18, 0.06], [70.0, -40.0, 1400.0]
    corners = project(objp, rvec, tvec, K, D)

    dist, _, err, rv, tv = solve_marker(corners, SIDE_MM, K, D)

    assert dist == pytest.approx(np.linalg.norm(tvec), rel=1e-4)
    assert np.allclose(tv.ravel(), tvec, rtol=1e-3, atol=0.5)
    assert np.allclose(rv.ravel(), rvec, rtol=1e-2, atol=1e-3)
    assert err < 0.01


def test_distance_is_proportional_to_the_assumed_marker_size(camera):
    """Getting the printed size wrong scales every distance by the same factor.

    This is why the measurement report can be recomputed for a different marker
    size without collecting anything again — and why a mistyped size is invisible
    in the reprojection error.
    """
    K, D, _ = camera
    objp = marker_object_points(SIDE_MM)
    corners = project(objp, [0.0, 0.0, 0.0], [0.0, 0.0, 1000.0], K, D)

    truth = solve_marker(corners, SIDE_MM, K, D)
    wrong = solve_marker(corners, SIDE_MM * 0.95, K, D)

    assert wrong[0] == pytest.approx(truth[0] * 0.95, rel=1e-6)
    assert wrong[2] == pytest.approx(truth[2], abs=1e-6), "reprojection error does not notice"


def test_reprojection_error_cannot_see_a_corner_ordering_mistake(camera):
    """Pins a blind spot rather than a behaviour.

    A square marker is symmetric under a quarter turn, so feeding the four
    corners in a rotated order gives the same distance and the same reprojection
    error while the recovered orientation is wrong by 90 degrees. Distance work
    is unaffected; anything using the rotation is not, and no self-check in this
    repository would catch it.
    """
    K, D, _ = camera
    objp = marker_object_points(SIDE_MM)
    corners = project(objp, [0.2, -0.15, 0.05], [30.0, 20.0, 1200.0], K, D)

    ok = solve_marker(corners, SIDE_MM, K, D)
    rolled = solve_marker(np.roll(corners, 1, axis=0), SIDE_MM, K, D)

    assert rolled[0] == pytest.approx(ok[0], rel=1e-6)
    assert rolled[2] == pytest.approx(ok[2], abs=1e-6)

    turn = cv2.Rodrigues(rolled[3])[0] @ cv2.Rodrigues(ok[3])[0].T
    angle = np.degrees(np.arccos(np.clip((np.trace(turn) - 1) / 2, -1, 1)))
    assert angle == pytest.approx(90.0, abs=1.0), "the orientation is the part that moved"


# --------------------------------------------------------------- calibration


def test_calibration_recovers_the_intrinsics_it_was_built_from():
    """End-to-end: invent a camera, generate corner observations, calibrate, compare."""
    cols, rows, square = 8, 5, 24.5
    size = (640, 360)
    K_true = np.array([[330.0, 0.0, 322.0], [0.0, 331.0, 178.0], [0.0, 0.0, 1.0]])
    D_true = np.array([-0.34, 0.12, 0.0006, -0.0004, -0.02])

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square

    rng = np.random.default_rng(5)
    obj_pts, img_pts = [], []
    while len(img_pts) < 18:
        rvec = rng.uniform(-0.45, 0.45, 3)
        tvec = [rng.uniform(-70, 70), rng.uniform(-50, 50), rng.uniform(280, 620)]
        pts = project(objp, rvec, tvec, K_true, D_true)
        if pts.min() < 5 or pts[:, 0].max() > size[0] - 5 or pts[:, 1].max() > size[1] - 5:
            continue
        obj_pts.append(objp.copy())
        img_pts.append(pts.reshape(-1, 1, 2).astype(np.float32))

    rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)

    assert rms < 0.05, "noiseless observations should fit almost exactly"
    assert K[0, 0] == pytest.approx(K_true[0, 0], rel=0.01)
    assert K[1, 1] == pytest.approx(K_true[1, 1], rel=0.01)
    assert K[0, 2] == pytest.approx(K_true[0, 2], abs=3.0)
    assert K[1, 2] == pytest.approx(K_true[1, 2], abs=3.0)
    assert D.ravel()[0] == pytest.approx(D_true[0], abs=0.03)


def test_undistortion_straightens_a_line_that_distortion_bent(camera):
    """The check straightness.py performs, on one synthetic line.

    Collinearity is a different question from reprojection error: it asks whether
    something known to be straight comes out straight, which is the property the
    distance work leans on.
    """
    K, D, size = camera
    w, h = size
    # kept within the radius where the distortion model is invertible; see
    # test_the_distortion_model_is_not_invertible_at_the_corners
    xs = np.linspace(w * 0.25, w * 0.75, 25)
    straight = np.stack([xs, np.full_like(xs, h * 0.30)], axis=1)

    # push the ideal line through the lens model, then pull it back out
    norm = np.stack([(straight[:, 0] - K[0, 2]) / K[0, 0],
                     (straight[:, 1] - K[1, 2]) / K[1, 1]], axis=1)
    bent = cv2.projectPoints(
        np.hstack([norm, np.zeros((len(norm), 1))]).astype(np.float64),
        np.zeros(3), np.array([0.0, 0.0, 1.0]), K, D)[0].reshape(-1, 2)
    # the default undistortPoints stops after a few iterations and leaves ~0.2 px
    # behind on this lens; that residual is the algorithm, not the model
    crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-10)
    fixed = cv2.undistortPointsIter(
        bent.reshape(-1, 1, 2), K, D, None, None, crit).reshape(-1, 2)
    fixed = np.stack([fixed[:, 0] * K[0, 0] + K[0, 2],
                      fixed[:, 1] * K[1, 1] + K[1, 2]], axis=1)

    def bow(points):
        p = points - points.mean(0)
        return float(np.abs(p @ np.linalg.svd(p, full_matrices=False)[2][1]).max())

    assert bow(bent) > 3.0, "the lens model should visibly bend this line"
    assert bow(fixed) < 0.05, "undistortion should put it back"


def test_the_distortion_model_is_not_invertible_at_the_corners(camera):
    """Pins a limit of the calibration rather than a property of the code.

    Undistorting a point and distorting it again should return it to where it
    started. Near the image corners it does not: the radial polynomial is no
    longer monotonic there, so the inverse has no unique solution and the
    iterative solver walks away from it. On the 40 calibration frames this
    affects 4% of corners, all at large radius, one frame in forty.

    cv2.undistortPoints with its default iteration count does not diverge, which
    is worse rather than better: it stops early and returns a wrong answer that
    looks reasonable.
    """
    K, D, size = camera
    w, h = size
    crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-10)

    def roundtrip(px):
        pts = np.array([px], dtype=np.float64).reshape(-1, 1, 2)
        norm = cv2.undistortPointsIter(pts, K, D, None, None, crit).reshape(-1, 2)
        obj = np.hstack([norm, np.zeros((len(norm), 1))])
        back, _ = cv2.projectPoints(obj, np.zeros(3), np.array([0.0, 0.0, 1.0]), K, D)
        return float(np.linalg.norm(back.reshape(-1, 2) - pts.reshape(-1, 2)))

    assert roundtrip([w / 2, h / 2]) < 1e-6, "the centre must round-trip exactly"
    assert roundtrip([w * 0.75, h * 0.7]) < 0.5, "mid-field is fine"
    assert roundtrip([4.0, 4.0]) > 1.0, (
        "the top-left corner does not round-trip; anything metric that reaches "
        "the image border needs to know this"
    )
