"""Synthetic checks for the pose solver: place a plate, project it, solve it back."""

import cv2
import numpy as np
import pytest

from replay.pipelines import LightBar
from replay.pose import image_points, load_camera, load_plate, object_points, solve, solve_armor

K, DIST = load_camera()
WIDTH, HEIGHT = load_plate("small")


def project(rvec, tvec):
    obj = object_points(WIDTH, HEIGHT)
    pts, _ = cv2.projectPoints(obj, np.array(rvec, float), np.array(tvec, float), K, DIST)
    return pts.reshape(-1, 2)


@pytest.mark.parametrize("distance_mm", [1500.0, 3000.0, 6000.0])
def test_recovers_a_head_on_plate(distance_mm):
    corners = project([0.0, 0.0, 0.0], [0.0, 0.0, distance_mm])
    r = solve(corners, K, DIST, WIDTH, HEIGHT)

    assert r["depth_mm"] == pytest.approx(distance_mm, rel=1e-3)
    assert r["reprojection_px"] < 0.1
    # the independent pinhole estimate agrees when the plate faces the camera
    assert r["pinhole_mm"] == pytest.approx(distance_mm, rel=0.02)


def test_recovers_a_turned_plate():
    corners = project([0.0, 0.6, 0.0], [300.0, -100.0, 2500.0])  # yawed 34 degrees, off centre
    r = solve(corners, K, DIST, WIDTH, HEIGHT)

    assert r["tvec_mm"][0] == pytest.approx(300.0, abs=5)
    assert r["tvec_mm"][2] == pytest.approx(2500.0, rel=5e-3)
    assert r["reprojection_px"] < 0.1
    # turned away, the plate looks narrower, so the pinhole estimate reads too far
    assert r["pinhole_mm"] > r["depth_mm"]


def test_wrong_plate_width_scales_the_distance():
    corners = project([0.0, 0.0, 0.0], [0.0, 0.0, 3000.0])
    half = solve(corners, K, DIST, WIDTH / 2, HEIGHT / 2)
    assert half["depth_mm"] == pytest.approx(1500.0, rel=1e-3)


def test_image_points_order_follows_the_left_bar():
    left = LightBar(100.0, 200.0, 8.0, 50.0, 5.0, top=(100.0, 175.0), bottom=(100.0, 225.0))
    right = LightBar(300.0, 200.0, 8.0, 50.0, 5.0, top=(300.0, 175.0), bottom=(300.0, 225.0))

    for a, b in ((left, right), (right, left)):          # order of arguments must not matter
        pts = image_points(a, b)
        assert pts[0].tolist() == [100.0, 175.0]          # left top
        assert pts[1].tolist() == [100.0, 225.0]          # left bottom
        assert pts[2].tolist() == [300.0, 225.0]          # right bottom
        assert pts[3].tolist() == [300.0, 175.0]          # right top


def test_solve_armor_matches_solve_on_the_same_corners():
    left = LightBar(560.0, 512.0, 8.0, 60.0, 3.0, top=(560.0, 482.0), bottom=(560.0, 542.0))
    right = LightBar(720.0, 512.0, 8.0, 60.0, 3.0, top=(720.0, 482.0), bottom=(720.0, 542.0))
    direct = solve(image_points(left, right), K, DIST, WIDTH, HEIGHT)
    wrapped = solve_armor(left, right, K, DIST, WIDTH, HEIGHT)
    assert wrapped["depth_mm"] == pytest.approx(direct["depth_mm"])
    assert direct["depth_mm"] > 0


def test_rescaling_the_frame_keeps_the_distance():
    """Resizing the frame must not move the plate: intrinsics scale with it."""
    from replay.pose import scale_camera

    corners = project([0.0, 0.3, 0.0], [100.0, 50.0, 3000.0])
    full = solve(corners, K, DIST, WIDTH, HEIGHT)
    for s in (0.5625, 0.5, 2.0):
        small = solve(corners * s, scale_camera(K, s), DIST, WIDTH, HEIGHT)
        assert small["depth_mm"] == pytest.approx(full["depth_mm"], rel=1e-3)


def test_forgetting_to_rescale_is_wrong_by_the_resize_factor():
    """The failure mode the rescaling guards against, pinned so it stays visible."""
    corners = project([0.0, 0.0, 0.0], [0.0, 0.0, 3000.0])
    s = 0.5
    wrong = solve(corners * s, K, DIST, WIDTH, HEIGHT)      # unscaled intrinsics
    assert wrong["depth_mm"] == pytest.approx(3000.0 / s, rel=0.05)
