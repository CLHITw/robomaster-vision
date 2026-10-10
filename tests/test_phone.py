"""Synthetic checks for the hand-held phone path: rotation, up direction, focal length, bright bars, timing."""

import cv2
import numpy as np
import pytest

from replay.egomotion import (intrinsics, rotations, self_calibrate_focal, to_reference, up_direction,
                              up_from_verticals, vertical_vanishing_point)
from replay.pipelines import detect_armors
from replay.trajectory import FloorTracker, fit_floor

SIZE = (720, 1280)
K = intrinsics(1000.0, SIZE)


def rot(axis, deg):
    v = np.asarray(axis, float)
    return cv2.Rodrigues(v / np.linalg.norm(v) * np.radians(deg))[0]


# ----------------------------------------------------------------------------- the floor, from a straight drive

def test_known_up_fixes_the_floor_when_the_robot_drives_in_a_straight_line():
    rng = np.random.default_rng(0)
    up = rot([1, 0, 0], 25) @ np.array([0.0, -1.0, 0.0])         # camera pitched 25 degrees down
    height = 700.0
    s = np.linspace(1500, 4500, 150)                               # a straight line on the floor
    fwd = np.cross(up, [1.0, 0, 0]); fwd /= np.linalg.norm(fwd)
    side = np.cross(fwd, up)
    pts = -height * up + np.outer(s, fwd + 0.4 * side) + rng.normal(0, 20, (150, 1)) * up

    floor = fit_floor(pts, up=up)
    assert floor.camera_height_mm == pytest.approx(height, abs=10)
    assert floor.pitch_deg == pytest.approx(25.0, abs=1e-6)


# ----------------------------------------------------------------------------- rotation and up

def test_rotation_is_recovered_and_undone():
    R = rot([0.3, 1.0, 0.1], 8.0)
    H = K @ R @ np.linalg.inv(K)
    (R0, R1) = rotations([np.eye(3), H], K)
    assert np.allclose(R0, np.eye(3))
    assert np.allclose(R1, R, atol=1e-9)

    point_ref = np.array([300.0, 200.0, 2500.0])
    point_k = R @ point_ref
    px_k = (K @ point_k)[:2] / point_k[2]
    px_ref = (K @ point_ref)[:2] / point_ref[2]
    px, pts = to_reference([px_k], [point_k], R, K)
    assert np.allclose(px[0], px_ref, atol=1e-6)
    assert np.allclose(pts[0], point_ref, atol=1e-6)


def verticals_image(vp, n=14, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((SIZE[1], SIZE[0]), 40, np.uint8)
    for _ in range(n):
        x0, y0 = rng.uniform(60, 660), rng.uniform(80, 500)
        d = np.array(vp) - np.array([x0, y0])
        d /= np.linalg.norm(d)
        end = np.array([x0, y0]) + d * rng.uniform(150, 350)
        cv2.line(img, (int(round(x0)), int(round(y0))), (int(round(end[0])), int(round(end[1]))), 230, 3)
    return img


def test_vertical_vanishing_point_gives_the_direction_of_gravity():
    true_vp = np.array([420.0, 3200.0])
    vp, agree, found = vertical_vanishing_point(verticals_image(true_vp))
    assert agree >= 8
    expected = up_direction(np.array([*true_vp, 1.0]), K)
    angle = np.degrees(np.arccos(np.clip(up_direction(vp, K) @ expected, -1, 1)))
    assert angle < 0.5


def test_up_votes_from_rotated_frames_agree_in_the_reference_view():
    up0 = up_direction(np.array([400.0, 3000.0, 1.0]), K)
    Rs = [np.eye(3), rot([1, 0, 0], 5), rot([0, 1, 0], -7), rot([0.2, 1, 0], 4)]
    vps = {}
    for k, R in enumerate(Rs):
        down_k = -(R @ up0)
        vps[k] = K @ down_k                                     # where that frame sees the verticals meet
    vps[4] = K @ -(rot([1, 0, 0], 15) @ up0)                    # one frame whose estimate is off
    Rs.append(np.eye(3))
    up, spread, used = up_from_verticals(vps, Rs, K)
    assert np.degrees(np.arccos(np.clip(up @ up0, -1, 1))) < 0.1
    assert used == 4                                            # the outlier is voted out


def test_rotation_self_calibration_works_for_a_camera_that_only_turns():
    f_true = 1000.0
    Kt = intrinsics(f_true, SIZE)
    Hs = [np.eye(3)] + [Kt @ rot([0.2, 1, 0.05], a) @ np.linalg.inv(Kt) for a in np.linspace(2, 10, 20)]
    f, _, pairs = self_calibrate_focal(Hs, SIZE)
    assert f == pytest.approx(f_true, rel=0.02)
    assert pairs >= 15


# ----------------------------------------------------------------------------- overexposed light bars

def bright_bar_frame():
    img = np.zeros((600, 800, 3), np.uint8)
    img[:] = (30, 30, 30)
    cv2.circle(img, (150, 450), 70, (20, 20, 220), -1)          # red glow on the floor, no white core
    for x in (370, 470):                                        # two bars of one plate: white core, red halo
        cv2.rectangle(img, (x - 9, 255), (x + 9, 345), (60, 60, 170), -1)   # dimmer than the glow
        cv2.rectangle(img, (x - 4, 260), (x + 4, 340), (255, 255, 255), -1)
    return img


def test_saturated_variant_finds_bars_with_white_cores():
    frame = bright_bar_frame()
    _, armors, _ = detect_armors(frame, "red", "saturated")
    assert len(armors) == 1
    assert armors[0].cx == pytest.approx(420, abs=3)
    assert armors[0].cy == pytest.approx(300, abs=3)


def test_original_threshold_follows_the_glow_instead():
    _, armors, binary = detect_armors(bright_bar_frame(), "red", "original")
    assert len(armors) == 0
    assert binary[450, 150] == 255                              # it lit up the glow


# ----------------------------------------------------------------------------- timestamps

def test_timestamps_keep_the_speed_right_across_a_dropped_frame():
    v = np.array([1500.0, 0.0])                                 # 1.5 m/s
    t = np.arange(60) / 30.0
    t[30:] += 1 / 30.0                                          # one frame dropped: the clock jumps
    pos = np.outer(t, v) + np.array([0.0, 3000.0])
    R = np.eye(2) * 5.0 ** 2

    def final_speed(use_time):
        tr = FloorTracker(30.0, accel_std_mm_s2=500.0)
        for f in range(60):
            tr.step(f, [(pos[f], R)], t=t[f] if use_time else None)
        (track,) = tr.finish()
        return np.hypot(*np.array(track.x_smooth)[-1][2:])

    assert final_speed(True) == pytest.approx(1500.0, rel=0.02)
    assert abs(final_speed(False) - 1500.0) > abs(final_speed(True) - 1500.0)
