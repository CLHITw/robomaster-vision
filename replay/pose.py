"""Where is the plate? Pose of an armor plate from its four light-bar endpoints.

This finishes, in Python, what the 2021 C++ `display()` started and never
compiled: given the four corners of an armor plate in the image, the physical
size of the plate and the camera calibration, `cv2.solvePnP` returns the
rotation and translation of the plate relative to the camera.

Two things to keep in mind when reading the numbers it produces:

* The plate dimensions are public RoboMaster figures (`config/armor_plates.json`),
  not measurements of the plates our team used. A wrong width scales every
  distance by the same factor.
* There is no ground-truth distance for the recording, so the module reports two
  self-checks instead: the reprojection error, and an independent pinhole
  estimate from the apparent width of the plate. They catch a wrong corner order
  or a wrong calibration, not a wrong plate size.
"""

import json
from pathlib import Path

import cv2
import numpy as np

CONFIG = Path(__file__).resolve().parents[1] / "config"


def load_camera(path=None):
    """Return (camera_matrix, distortion) as float64 arrays."""
    data = json.loads(Path(path or CONFIG / "camera_2021.json").read_text(encoding="utf-8"))
    return (np.array(data["camera_matrix"], dtype=np.float64),
            np.array(data["distortion"], dtype=np.float64).reshape(-1, 1))


def scale_camera(camera_matrix, scale):
    """Rescale intrinsics when the frame is resized: fx, fy, cx, cy all scale.

    Distortion coefficients are defined in normalised coordinates and do not.
    Forgetting this is an easy way to get distances that are wrong by exactly
    the resize factor.
    """
    scaled = camera_matrix.copy()
    scaled[:2, :] *= float(scale)
    return scaled


def load_plate(kind="small", path=None):
    """Return (width_mm, height_mm) of an armor plate."""
    data = json.loads(Path(path or CONFIG / "armor_plates.json").read_text(encoding="utf-8"))
    plate = data["plates"][kind]
    return float(plate["width"]), float(plate["height"])


def object_points(width_mm, height_mm):
    """The four plate corners in plate coordinates, z = 0, in the corner order below."""
    w, h = width_mm / 2.0, height_mm / 2.0
    return np.array([[-w, -h, 0.0],    # left bar, top
                     [-w, +h, 0.0],    # left bar, bottom
                     [+w, +h, 0.0],    # right bar, bottom
                     [+w, -h, 0.0]],   # right bar, top
                    dtype=np.float64)


def image_points(bar_a, bar_b):
    """The same four corners in the image, taken from the two light bars."""
    left, right = (bar_a, bar_b) if bar_a.cx <= bar_b.cx else (bar_b, bar_a)
    return np.array([left.top, left.bottom, right.bottom, right.top], dtype=np.float64)


def solve(corners, camera_matrix, distortion, width_mm, height_mm):
    """Solve the plate pose. Returns None if OpenCV cannot solve it.

    The result holds the translation in millimetres, the distance along the
    optical axis and in a straight line, the reprojection error in pixels, and
    an independent pinhole distance from the apparent width of the plate.
    """
    obj = object_points(width_mm, height_mm)
    img = np.ascontiguousarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None

    projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
    reprojection_px = float(np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1).mean())

    t = tvec.reshape(3)
    # apparent width between the two bar centres, in pixels
    left_mid = (corners[0] + corners[1]) / 2.0
    right_mid = (corners[2] + corners[3]) / 2.0
    width_px = float(np.linalg.norm(right_mid - left_mid))
    fx = float(camera_matrix[0, 0])
    pinhole_mm = fx * width_mm / width_px if width_px > 0 else float("nan")

    return {
        "tvec_mm": (float(t[0]), float(t[1]), float(t[2])),
        "rvec": tuple(float(v) for v in rvec.reshape(3)),
        "distance_mm": float(np.linalg.norm(t)),
        "depth_mm": float(t[2]),
        "reprojection_px": reprojection_px,
        "pinhole_mm": float(pinhole_mm),
        "width_px": width_px,
    }


def solve_armor(bar_a, bar_b, camera_matrix, distortion, width_mm, height_mm):
    """Convenience wrapper: two light bars in, pose out."""
    return solve(image_points(bar_a, bar_b), camera_matrix, distortion, width_mm, height_mm)
