"""A hand-held phone: how it rotated, which way is up, and its focal length (not part of the original code).

The floor reconstruction assumes one static camera. A phone held in the hand
pans and tilts while it films, and for a camera standing in one place that
motion is almost a pure rotation:

* **The rotation can be undone.** Between two frames of a rotating camera the
  static background moves by the homography ``H = K R K^-1``. Estimating ``H``
  from background features and taking ``R`` out of it turns every frame back
  into the view of the first one, so the rest of the pipeline sees a camera
  that did not move.
* **The room says which way is up.** Vertical edges (door frames, cabinet
  sides, chair legs) meet in one vanishing point, and ``K^-1`` of that point is
  the direction of gravity in camera coordinates: the floor's normal, without
  fitting it to the robot. One frame is not enough (a person or the robot can
  hide most of the edges), so every few frames vote, each rotated back into the
  view of frame 0.
* **The plates say what the focal length is.** The plate size is fixed by the
  competition rules, so PnP depth scales with the focal length while the
  vertical direction tilts with it in the other sense. Only one focal length
  puts every plate centre at the same height above the floor; that is the
  estimate, and the spread of the plate heights is its check.

A classic alternative is self-calibration from the rotation alone (Hartley,
1994): ``K^-1 H K`` is a rotation only for the right ``K``. It is kept here for
comparison because it fails on hand-held footage of a floor: most features are
on the nearby floor, and a hand that moves a few centimetres makes that
homography a plane-induced one rather than a rotation.
"""

import cv2
import numpy as np


def intrinsics(f, size):
    w, h = size
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def _features(gray, sift):
    kp, des = sift.detectAndCompute(gray, None)
    return np.float32([k.pt for k in kp]), des


def _homography(a, b, matcher, ratio=0.75, reproj_px=2.0):
    (pa, da), (pb, db) = a, b
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return None, 0, np.nan
    pairs = matcher.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 12:
        return None, 0, np.nan
    A = pa[[m.queryIdx for m in good]]
    B = pb[[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(A, B, cv2.RANSAC, reproj_px)
    if H is None:
        return None, 0, np.nan
    inl = mask.ravel().astype(bool)
    err = np.linalg.norm(cv2.perspectiveTransform(A[inl].reshape(-1, 1, 2), H).reshape(-1, 2) - B[inl], axis=1)
    return H, int(inl.sum()), float(np.median(err))


def background_homographies(video, n_features=3000, min_inliers=150):
    """Homography from frame 0 to every frame, from background features.

    Each frame is matched to frame 0 directly; when too few features survive
    (the view has moved on, or something covers the background) it is chained
    through the previous frame instead. Returns (H list, inlier counts,
    median inlier errors, frame size).
    """
    cap = cv2.VideoCapture(str(video))
    sift = cv2.SIFT_create(n_features)
    matcher = cv2.BFMatcher()
    ref = prev = None
    Hs, inliers, errors, size = [], [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        feat = _features(gray, sift)
        if ref is None:
            ref = prev = feat
            Hs.append(np.eye(3)); inliers.append(len(feat[0])); errors.append(0.0)
            continue
        H, n, e = _homography(ref, feat, matcher)
        if H is None or n < min_inliers:
            Hp, n, e = _homography(prev, feat, matcher)
            H = Hp @ Hs[-1] if Hp is not None else Hs[-1]
        Hs.append(H); inliers.append(n); errors.append(e)
        prev = feat
    cap.release()
    return Hs, np.array(inliers), np.array(errors), size


def rotation_spread(H, K):
    """How far ``K^-1 H K`` is from a rotation: spread of its log singular values."""
    M = np.linalg.inv(K) @ H @ K
    M = M / np.cbrt(np.linalg.det(M))
    return float(np.log(np.linalg.svd(M, compute_uv=False)).std())


def self_calibrate_focal(Hs, size, f_grid=None, min_angle_deg=3.0, rng=None, n_boot=200):
    """Focal length (px) that makes the frame-0-to-frame-k homographies rotations.

    Only pairs that rotated by at least ``min_angle_deg`` carry information.
    Returns (focal, (p5, p95) bootstrap interval, number of pairs used).
    """
    if f_grid is None:
        f_grid = np.arange(0.3, 3.0, 0.005) * max(size)
    rough = intrinsics(np.median(f_grid), size)
    pairs = []
    for H in Hs[1:]:
        R = nearest_rotation(np.linalg.inv(rough) @ H @ rough)
        if np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) >= min_angle_deg:
            pairs.append(H)
    if len(pairs) < 3:
        raise ValueError("the camera did not rotate enough to calibrate itself")
    cost = np.array([[rotation_spread(H, intrinsics(f, size)) for f in f_grid] for H in pairs])
    focal = float(f_grid[np.argmin(cost.mean(axis=0))])
    rng = rng or np.random.default_rng(0)
    boot = [f_grid[np.argmin(cost[rng.integers(0, len(pairs), len(pairs))].mean(axis=0))] for _ in range(n_boot)]
    return focal, (float(np.percentile(boot, 5)), float(np.percentile(boot, 95))), len(pairs)


def nearest_rotation(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R


def rotations(Hs, K):
    """Camera rotation of each frame relative to frame 0: x_k = R_k x_0 in camera coordinates."""
    Ki = np.linalg.inv(K)
    return [nearest_rotation(Ki @ H @ K) for H in Hs]


def to_reference(pixels, points_cam, R, K):
    """Undo one frame's rotation: its pixels and camera-frame points, as seen from frame 0."""
    H = K @ R.T @ np.linalg.inv(K)
    px = cv2.perspectiveTransform(np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3) @ R      # R^T applied to each row
    return px, pts


def vertical_vanishing_point(frame, min_length=40, max_tilt_deg=20, tol_deg=1.0, iters=3000, rng=None):
    """Vanishing point of the near-vertical line segments in a frame (homogeneous 3-vector).

    Segments from OpenCV's line segment detector, RANSAC over pairs, then a
    length-weighted least-squares point. Returns (vp, number of segments that
    agree, number of candidate segments).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    seg = cv2.createLineSegmentDetector().detect(gray)[0]
    if seg is None:
        raise ValueError("no line segments found")
    seg = seg.reshape(-1, 4)
    d = seg[:, 2:] - seg[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    tilt = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))
    keep = (length > min_length) & (tilt < max_tilt_deg)
    seg, d, length = seg[keep], d[keep], length[keep]
    if len(seg) < 3:
        raise ValueError("fewer than three vertical segments")
    lines = np.cross(np.column_stack([seg[:, :2], np.ones(len(seg))]),
                     np.column_stack([seg[:, 2:], np.ones(len(seg))]))
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    mid = (seg[:, :2] + seg[:, 2:]) / 2
    rng = rng or np.random.default_rng(0)

    def agree(v):
        to_vp = v[:2] - mid * v[2]
        c = np.abs((to_vp * d).sum(1)) / (np.linalg.norm(to_vp, axis=1) * length + 1e-12)
        return np.degrees(np.arccos(np.clip(c, 0, 1))) < tol_deg

    best, best_score = None, -1.0
    for _ in range(iters):
        i, j = rng.choice(len(lines), 2, replace=False)
        inl = agree(np.cross(lines[i], lines[j]))
        if length[inl].sum() > best_score:
            best, best_score = inl, length[inl].sum()
    vp = np.linalg.svd(lines[best] * length[best, None])[2][-1]
    return vp, int(best.sum()), int(len(seg))


def up_direction(vp, K):
    """Unit 'up' vector in camera coordinates from the vertical vanishing point (camera y points down)."""
    down = np.linalg.inv(K) @ vp
    down /= np.linalg.norm(down)
    return -down if down[1] > 0 else down


def vertical_points(video, every=5, min_agree=8):
    """Vertical vanishing point of every ``every``-th frame where enough segments agree: {frame: vp}."""
    cap = cv2.VideoCapture(str(video))
    out, i = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            try:
                vp, n, _ = vertical_vanishing_point(frame)
                if n >= min_agree:
                    out[i] = vp
            except ValueError:
                pass
        i += 1
    cap.release()
    return out


def up_from_verticals(vps, frame_rotations, K, rounds=3):
    """Robust mean of the per-frame 'up' directions, in the coordinates of frame 0.

    Returns (up, median angle of the frames to it in degrees, frames used).
    """
    U = np.array([frame_rotations[k].T @ up_direction(v, K) for k, v in vps.items()])
    u = np.median(U, axis=0)
    u /= np.linalg.norm(u)
    keep = np.ones(len(U), dtype=bool)
    for _ in range(rounds):
        ang = np.degrees(np.arccos(np.clip(U @ u, -1, 1)))
        keep = ang < 3 * max(1.4826 * np.median(ang), 0.3)
        u = U[keep].mean(axis=0)
        u /= np.linalg.norm(u)
    ang = np.degrees(np.arccos(np.clip(U[keep] @ u, -1, 1)))
    return u, float(np.median(ang)), int(keep.sum())
