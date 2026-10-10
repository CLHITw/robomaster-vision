"""From plate positions in the camera frame to a trajectory on the floor.

PnP gives the armor plate in camera coordinates, one frame at a time. This
module turns those into a top-down track in metres:

1. **The floor, from the robot itself.** The robot drives on a flat floor, so
   its plate centres stay at one height and lie on one plane. A robust plane
   fit over the PnP positions gives the floor's orientation and the camera's
   height above the plates, with no marker or survey needed. (DeepScenario's
   GroundMix fits its ground plane from object ground centres in the same
   spirit.)
2. **Position from the ray, not from PnP depth.** PnP depth comes from how big
   the plate looks, and a 135 x 55 mm plate a few metres away is only tens of
   pixels across; when it turns, its width foreshortens and planar PnP cannot
   tell which way it turned. On this footage that leaves PnP depth scattered
   by about half a metre at 4-5 m. Once the plane is known, the pixel alone
   fixes the position: the viewing ray through the plate centre meets the
   plane once. With the camera 0.6 m above the plates, a pixel of vertical
   error moves that point by about a centimetre at 4 m. This is the step
   DeepScenario's pipeline takes when it replaces the detector's depth with
   the intersection of a ray and the reconstructed ground.
3. **Noise that has a direction.** Across the line of sight a pixel is
   ``r / f``; along it the error grows as the ray grazes the plane. The
   measurement covariance carries that shape, so the filter trusts each
   direction for what it is.
4. **Filter forward, smooth backward.** A constant-velocity Kalman filter on
   the floor gates and associates the candidates of each frame; a
   Rauch-Tung-Striebel pass then smooths each track using the frames after as
   well as before, which also fills short detection gaps.
5. **The robot, not the plate.** The robot in this footage spins while it
   drives, so the visible plate circles the chassis. ``robot_path`` runs a
   second, physically slow filter over the plate track that treats the
   circling as noise and keeps the translation.

What it cannot do exactly: put the track on the robot's centre. That needs the
plate's facing direction, and at this range planar PnP returns two mirror
solutions of almost equal error, so the direction is not observable from one
small plate. Only camera-facing plates are seen, so the robot path sits
towards the camera from the true centre by up to a chassis radius.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

CHI2_2DOF_99 = 9.21


# ----------------------------------------------------------------------------- floor plane

@dataclass
class Floor:
    """The plane of the plate centres, and a floor coordinate frame on it.

    Floor frame: origin directly below the camera on the plane, X to the right,
    Y forward (the optical axis projected onto the plane), Z up.
    """
    up: np.ndarray            # unit normal, pointing from the plane towards the camera side
    offset: float             # up . P on the plane (negative: the plane is below the camera)
    right: np.ndarray
    forward: np.ndarray
    residual_mm: float        # robust spread of the points about the plane (1.4826 * MAD)
    inliers: np.ndarray       # mask over the points the plane was fitted to

    @property
    def camera_height_mm(self):
        return -self.offset

    @property
    def pitch_deg(self):
        """How far the optical axis points down from the plane."""
        return float(np.degrees(np.arcsin(-self.up[2])))

    def to_floor(self, points_cam):
        """Camera-frame points (N x 3, mm) to floor coordinates (N x 3, mm)."""
        p = np.atleast_2d(np.asarray(points_cam, dtype=float))
        return np.column_stack([p @ self.right, p @ self.forward, p @ self.up - self.offset])

    def to_camera(self, points_floor):
        """Floor coordinates (N x 2 or N x 3, mm) back to the camera frame."""
        p = np.atleast_2d(np.asarray(points_floor, dtype=float))
        z = p[:, 2] if p.shape[1] > 2 else np.zeros(len(p))
        return (np.outer(p[:, 0], self.right) + np.outer(p[:, 1], self.forward)
                + np.outer(z + self.offset, self.up))


def _plane(points):
    centre = points.mean(axis=0)
    normal = np.linalg.svd(points - centre)[2][2]
    return normal, centre


def fit_floor(points_cam, rounds=4, k=3.0):
    """Robust plane through camera-frame points: least squares, then refit on inliers.

    Each round keeps the points within ``k`` robust standard deviations of the
    current plane. Raises ValueError with fewer than three usable points.
    """
    p = np.asarray(points_cam, dtype=float)
    if len(p) < 3:
        raise ValueError("a plane needs at least three points")
    mask = np.ones(len(p), dtype=bool)
    for _ in range(rounds):
        normal, centre = _plane(p[mask])
        dist = (p - centre) @ normal
        spread = 1.4826 * np.median(np.abs(dist[mask] - np.median(dist[mask])))
        new = np.abs(dist) <= k * max(spread, 1e-9)
        if new.sum() < 3 or np.array_equal(new, mask):
            break
        mask = new
    normal, centre = _plane(p[mask])
    if normal[1] > 0:                  # camera y points down; the plane normal should point up
        normal = -normal
    offset = float(centre @ normal)
    forward = np.array([0.0, 0.0, 1.0]) - normal[2] * normal
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, normal)
    dist = (p - centre) @ normal
    residual = 1.4826 * float(np.median(np.abs(dist[mask])))
    return Floor(normal, offset, right, forward, residual, mask)


# ----------------------------------------------------------------------------- measurement

UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-10)


def rays(pixels, camera_matrix, distortion):
    """Pixels (N x 2) to viewing-ray directions in the camera frame (N x 3, z = 1).

    ``undistortPointsIter`` with explicit criteria: the default iteration
    count of ``undistortPoints`` does not converge on strong distortion.
    """
    px = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPointsIter(px, camera_matrix, distortion, None, None,
                                   UNDISTORT_CRITERIA).reshape(-1, 2)
    return np.column_stack([norm, np.ones(len(norm))])


def ray_to_floor(pixels, camera_matrix, distortion, floor):
    """Where each pixel's viewing ray meets the plate plane.

    Returns (camera-frame points N x 3 in mm, valid mask); a ray that points
    above the horizon never meets the plane and is marked invalid.
    """
    d = rays(pixels, camera_matrix, distortion)
    denom = d @ floor.up
    valid = denom < -1e-9                      # the ray has to go down towards the plane
    s = np.where(valid, floor.offset / np.where(valid, denom, -1.0), np.nan)
    return d * s[:, None], valid


def measurement_cov(point_floor_xy, floor, fx, pixel_std=1.0, height_std_mm=10.0):
    """2 x 2 covariance (mm^2), in floor X/Y, of a ray-plane intersection.

    With the camera ``h`` above the plane and the point at horizontal range
    ``r``, the ray meets the plane at depression ``atan(h / r)``. A vertical
    pixel error changes that angle by ``pixel_std / fx`` and the range by
    ``(r**2 + h**2) / h`` times as much; a plate that sits ``dh`` off the plane
    moves it by ``r / h * dh``. Across the line of sight a pixel is just
    ``sqrt(r**2 + h**2) / fx``.
    """
    xy = np.asarray(point_floor_xy, dtype=float)
    r = float(np.hypot(*xy))
    h = floor.camera_height_mm
    s_along = np.hypot((r ** 2 + h ** 2) / h * pixel_std / fx, r / h * height_std_mm)
    s_across = np.hypot(r, h) * pixel_std / fx
    a = xy / r
    b = np.array([-a[1], a[0]])
    return s_along ** 2 * np.outer(a, a) + s_across ** 2 * np.outer(b, b)


# ----------------------------------------------------------------------------- filter and smoother

class CVKalmanFloor:
    """State [X, Y, vX, vY] in mm and mm/s; white-acceleration process noise."""

    def __init__(self, z, R, accel_std_mm_s2=3000.0, speed_std_mm_s=2000.0):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.zeros((4, 4))
        self.P[:2, :2] = R
        self.P[2, 2] = self.P[3, 3] = speed_std_mm_s ** 2
        self.q = accel_std_mm_s2
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    @staticmethod
    def transition(dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        return F

    def process_noise(self, dt):
        g = np.array([[0.5 * dt ** 2, 0], [0, 0.5 * dt ** 2], [dt, 0], [0, dt]])
        return g @ g.T * self.q ** 2

    def predict(self, dt):
        F = self.transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.process_noise(dt)
        return F

    def innovation(self, z, R):
        S = self.H @ self.P @ self.H.T + R
        d = np.asarray(z, dtype=float) - self.H @ self.x
        return d, S

    def mahalanobis2(self, z, R):
        d, S = self.innovation(z, R)
        return float(d @ np.linalg.solve(S, d))

    def update(self, z, R):
        d, S = self.innovation(z, R)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ d
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form, stays symmetric


def rts_smooth(xs_filt, Ps_filt, xs_pred, Ps_pred, Fs):
    """Rauch-Tung-Striebel backward pass.

    ``xs_pred[k], Ps_pred[k]`` is the prediction for step k made with ``Fs[k]``
    from step k-1; index 0 is unused. Returns smoothed states and covariances.
    """
    n = len(xs_filt)
    xs = [None] * n
    Ps = [None] * n
    xs[-1], Ps[-1] = xs_filt[-1].copy(), Ps_filt[-1].copy()
    for k in range(n - 2, -1, -1):
        C = Ps_filt[k] @ Fs[k + 1].T @ np.linalg.inv(Ps_pred[k + 1])
        xs[k] = xs_filt[k] + C @ (xs[k + 1] - xs_pred[k + 1])
        Ps[k] = Ps_filt[k] + C @ (Ps[k + 1] - Ps_pred[k + 1]) @ C.T
    return xs, Ps


# ----------------------------------------------------------------------------- tracking

@dataclass
class Track:
    """One continuous track: per frame, the filtered and smoothed floor state."""
    frames: list = field(default_factory=list)
    measured: list = field(default_factory=list)      # True where a measurement was used
    z: list = field(default_factory=list)              # the measurement used, or None
    x_filt: list = field(default_factory=list)
    P_filt: list = field(default_factory=list)
    x_pred: list = field(default_factory=list)
    P_pred: list = field(default_factory=list)
    F: list = field(default_factory=list)
    nis: list = field(default_factory=list)            # normalised innovation squared, per update
    x_smooth: list = field(default_factory=list)
    P_smooth: list = field(default_factory=list)

    def trim_and_smooth(self):
        """Drop the coasting tail after the last measurement, then run the smoother."""
        last = max(i for i, m in enumerate(self.measured) if m)
        for name in ("frames", "measured", "z", "x_filt", "P_filt", "x_pred", "P_pred", "F"):
            setattr(self, name, getattr(self, name)[:last + 1])
        self.x_smooth, self.P_smooth = rts_smooth(self.x_filt, self.P_filt,
                                                  self.x_pred, self.P_pred, self.F)
        return self


def robot_path(track, fps, accel_std_mm_s2=5000.0, spin_std_mm=150.0):
    """The robot's own path, from the plate measurements of one track.

    The robot in this footage spins in place while it drives, so the visible
    plate circles the chassis centre several times a second. A second filter
    with a physical process noise treats that circling as measurement noise of
    ``spin_std_mm`` (about the chassis radius) and smooths it out, leaving the
    translation. Only the camera-facing plates are ever visible, so the result
    sits towards the camera from the true centre by up to a chassis radius.

    Returns (frames, smoothed states N x 4).
    """
    R = np.eye(2) * spin_std_mm ** 2
    dt = 1.0 / float(fps)
    kf, frames = None, []
    xs_f, Ps_f, xs_p, Ps_p, Fs = [], [], [None], [None], [None]
    for f, z in zip(track.frames, track.z):
        if kf is None:
            if z is None:
                continue
            kf = CVKalmanFloor(z, R, accel_std_mm_s2)
        else:
            Fs.append(kf.predict(dt))
            xs_p.append(kf.x.copy())
            Ps_p.append(kf.P.copy())
            if z is not None:
                kf.update(z, R)
        frames.append(f)
        xs_f.append(kf.x.copy())
        Ps_f.append(kf.P.copy())
    xs, _ = rts_smooth(xs_f, Ps_f, xs_p, Ps_p, Fs)
    return np.array(frames), np.array(xs)


class FloorTracker:
    """Single-target tracker on the floor, fed with every plate candidate per frame.

    A track starts from a frame with exactly one candidate, takes the candidate
    with the smallest Mahalanobis distance inside the chi-square gate, coasts
    through frames with none, and ends after ``max_gap`` frames without one.
    Ended tracks are trimmed to their last measurement and smoothed.
    """

    def __init__(self, fps, gate=CHI2_2DOF_99, max_gap=12,
                 accel_std_mm_s2=3000.0, speed_std_mm_s=2000.0):
        self.dt = 1.0 / float(fps)
        self.gate = gate
        self.max_gap = max_gap
        self.accel = accel_std_mm_s2
        self.speed = speed_std_mm_s
        self.kf = None
        self.track = None
        self.gap = 0
        self.done = []

    def _close(self):
        if self.track is not None and any(self.track.measured):
            self.done.append(self.track.trim_and_smooth())
        self.kf, self.track, self.gap = None, None, 0

    def step(self, frame, candidates):
        """``candidates``: list of (z_xy, R) pairs in floor millimetres."""
        if self.kf is not None:
            F = self.kf.predict(self.dt)
            x_pred, P_pred = self.kf.x.copy(), self.kf.P.copy()
            scored = sorted((self.kf.mahalanobis2(z, R), k) for k, (z, R) in enumerate(candidates))
            used = None
            if scored and scored[0][0] < self.gate:
                used = candidates[scored[0][1]]
                self.track.nis.append(scored[0][0])
                self.kf.update(*used)
                self.gap = 0
            else:
                self.gap += 1
            t = self.track
            t.frames.append(frame)
            t.measured.append(used is not None)
            t.z.append(None if used is None else np.asarray(used[0], dtype=float))
            t.x_pred.append(x_pred)
            t.P_pred.append(P_pred)
            t.F.append(F)
            t.x_filt.append(self.kf.x.copy())
            t.P_filt.append(self.kf.P.copy())
            if self.gap > self.max_gap:
                self._close()
            return

        if len(candidates) == 1:
            z, R = candidates[0]
            self.kf = CVKalmanFloor(z, R, self.accel, self.speed)
            self.track = Track(frames=[frame], measured=[True], z=[np.asarray(z, dtype=float)],
                               x_filt=[self.kf.x.copy()], P_filt=[self.kf.P.copy()],
                               x_pred=[None], P_pred=[None], F=[None])
            self.gap = 0

    def finish(self):
        self._close()
        return self.done
