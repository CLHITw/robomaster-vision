"""Constant-velocity Kalman tracking of the armor centre (2026 addition).

Not part of the 2021 code. The idea (predict, gate, nearest-neighbour
association, coast through missed frames) follows the tracking modules used by
later HITCRT vision code; this is a minimal 2-D image-plane version.
"""

from dataclasses import dataclass

import numpy as np

CHI2_2DOF_99 = 9.21


class CVKalman2D:
    """State [x, y, vx, vy] in pixels and pixels/frame, dt = 1 frame."""

    def __init__(self, z, accel_std=4.0, meas_std=15.0):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([meas_std ** 2, meas_std ** 2, 50.0 ** 2, 50.0 ** 2])
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        g = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]])
        self.Q = g @ g.T * accel_std ** 2
        self.R = np.eye(2) * meas_std ** 2

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def mahalanobis2(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        d = np.asarray(z, dtype=float) - self.H @ self.x
        return float(d @ np.linalg.solve(S, d))

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z, dtype=float) - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P


@dataclass
class TrackStep:
    state: str             # "none" | "init" | "update" | "coast" | "lost"
    measurement: tuple     # (x, y) used for the update, or None
    estimate: tuple        # (x, y) after this step, or None


class ArmorTracker:
    """Single-target tracker fed with the armor candidates of each frame.

    - A track starts only from a frame with exactly one candidate.
    - Each frame, the candidate closest to the prediction (Mahalanobis
      distance) is used if it falls inside the gate; otherwise the track coasts.
    - After ``max_coast`` consecutive coasting frames the track is dropped.
    """

    def __init__(self, gate=CHI2_2DOF_99, max_coast=15, accel_std=4.0, meas_std=15.0):
        self.gate = gate
        self.max_coast = max_coast
        self.accel_std = accel_std
        self.meas_std = meas_std
        self.kf = None
        self.coast = 0

    def step(self, candidates):
        candidates = [tuple(map(float, c)) for c in candidates]
        state, used = "none", None

        if self.kf is not None:
            self.kf.predict()
            scored = sorted((self.kf.mahalanobis2(z), k) for k, z in enumerate(candidates))
            if scored and scored[0][0] < self.gate:
                used = candidates[scored[0][1]]
                self.kf.update(used)
                self.coast, state = 0, "update"
            else:
                self.coast += 1
                state = "coast"
                if self.coast > self.max_coast:
                    self.kf, state = None, "lost"

        if self.kf is None and len(candidates) == 1:
            self.kf = CVKalman2D(candidates[0], self.accel_std, self.meas_std)
            self.coast, used, state = 0, candidates[0], "init"

        estimate = tuple(self.kf.x[:2]) if self.kf is not None else None
        return TrackStep(state, used, estimate)
