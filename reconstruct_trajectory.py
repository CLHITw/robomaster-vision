"""Reconstruct the robot's path on the floor, in metres, and check it against the labels.

    python reconstruct_trajectory.py clips/robot_clip.mp4
    python reconstruct_trajectory.py video2.avi --figure docs/figures/trajectory.png
    python reconstruct_trajectory.py phone.mp4 --enemy red --phone     # hand-held, uncalibrated

``--phone`` is for a hand-held phone recording with no calibration: the focal
length is estimated from the camera's own rotation, every frame is rotated back
into the view of the first one, and the light bars are found by the variant
that copes with overexposed bar cores (replay/egomotion.py,
replay/pipelines.py). Times always come from the video's own frame timestamps.

Every armor candidate of every frame is solved with PnP, the floor plane is
fitted to the plate centres themselves, candidates that do not sit on that
plane are dropped, and a Kalman filter on the floor followed by an RTS smoother
turns the rest into tracks (replay/trajectory.py explains each step).

There is no ground truth in metres for this footage. What can be checked is
where the reconstructed plate lands in the image: the smoothed 3-D track is
projected back through the calibration and compared with the plates clicked
by hand, including on frames where the detector found nothing and the
position comes from the smoother alone.

Writes outputs/<video>/candidates.csv, frame_times.csv, trajectory.csv,
trajectory.png and trajectory.json (and camera_selfcal.json with --phone).
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import load_labels
from replay.egomotion import (background_homographies, intrinsics, rotations, self_calibrate_focal,
                              to_reference, up_from_verticals, vertical_points)
from replay.pipelines import detect_armors
from replay.pose import image_points, load_camera, load_plate, scale_camera, solve
from replay.trajectory import FloorTracker, fit_floor, measurement_cov, ray_to_floor, robot_path

CONFIG_DIR = Path(__file__).resolve().parent / "config"


# ----------------------------------------------------------------------------- pass 1: candidates

def detect_all(video, enemy, preprocess="original"):
    """Every armor candidate of every frame: (frame, four corners, centre), plus frame timestamps (s)."""
    cap = cv2.VideoCapture(str(video))
    dets, times, first = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if first is None:
            first = frame
        frame_id = len(times)
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        bars, armors, _ = detect_armors(frame, enemy, preprocess)
        for a in armors:
            dets.append((frame_id, image_points(bars[a.bar_i], bars[a.bar_j]), (a.cx, a.cy)))
    cap.release()
    return dets, np.array(times), first


def solve_candidates(dets, camera_matrix, distortion, width_mm, height_mm, frame_rotations=None):
    """PnP for every candidate.

    With ``frame_rotations`` each candidate is also rotated into the view of
    frame 0; ``cx, cy, x_mm, y_mm, z_mm`` are then reference-view values and
    ``u_px, v_px`` keep where it was in its own frame.
    """
    rows = []
    for frame_id, corners, (cx, cy) in dets:
        r = solve(corners, camera_matrix, distortion, width_mm, height_mm)
        if r is None:
            continue
        centre, point = np.array([cx, cy]), np.array(r["tvec_mm"])
        if frame_rotations is not None:
            c, pt = to_reference([centre], [point], frame_rotations[frame_id], camera_matrix)
            centre, point = c[0], pt[0]
        rows.append({"frame": frame_id, "cx": round(float(centre[0]), 2), "cy": round(float(centre[1]), 2),
                     "x_mm": round(float(point[0]), 2), "y_mm": round(float(point[1]), 2),
                     "z_mm": round(float(point[2]), 2),
                     "u_px": round(float(cx), 2), "v_px": round(float(cy), 2),
                     "reprojection_px": round(r["reprojection_px"], 3),
                     "width_px": round(r["width_px"], 2)})
    return rows


def focal_from_plates(dets, Hs, vps, size, width_mm, height_mm, grid):
    """The focal length that puts every plate centre at one height above the floor.

    For each candidate focal length: undo the camera rotation, take 'up' from
    the frames' vertical vanishing points, solve every plate with PnP, and
    measure the robust spread of the plate heights. Returns (focal, spread
    curve, focal from the first and from the second half of the frames).
    """
    split = np.median([d[0] for d in dets])

    def spreads(sel):
        out = []
        for f in grid:
            K = intrinsics(f, size)
            Rs = rotations(Hs, K)
            up, _, _ = up_from_verticals(vps, Rs, K)
            h = []
            for frame_id, corners, _ in dets:
                if not sel(frame_id):
                    continue
                r = solve(corners, K, np.zeros((5, 1)), width_mm, height_mm)
                if r is not None:
                    h.append(np.array(r["tvec_mm"]) @ Rs[frame_id] @ up)
            h = np.array(h)
            out.append(1.4826 * float(np.median(np.abs(h - np.median(h)))))
        return np.array(out)

    curve = spreads(lambda f: True)
    halves = [float(grid[np.argmin(spreads(sel))]) for sel in (lambda f: f < split, lambda f: f >= split)]
    return float(grid[np.argmin(curve)]), curve, halves


def read_candidates(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["frame"] = int(r["frame"])
        for k in ("cx", "cy", "x_mm", "y_mm", "z_mm", "u_px", "v_px", "reprojection_px", "width_px"):
            r[k] = float(r[k])
    return rows


# ----------------------------------------------------------------------------- pass 2: floor and tracks

def reconstruct(rows, times, camera_matrix, distortion, pixel_std, height_std, accel, max_gap,
                consistency_k, hide=None, up=None):
    """Floor plane, ray-plane measurements, consistency gate, tracks.

    ``hide``: optional set of frames whose measurements are withheld from the
    tracker (for the hold-out check). ``up``: the floor normal in camera
    coordinates, if known from elsewhere; otherwise it is fitted.
    """
    n_frames = len(times)
    fps = 1.0 / float(np.median(np.diff(times))) if n_frames > 1 else 30.0
    fx = float(camera_matrix[0, 0])
    pts = np.array([[r["x_mm"], r["y_mm"], r["z_mm"]] for r in rows])
    frames = np.array([r["frame"] for r in rows])

    # fit the plane on frames with a single candidate, where mis-paired light bars are rarest
    counts = np.bincount(frames, minlength=n_frames)
    floor = fit_floor(pts[counts[frames] == 1], up=up)

    centres = np.array([[r["cx"], r["cy"]] for r in rows])
    ray_pts, valid = ray_to_floor(centres, camera_matrix, distortion, floor)

    # two independent ranges for the same plate: from its apparent size (PnP) and from the
    # plane (the ray). PnP noise grows with range squared, so compare on that scale.
    rng_pnp = np.linalg.norm(pts, axis=1)
    rng_ray = np.where(valid, np.linalg.norm(np.nan_to_num(ray_pts), axis=1), np.nan)
    disagreement = (rng_pnp - rng_ray) / np.where(valid, rng_ray, 1.0) ** 2 * 1e6     # mm per (m^2)
    ok = valid & np.isfinite(disagreement)
    centre = np.median(disagreement[ok])
    spread = 1.4826 * np.median(np.abs(disagreement[ok] - centre))
    consistent = ok & (np.abs(disagreement - centre) <= consistency_k * spread)

    fl = np.full((len(rows), 2), np.nan)
    fl[valid] = floor.to_floor(ray_pts[valid])[:, :2]

    per_frame = [[] for _ in range(n_frames)]
    for i in np.flatnonzero(consistent):
        if hide is not None and frames[i] in hide:
            continue
        per_frame[frames[i]].append((fl[i], measurement_cov(fl[i], floor, fx, pixel_std, height_std)))

    tracker = FloorTracker(fps, max_gap=max_gap, accel_std_mm_s2=accel)
    for f in range(n_frames):
        tracker.step(f, per_frame[f], t=times[f])
    tracks = tracker.finish()
    check = {"pnp_minus_ray_mm_per_m2": {"median": round(float(centre), 1), "robust_sd": round(float(spread), 1)},
             "pnp_vs_ray_range_ratio_median": round(float(np.median(rng_pnp[ok] / rng_ray[ok])), 4)}
    return floor, fl, consistent, tracks, check


def holdout(rows, times, camera_matrix, distortion, settings, block, up=None):
    """Hide one block of ``block`` frames in every three, re-run, and score the hidden frames.

    The truth for a hidden frame is its own ray-plane measurement, used only on
    frames with exactly one consistent candidate so that the truth is not a
    choice between plates. Returns median and p90 floor error (mm) of the
    smoothed and of the forward-only filtered track.
    """
    frames = np.array([r["frame"] for r in rows])
    hide = {f for f in range(len(times)) if (f // block) % 3 == 1}
    floor, fl, consistent, tracks, _ = reconstruct(rows, times, camera_matrix, distortion, hide=hide, up=up,
                                                   **settings)
    truth = {}
    for f in hide:
        idx = np.flatnonzero((frames == f) & consistent)
        if len(idx) == 1:
            truth[f] = fl[idx[0]]
    err_s, err_f = [], []
    for t in tracks:
        for f, xs, xf in zip(t.frames, t.x_smooth, t.x_filt):
            if f in truth:
                err_s.append(np.hypot(*(xs[:2] - truth[f])))
                err_f.append(np.hypot(*(xf[:2] - truth[f])))
    if not err_s:
        return {"block_frames": block, "scored": 0}
    return {"block_frames": block, "scored": len(err_s),
            "smoothed_mm": {"median": round(float(np.median(err_s)), 1), "p90": round(float(np.percentile(err_s, 90)), 1)},
            "filtered_mm": {"median": round(float(np.median(err_f)), 1), "p90": round(float(np.percentile(err_f, 90)), 1)}}


def robot_paths(tracks, fps, accel, spin_std, min_measured=10):
    """Robot path per track with enough measurements: {track id: (frames, states)}."""
    return {tid: robot_path(t, fps, accel, spin_std) for tid, t in enumerate(tracks)
            if sum(t.measured) >= min_measured}


def track_rows(tracks, paths, fps):
    out = []
    for tid, t in enumerate(tracks):
        path = {}
        if tid in paths:
            path = {f: s for f, s in zip(*paths[tid])}
        for f, tt, m, xf, xs, Ps in zip(t.frames, t.times, t.measured, t.x_filt, t.x_smooth, t.P_smooth):
            rp = path.get(f)
            out.append({"frame": f, "time_s": round(tt, 4), "track": tid, "measured": int(m),
                        "X_filt_mm": round(xf[0], 1), "Y_filt_mm": round(xf[1], 1),
                        "X_mm": round(xs[0], 1), "Y_mm": round(xs[1], 1),
                        "sigma_X_mm": round(float(np.sqrt(Ps[0, 0])), 1),
                        "sigma_Y_mm": round(float(np.sqrt(Ps[1, 1])), 1),
                        "robot_X_mm": "" if rp is None else round(rp[0], 1),
                        "robot_Y_mm": "" if rp is None else round(rp[1], 1),
                        "robot_speed_m_s": "" if rp is None else round(float(np.hypot(rp[2], rp[3])) / 1000.0, 3)})
    return out


# ----------------------------------------------------------------------------- check against the labels

def project(floor, xy, camera_matrix, distortion):
    cam = floor.to_camera(np.asarray(xy, dtype=float).reshape(-1, 2))
    px, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), camera_matrix, distortion)
    return px.reshape(-1, 2)


def score_against_labels(labels, rows, floor, camera_matrix, distortion, tol_px):
    """Per labelled frame with a visible plate: pixel error of the smoothed and the filtered track."""
    by_frame = {r["frame"]: r for r in rows}
    result = []
    for f, entry in sorted(labels.items()):
        lab = entry["points"]
        if not lab:
            continue
        r = by_frame.get(f)
        if r is None:
            result.append({"frame": f, "covered": False})
            continue
        targets = np.asarray(lab, dtype=float)
        sm = project(floor, [r["X_mm"], r["Y_mm"]], camera_matrix, distortion)[0]
        fi = project(floor, [r["X_filt_mm"], r["Y_filt_mm"]], camera_matrix, distortion)[0]
        result.append({"frame": f, "covered": True, "measured": bool(r["measured"]),
                       "err_smooth": float(np.min(np.linalg.norm(targets - sm, axis=1))),
                       "err_filt": float(np.min(np.linalg.norm(targets - fi, axis=1)))})
    n = len(result)

    def part(sel, key):
        e = np.array([x[key] for x in sel])
        if len(e) == 0:
            return {"frames": 0}
        return {"frames": int(len(e)), "correct": int((e <= tol_px).sum()),
                "median_px": round(float(np.median(e)), 1), "p90_px": round(float(np.percentile(e, 90)), 1)}

    cov = [x for x in result if x["covered"]]
    meas = [x for x in cov if x["measured"]]
    gap = [x for x in cov if not x["measured"]]
    smooth_correct = sum(x["err_smooth"] <= tol_px for x in cov)
    filt_correct = sum(x["err_filt"] <= tol_px for x in cov)
    return {
        "tolerance_px": tol_px,
        "labelled_frames_with_target": n,
        "covered_by_a_track": len(cov),
        "correct_of_target_frames_smoothed": round(smooth_correct / n, 4) if n else None,
        "correct_of_target_frames_filtered": round(filt_correct / n, 4) if n else None,
        "smoothed": {"measured_frames": part(meas, "err_smooth"), "gap_frames": part(gap, "err_smooth")},
        "filtered": {"measured_frames": part(meas, "err_filt"), "gap_frames": part(gap, "err_filt")},
    }


# ----------------------------------------------------------------------------- figure

def figure(path, title, fl, on_plane, tracks, paths, times):
    from matplotlib.collections import LineCollection

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={"width_ratios": [1, 1.25]})
    m = 1 / 1000.0
    ax.scatter(fl[~on_plane, 0] * m, fl[~on_plane, 1] * m, s=7, c="0.7", marker="x",
               label=f"candidate dropped: PnP and ray disagree ({(~on_plane).sum()})")
    used = np.array([z for t in tracks for z in t.z if z is not None])
    if len(used):
        ax.scatter(used[:, 0] * m, used[:, 1] * m, s=5, c="tab:blue", alpha=0.3,
                   label=f"plate centre, ray meets floor plane ({len(used)})")
    t_end = max((times[f[-1]] for f, _ in paths.values()), default=1.0)
    lc = None
    for frames, xs in paths.values():
        pts = xs[:, :2] * m
        lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap="viridis",
                            norm=plt.Normalize(0, t_end), linewidths=2.6)
        lc.set_array(times[frames[:-1]])
        ax.add_collection(lc)
    if lc is not None:
        fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02).set_label("time (s)")
    ax.plot([], [], color=plt.get_cmap("viridis")(0.6), lw=2.6, label="robot path, spin smoothed out (colour: time)")
    ax.plot(0, 0, "k^", ms=10)
    ax.annotate("camera", (0, 0), xytext=(9, -3), textcoords="offset points")
    for rad in (2, 4, 6):
        a = np.linspace(np.radians(62), np.radians(118), 60)
        ax.plot(rad * np.cos(a), rad * np.sin(a), color="0.85", lw=0.8, zorder=0)
        ax.text(rad * np.cos(np.radians(62)) + 0.05, rad * np.sin(np.radians(62)), f"{rad} m", color="0.55", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("right of the camera (m)")
    ax.set_ylabel("in front of the camera (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Top view, on the floor plane fitted to the plates", fontsize=10)

    for i, t in enumerate(tracks):
        f = np.array(t.times)
        xs = np.array(t.x_smooth)
        meas = np.array([(fr, z[0], z[1]) for fr, z in zip(t.frames, t.z) if z is not None])
        if len(meas):
            ax2.scatter(times[meas[:, 0].astype(int)], meas[:, 2] * m, s=4, c="tab:blue", alpha=0.35,
                        label="measured plate, forward" if i == 0 else None)
            ax2.scatter(times[meas[:, 0].astype(int)], meas[:, 1] * m, s=4, c="tab:orange", alpha=0.35,
                        label="measured plate, right" if i == 0 else None)
        ax2.plot(f, xs[:, 1] * m, color="tab:blue", lw=0.6, alpha=0.6)
        ax2.plot(f, xs[:, 0] * m, color="tab:orange", lw=0.6, alpha=0.6)
    for j, (frames, xs) in enumerate(paths.values()):
        ax2.plot(times[frames], xs[:, 1] * m, color="navy", lw=2.2, label="robot path, forward" if j == 0 else None)
        ax2.plot(times[frames], xs[:, 0] * m, color="saddlebrown", lw=2.2, label="robot path, right" if j == 0 else None)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("position on the floor (m)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax2.set_title("The same, over time: thin lines are the plate track, thick the robot", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def robot_summary(paths, fps):
    if not paths:
        return {"frames": 0}
    xs = np.concatenate([s for _, s in paths.values()])
    steps = sum(float(np.sum(np.hypot(*np.diff(s[:, :2], axis=0).T))) for _, s in paths.values())
    speed = np.hypot(xs[:, 2], xs[:, 3]) / 1000.0
    return {"frames": int(len(xs)), "length_m": round(steps / 1000.0, 2),
            "extent_m": [round(float(np.ptp(xs[:, 0])) / 1000.0, 2), round(float(np.ptp(xs[:, 1])) / 1000.0, 2)],
            "speed_m_s": {"median": round(float(np.median(speed)), 3), "p95": round(float(np.percentile(speed, 95)), 3),
                          "max": round(float(speed.max()), 3)},
            "note": "times are the video's own frame timestamps"}


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--enemy", choices=("blue", "red"), default="blue")
    p.add_argument("--plate", choices=("small", "large"), default="small")
    p.add_argument("--camera", type=Path, default=None)
    p.add_argument("--plates", type=Path, default=None)
    p.add_argument("--phone", action="store_true",
                   help="hand-held, uncalibrated: self-calibrate the focal length, undo the camera rotation, "
                        "and use the detector variant for overexposed light bars")
    p.add_argument("--focal", type=float, default=None, help="with --phone: use this focal length (px) instead")
    p.add_argument("--labels", type=Path, default=None, help="default: labels/<video>.csv")
    p.add_argument("--pixel-std", type=float, default=2.0, help="pixel noise of the plate centre")
    p.add_argument("--height-std", type=float, default=30.0, help="how far a plate centre strays off the plane, mm")
    p.add_argument("--accel", type=float, default=200000.0,
                   help="process noise of the plate track, mm/s^2 (large: the visible plate jumps as the robot spins)")
    p.add_argument("--robot-accel", type=float, default=5000.0, help="process noise of the robot path, mm/s^2")
    p.add_argument("--spin-std", type=float, default=150.0,
                   help="plate circling about the chassis, treated as noise by the robot path, mm")
    p.add_argument("--max-gap", type=int, default=12, help="frames without a measurement before a track ends")
    p.add_argument("--consistency-k", type=float, default=3.0,
                   help="drop candidates whose PnP range disagrees with the ray range by more than k robust sigmas")
    p.add_argument("--tol-px", type=float, default=None, help="default: 2%% of the frame width")
    p.add_argument("--reuse", action="store_true", help="reuse outputs/<video>/candidates.csv")
    p.add_argument("--figure", type=Path, default=None, help="also write the figure here")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    width_mm, height_mm = load_plate(a.plate, a.plates)
    out_dir = a.out or Path("outputs") / a.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path, times_path = out_dir / "candidates.csv", out_dir / "frame_times.csv"

    cap = cv2.VideoCapture(str(a.video))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_size = (video_width, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    camera_note, frame_rots, motion, up = None, None, None, None
    if a.phone:
        if a.reuse:
            raise SystemExit("--reuse is not supported with --phone")
        dets, times, _ = detect_all(a.video, a.enemy, "saturated")
        Hs, inliers, errors, size = background_homographies(a.video)
        vps = vertical_points(a.video)
        grid = np.arange(0.3, 2.5, 0.01) * max(size)
        focal, curve, halves = focal_from_plates(dets, Hs, vps, size, width_mm, height_mm, grid)
        if a.focal:
            focal = a.focal
        try:
            rot_focal, _, rot_pairs = self_calibrate_focal(Hs, size)
        except ValueError:
            rot_focal, rot_pairs = None, 0
        camera_matrix, distortion = intrinsics(focal, size), np.zeros((5, 1))
        frame_rots = rotations(Hs, camera_matrix)
        up, up_spread_deg, up_frames = up_from_verticals(vps, frame_rots, camera_matrix)
        angles = [np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) for R in frame_rots]
        k = int(np.argmin(np.abs(grid - focal)))
        motion = {"focal_px": round(focal, 1), "focal_px_first_half": round(halves[0], 1),
                  "focal_px_second_half": round(halves[1], 1),
                  "plate_height_spread_mm_at_focal": round(float(curve[k]), 1),
                  "plate_height_spread_mm_at_1.5x_focal": round(float(np.interp(1.5 * focal, grid, curve)), 1),
                  "horizontal_fov_deg_long_side": round(float(np.degrees(2 * np.arctan(max(size) / 2 / focal))), 1),
                  "frames_voting_for_up": up_frames, "frames_with_verticals": len(vps),
                  "up_agreement_deg_median": round(up_spread_deg, 2),
                  "rotation_only_self_calibration_px": None if rot_focal is None else round(rot_focal, 1),
                  "max_rotation_deg": round(float(max(angles)), 2),
                  "background_inliers_median": int(np.median(inliers[1:])),
                  "background_error_px_median": round(float(np.nanmedian(errors[1:])), 3)}
        (out_dir / "camera_selfcal.json").write_text(json.dumps({
            "name": f"estimated from {a.video.name}: vertical edges of the room and the fixed plate size",
            "image_size": list(size), "camera_matrix": camera_matrix.tolist(),
            "distortion": [0.0] * 5, "up_in_frame0": up.tolist(), **motion}, indent=2), encoding="utf-8")
        camera_note = f"focal length {focal:.0f} px from the room's vertical edges and the plate size"
        rows = solve_candidates(dets, camera_matrix, distortion, width_mm, height_mm, frame_rots)
    else:
        camera_path = a.camera or CONFIG_DIR / "camera_2021.json"
        camera_matrix, distortion = load_camera(camera_path)
        calibrated_width = json.loads(Path(camera_path).read_text(encoding="utf-8"))["image_size"][0]
        if video_width != calibrated_width:
            camera_matrix = scale_camera(camera_matrix, video_width / calibrated_width)
            print(f"note: intrinsics rescaled by {video_width / calibrated_width:.3f} to {video_width} px")
        if a.reuse and cand_path.exists() and times_path.exists():
            rows = read_candidates(cand_path)
            times = np.loadtxt(times_path, delimiter=",", skiprows=1, usecols=1)
        else:
            dets, times, _ = detect_all(a.video, a.enemy, "original")
            rows = solve_candidates(dets, camera_matrix, distortion, width_mm, height_mm)
    if rows:
        with open(cand_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        np.savetxt(times_path, np.column_stack([np.arange(len(times)), times]), delimiter=",",
                   header="frame,time_s", comments="", fmt=["%d", "%.6f"])
    if not rows:
        raise SystemExit("no armor candidate was solved on any frame")
    n_frames = len(times)
    intervals = np.diff(times)
    fps = 1.0 / float(np.median(intervals))

    settings = {"pixel_std": a.pixel_std, "height_std": a.height_std, "accel": a.accel,
                "max_gap": a.max_gap, "consistency_k": a.consistency_k}
    floor, fl, on_plane, tracks, check = reconstruct(rows, times, camera_matrix, distortion, up=up, **settings)
    paths = robot_paths(tracks, fps, a.robot_accel, a.spin_std)
    trows = track_rows(tracks, paths, fps)
    with open(out_dir / "trajectory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(trows[0].keys()))
        w.writeheader()
        w.writerows(trows)

    nis = np.concatenate([t.nis for t in tracks]) if tracks else np.array([])
    measured = sum(r["measured"] for r in trows)
    spans = [len(t.frames) for t in tracks]
    summary = {
        "video": a.video.name, "frames": n_frames, "median_frame_rate": round(fps, 3),
        "frame_intervals_ms": {"median": round(1000 * float(np.median(intervals)), 2),
                               "max": round(1000 * float(intervals.max()), 2),
                               "longer_than_1.5x_median": int((intervals > 1.5 * np.median(intervals)).sum())},
        "camera": camera_note or "calibration file", "camera_motion": motion,
        "plate_mm": [width_mm, height_mm],
        "floor": {"camera_height_above_plates_mm": round(floor.camera_height_mm, 1),
                  "pitch_down_deg": round(floor.pitch_deg, 2),
                  "plane_residual_mm": round(floor.residual_mm, 1),
                  "inliers_of_single_candidate_frames": round(float(floor.inliers.mean()), 4)},
        "candidates": len(rows), "dropped_inconsistent": int((~on_plane).sum()),
        "pnp_vs_ray": check,
        "tracks": len(tracks), "frames_in_tracks": len(trows), "frames_measured": int(measured),
        "longest_track_frames": max(spans) if spans else 0,
        "nis_mean": round(float(nis.mean()), 3) if len(nis) else None,
        "robot_path": robot_summary(paths, fps),
        "settings": {**settings, "robot_accel": a.robot_accel, "spin_std": a.spin_std},
        "holdout": [holdout(rows, times, camera_matrix, distortion, settings, b, up) for b in (2, 5, 10)],
    }

    label_path = a.labels or Path("labels") / f"{a.video.stem}.csv"
    if label_path.exists():
        tol = a.tol_px or round(0.02 * video_width, 1)
        summary["against_labels"] = score_against_labels(load_labels(label_path, 1.0), trows, floor,
                                                         camera_matrix, distortion, tol)

    (out_dir / "trajectory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    kind = "a hand-held phone, rotation undone" if a.phone else "one static camera"
    title = f"{a.video.name}: the robot's path on the floor, reconstructed from {kind}"
    figure(out_dir / "trajectory.png", title, fl, on_plane, tracks, paths, times)
    if a.figure:
        a.figure.parent.mkdir(parents=True, exist_ok=True)
        figure(a.figure, title, fl, on_plane, tracks, paths, times)

    if motion:
        mo = motion
        print(f"\nphone: focal length {mo['focal_px']:.0f} px ({mo['horizontal_fov_deg_long_side']:.0f} deg across "
              f"the long side), {mo['focal_px_first_half']:.0f} / {mo['focal_px_second_half']:.0f} px from either half; "
              f"plate heights spread {mo['plate_height_spread_mm_at_focal']:.0f} mm there, "
              f"{mo['plate_height_spread_mm_at_1.5x_focal']:.0f} mm at 1.5x")
        print(f"  up from the vertical edges of {mo['frames_voting_for_up']} frames, agreeing to "
              f"{mo['up_agreement_deg_median']:.1f} deg; rotation-only self-calibration would say "
              f"{mo['rotation_only_self_calibration_px']} px")
        print(f"  camera turned up to {mo['max_rotation_deg']:.1f} deg; background homography error "
              f"{mo['background_error_px_median']:.2f} px median")
    fi = summary["frame_intervals_ms"]
    print(f"frame intervals: median {fi['median']:.1f} ms, longest {fi['max']:.1f} ms, "
          f"{fi['longer_than_1.5x_median']} longer than 1.5x the median")
    f = summary["floor"]
    print(f"\nfloor plane from the plates: camera {f['camera_height_above_plates_mm'] / 1000:.2f} m above them, "
          f"pitched {f['pitch_down_deg']:.1f} deg down, plates within {f['plane_residual_mm']:.0f} mm of the plane")
    print(f"candidates {len(rows)}, dropped because PnP range and ray range disagree {summary['dropped_inconsistent']}"
          f"  (PnP/ray range ratio, median {check['pnp_vs_ray_range_ratio_median']:.3f})")
    print(f"tracks {len(tracks)}, covering {len(trows)} of {n_frames} frames ({measured} measured), "
          f"longest {summary['longest_track_frames']} frames")
    print(f"mean NIS {summary['nis_mean']} (2.0 if the measurement noise model is right)")
    rp = summary["robot_path"]
    if rp["frames"]:
        print(f"robot path: {rp['frames']} frames, {rp['length_m']:.1f} m driven, extent {rp['extent_m'][0]:.1f} x "
              f"{rp['extent_m'][1]:.1f} m; speed from the frame timestamps median {rp['speed_m_s']['median']:.2f}, "
              f"p95 {rp['speed_m_s']['p95']:.2f} m/s")
    print("\nhold-out: hide one block of frames in three, predict it from the rest")
    for h in summary["holdout"]:
        if h["scored"]:
            print(f"  blocks of {h['block_frames']:2d} frames, n={h['scored']:3d}: smoothed median {h['smoothed_mm']['median']:.0f} mm"
                  f" (p90 {h['smoothed_mm']['p90']:.0f})   filtered median {h['filtered_mm']['median']:.0f} mm"
                  f" (p90 {h['filtered_mm']['p90']:.0f})")
    if "against_labels" in summary:
        s = summary["against_labels"]
        print(f"\nagainst {s['labelled_frames_with_target']} labelled frames with a target "
              f"(tolerance {s['tolerance_px']} px):")
        print(f"  correct, of all target frames: smoothed {100 * s['correct_of_target_frames_smoothed']:.1f}%"
              f"   filtered only {100 * s['correct_of_target_frames_filtered']:.1f}%")
        for kind in ("measured_frames", "gap_frames"):
            sm, fi = s["smoothed"][kind], s["filtered"][kind]
            if sm["frames"]:
                print(f"  {kind:16s} n={sm['frames']:3d}  smoothed {sm['correct']:3d} correct, median {sm['median_px']} px"
                      f"   | filtered {fi['correct']:3d} correct, median {fi['median_px']} px")
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
