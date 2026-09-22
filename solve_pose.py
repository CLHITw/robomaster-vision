"""Solve the armor plate pose on a video, and check the result against itself.

    python solve_pose.py clips/robot_clip.mp4

Writes outputs/<video>/pose.csv and pose_plot.png, and prints the two
self-checks: the reprojection error, and how the PnP depth compares with an
independent pinhole estimate from the apparent width of the plate.

There is no ground-truth distance for this recording, so nothing here proves
the distances are right in metres; the plate dimensions come from public
RoboMaster figures (config/armor_plates.json) and scale every distance.
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

from replay.pipelines import armor_status, detect_armors
from replay.pose import load_camera, load_plate, scale_camera, solve_armor
from replay.tracking import ArmorTracker

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def run(video, enemy, plate, camera_path, plate_path, max_coast):
    camera_matrix, distortion = load_camera(camera_path)
    width_mm, height_mm = load_plate(plate, plate_path)

    cap = cv2.VideoCapture(str(video))
    calibrated_width = json.loads(Path(camera_path or CONFIG_DIR / "camera_2021.json")
                                  .read_text(encoding="utf-8"))["image_size"][0]
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if video_width != calibrated_width:
        camera_matrix = scale_camera(camera_matrix, video_width / calibrated_width)
        print(f"note: video is {video_width} px wide, the calibration is for {calibrated_width} px; "
              f"intrinsics rescaled by {video_width / calibrated_width:.3f}")
    tracker, rows, frame_id = None, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if tracker is None:
            tracker = ArmorTracker(max_coast=max_coast, meas_std=round(0.008 * frame.shape[1], 1))
        bars, armors, _ = detect_armors(frame, enemy)
        status = armor_status(bars, armors)
        step = tracker.step([(a.cx, a.cy) for a in armors])

        # solve the plate the tracker accepted this frame, so the series follows one target
        chosen = None
        if step.measurement is not None:
            for a in armors:
                if abs(a.cx - step.measurement[0]) < 1e-6 and abs(a.cy - step.measurement[1]) < 1e-6:
                    chosen = a
                    break
        if chosen is not None:
            r = solve_armor(bars[chosen.bar_i], bars[chosen.bar_j],
                            camera_matrix, distortion, width_mm, height_mm)
            if r is not None:
                rows.append({"frame": frame_id, "status": status,
                             "x_mm": round(r["tvec_mm"][0], 1), "y_mm": round(r["tvec_mm"][1], 1),
                             "depth_mm": round(r["depth_mm"], 1),
                             "distance_mm": round(r["distance_mm"], 1),
                             "pinhole_mm": round(r["pinhole_mm"], 1),
                             "reprojection_px": round(r["reprojection_px"], 3),
                             "width_px": round(r["width_px"], 1)})
        frame_id += 1
    cap.release()
    return rows, frame_id, (width_mm, height_mm)


def plot(rows, path, title):
    f = [r["frame"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(f, [r["depth_mm"] / 1000 for r in rows], ".-", ms=3, lw=1, color="tab:blue",
             label="PnP depth (z)")
    ax1.plot(f, [r["pinhole_mm"] / 1000 for r in rows], ".", ms=3, color="tab:orange",
             label="pinhole estimate from plate width")
    ax1.set_ylabel("distance (m)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title(title, fontsize=10)

    ax2.plot(f, [r["reprojection_px"] for r in rows], ".", ms=3, color="tab:green")
    ax2.set_ylabel("reprojection (px)")
    ax2.set_xlabel("frame")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--enemy", choices=("blue", "red"), default="blue")
    p.add_argument("--plate", choices=("small", "large"), default="small")
    p.add_argument("--camera", type=Path, default=None, help="camera json (default: config/camera_2021.json)")
    p.add_argument("--plates", type=Path, default=None, help="plate json (default: config/armor_plates.json)")
    p.add_argument("--max-coast", type=int, default=2)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows, frames, (width_mm, height_mm) = run(args.video, args.enemy, args.plate,
                                              args.camera, args.plates, args.max_coast)
    if not rows:
        raise SystemExit("no armor plate was accepted on any frame, nothing to solve")

    out_dir = args.out or Path("outputs") / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pose.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot(rows, out_dir / "pose_plot.png", f"{args.video.name}: armor plate distance and reprojection error")

    depth = np.array([r["depth_mm"] for r in rows]) / 1000
    pinhole = np.array([r["pinhole_mm"] for r in rows]) / 1000
    reproj = np.array([r["reprojection_px"] for r in rows])
    ratio = pinhole / depth

    print(f"\nplate {args.plate}: {width_mm:.0f} x {height_mm:.0f} mm (from config, unverified)")
    print(f"solved on {len(rows)} of {frames} frames\n")
    print(f"  distance          median {np.median(depth):.2f} m   range {depth.min():.2f} - {depth.max():.2f} m")
    print(f"  reprojection      median {np.median(reproj):.2f} px   p90 {np.percentile(reproj, 90):.2f} px   "
          f"max {reproj.max():.2f} px")
    print(f"  pinhole / PnP     median {np.median(ratio):.3f}   p10 {np.percentile(ratio, 10):.3f}   "
          f"p90 {np.percentile(ratio, 90):.3f}")
    first, last = depth[:max(len(depth) // 5, 1)], depth[-max(len(depth) // 5, 1):]
    print(f"  first fifth {np.median(first):.2f} m  ->  last fifth {np.median(last):.2f} m")
    print(f"\nwritten to {out_dir / 'pose.csv'} and {out_dir / 'pose_plot.png'}")
    print("No ground-truth distance exists for this recording: the two checks above test whether the\n"
          "solution is self-consistent, not whether the metres are right.")


if __name__ == "__main__":
    main()
