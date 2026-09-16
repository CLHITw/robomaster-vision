"""Tune the tracker against the clicked ground truth.

    python tune_tracker.py video2.avi

Detection is the expensive part and does not depend on the tracker, so the
video is decoded and detected once; every (gate width, coasting patience)
combination is then scored on the cached detections.

Prints a table sorted by the share of labelled target frames for which the
tracker gives a position within tolerance, and writes tuning.csv.
"""

import argparse
import csv
from pathlib import Path

import cv2

from evaluate import load_labels
from replay.metrics import evaluate
from replay.pipelines import armor_status, detect_armors
from replay.tracking import ArmorTracker


def detect_all(video, enemy, proc_width):
    cap = cv2.VideoCapture(str(video))
    frames, width = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if proc_width and frame.shape[1] > proc_width:
            h = round(frame.shape[0] * proc_width / frame.shape[1])
            frame = cv2.resize(frame, (proc_width, h), interpolation=cv2.INTER_AREA)
        width = frame.shape[1]
        bars, armors, _ = detect_armors(frame, enemy)
        frames.append(([(a.cx, a.cy) for a in armors], armor_status(bars, armors)))
    cap.release()
    return frames, width


def run_tracker(detections, meas_std, max_coast):
    tracker = ArmorTracker(meas_std=meas_std, max_coast=max_coast)
    records = []
    for candidates, status in detections:
        step = tracker.step(candidates)
        records.append({"candidates": candidates, "status": status,
                        "state": step.state, "estimate": step.estimate})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--enemy", choices=("blue", "red"), default="blue")
    p.add_argument("--proc-width", type=int, default=0)
    p.add_argument("--tol-frac", type=float, default=0.02)
    p.add_argument("--meas-std", type=float, nargs="+", default=[10, 15, 25, 35, 50])
    p.add_argument("--max-coast", type=int, nargs="+", default=[0, 2, 5, 10, 15])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    labels_path = args.labels or Path("labels") / f"{args.video.stem}.csv"
    detections, width = detect_all(args.video, args.enemy, args.proc_width)
    src_width = int(cv2.VideoCapture(str(args.video)).get(cv2.CAP_PROP_FRAME_WIDTH))
    labels = load_labels(labels_path, width / src_width)
    tol_px = args.tol_frac * width
    print(f"{len(detections)} frames detected once, scoring {len(args.meas_std) * len(args.max_coast)} "
          f"tracker settings against {sum(1 for l in labels.values() if l['points'])} labelled target frames "
          f"(tolerance {tol_px:.1f} px)\n")

    rows = []
    for meas_std in args.meas_std:
        for max_coast in args.max_coast:
            r = evaluate(run_tracker(detections, meas_std, max_coast), labels, tol_px)
            rows.append({
                "meas_std": meas_std, "max_coast": max_coast,
                "usable_pct": r["tracker_usable_pct"], "precision_pct": r["tracker_precision_pct"],
                "median_err_px": r["tracker_error"]["median_px"],
                "err_measuring_px": r["tracker_error_when_measuring"]["median_px"],
                "err_coasting_px": r["tracker_error_when_coasting"]["median_px"],
                "position_when_empty_pct": r.get("tracker_position_when_empty_pct"),
            })

    baseline = {"detector_usable_pct": evaluate(run_tracker(detections, 25, 0), labels, tol_px)["detector_usable_pct"]}
    rows.sort(key=lambda r: (-(r["usable_pct"] or 0), r["median_err_px"] or 1e9))
    header = ["meas_std", "max_coast", "usable_pct", "precision_pct", "median_err_px",
              "err_measuring_px", "err_coasting_px", "position_when_empty_pct"]
    print(" ".join(f"{h:>18s}" for h in header))
    for row in rows:
        print(" ".join(f"{str(row[h]):>18s}" for h in header))
    print(f"\nfor reference, the 2021 detector alone is usable in {baseline['detector_usable_pct']}% "
          f"of labelled target frames")

    out = args.out or Path("outputs") / args.video.stem / "tuning.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
