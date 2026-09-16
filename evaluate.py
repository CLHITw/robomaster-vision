"""Score the detector and the tracker against clicked ground truth.

    python label_frames.py video2.avi --every 5     # first, click the frames
    python evaluate.py video2.avi                   # then score them

Writes ``outputs/<video>/evaluation.json`` and an error plot, and prints a
summary. A position counts as correct when it lies within the tolerance of the
clicked armor centre (default 2% of the frame width, 25.6 px at 1280).
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

from replay.metrics import evaluate, nearest
from replay.pipelines import armor_status, detect_armors
from replay.tracking import ArmorTracker


def load_labels(path, scale):
    labels = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            visible = int(row["visible"])
            labels[int(row["frame"])] = {
                "visible": visible,
                "x": float(row["x"]) * scale if visible and row["x"] else None,
                "y": float(row["y"]) * scale if visible and row["y"] else None,
            }
    return labels


def replay(video, enemy, proc_width, meas_std, max_coast):
    cap = cv2.VideoCapture(str(video))
    records, tracker, width = [], None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if proc_width and frame.shape[1] > proc_width:
            h = round(frame.shape[0] * proc_width / frame.shape[1])
            frame = cv2.resize(frame, (proc_width, h), interpolation=cv2.INTER_AREA)
        if tracker is None:
            width = frame.shape[1]
            tracker = ArmorTracker(max_coast=max_coast, meas_std=meas_std or round(0.02 * width, 1))
        bars, armors, _ = detect_armors(frame, enemy)
        candidates = [(a.cx, a.cy) for a in armors]
        step = tracker.step(candidates)
        records.append({"candidates": candidates, "status": armor_status(bars, armors),
                        "state": step.state, "estimate": step.estimate})
    cap.release()
    return records, width


def error_plot(records, labels, tol_px, path, title):
    frames = sorted(f for f, lab in labels.items() if lab["visible"] and f < len(records))
    det, trk = [], []
    for f in frames:
        point = (labels[f]["x"], labels[f]["y"])
        r = records[f]
        det.append(nearest(point, r["candidates"]) if r["status"] == "one_armor" else np.nan)
        trk.append(nearest(point, [r["estimate"]]) if r["estimate"] is not None else np.nan)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axhline(tol_px, color="grey", ls="--", lw=1, label=f"tolerance {tol_px:.0f} px")
    ax.plot(frames, det, "o", ms=4, color="tab:orange", label="2021 detector (single armor frames)")
    ax.plot(frames, trk, "-", lw=1.2, color="tab:blue", label="tracker estimate")
    ax.set_yscale("symlog", linthresh=50)
    ax.set_xlabel("frame")
    ax.set_ylabel("distance to clicked centre (px)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--labels", type=Path, default=None, help="default: labels/<video>.csv")
    p.add_argument("--enemy", choices=("blue", "red"), default="blue")
    p.add_argument("--proc-width", type=int, default=0, help="0 = native resolution")
    p.add_argument("--meas-std", type=float, default=0.0, help="0 = 2%% of the frame width")
    p.add_argument("--max-coast", type=int, default=15)
    p.add_argument("--tol-frac", type=float, default=0.02, help="tolerance as a fraction of frame width")
    p.add_argument("--tol-px", type=float, default=0.0, help="tolerance in px (overrides --tol-frac)")
    p.add_argument("--out", type=Path, default=None, help="default: outputs/<video>/")
    args = p.parse_args()

    labels_path = args.labels or Path("labels") / f"{args.video.stem}.csv"
    if not labels_path.exists():
        raise SystemExit(f"no labels at {labels_path} - run label_frames.py first")

    records, width = replay(args.video, args.enemy, args.proc_width, args.meas_std, args.max_coast)
    src_width = int(cv2.VideoCapture(str(args.video)).get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = width / src_width
    labels = load_labels(labels_path, scale)
    tol_px = args.tol_px or args.tol_frac * width

    result = evaluate(records, labels, tol_px)
    result["video"] = args.video.name
    result["processed_width"] = width
    result["labels"] = str(labels_path)

    out_dir = args.out or Path("outputs") / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "evaluation.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    error_plot(records, labels, tol_px, out_dir / "error_plot.png",
               f"{args.video.name}: distance to the clicked armor centre")

    r = result
    print(f"\n{r['labelled_frames']} labelled frames: {r['frames_with_target']} with a target, "
          f"{r['frames_without_target']} without. Tolerance {r['tolerance_px']} px.\n")
    print(f"{'':34s} {'correct':>10s} {'of frames':>11s} {'median err':>11s}")
    print(f"{'2021 detector, single-armor frames':34s} "
          f"{str(r['detector_single_correct']) + '/' + str(r['detector_single_frames']):>10s} "
          f"{str(r['detector_single_precision_pct']) + '%':>11s} "
          f"{str(r['detector_error']['median_px']) + ' px':>11s}")
    print(f"{'  -> of all frames with a target':34s} {'':>10s} {str(r['detector_usable_pct']) + '%':>11s}")
    print(f"{'  best candidate (upper bound)':34s} {'':>10s} {str(r['detector_best_candidate_pct']) + '%':>11s}")
    print(f"{'tracker estimate':34s} "
          f"{str(r['tracker_correct']) + '/' + str(r['tracker_frames_with_estimate']):>10s} "
          f"{str(r['tracker_precision_pct']) + '%':>11s} "
          f"{str(r['tracker_error']['median_px']) + ' px':>11s}")
    print(f"{'  -> of all frames with a target':34s} {'':>10s} {str(r['tracker_usable_pct']) + '%':>11s}")
    print(f"{'  when measuring':34s} {'':>10s} {'':>11s} "
          f"{str(r['tracker_error_when_measuring']['median_px']) + ' px':>11s}")
    print(f"{'  when coasting':34s} {'':>10s} {'':>11s} "
          f"{str(r['tracker_error_when_coasting']['median_px']) + ' px':>11s}")
    if "false_target_when_empty_pct" in r:
        print(f"\nframes labelled empty: detector reported a target in {r['false_target_when_empty_pct']}%, "
              f"tracker held a position in {r['tracker_position_when_empty_pct']}%")
    print(f"\nwritten to {out_dir / 'evaluation.json'} and {out_dir / 'error_plot.png'}")


if __name__ == "__main__":
    main()
