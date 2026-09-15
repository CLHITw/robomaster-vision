"""Run the 2021 pipelines (and the 2026 tracker) on one or more videos.

Examples:
    python run_videos.py path/to/video1.avi
    python run_videos.py videos/ --enemy red --save-video
    python run_videos.py "videos/*.mp4" --out outputs

For every video, ``<out>/<video name>/`` receives:
    summary.json      per-video statistics
    frames.csv        per-frame results of all three stages
    v1_sheet.jpg      sampled frames, carcarcar.cpp boxes (top) and its binary image
    v2_sheet.jpg      sampled frames, armor_plate light bars / armors and its binary image
    track_sheet.jpg   sampled frames with candidates, used measurement and Kalman estimate
    track_plot.png    armor-centre trajectory over time
    annotated.mp4     (with --save-video) v2 + tracker overlay for every frame
and ``<out>/summary.csv`` collects one row per video.
"""

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np

from replay.pipelines import armor_status, detect_armors, threshold_boxes
from replay.tracking import ArmorTracker
from replay.visualize import contact_sheet, draw_track, draw_v1, draw_v2, label, trajectory_plot

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".m4v", ".wmv"}


def collect_videos(inputs):
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in VIDEO_EXTS)
        elif any(ch in item for ch in "*?["):
            paths += sorted(Path(q) for q in glob.glob(item) if Path(q).suffix.lower() in VIDEO_EXTS)
        else:
            paths.append(p)
    return paths


def count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    return n


def pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def process(path, out_root, args):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[skip] cannot open {path}")
        return None
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = count_frames(path)
    if args.max_frames:
        total = min(total, args.max_frames)
    sample_ids = set(np.linspace(0, max(total - 1, 0), args.samples).astype(int).tolist())

    out_dir = out_root / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = ArmorTracker(max_coast=args.max_coast, accel_std=args.accel_std, meas_std=args.meas_std)
    records, samples = [], {"v1": [], "v1_bin": [], "v2": [], "v2_bin": [], "track": []}
    t_start = time.perf_counter()
    frame_id = 0
    while frame_id < total:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, v1_bin = threshold_boxes(frame)
        bars, armors, v2_bin = detect_armors(frame, args.enemy)
        status = armor_status(bars, armors)
        candidates = [(a.cx, a.cy) for a in armors]
        step = tracker.step(candidates)

        full_frame = any(w * h > 0.5 * width * height for (_, _, w, h) in boxes)
        records.append({
            "frame": frame_id,
            "v1_boxes": len(boxes), "v1_full_frame_box": int(full_frame),
            "v2_bars": len(bars), "v2_armors": len(armors), "v2_status": status,
            "track_state": step.state,
            "meas_x": None if step.measurement is None else round(step.measurement[0], 1),
            "meas_y": None if step.measurement is None else round(step.measurement[1], 1),
            "est_x": None if step.estimate is None else round(step.estimate[0], 1),
            "est_y": None if step.estimate is None else round(step.estimate[1], 1),
        })

        if frame_id in sample_ids or writer is not None:
            v2_img = draw_v2(frame, bars, armors)
            track_img = draw_track(v2_img, candidates, step)
            if writer is not None:
                writer.write(label(track_img.copy(), f"#{frame_id} {status} / {step.state}"))
            if frame_id in sample_ids:
                samples["v1"].append(label(draw_v1(frame, boxes), f"#{frame_id} boxes={len(boxes)}"))
                samples["v1_bin"].append(v1_bin)
                samples["v2"].append(label(v2_img, f"#{frame_id} bars={len(bars)} armors={len(armors)}"))
                samples["v2_bin"].append(v2_bin)
                samples["track"].append(label(draw_track(frame, candidates, step), f"#{frame_id} {step.state}"))
        frame_id += 1
    elapsed = time.perf_counter() - t_start
    cap.release()
    if writer is not None:
        writer.release()

    n = len(records)
    if n == 0:
        print(f"[skip] no frames read from {path}")
        return None

    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    cv2.imwrite(str(out_dir / "v1_sheet.jpg"), contact_sheet([samples["v1"], samples["v1_bin"]]))
    cv2.imwrite(str(out_dir / "v2_sheet.jpg"), contact_sheet([samples["v2"], samples["v2_bin"]]))
    cv2.imwrite(str(out_dir / "track_sheet.jpg"), contact_sheet([samples["track"]]))
    trajectory_plot(records, out_dir / "track_plot.png", f"{path.name} (enemy={args.enemy})")

    status_count = {s: sum(1 for r in records if r["v2_status"] == s)
                    for s in ("lt2_bars", "no_armor", "one_armor", "too_many")}
    state_count = {s: sum(1 for r in records if r["track_state"] == s)
                   for s in ("none", "init", "update", "coast", "lost")}
    too_many = [r for r in records if r["v2_status"] == "too_many"]
    summary = {
        "video": path.name,
        "enemy": args.enemy,
        "frames": n,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "ms_per_frame": round(1000 * elapsed / n, 1),
        "v1_boxes_median": float(np.median([r["v1_boxes"] for r in records])),
        "v1_full_frame_box_pct": pct(sum(r["v1_full_frame_box"] for r in records), n),
        "v2_bars_median": float(np.median([r["v2_bars"] for r in records])),
        **{f"v2_{k}_pct": pct(v, n) for k, v in status_count.items()},
        "track_measurement_used_pct": pct(sum(1 for r in records if r["meas_x"] is not None), n),
        "track_estimate_pct": pct(sum(1 for r in records if r["est_x"] is not None), n),
        "track_coast_pct": pct(state_count["coast"], n),
        "too_many_resolved": f"{sum(1 for r in too_many if r['meas_x'] is not None)}/{len(too_many)}",
        "single_detection_rejected": sum(1 for r in records
                                         if r["v2_status"] == "one_armor" and r["meas_x"] is None),
        "track_inits": state_count["init"],
        "track_lost": state_count["lost"],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="video files, directories or glob patterns")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--enemy", choices=("blue", "red"), default="blue",
                        help="light-bar colour to detect (the 2021 code only implemented blue)")
    parser.add_argument("--samples", type=int, default=8, help="frames shown in the contact sheets")
    parser.add_argument("--save-video", action="store_true", help="write an annotated mp4 per video")
    parser.add_argument("--max-frames", type=int, default=0, help="process at most this many frames (0 = all)")
    parser.add_argument("--max-coast", type=int, default=15, help="frames without a match before the track is dropped")
    parser.add_argument("--meas-std", type=float, default=15.0, help="measurement noise of the armor centre (px)")
    parser.add_argument("--accel-std", type=float, default=4.0, help="process noise (px / frame^2)")
    args = parser.parse_args()

    videos = collect_videos(args.inputs)
    if not videos:
        parser.error("no videos found")
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in videos:
        print(f"[run] {path}")
        s = process(path, args.out, args)
        if s:
            summaries.append(s)
            print(f"      {s['frames']} frames {s['resolution']} | v2 one armor {s['v2_one_armor_pct']}% "
                  f"| tracker estimate {s['track_estimate_pct']}% (measured {s['track_measurement_used_pct']}%) "
                  f"| too many resolved {s['too_many_resolved']} -> {args.out / path.stem}")

    if summaries:
        with open(args.out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"[done] {len(summaries)} video(s), summary in {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
