"""Run the 2021 pipelines (and the 2026 tracker) on one or more videos.

By default every video is played back in a window with the detections drawn on
top, and the same annotated frames are written to an mp4 next to the results.

Examples:
    python run_videos.py                        # every video in the current folder
    python run_videos.py video0.avi
    python run_videos.py .                      # every video in this folder
    python run_videos.py "video*.mp4" --enemy red
    python run_videos.py . --no-show            # batch mode, no window

Keys while a window is open:
    space  pause / resume        q or Esc  stop everything
    n      skip to next video    s         save the current frame as a png

For every video, ``<out>/<video name>/`` receives:
    annotated.mp4     the full annotated video (disable with --no-video)
    summary.json      per-video statistics
    frames.csv        per-frame results of all three stages
    v1_sheet.jpg      sampled frames, carcarcar.cpp boxes and its binary image
    v2_sheet.jpg      sampled frames, armor_plate light bars / armors and its binary image
    track_sheet.jpg   sampled frames with candidates, used measurement and Kalman estimate
    track_plot.png    armor-centre trajectory over time
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
from replay.visualize import (contact_sheet, draw_hud, draw_track, draw_v1, draw_v2,
                              label, trajectory_plot)

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


def probe_enemy(path, proc_width, n_probe=30):
    """Decide the light-bar colour by running both variants on a few frames."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    ids = np.linspace(0, max(total - 1, 0), min(n_probe, max(total, 1))).astype(int)
    score = {"blue": 0, "red": 0}
    for fid in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok:
            continue
        if proc_width and frame.shape[1] > proc_width:
            h = round(frame.shape[0] * proc_width / frame.shape[1])
            frame = cv2.resize(frame, (proc_width, h), interpolation=cv2.INTER_AREA)
        for enemy in score:
            bars, armors, _ = detect_armors(frame, enemy)
            score[enemy] += len(armors) + 0.1 * len(bars)
    cap.release()
    return ("red" if score["red"] > score["blue"] else "blue"), score


def pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def process(path, out_root, args):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[skip] cannot open {path}")
        return None, False
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scale = args.proc_width / src_w if args.proc_width and src_w > args.proc_width else 1.0
    width, height = round(src_w * scale), round(src_h * scale)

    total = count_frames(path)
    if args.max_frames:
        total = min(total, args.max_frames)
    sample_ids = set(np.linspace(0, max(total - 1, 0), args.samples).astype(int).tolist())

    out_dir = out_root / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    window = None
    if not args.no_show:
        window = f"{path.name}  [space] pause  [n] next  [q] quit  [s] snapshot"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        disp_h = min(args.display_height, height)
        cv2.resizeWindow(window, max(1, round(width * disp_h / height)), disp_h)

    enemy = args.enemy
    if enemy == "auto":
        enemy, score = probe_enemy(path, args.proc_width)
        print(f"      enemy colour: {enemy} (probe score blue={score['blue']:.1f} red={score['red']:.1f})")

    meas_std = args.meas_std if args.meas_std else round(0.02 * width, 1)
    tracker = ArmorTracker(max_coast=args.max_coast, accel_std=args.accel_std, meas_std=meas_std)
    records, samples = [], {"v1": [], "v1_bin": [], "v2": [], "v2_bin": [], "track": []}
    trail = []
    quit_all, paused = False, False
    frame_delay = max(1, int(1000.0 / max(fps, 1.0) / max(args.speed, 0.01)))
    t_start = time.perf_counter()
    frame_id = 0
    while frame_id < total:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        boxes, v1_bin = threshold_boxes(frame)
        bars, armors, v2_bin = detect_armors(frame, enemy)
        status = armor_status(bars, armors)
        candidates = [(a.cx, a.cy) for a in armors]
        step = tracker.step(candidates)

        trail.append(step.estimate)
        trail = trail[-args.trail:]

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

        annotated = None
        if writer is not None or window is not None or frame_id in sample_ids:
            annotated = draw_track(draw_v2(frame, bars, armors), candidates, step, trail)
            draw_hud(annotated, frame_id, total, status, step, len(bars))
        if writer is not None:
            writer.write(annotated)
        if frame_id in sample_ids:
            samples["v1"].append(label(draw_v1(frame, boxes), f"#{frame_id} boxes={len(boxes)}"))
            samples["v1_bin"].append(v1_bin)
            samples["v2"].append(label(draw_v2(frame, bars, armors),
                                       f"#{frame_id} bars={len(bars)} armors={len(armors)}"))
            samples["v2_bin"].append(v2_bin)
            samples["track"].append(annotated.copy())

        if window is not None:
            cv2.imshow(window, annotated)
            while True:
                key = cv2.waitKey(0 if paused else frame_delay) & 0xFF
                if key in (ord("q"), 27):
                    quit_all = True
                elif key == ord("n"):
                    frame_id = total
                elif key == ord(" "):
                    paused = not paused
                    continue
                elif key == ord("s"):
                    snap = out_dir / f"frame_{frame_id:05d}.png"
                    cv2.imwrite(str(snap), annotated)
                    print(f"      saved {snap}")
                    continue
                break
            if quit_all or frame_id >= total:
                break
        frame_id += 1
    elapsed = time.perf_counter() - t_start
    cap.release()
    if writer is not None:
        writer.release()
    if window is not None:
        cv2.destroyWindow(window)
        cv2.waitKey(1)

    n = len(records)
    if n == 0:
        print(f"[skip] no frames read from {path}")
        return None, quit_all

    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    if samples["v1"]:
        cv2.imwrite(str(out_dir / "v1_sheet.jpg"), contact_sheet([samples["v1"], samples["v1_bin"]]))
        cv2.imwrite(str(out_dir / "v2_sheet.jpg"), contact_sheet([samples["v2"], samples["v2_bin"]]))
        cv2.imwrite(str(out_dir / "track_sheet.jpg"), contact_sheet([samples["track"]]))
    trajectory_plot(records, out_dir / "track_plot.png", f"{path.name} (enemy={enemy})")

    status_count = {s: sum(1 for r in records if r["v2_status"] == s)
                    for s in ("lt2_bars", "no_armor", "one_armor", "too_many")}
    state_count = {s: sum(1 for r in records if r["track_state"] == s)
                   for s in ("none", "init", "update", "coast", "lost")}
    too_many = [r for r in records if r["v2_status"] == "too_many"]
    resolved = sum(1 for r in too_many if r["meas_x"] is not None)
    summary = {
        "video": path.name,
        "enemy": enemy,
        "frames": n,
        "source_resolution": f"{src_w}x{src_h}",
        "processed_resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "ms_per_frame": round(1000 * elapsed / n, 1),
        "meas_std_px": meas_std,
        "v1_boxes_median": float(np.median([r["v1_boxes"] for r in records])),
        "v1_full_frame_box_pct": pct(sum(r["v1_full_frame_box"] for r in records), n),
        "v2_bars_median": float(np.median([r["v2_bars"] for r in records])),
        **{f"v2_{k}_pct": pct(v, n) for k, v in status_count.items()},
        "track_measurement_used_pct": pct(sum(1 for r in records if r["meas_x"] is not None), n),
        "track_estimate_pct": pct(sum(1 for r in records if r["est_x"] is not None), n),
        "track_coast_pct": pct(state_count["coast"], n),
        "too_many_resolved": f"{resolved}/{len(too_many)}",
        "single_detection_rejected": sum(1 for r in records
                                         if r["v2_status"] == "one_armor" and r["meas_x"] is None),
        "track_inits": state_count["init"],
        "track_lost": state_count["lost"],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary, quit_all


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", default=["."],
                        help="video files, directories or glob patterns (default: every video in this folder)")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--enemy", choices=("auto", "blue", "red"), default="auto",
                        help="light-bar colour to detect; auto probes both on a few frames "
                             "(the 2021 code only implemented blue)")
    parser.add_argument("--no-show", action="store_true", help="do not open a playback window")
    parser.add_argument("--no-video", action="store_true", help="do not write annotated.mp4")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed factor for the window")
    parser.add_argument("--display-height", type=int, default=900, help="window height in pixels")
    parser.add_argument("--proc-width", type=int, default=720,
                        help="downscale frames to this width before detection; the 2021 thresholds "
                             "(light bar height 10-150 px) were written for roughly this size. 0 keeps the original")
    parser.add_argument("--trail", type=int, default=30, help="number of past estimates drawn as a trail")
    parser.add_argument("--samples", type=int, default=8, help="frames shown in the contact sheets")
    parser.add_argument("--max-frames", type=int, default=0, help="process at most this many frames (0 = all)")
    parser.add_argument("--max-coast", type=int, default=15, help="frames without a match before the track is dropped")
    parser.add_argument("--meas-std", type=float, default=0.0,
                        help="measurement noise of the armor centre in px; 0 (default) uses 2%% of the frame "
                             "width, which keeps the gate the same size relative to the image")
    parser.add_argument("--accel-std", type=float, default=4.0, help="process noise (px / frame^2)")
    args = parser.parse_args()

    videos = collect_videos(args.inputs or ["."])
    missing = [p for p in videos if not p.exists()]
    videos = [p for p in videos if p.exists()]
    for p in missing:
        print(f"[skip] no such file: {p}")
    if not videos:
        parser.error(f"no videos found in {', '.join(str(i) for i in (args.inputs or ['.']))} "
                     f"(looked for {', '.join(sorted(VIDEO_EXTS))})")
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in videos:
        print(f"[run] {path}")
        summary, quit_all = process(path, args.out, args)
        if summary:
            summaries.append(summary)
            print(f"      {summary['frames']} frames {summary['processed_resolution']} | "
                  f"v2 one armor {summary['v2_one_armor_pct']}% | "
                  f"tracker estimate {summary['track_estimate_pct']}% "
                  f"(measured {summary['track_measurement_used_pct']}%) | "
                  f"too many resolved {summary['too_many_resolved']} -> {args.out / path.stem}")
        if quit_all:
            print("[stop] interrupted")
            break

    if summaries:
        with open(args.out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"[done] {len(summaries)} video(s), summary in {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
