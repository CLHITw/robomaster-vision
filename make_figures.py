"""Build the README figures for one video.

    python make_figures.py video2.avi --proc-width 0 --gif-start 300 --gif-frames 200

Writes into docs/figures/:
    three_stages.jpg   the same frames through v1, v2 and v2 + tracker
    timeline.png       per-frame detector status and tracker state over the whole video
    demo.gif           a short annotated clip
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from replay.pipelines import armor_status, detect_armors, threshold_boxes
from replay.tracking import ArmorTracker
from replay.visualize import contact_sheet, draw_hud, draw_track, draw_v1, draw_v2, label

STATUS_COLOUR = {"lt2_bars": "#d9d9d9", "no_armor": "#f0f0f0", "one_armor": "#2ca02c", "too_many": "#ff7f0e"}
STATE_COLOUR = {"none": "#f0f0f0", "init": "#9467bd", "update": "#1f77b4", "coast": "#9ecae1", "lost": "#d62728"}


def run(video, enemy, proc_width, keep_ids=()):
    """Replay the whole video once.

    Only the frames in ``keep_ids`` are kept in memory; every frame contributes
    its per-frame result, so the tracker state is identical to a full run.
    """
    keep_ids = set(keep_ids)
    cap = cv2.VideoCapture(str(video))
    tracker = ArmorTracker()
    trail, records, kept = [], [], {}
    meas_std = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if proc_width and frame.shape[1] > proc_width:
            h = round(frame.shape[0] * proc_width / frame.shape[1])
            frame = cv2.resize(frame, (proc_width, h), interpolation=cv2.INTER_AREA)
        if meas_std is None:  # same resolution-relative gate as run_videos.py
            meas_std = round(0.02 * frame.shape[1], 1)
            tracker = ArmorTracker(meas_std=meas_std)
        boxes, _ = threshold_boxes(frame)
        bars, armors, _ = detect_armors(frame, enemy)
        status = armor_status(bars, armors)
        step = tracker.step([(a.cx, a.cy) for a in armors])
        trail.append(step.estimate)
        trail = trail[-30:]
        records.append({"status": status, "state": step.state, "boxes": boxes, "bars": bars,
                        "armors": armors, "step": step, "trail": list(trail)})
        if i in keep_ids:
            kept[i] = frame
        i += 1
    cap.release()
    return kept, records


def three_stages(frames, records, ids, path):
    rows = [[], [], []]
    for i in ids:
        r = records[i]
        rows[0].append(label(draw_v1(frames[i], r["boxes"]), f"#{i} v1: {len(r['boxes'])} boxes"))
        rows[1].append(label(draw_v2(frames[i], r["bars"], r["armors"]),
                             f"#{i} v2: {len(r['bars'])} bars -> {r['status']}"))
        track = draw_track(draw_v2(frames[i], r["bars"], r["armors"]), [(a.cx, a.cy) for a in r["armors"]],
                           r["step"], r["trail"])
        rows[2].append(label(track, f"#{i} + tracker: {r['state']}"))
    cv2.imwrite(str(path), contact_sheet(rows, tile_height=260), [cv2.IMWRITE_JPEG_QUALITY, 88])


def timeline(records, path, title):
    n = len(records)
    fig, ax = plt.subplots(figsize=(11, 2.6))
    for y, key, colours in ((1, "status", STATUS_COLOUR), (0, "state", STATE_COLOUR)):
        for i, r in enumerate(records):
            ax.add_patch(plt.Rectangle((i, y), 1, 0.8, color=colours[r[key]], linewidth=0))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 2)
    ax.set_yticks([0.4, 1.4])
    ax.set_yticklabels(["tracker", "armor_plate"], fontsize=9)
    ax.set_xlabel("frame")
    ax.set_title(title, fontsize=10)
    handles = [mpatches.Patch(color=c, label=k) for k, c in
               list(STATUS_COLOUR.items())[2:] + list(STATE_COLOUR.items())[1:]]
    ax.legend(handles=handles, ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.45), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def gif(frames, records, path, start, count, step_every, width, fps):
    images = []
    total = len(records)
    for i in range(start, min(start + count, total), step_every):
        r = records[i]
        img = draw_track(draw_v2(frames[i], r["bars"], r["armors"]), [(a.cx, a.cy) for a in r["armors"]],
                         r["step"], r["trail"])
        draw_hud(img, i, total, r["status"], r["step"], len(r["bars"]))
        h = round(img.shape[0] * width / img.shape[1])
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
        images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).quantize(colors=128, method=Image.MEDIANCUT))
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--enemy", choices=("blue", "red"), default="blue")
    p.add_argument("--proc-width", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    p.add_argument("--stage-frames", type=int, nargs="*", default=None, help="frame ids for three_stages.jpg")
    p.add_argument("--gif-start", type=int, default=0)
    p.add_argument("--gif-frames", type=int, default=240)
    p.add_argument("--gif-every", type=int, default=3)
    p.add_argument("--gif-width", type=int, default=420)
    p.add_argument("--gif-fps", type=int, default=12)
    args = p.parse_args()

    ids = args.stage_frames
    if ids is None:
        probe = cv2.VideoCapture(str(args.video))
        n_probe = int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        probe.release()
        ids = np.linspace(0, n_probe - 1, 4).astype(int).tolist()
    gif_ids = range(args.gif_start, args.gif_start + args.gif_frames, args.gif_every)
    frames, records = run(args.video, args.enemy, args.proc_width, set(ids) | set(gif_ids))
    print(f"{len(records)} frames replayed, {len(frames)} kept in memory")
    args.out.mkdir(parents=True, exist_ok=True)

    three_stages(frames, records, ids, args.out / "three_stages.jpg")
    timeline(records, args.out / "timeline.png", f"{args.video.name}: detector output and tracker state per frame")
    gif(frames, records, args.out / "demo.gif", args.gif_start, args.gif_frames,
        args.gif_every, args.gif_width, args.gif_fps)
    for f in ("three_stages.jpg", "timeline.png", "demo.gif"):
        print(f"  {args.out / f}: {(args.out / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
