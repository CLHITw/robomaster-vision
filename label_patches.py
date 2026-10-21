"""Label the number on each candidate armor plate, for train_armor_models.py.

    python label_frames.py video2.avi --every 5     # first, click the plate centres
    python label_patches.py video2.avi              # then name what each pair of bars shows

Only frames that already have clicks are visited, and in them only the pairs
of light bars whose centre lies near a click: pairs far from every click are
taken as "not a plate" without asking. For each pair the window shows the
straightened patch the number classifier sees, next to the frame.

Keys: **1-9** the number on the plate, **x** not a plate (two bars of different
plates, a plate seen edge-on, ...), **n** skip, **b** back, **q** save and quit.
Labels go to ``labels/<video>.patches.csv`` after every key, and a second run
resumes at the first unlabelled pair.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from evaluate import load_labels
from replay.armor_patch import plate_quad
from replay.learned import NEGATIVE, all_pairs, load_patch_labels, lookup_patch_label, pair_centre, warp_patch
from replay.pipelines import detect_armors

FIELDS = ["frame", "cx", "cy", "label", "video"]


def save(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["frame"], r["cx"])))


def queue(video, clicks, tol_px, enemy):
    """(frame index, frame, bars, i, j) for every pair near a click."""
    cap = cv2.VideoCapture(str(video))
    items, f = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        points = clicks.get(f, {}).get("points")
        if points:
            bars, _, _ = detect_armors(frame, enemy)
            for i, j in all_pairs(bars):
                c = pair_centre(bars[i], bars[j])
                if any(math.hypot(c[0] - x, c[1] - y) <= tol_px for x, y in points):
                    items.append((f, frame, bars, i, j))
        f += 1
    cap.release()
    return items


def view(frame, bars, i, j, text, display_height):
    out = frame.copy()
    cv2.polylines(out, [plate_quad(bars[i], bars[j]).astype(np.int32)], True, (0, 0, 255), 2)
    s = display_height / out.shape[0]
    out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    patch = (warp_patch(frame, bars[i], bars[j]) * 255).astype(np.uint8)
    patch = cv2.cvtColor(cv2.resize(patch, (256, 256), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    panel = np.zeros((out.shape[0], 256, 3), np.uint8)
    panel[:256] = patch
    cv2.putText(panel, text, (8, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([out, panel])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--enemy", choices=("blue", "red"), default="blue")
    p.add_argument("--tol-frac", type=float, default=0.02, help="same tolerance as evaluate.py")
    p.add_argument("--display-height", type=int, default=720)
    args = p.parse_args()

    clicks_path = Path("labels") / f"{args.video.stem}.csv"
    if not clicks_path.exists():
        raise SystemExit(f"no labels at {clicks_path} - run label_frames.py first")
    out_path = Path("labels") / f"{args.video.stem}.patches.csv"
    clicks = load_labels(clicks_path, 1.0)   # labelling runs at native resolution
    width = int(cv2.VideoCapture(str(args.video)).get(cv2.CAP_PROP_FRAME_WIDTH))
    items = queue(args.video, clicks, args.tol_frac * width, args.enemy)
    existing = load_patch_labels(out_path)
    rows = [{"frame": f, "cx": cx, "cy": cy, "label": lab, "video": args.video.name}
            for f, entries in existing.items() for cx, cy, lab in entries]
    print(f"{len(items)} pairs of bars near a clicked plate, {len(rows)} already labelled")

    k = next((n for n, (f, _, bars, i, j) in enumerate(items)
              if lookup_patch_label(existing, f, pair_centre(bars[i], bars[j])) is None), len(items))
    window = f"{args.video.name}  [1-9] number  [x] not a plate  [n] skip  [b] back  [q] quit"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    while 0 <= k < len(items):
        f, frame, bars, i, j = items[k]
        cx, cy = pair_centre(bars[i], bars[j])
        current = lookup_patch_label(load_patch_labels(out_path), f, (cx, cy))
        cv2.imshow(window, view(frame, bars, i, j, f"#{k + 1}/{len(items)} frame {f}  [{current or '-'}]",
                                args.display_height))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("b"):
            k = max(0, k - 1)
            continue
        if key == ord("n"):
            k += 1
            continue
        if ord("1") <= key <= ord("9") or key == ord("x"):
            label = NEGATIVE if key == ord("x") else chr(key)
            rows = [r for r in rows if not (r["frame"] == f and math.hypot(r["cx"] - cx, r["cy"] - cy) <= 3)]
            rows.append({"frame": f, "cx": round(cx, 1), "cy": round(cy, 1), "label": label,
                         "video": args.video.name})
            save(out_path, rows)
            k += 1
    cv2.destroyAllWindows()
    print(f"{len(rows)} labels in {out_path}")


if __name__ == "__main__":
    main()
