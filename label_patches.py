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

    python label_patches.py video2.avi --auto       # let the current models suggest the labels

``--auto`` opens no window: the number classifier in ``models/`` labels every
pair that has no label yet, and a contact sheet of the patches with their
suggested labels goes to ``outputs/<video>/patch_review.jpg``. Look through it,
then fix wrong ones in the window (a normal run shows the stored label in
brackets and overwrites it on a key press, ``b`` steps back).
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


def auto_label(items, rows, existing, video, models_dir):
    """Label every unlabelled pair with the number classifier; returns the new rows."""
    from replay.learned import LearnedMatcher
    m = LearnedMatcher.load(models_dir)
    neg = m.classes.index(NEGATIVE)
    todo = [(f, frame, bars, i, j) for f, frame, bars, i, j in items
            if lookup_patch_label(existing, f, pair_centre(bars[i], bars[j])) is None]
    if not todo:
        return []
    probs = m.number_probs([warp_patch(frame, bars[i], bars[j]) for _, frame, bars, i, j in todo])
    new = []
    for (f, _, bars, i, j), pr in zip(todo, probs):
        pr = pr.copy()
        if 1 - pr[neg] < m.number_threshold:
            label = NEGATIVE
        else:
            pr[neg] = -1
            label = m.classes[int(np.argmax(pr))]
        cx, cy = pair_centre(bars[i], bars[j])
        new.append({"frame": f, "cx": round(cx, 1), "cy": round(cy, 1), "label": label, "video": video.name})
    return new


def review_sheet(items, rows, path, per_row=16):
    """Every pair near a click with its stored label, for checking by eye."""
    tiles = []
    labelled = {}
    for r in rows:
        labelled.setdefault(int(r["frame"]), []).append((float(r["cx"]), float(r["cy"]), r["label"]))
    for f, frame, bars, i, j in items:
        lab = lookup_patch_label(labelled, f, pair_centre(bars[i], bars[j])) or "-"
        t = cv2.cvtColor((cv2.resize(warp_patch(frame, bars[i], bars[j]), (72, 72),
                                     interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        colour = (0, 0, 255) if lab == NEGATIVE else (0, 255, 0)
        cv2.putText(t, f"{f}:{lab}", (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        return
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[k:k + per_row]) for k in range(0, len(tiles), per_row)]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--enemy", choices=("blue", "red"), default="blue")
    p.add_argument("--tol-frac", type=float, default=0.02, help="same tolerance as evaluate.py")
    p.add_argument("--display-height", type=int, default=720)
    p.add_argument("--auto", action="store_true", help="label with the models in --models, no window")
    p.add_argument("--models", type=Path, default=Path("models"))
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

    if args.auto:
        new = auto_label(items, rows, existing, args.video, args.models)
        rows += new
        save(out_path, rows)
        sheet = Path("outputs") / args.video.stem / "patch_review.jpg"
        review_sheet(items, rows, sheet)
        counts = {}
        for r in new:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        print(f"{len(new)} pairs labelled by the models ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}) "
              f"-> {out_path}\ncheck them in {sheet}")
        return

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
