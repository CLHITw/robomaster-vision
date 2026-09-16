"""Click the armor-plate centre on sampled frames to build ground truth.

    python label_frames.py video2.avi --every 5

A window shows one sampled frame at a time:

    left click   add the centre of a visible armor plate (click each one)
    n / space    next frame          b / p   previous frame
    x            no armor plate visible in this frame
    u            clear this frame    q / Esc save and quit

Labels are written to ``labels/<video>.csv`` after every change, so the work
survives a crash and can be resumed by running the same command again: the tool
starts at the first unlabelled frame.

The CSV stores pixel coordinates in the full resolution of the video together
with that resolution, so evaluation can rescale them.
"""

import argparse
import csv
from pathlib import Path

import cv2

HELP_LINES = ["click: add armor centre", "n/space: next   b: back",
              "x: no target   u: clear", "q: save and quit"]


def load_labels(path):
    """frame -> {"visible": 0/1, "points": [(x, y), ...]}; one row per point."""
    labels = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                frame = int(row["frame"])
                entry = labels.setdefault(frame, {"visible": int(row["visible"]), "points": []})
                if row["x"]:
                    entry["visible"] = 1
                    entry["points"].append((float(row["x"]), float(row["y"])))
    return labels


def save_labels(path, labels, width, height, video):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "visible", "x", "y", "width", "height", "video"])
        for frame in sorted(labels):
            lab = labels[frame]
            if lab["points"]:
                for x, y in lab["points"]:
                    w.writerow([frame, 1, round(x, 1), round(y, 1), width, height, video])
            else:
                w.writerow([frame, 0, "", "", width, height, video])


def draw(frame, label, index, total, frame_id, n_done):
    img = frame.copy()
    s = max(1, round(img.shape[1] / 640))
    for k, (x, y) in enumerate(label["points"] if label else []):
        p = (int(x), int(y))
        cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_CROSS, 40 * s, 2 * s)
        cv2.circle(img, p, 14 * s, (0, 0, 255), 2 * s)
        cv2.putText(img, str(k + 1), (p[0] + 16 * s, p[1] - 10 * s), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * s, (0, 0, 255), s, cv2.LINE_AA)
    if label is None:
        status = "unlabelled"
    elif not label["points"]:
        status = "no target"
    else:
        status = f"{len(label['points'])} armor plate(s) marked"
    lines = [f"sample {index + 1}/{total}   frame {frame_id}   {n_done} done", status] + HELP_LINES
    pad, line_h = 8 * s, 20 * s
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (300 * s, int(pad + line_h * len(lines))), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    for k, text in enumerate(lines):
        colour = (0, 255, 255) if k == 0 else (255, 255, 255)
        cv2.putText(img, text, (pad, int(pad + line_h * (k + 0.8))), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * s, colour, s, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--every", type=int, default=5, help="label every Nth frame (default 5)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=0, help="last frame to consider (0 = end of video)")
    p.add_argument("--max", type=int, default=0, help="stop after this many sampled frames (0 = all)")
    p.add_argument("--out", type=Path, default=None, help="default: labels/<video>.csv")
    p.add_argument("--display-height", type=int, default=900)
    args = p.parse_args()

    out = args.out or Path("labels") / f"{args.video.stem}.csv"
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = args.end or total_frames
    frame_ids = list(range(args.start, end, args.every))
    if args.max:
        frame_ids = frame_ids[:args.max]

    labels = load_labels(out)
    print(f"{len(frame_ids)} frames to label, {len(labels)} already in {out}")

    index = next((i for i, f in enumerate(frame_ids) if f not in labels), 0)
    scale = min(1.0, args.display_height / height)
    window = "label armor centre"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, round(width * scale), round(height * scale))

    click = {}

    def on_mouse(event, x, y, flags, _):
        # OpenCV reports image coordinates for a WINDOW_NORMAL window, not window pixels
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pos"] = (min(max(x, 0), width - 1), min(max(y, 0), height - 1))

    cv2.setMouseCallback(window, on_mouse)

    cached = {}
    while 0 <= index < len(frame_ids):
        frame_id = frame_ids[index]
        if frame_id not in cached:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                print(f"cannot read frame {frame_id}, skipping")
                index += 1
                continue
            cached = {frame_id: frame}  # keep only the current frame
        frame = cached[frame_id]

        while True:
            cv2.imshow(window, draw(frame, labels.get(frame_id), index, len(frame_ids), frame_id, len(labels)))
            key = cv2.waitKey(20) & 0xFF
            if "pos" in click:
                x, y = click.pop("pos")
                entry = labels.setdefault(frame_id, {"visible": 1, "points": []})
                entry["visible"] = 1
                entry["points"].append((x, y))
                save_labels(out, labels, width, height, args.video.name)
                continue  # stay on this frame so a second plate can be marked
            if key == ord("x"):
                labels[frame_id] = {"visible": 0, "points": []}
                save_labels(out, labels, width, height, args.video.name)
                index += 1
                break
            if key in (ord("n"), ord(" ")):
                index += 1
                break
            if key in (ord("b"), ord("p")):
                index = max(0, index - 1)
                break
            if key == ord("u"):
                labels.pop(frame_id, None)
                save_labels(out, labels, width, height, args.video.name)
                break
            if key in (ord("q"), 27):
                index = len(frame_ids)
                break

    cap.release()
    cv2.destroyAllWindows()
    save_labels(out, labels, width, height, args.video.name)
    points = sum(len(v["points"]) for v in labels.values())
    visible = sum(1 for v in labels.values() if v["points"])
    print(f"saved {len(labels)} frames ({visible} with a target, {len(labels) - visible} marked empty, "
          f"{points} armor plates in total) to {out}")


if __name__ == "__main__":
    main()
