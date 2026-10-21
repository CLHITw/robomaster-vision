"""Watch the 2021 pairing rules and the learned pairing side by side.

    python compare_matchers.py video2.avi
    python compare_matchers.py video2.avi --no-show            # only write the video
    python compare_matchers.py video2.avi --figure docs/figures/learned_vs_rules.jpg

Left: the 2021 rules. Right: the models in ``models/`` (``train_armor_models.py``),
with the number they read and their score above each plate. Both feed the same
tracker. On frames that have a clicked label (``labels/<video>.csv``) a white
circle marks the tolerance around each clicked plate, so a hit or a miss is
visible at a glance; the counters at the bottom add up hits over the labelled
frames seen so far.

Keys: **space** pause, **d** / **a** one frame forward / back while paused,
**s** save the current frame as a png, **q** quit. The side-by-side video goes
to ``outputs/<video>/compare.mp4``.

The models were trained on video2.avi and the committed clip, so on those the
right-hand side shows what it learned, not how it generalises; the held-out
numbers come from ``train_armor_models.py``.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from evaluate import load_labels
from replay.learned import LearnedMatcher
from replay.metrics import closest_pair
from replay.pipelines import armor_status, detect_armors
from replay.tracking import ArmorTracker
from replay.visualize import draw_track, draw_v2, label

WHITE = (255, 255, 255)


def draw_clicks(img, points, tol):
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), int(tol), WHITE, 1, cv2.LINE_AA)


def panel(frame, title, bars, armors, step, trail, points, tol, score):
    out = draw_track(draw_v2(frame, bars, armors), [(a.cx, a.cy) for a in armors], step, trail)
    if points:
        draw_clicks(out, points, tol)
    label(out, title)
    hits, seen = score
    text = f"correct plate on {hits}/{seen} labelled frames" if seen else "no labelled frame yet"
    s = max(1, round(max(out.shape[:2]) / 640))
    cv2.putText(out, text, (10 * s, out.shape[0] - 12 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 0, 0),
                3 * s, cv2.LINE_AA)
    cv2.putText(out, text, (10 * s, out.shape[0] - 12 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, WHITE,
                s, cv2.LINE_AA)
    return out


def run(args):
    labels_path = Path("labels") / f"{args.video.stem}.csv"
    labels = load_labels(labels_path, 1.0) if labels_path.exists() else {}
    matcher = LearnedMatcher.load(args.models)
    if any(Path(v.replace("\\", "/")).name == args.video.name for v in matcher.meta.get("trained_on", [])):
        print(f"note: the models were trained on {args.video.name}, so the right-hand side shows what they "
              f"learned; train_armor_models.py --cv reports held-out scores.")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    tol = args.tol_frac * width
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    trackers = {k: ArmorTracker(max_coast=2, meas_std=round(0.008 * width, 1)) for k in ("rules", "learned")}
    trails = {k: [] for k in trackers}
    score = {k: [0, 0] for k in trackers}
    shown, figure_rows = [], []

    out_dir = Path("outputs") / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    window = None if args.no_show else f"{args.video.name}: 2021 rules | learned   [space] [a/d] [s] [q]"
    if window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bars, rule_armors, _ = detect_armors(frame, args.enemy)
        found = {"rules": rule_armors, "learned": matcher.match(frame, bars)}
        points = labels.get(f, {}).get("points") or []
        panels, hit = [], {}
        for key, armors in found.items():
            step = trackers[key].step([(a.cx, a.cy) for a in armors])
            trails[key] = (trails[key] + [step.estimate])[-30:]
            # what each detector reports: the rules only with exactly one plate, the models their best plate
            if key == "rules":
                target = (armors[0].cx, armors[0].cy) if armor_status(bars, armors) == "one_armor" else None
            else:
                target = (armors[0].cx, armors[0].cy) if armors else None
            hit[key] = target is not None and closest_pair(points, [target]) <= tol
            if points:
                score[key][0] += hit[key]
                score[key][1] += 1
            title = f"{'2021 rules' if key == 'rules' else 'learned'}  #{f}"
            panels.append(panel(frame, title, bars, armors, step, trails[key], points, tol, score[key]))
        view = np.hstack(panels)
        if points and hit["learned"] and not hit["rules"]:
            figure_rows.append(view)

        if not args.no_video:
            if writer is None:
                scale = min(1.0, 1920 / view.shape[1])   # two full-size frames side by side are 2560 px wide
                size = (int(view.shape[1] * scale), int(view.shape[0] * scale))
                writer = cv2.VideoWriter(str(out_dir / "compare.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            writer.write(cv2.resize(view, size, interpolation=cv2.INTER_AREA))
        if window:
            shown = (shown + [view])[-200:]
            if not show(window, shown, fps, out_dir, args):
                break
        f += 1
    cap.release()
    if writer is not None:
        writer.release()
    if window:
        cv2.destroyAllWindows()

    for key in ("rules", "learned"):
        hits, seen = score[key]
        if seen:
            print(f"{key:8s}: correct plate on {hits}/{seen} labelled frames ({100 * hits / seen:.1f}%)")
    if not args.no_video:
        print(f"side-by-side video: {out_dir / 'compare.mp4'}")
    if args.figure and figure_rows:
        picks = [figure_rows[k] for k in np.linspace(0, len(figure_rows) - 1, min(args.figure_rows,
                                                                                  len(figure_rows))).astype(int)]
        fig = np.vstack(picks)
        fig = cv2.resize(fig, None, fx=args.figure_width / fig.shape[1], fy=args.figure_width / fig.shape[1],
                         interpolation=cv2.INTER_AREA)
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.figure), fig, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"figure: {len(picks)} of the {len(figure_rows)} labelled frames where only the learned "
              f"pairing is right -> {args.figure}")


def show(window, shown, fps, out_dir, args):
    """Display the newest frame; returns False to stop. Paused, a/d step through the last 200 frames."""
    k = len(shown) - 1
    paused = False
    while True:
        cv2.imshow(window, shown[k])
        key = cv2.waitKey(0 if paused else max(1, int(1000 / fps))) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord(" "):
            paused = not paused
        elif key == ord("s"):
            path = out_dir / f"compare_{len(list(out_dir.glob('compare_*.png'))):03d}.png"
            cv2.imwrite(str(path), shown[k])
            print(f"saved {path}")
        elif paused and key == ord("a"):
            k = max(0, k - 1)
        elif paused and key == ord("d"):
            if k == len(shown) - 1:
                return True
            k += 1
        elif not paused:
            return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--enemy", choices=("blue", "red"), default="blue")
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--tol-frac", type=float, default=0.02, help="same tolerance as evaluate.py")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    p.add_argument("--no-video", action="store_true", help="do not write compare.mp4")
    p.add_argument("--figure", type=Path, default=None,
                   help="write a still of labelled frames where only the learned pairing is right")
    p.add_argument("--figure-rows", type=int, default=3)
    p.add_argument("--figure-width", type=int, default=1600)
    run(p.parse_args())


if __name__ == "__main__":
    main()
