"""Train the learned light-bar pairing and compare it with the 2021 rules.

    python train_armor_models.py clips/robot_clip.mp4
    python train_armor_models.py video2.avi clips/robot_clip.mp4 --cv 5

Needs PyTorch (``pip install -r requirements-ml.txt``); running the trained
models afterwards does not.

For each video it reads the clicked plate centres (``labels/<video>.csv``) and,
if present, the hand labels of individual bar pairs
(``labels/<video>.patches.csv``, written by ``label_patches.py``). Every pair of
light bars in a labelled frame becomes a training sample: a pair whose centre
is far from every click is "not a plate", a pair near a click is a plate with
the number from its hand label, or the generic class ``plate`` when it has none.

Training and scoring on the same frames would only show how well the models
memorise them, so the comparison is cross-validated in time: the video is cut
into ``--cv`` contiguous blocks, and each block is scored with models trained
on the labelled frames of the other blocks. Neighbouring frames look almost
the same, which is why the folds are blocks and not every k-th frame.

Then both models are trained on all samples and written to ``models/`` as
ONNX, with ``armor_models.json`` holding the class names, the feature scaling
and the thresholds that ``replay.learned.LearnedMatcher`` reads.
"""

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from evaluate import load_labels
from replay.learned import (FEATURE_NAMES, NEGATIVE, all_pairs, frame_samples, load_patch_labels,
                            pair_features, select_plates)
from replay.armor_patch import PATCH_SIZE, warp_patch
from replay.metrics import evaluate
from replay.pipelines import armor_status, detect_armors
from replay.tracking import ArmorTracker

GENERIC = "plate"   # class of a clicked plate whose number was not labelled


# --------------------------------------------------------------------------
# decode every video once
# --------------------------------------------------------------------------

@dataclass
class FrameCache:
    bars: list
    pairs: list
    features: np.ndarray      # (n_pairs, n_features)
    patches: np.ndarray       # (n_pairs, PATCH_SIZE, PATCH_SIZE)
    rule_armors: list         # (cx, cy) from the 2021 rules
    rule_status: str


def decode(video, enemy, proc_width, clicks, tol_px, patch_labels):
    cap = cv2.VideoCapture(str(video))
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames, samples, width = [], [], None
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if proc_width and frame.shape[1] > proc_width:
            h = round(frame.shape[0] * proc_width / frame.shape[1])
            frame = cv2.resize(frame, (proc_width, h), interpolation=cv2.INTER_AREA)
        width = frame.shape[1]
        scale = width / src_width
        bars, armors, _ = detect_armors(frame, enemy)
        pairs = all_pairs(bars)
        feats = np.array([pair_features(bars[i], bars[j]) for i, j in pairs], dtype=np.float32)
        patches = np.array([warp_patch(frame, bars[i], bars[j]) for i, j in pairs], dtype=np.float32)
        frames.append(FrameCache(bars, pairs, feats.reshape(-1, len(FEATURE_NAMES)),
                                 patches.reshape(-1, PATCH_SIZE, PATCH_SIZE),
                                 [(a.cx, a.cy) for a in armors], armor_status(bars, armors)))
        if f in clicks:
            samples += frame_samples(frame, bars, f, clicks[f]["points"], tol_px, patch_labels, scale)
        f += 1
    cap.release()
    return frames, samples, width


def number_target(s):
    if s.label is not None:
        return s.label
    return GENERIC if s.near_click else NEGATIVE


# --------------------------------------------------------------------------
# the two models
# --------------------------------------------------------------------------

def build_pair_net(n_in):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))


def build_number_net(n_classes):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),     # 16x16
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 8x8
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 4x4
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, n_classes))


def augment(patch, rng):
    """Small rotation, scale and shift, then gamma, blur and noise."""
    c = PATCH_SIZE / 2
    m = cv2.getRotationMatrix2D((c, c), rng.uniform(-8, 8), rng.uniform(0.9, 1.1))
    m[:, 2] += rng.uniform(-2, 2, size=2)
    p = cv2.warpAffine(patch, m, (PATCH_SIZE, PATCH_SIZE), borderMode=cv2.BORDER_REFLECT)
    p = np.clip(p, 0, 1) ** rng.uniform(0.6, 1.6)
    if rng.random() < 0.3:
        p = cv2.GaussianBlur(p, (3, 3), 0)
    p = p + rng.normal(0, 0.04, p.shape)
    return np.clip(p, 0, 1).astype(np.float32)


def train_pair_net(x, y, epochs, seed):
    import torch
    torch.manual_seed(seed)
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    xt = torch.tensor((x - mean) / std)
    yt = torch.tensor(y, dtype=torch.float32)
    net = build_pair_net(x.shape[1])
    pos = max(float(y.sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / pos))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(xt).squeeze(1), yt)
        loss.backward()
        opt.step()
    net.eval()
    return net, mean.astype(np.float32), std.astype(np.float32)


def train_number_net(patches, targets, classes, epochs, seed):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = np.array([classes.index(t) for t in targets], dtype=np.int64)   # int32 on Windows; torch wants int64
    counts = np.bincount(y, minlength=len(classes)).astype(float)
    # every class is drawn about equally often, so the many negatives do not drown the numbers
    weights = 1.0 / counts[y]
    weights /= weights.sum()
    net = build_number_net(len(classes))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    batch = 64
    steps = max(1, len(y) // batch)
    for _ in range(epochs):
        net.train()
        for _ in range(steps):
            idx = rng.choice(len(y), size=batch, p=weights)
            xb = torch.tensor(np.stack([augment(patches[k], rng) for k in idx])[:, None])
            opt.zero_grad()
            loss = loss_fn(net(xb), torch.tensor(y[idx]))
            loss.backward()
            opt.step()
    net.eval()
    return net


@dataclass
class Models:
    pair_net: object
    number_net: object
    classes: list
    mean: np.ndarray
    std: np.ndarray

    def outputs(self, fc):
        import torch
        if not fc.pairs:
            return np.zeros(0), np.zeros((0, len(self.classes)))
        with torch.no_grad():
            logit = self.pair_net(torch.tensor((fc.features - self.mean) / self.std)).squeeze(1).numpy()
            probs = torch.softmax(self.number_net(torch.tensor(fc.patches[:, None])), dim=1).numpy()
        return 1 / (1 + np.exp(-logit)), probs


def fit(samples, classes, args, seed):
    x = np.stack([s.features for s in samples])
    y = np.array([s.is_plate for s in samples], dtype=np.float32)
    pair_net, mean, std = train_pair_net(x, y, args.pair_epochs, seed)
    numbered = [s for s in samples if number_target(s) in classes]
    number_net = train_number_net(np.stack([s.patch for s in numbered]),
                                  [number_target(s) for s in numbered], classes, args.number_epochs, seed)
    return Models(pair_net, number_net, classes, mean, std)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def track(frame_outputs, width, max_coast=2):
    """frame_outputs: list of (candidates best first, status, target) -> evaluate() records."""
    tracker = ArmorTracker(max_coast=max_coast, meas_std=round(0.008 * width, 1))
    records = []
    for candidates, status, target in frame_outputs:
        step = tracker.step(candidates)
        rec = {"candidates": candidates, "status": status, "state": step.state, "estimate": step.estimate}
        if target is not ...:
            rec["target"] = target
        records.append(rec)
    return records


def learned_outputs(models, frames, args):
    out, numbers = [], []
    for fc in frames:
        p_pair, probs = models.outputs(fc)
        plates = select_plates(fc.bars, fc.pairs, p_pair, probs, models.classes,
                               args.pair_threshold, args.number_threshold)
        cands = [(p.cx, p.cy) for p in plates]
        status = "one_armor" if len(plates) == 1 else ("too_many" if plates else "no_armor")
        out.append((cands, status, cands[0] if cands else None))
        numbers.append(plates)
    return out, numbers


def cross_validate(videos, args, classes):
    """Out-of-fold records for every video, plus out-of-fold number predictions."""
    results = {}
    confusion = Counter()
    for name, v in videos.items():
        n = len(v["frames"])
        edges = np.linspace(0, n, args.cv + 1).astype(int)
        learned = [None] * n
        for k in range(args.cv):
            lo, hi = edges[k], edges[k + 1]
            train = [s for other, w in videos.items() for s in w["samples"]
                     if other != name or not lo <= s.frame < hi]
            models = fit(train, classes, args, args.seed + k)
            out, _ = learned_outputs(models, v["frames"][lo:hi], args)
            learned[lo:hi] = out
            # number accuracy on the hand-labelled plates of this block
            for s in v["samples"]:
                if lo <= s.frame < hi and s.label not in (None, NEGATIVE):
                    fc = v["frames"][s.frame]
                    k_pair = min(range(len(fc.pairs)), key=lambda q: _centre_dist(fc, q, s))
                    _, probs = models.outputs(FrameCache(fc.bars, [fc.pairs[k_pair]], fc.features[[k_pair]],
                                                         fc.patches[[k_pair]], [], ""))
                    pr = probs[0].copy()
                    pr[classes.index(NEGATIVE)] = -1
                    confusion[(s.label, classes[int(np.argmax(pr))])] += 1
        rules = [(fc.rule_armors, fc.rule_status, ...) for fc in v["frames"]]
        results[name] = {
            "rules": evaluate(track(rules, v["width"]), v["clicks"], v["tol"]),
            "learned": evaluate(track(learned, v["width"]), v["clicks"], v["tol"]),
        }
    return results, confusion


def _centre_dist(fc, q, s):
    i, j = fc.pairs[q]
    a, b = fc.bars[i], fc.bars[j]
    return math.hypot((a.cx + b.cx) / 2 - s.cx, (a.cy + b.cy) / 2 - s.cy)


def print_comparison(name, res):
    rows = [
        ("detector reports a plate, correct", "detector_single_precision_pct", "detector_single_correct",
         "detector_single_frames"),
        ("  -> of all frames with a target", "detector_usable_pct", None, None),
        ("tracker, correct", "tracker_precision_pct", "tracker_correct", "tracker_frames_with_estimate"),
        ("  -> of all frames with a target", "tracker_usable_pct", None, None),
        ("false target on empty frames", "false_target_when_empty_pct", None, None),
    ]
    print(f"\n{name}: {res['rules']['frames_with_target']} labelled frames with a target, "
          f"tolerance {res['rules']['tolerance_px']} px")
    print(f"{'':36s} {'2021 rules':>16s} {'learned (CV)':>16s}")
    for title, pct, num, den in rows:
        cells = []
        for key in ("rules", "learned"):
            r = res[key]
            v = r.get(pct)
            cell = "-" if v is None else f"{v}%"
            if num:
                cell = f"{r[num]}/{r[den]}  {cell}"
            cells.append(cell)
        print(f"{title:36s} {cells[0]:>16s} {cells[1]:>16s}")
    for key in ("rules", "learned"):
        e = res[key]["detector_error"]
        print(f"  median error, {key:8s}: {e['median_px']} px (p90 {e['p90_px']} px)")


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def export(models, args):
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(models.pair_net, torch.zeros(1, len(FEATURE_NAMES)), out / "pair_mlp.onnx",
                      input_names=["features"], output_names=["logit"],
                      dynamic_axes={"features": {0: "n"}, "logit": {0: "n"}}, dynamo=False)
    torch.onnx.export(models.number_net, torch.zeros(1, 1, PATCH_SIZE, PATCH_SIZE), out / "number_cnn.onnx",
                      input_names=["patch"], output_names=["logits"],
                      dynamic_axes={"patch": {0: "n"}, "logits": {0: "n"}}, dynamo=False)
    meta = {
        "pair_model": "pair_mlp.onnx",
        "number_model": "number_cnn.onnx",
        "classes": models.classes,
        "features": list(FEATURE_NAMES),
        "feature_mean": models.mean.tolist(),
        "feature_std": models.std.tolist(),
        "patch_size": PATCH_SIZE,
        "pair_threshold": args.pair_threshold,
        "number_threshold": args.number_threshold,
        "trained_on": [str(v) for v in args.videos],
    }
    (out / "armor_models.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", type=Path, nargs="+")
    p.add_argument("--enemy", choices=("blue", "red"), default="blue")
    p.add_argument("--proc-width", type=int, default=0, help="0 = native resolution")
    p.add_argument("--tol-frac", type=float, default=0.02)
    p.add_argument("--cv", type=int, default=5, help="number of time blocks; 0 skips the comparison")
    p.add_argument("--pair-epochs", type=int, default=400)
    p.add_argument("--number-epochs", type=int, default=60)
    p.add_argument("--pair-threshold", type=float, default=0.3)
    p.add_argument("--number-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--no-export", action="store_true")
    args = p.parse_args()

    videos = {}
    for video in args.videos:
        clicks_path = Path("labels") / f"{video.stem}.csv"
        if not clicks_path.exists():
            raise SystemExit(f"no labels at {clicks_path} - run label_frames.py first")
        src_width = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_WIDTH))
        if not src_width:
            raise SystemExit(f"cannot read {video}")
        width = min(src_width, args.proc_width) if args.proc_width else src_width
        clicks = load_labels(clicks_path, width / src_width)
        tol = args.tol_frac * width
        patch_labels = load_patch_labels(Path("labels") / f"{video.stem}.patches.csv")
        print(f"decoding {video} ...")
        frames, samples, width = decode(video, args.enemy, args.proc_width, clicks, tol, patch_labels)
        videos[video.name] = {"frames": frames, "samples": samples, "width": width,
                              "clicks": clicks, "tol": tol}

    samples = [s for v in videos.values() for s in v["samples"]]
    counts = Counter(number_target(s) for s in samples)
    classes = [NEGATIVE] + sorted(c for c in counts if c != NEGATIVE)
    print(f"{len(samples)} bar pairs in labelled frames: "
          + ", ".join(f"{c} {counts[c]}" for c in classes))
    if len(classes) < 2:
        raise SystemExit("no pair of bars lies near a clicked plate - nothing to learn")

    if args.cv:
        results, confusion = cross_validate(videos, args, classes)
        for name, res in results.items():
            print_comparison(name, res)
        if confusion:
            total = sum(confusion.values())
            right = sum(n for (t, pr), n in confusion.items() if t == pr)
            print(f"\nnumber of the plate, out of fold: {right}/{total} correct")
            for (t, pr), n in sorted(confusion.items()):
                if t != pr:
                    print(f"  labelled {t}, predicted {pr}: {n}")

    if not args.no_export:
        models = fit(samples, classes, args, args.seed)
        out = export(models, args)
        print(f"\nmodels written to {out}/ (pair_mlp.onnx, number_cnn.onnx, armor_models.json)")


if __name__ == "__main__":
    main()
