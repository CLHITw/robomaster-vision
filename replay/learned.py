"""Learned light-bar pairing (2026 addition, not part of the 2021 code).

The 2021 detector pairs light bars with four cascaded hand-written rules
(``match_armors`` in ``replay/pipelines.py``), and the evaluation showed those
rules are its bottleneck: a correct pair of bars is found far more often than
the rules accept it. This module replaces the rules with two small models:

* a **pair classifier**, a small MLP on scale-free geometry of the two bars
  (spacing, height ratio, angle difference, ...), that says whether the two
  bars could belong to one plate;
* a **number classifier**, a small CNN on the straightened patch between the
  bars (``replay/armor_patch.py``), that says which number is printed there,
  or that there is no armor plate at all.

Both are trained by ``train_armor_models.py`` and stored as ONNX, so running
them needs only OpenCV's ``cv2.dnn`` (in Python here, and the same files load
from OpenCV in C++, OpenVINO or TensorRT on the robot). PyTorch is only needed
to train.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .armor_patch import warp_patch

NEGATIVE = "neg"
FEATURE_NAMES = (
    "spacing",          # distance between bar centres / mean bar length
    "dx",               # horizontal offset / mean bar length
    "dy",               # vertical offset / mean bar length
    "length_ratio",     # shorter bar / longer bar
    "width_ratio",      # thinner bar / thicker bar
    "angle_diff",       # difference of the two bar tilts, degrees / 90
    "mean_tilt",        # mean tilt of the bars from vertical, degrees / 90
    "skew",             # angle between the line joining the bars and the bars' normal, degrees / 90
    "aspect_left",      # bar width / bar length, left bar
    "aspect_right",     # same, right bar
)


def _tilt(bar):
    """Signed tilt from vertical in degrees; fitEllipse gives [0, 180) with 0 = vertical."""
    return bar.angle if bar.angle < 90 else bar.angle - 180


def pair_features(a, b):
    """Scale-free description of two light bars, in the order of ``FEATURE_NAMES``."""
    left, right = (a, b) if a.cx <= b.cx else (b, a)
    length = (left.height + right.height) / 2
    dx, dy = right.cx - left.cx, right.cy - left.cy
    t_left, t_right = _tilt(left), _tilt(right)
    mean_tilt = (t_left + t_right) / 2
    # direction of the line joining the bars, against the normal of the mean bar direction
    join = math.degrees(math.atan2(dy, dx))
    skew = join - mean_tilt
    return np.array([
        math.hypot(dx, dy) / length,
        abs(dx) / length,
        abs(dy) / length,
        min(left.height, right.height) / max(left.height, right.height),
        min(left.width, right.width) / max(left.width, right.width, 1e-6),
        abs(t_left - t_right) / 90,
        abs(mean_tilt) / 90,
        abs(skew) / 90,
        left.width / left.height,
        right.width / right.height,
    ], dtype=np.float32)


def all_pairs(bars):
    return [(i, j) for i in range(len(bars)) for j in range(i + 1, len(bars))]


def pair_centre(a, b):
    return (a.cx + b.cx) / 2, (a.cy + b.cy) / 2


# --------------------------------------------------------------------------
# training samples
# --------------------------------------------------------------------------

def load_patch_labels(path):
    """``labels/<video>.patches.csv`` -> {frame: [(cx, cy, label), ...]}.

    Written by ``label_patches.py``. A label is a class name (``"3"``) or
    ``"neg"`` for a pair of bars that is not an armor plate. Positions are in
    pixels of the original video.
    """
    import csv
    out = {}
    if not Path(path).exists():
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(int(row["frame"]), []).append((float(row["cx"]), float(row["cy"]), row["label"]))
    return out


def lookup_patch_label(patch_labels, frame, centre, scale=1.0, tol=3.0):
    """Label of the pair whose centre is within ``tol`` px of ``centre``, or None."""
    best, best_d = None, tol
    for cx, cy, lab in patch_labels.get(frame, ()):
        d = math.hypot(cx * scale - centre[0], cy * scale - centre[1])
        if d <= best_d:
            best, best_d = lab, d
    return best


@dataclass
class PairSample:
    frame: int
    cx: float
    cy: float
    features: np.ndarray
    patch: np.ndarray
    near_click: bool            # pair centre within tolerance of a clicked plate
    label: str = None           # number, NEGATIVE, or None when not labelled by hand

    @property
    def is_plate(self):
        """Target for the pair classifier: a hand label wins over the click distance."""
        if self.label is not None:
            return self.label != NEGATIVE
        return self.near_click


def frame_samples(frame_img, bars, frame, clicks, tol_px, patch_labels=None, scale=1.0):
    """One ``PairSample`` per pair of bars in a labelled frame."""
    samples = []
    for i, j in all_pairs(bars):
        a, b = bars[i], bars[j]
        c = pair_centre(a, b)
        near = any(math.hypot(c[0] - x, c[1] - y) <= tol_px for x, y in clicks)
        lab = lookup_patch_label(patch_labels or {}, frame, c, scale) if patch_labels else None
        if lab is None and not near:
            lab = NEGATIVE   # far from every clicked plate
        samples.append(PairSample(frame, c[0], c[1], pair_features(a, b),
                                  warp_patch(frame_img, a, b), near, lab))
    return samples


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

@dataclass
class LearnedArmor:
    """Same fields as ``pipelines.Armor`` plus the two model outputs."""
    cx: float
    cy: float
    width: int
    height: int
    bar_i: int
    bar_j: int
    pair_score: float = 0.0
    number: str = None
    number_score: float = 0.0

    @property
    def score(self):
        return self.pair_score * self.number_score


@dataclass
class LearnedMatcher:
    """Runs both ONNX models with ``cv2.dnn``; drop-in replacement for ``match_armors``."""
    pair_net: object
    number_net: object
    classes: list
    feature_mean: np.ndarray
    feature_std: np.ndarray
    pair_threshold: float = 0.5
    number_threshold: float = 0.5
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, model_dir="models"):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "armor_models.json").read_text(encoding="utf-8"))
        return cls(cv2.dnn.readNetFromONNX(str(model_dir / meta["pair_model"])),
                   cv2.dnn.readNetFromONNX(str(model_dir / meta["number_model"])),
                   meta["classes"],
                   np.array(meta["feature_mean"], dtype=np.float32),
                   np.array(meta["feature_std"], dtype=np.float32),
                   meta.get("pair_threshold", 0.5), meta.get("number_threshold", 0.5), meta)

    def pair_scores(self, features):
        x = (np.asarray(features, dtype=np.float32) - self.feature_mean) / self.feature_std
        self.pair_net.setInput(x)
        logits = self.pair_net.forward().reshape(-1)
        return 1 / (1 + np.exp(-logits))

    def number_probs(self, patches):
        blob = np.asarray(patches, dtype=np.float32)[:, None, :, :]
        self.number_net.setInput(blob)
        logits = self.number_net.forward()
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def match(self, frame, bars):
        """Accepted plates, best first; no light bar is used by two plates."""
        pairs = all_pairs(bars)
        if not pairs:
            return []
        p_pair = self.pair_scores([pair_features(bars[i], bars[j]) for i, j in pairs])
        probs = np.zeros((len(pairs), len(self.classes)), dtype=np.float32)
        probs[:, self.classes.index(NEGATIVE)] = 1.0
        keep = [k for k, p in enumerate(p_pair) if p >= self.pair_threshold]
        if keep:
            probs[keep] = self.number_probs([warp_patch(frame, bars[pairs[k][0]], bars[pairs[k][1]])
                                             for k in keep])
        return select_plates(bars, pairs, p_pair, probs, self.classes,
                             self.pair_threshold, self.number_threshold)


def select_plates(bars, pairs, p_pair, probs, classes, pair_threshold, number_threshold):
    """Turn per-pair model outputs into plates: threshold both, then claim bars best first."""
    neg = classes.index(NEGATIVE)
    found = []
    for (i, j), pp, pr in zip(pairs, p_pair, probs):
        p_plate = 1 - float(pr[neg])
        if pp < pair_threshold or p_plate < number_threshold:
            continue
        pr = np.array(pr, dtype=float)
        pr[neg] = -1
        a, b = bars[i], bars[j]
        cx, cy = pair_centre(a, b)
        found.append(LearnedArmor(cx, cy, int(math.hypot(a.cx - b.cx, a.cy - b.cy)),
                                  int((a.height + b.height) / 2), i, j,
                                  float(pp), classes[int(np.argmax(pr))], p_plate))
    found.sort(key=lambda arm: arm.score, reverse=True)
    used, result = set(), []
    for arm in found:
        if arm.bar_i in used or arm.bar_j in used:
            continue
        used.update((arm.bar_i, arm.bar_j))
        result.append(arm)
    return result
