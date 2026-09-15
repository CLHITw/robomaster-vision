"""Python replays of the two 2021 C++ programs in ``original_2021/``.

The functions use the same OpenCV calls, parameters and branch conditions as
the C++ code, so both programs can be evaluated on videos without a C++
toolchain. Places where the replay has to differ from the C++ code are marked
with ``NOTE``.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------
# v1: original_2021/carcarcar.cpp, main() loop body
# --------------------------------------------------------------------------

def threshold_boxes(frame):
    """Grayscale -> three blurs -> fixed threshold 40 -> contour bounding boxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    gray = cv2.GaussianBlur(gray, (9, 9), 2, 2)
    gray = cv2.medianBlur(gray, 15)
    _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cv2.approxPolyDP(c, 3, True)) for c in contours]
    return boxes, binary


# --------------------------------------------------------------------------
# v2: original_2021/armor_plate/armor_plate.cpp, ImgPreprocess + FindArmor
# --------------------------------------------------------------------------

# Thresholds from armor_plate.h
T_ANGLE_THRE = 5
T_ANGLE_THRE180 = 3
T_ANGLE_THREMIN = 3
T_ANGLE_THRE180MIN = 2
T_HIGH_RAT = 0.2
T_HIGH_RAT_ANGLE = 0.34
T_WHIDTH_RAT = 0.4
T_WHIDTH_RAT_ANGLE = 0.55
L_WH_RAT = 0.8


@dataclass
class LightBar:
    cx: float
    cy: float
    width: float   # short side of the min-area rectangle
    height: float  # long side
    angle: float   # from fitEllipse, degrees in [0, 180)


@dataclass
class Armor:
    cx: float
    cy: float
    width: int     # distance between the two bar centres (nW in the C++ code)
    height: int    # mean bar height (nL)
    bar_i: int
    bar_j: int


def armor_preprocess(frame, enemy="blue"):
    """Colour-difference image -> threshold at 0.65 * max -> 3x dilation."""
    b, _, r = cv2.split(frame)
    if enemy == "blue":
        # C++ else-branch (our_team_ = TEAMBLUE): blue - red, saturating like Mat subtraction
        diff = cv2.subtract(b, r)
    elif enemy == "red":
        # NOTE: the C++ red branch never wrote its output image, so it could not
        # run. This mirrors the blue branch with the channels swapped.
        diff = cv2.subtract(r, b)
    else:
        raise ValueError("enemy must be 'blue' or 'red'")
    _, max_value, _, _ = cv2.minMaxLoc(diff)
    _, binary = cv2.threshold(diff, max_value * 0.65, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(binary, kernel, iterations=3)


def find_light_bars(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rows, cols = binary.shape
    bars = []
    for c in contours:
        if len(c) < 10:
            continue
        (cx, cy), _, angle = cv2.fitEllipse(c)
        _, (w, h), _ = cv2.minAreaRect(c)
        height, width = max(w, h), min(w, h)
        if width / height > L_WH_RAT:
            continue
        x = int(cx - width)
        if x < 0:
            continue
        y = int(cy - height)
        if y < 0:
            continue
        w2 = int(width + width)
        if w2 > cols - x:
            continue
        if w2 > rows - y:  # the C++ code compares the width here, not the height
            continue
        if (angle < 45 or angle > 135) and 10 < height < 150:
            bars.append(LightBar(cx, cy, width, height, angle))
    return bars


def match_armors(bars):
    """Pair light bars with the four cascaded rules of FindArmor."""
    armors = []
    if len(bars) < 2:
        return armors
    for i in range(len(bars) - 1):
        for j in range(i + 1, len(bars)):
            a, b = bars[i], bars[j]
            height_diff, height_sum = abs(a.height - b.height), a.height + b.height
            width_diff, width_sum = abs(a.width - b.width), a.width + b.width
            angle_diff = abs(a.angle - b.angle)
            y_diff, x_diff = abs(a.cy - b.cy), abs(a.cx - b.cx)
            mh_diff = min(a.height, b.height) * 2 / 3
            height_max = max(a.height, b.height)
            n_w = int(math.hypot(a.cx - b.cx, a.cy - b.cy))  # int in C++
            parallel = angle_diff < T_ANGLE_THREMIN or 180 - angle_diff < T_ANGLE_THRE180MIN

            if (y_diff < mh_diff and x_diff < height_max * 4
                    and (angle_diff < T_ANGLE_THRE or 180 - angle_diff < T_ANGLE_THRE)
                    and height_diff / height_sum < T_HIGH_RAT and width_diff / width_sum < T_WHIDTH_RAT):
                ok = y_diff < n_w // 3
            elif (parallel and y_diff < mh_diff * 3 / 2 and x_diff < height_max * 4
                    and height_diff / height_sum < T_HIGH_RAT_ANGLE
                    and width_diff / width_sum < T_WHIDTH_RAT_ANGLE):
                ok = y_diff < n_w // 2
            elif parallel and y_diff < mh_diff * 2 and x_diff < height_max * 4:
                ok = y_diff < n_w // 2
            elif parallel and y_diff < mh_diff * 3 and x_diff < height_max * 5:
                ok = y_diff < n_w // 2
            else:
                ok = False

            if ok:
                armors.append(Armor((a.cx + b.cx) / 2, (a.cy + b.cy) / 2, n_w,
                                    int((a.height + b.height) / 2), i, j))
    return armors


def armor_status(bars, armors):
    """The C++ program only reports a target when exactly one armor is found."""
    if len(bars) < 2:
        return "lt2_bars"
    if not armors:
        return "no_armor"
    if len(armors) == 1:
        return "one_armor"
    return "too_many"


def detect_armors(frame, enemy="blue"):
    binary = armor_preprocess(frame, enemy)
    bars = find_light_bars(binary)
    armors = match_armors(bars)
    return bars, armors, binary
