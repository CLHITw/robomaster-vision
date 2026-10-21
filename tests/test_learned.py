from pathlib import Path

import cv2
import numpy as np
import pytest

from replay.armor_patch import PATCH_SIZE, warp_patch
from replay.learned import (NEGATIVE, LearnedMatcher, frame_samples, lookup_patch_label, pair_features,
                            select_plates)
from replay.metrics import evaluate, reported_target
from replay.pipelines import LightBar

MODELS = Path(__file__).resolve().parent.parent / "models"


def bar(cx, cy, length=40.0, width=8.0, angle=0.0):
    return LightBar(cx, cy, width, length, angle, (cx, cy - length / 2), (cx, cy + length / 2))


def test_pair_features_do_not_depend_on_scale_or_order():
    a, b = bar(100, 200), bar(180, 205, length=36, angle=3)
    a2, b2 = bar(200, 400, 80, 16), bar(360, 410, length=72, width=16, angle=3)
    np.testing.assert_allclose(pair_features(a, b), pair_features(a2, b2), rtol=1e-5)
    np.testing.assert_allclose(pair_features(a, b), pair_features(b, a))


def test_patch_shows_what_lies_between_the_bars():
    frame = np.zeros((300, 400, 3), np.uint8)
    cv2.rectangle(frame, (180, 130), (220, 170), (255, 255, 255), -1)   # a bright square between the bars
    patch = warp_patch(frame, bar(150, 150), bar(250, 150))
    assert patch.shape == (PATCH_SIZE, PATCH_SIZE)
    c = PATCH_SIZE // 2
    assert patch[c, c] == pytest.approx(1.0) and patch[1, 1] == pytest.approx(0.0)


def test_samples_far_from_a_click_are_negative_and_hand_labels_win():
    frame = np.zeros((300, 600, 3), np.uint8)
    bars = [bar(100, 150), bar(180, 150), bar(400, 150)]
    hand = {7: [(140.0, 150.0, "3")]}
    samples = frame_samples(frame, bars, 7, [(140, 150)], 25, hand)
    by_centre = {(round(s.cx), round(s.cy)): s for s in samples}
    assert by_centre[(140, 150)].label == "3" and by_centre[(140, 150)].is_plate
    assert by_centre[(290, 150)].label == NEGATIVE and not by_centre[(290, 150)].is_plate
    assert lookup_patch_label(hand, 7, (141, 151)) == "3"
    assert lookup_patch_label(hand, 7, (150, 150)) is None


def test_select_plates_thresholds_and_never_shares_a_bar():
    bars = [bar(100, 150), bar(180, 150), bar(260, 150)]
    pairs = [(0, 1), (0, 2), (1, 2)]
    classes = [NEGATIVE, "3", "4"]
    p_pair = np.array([0.9, 0.1, 0.8])
    probs = np.array([[0.1, 0.8, 0.1],     # good plate, number 3
                      [0.0, 0.0, 1.0],     # rejected by the pair score
                      [0.3, 0.1, 0.6]])    # shares bar 1 with the better plate
    plates = select_plates(bars, pairs, p_pair, probs, classes, 0.3, 0.5)
    assert [(p.bar_i, p.bar_j, p.number) for p in plates] == [(0, 1, "3")]
    assert plates[0].score == pytest.approx(0.9 * 0.9)


def test_a_ranked_detector_reports_its_target():
    records = [{"candidates": [(300, 300), (100, 100)], "status": "too_many", "state": "none",
                "estimate": None, "target": (100, 100)},
               {"candidates": [(100, 100), (300, 300)], "status": "too_many", "state": "none", "estimate": None}]
    assert reported_target(records[0]) == (100, 100) and reported_target(records[1]) is None
    r = evaluate(records, {0: {"points": [(100, 100)]}, 1: {"points": [(100, 100)]}}, tol_px=25)
    assert r["detector_single_frames"] == 1 and r["detector_single_correct"] == 1


@pytest.mark.skipif(not (MODELS / "armor_models.json").exists(), reason="no trained models")
def test_committed_models_load_and_run_with_opencv_alone():
    m = LearnedMatcher.load(MODELS)
    frame = np.zeros((300, 600, 3), np.uint8)
    plates = m.match(frame, [bar(100, 150), bar(180, 150), bar(400, 150)])
    assert all(p.number in m.classes and p.number != NEGATIVE for p in plates)
    probs = m.number_probs(np.zeros((3, PATCH_SIZE, PATCH_SIZE), np.float32))
    np.testing.assert_allclose(probs.sum(axis=1), 1, rtol=1e-5)
