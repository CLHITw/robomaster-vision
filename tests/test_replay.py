import numpy as np
import cv2

from replay.pipelines import armor_status, detect_armors, threshold_boxes
from replay.tracking import ArmorTracker


def frame_with_bars(colour, centres, size=(1280, 720), bar=(12, 60)):
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for cx, cy in centres:
        cv2.rectangle(img, (cx - bar[0] // 2, cy - bar[1] // 2), (cx + bar[0] // 2, cy + bar[1] // 2), colour, -1)
    return img


BLUE, RED = (255, 60, 0), (0, 60, 255)


def test_pair_of_blue_bars_is_one_armor():
    img = frame_with_bars(BLUE, [(300, 600), (380, 600)])
    bars, armors, _ = detect_armors(img, "blue")
    assert len(bars) == 2
    assert armor_status(bars, armors) == "one_armor"
    assert abs(armors[0].cx - 340) < 2 and abs(armors[0].cy - 600) < 2


def test_single_bar_is_not_an_armor():
    bars, armors, _ = detect_armors(frame_with_bars(BLUE, [(300, 600)]), "blue")
    assert armor_status(bars, armors) == "lt2_bars"


def test_enemy_colour_selects_channel():
    img = frame_with_bars(RED, [(300, 600), (380, 600)])
    assert armor_status(*detect_armors(img, "red")[:2]) == "one_armor"
    assert armor_status(*detect_armors(img, "blue")[:2]) == "lt2_bars"


def test_bars_at_different_heights_are_not_paired():
    img = frame_with_bars(BLUE, [(300, 300), (380, 900)])
    bars, armors, _ = detect_armors(img, "blue")
    assert len(bars) == 2 and armors == []


def test_v1_threshold_boxes_include_whole_frame_on_bright_image():
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), -1)
    boxes, binary = threshold_boxes(img)
    assert any(w >= 639 and h >= 479 for (_, _, w, h) in boxes)
    assert binary.mean() > 200


def test_tracker_initialises_only_on_single_candidate():
    t = ArmorTracker()
    assert t.step([(10, 10), (500, 500)]).state == "none"
    assert t.step([(100, 100)]).state == "init"


def test_tracker_picks_candidate_nearest_prediction():
    t = ArmorTracker()
    for k in range(5):
        t.step([(100 + 5 * k, 200)])
    step = t.step([(126, 200), (600, 900)])
    assert step.state == "update" and step.measurement == (126.0, 200.0)


def test_tracker_rejects_outlier_and_drops_after_max_coast():
    t = ArmorTracker(max_coast=3)
    for k in range(5):
        t.step([(100, 200)])
    assert t.step([(700, 1000)]).state == "coast"
    states = [t.step([]).state for _ in range(3)]
    assert states == ["coast", "coast", "lost"]
    assert t.step([]).estimate is None
