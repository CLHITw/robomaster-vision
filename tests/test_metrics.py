import math

from replay.metrics import evaluate, nearest


def rec(candidates, status, state, estimate):
    return {"candidates": candidates, "status": status, "state": state, "estimate": estimate}


def test_nearest_picks_closest_and_handles_empty():
    assert nearest((0, 0), [(3, 4), (10, 0)]) == 5
    assert math.isinf(nearest((0, 0), []))


def test_counts_hits_within_tolerance():
    records = [
        rec([(100, 100)], "one_armor", "update", (100, 100)),   # both correct
        rec([(300, 300)], "one_armor", "update", (300, 300)),   # both far from the label
        rec([], "no_armor", "coast", (105, 100)),               # only the tracker, correct
    ]
    labels = {0: {"visible": 1, "x": 100, "y": 100},
              1: {"visible": 1, "x": 100, "y": 100},
              2: {"visible": 1, "x": 100, "y": 100}}
    r = evaluate(records, labels, tol_px=25)

    assert r["frames_with_target"] == 3
    assert r["detector_single_frames"] == 2 and r["detector_single_correct"] == 1
    assert r["detector_single_precision_pct"] == 50.0      # of the frames it spoke up in
    assert r["detector_usable_pct"] == round(100 / 3, 1)   # of all frames with a target
    assert r["tracker_correct"] == 2 and r["tracker_usable_pct"] == round(200 / 3, 1)
    assert r["tracker_error_when_coasting"]["median_px"] == 5.0


def test_best_candidate_is_an_upper_bound_on_the_detector():
    records = [rec([(500, 500), (100, 100)], "too_many", "coast", None)]
    labels = {0: {"visible": 1, "x": 100, "y": 100}}
    r = evaluate(records, labels, tol_px=25)

    assert r["detector_single_frames"] == 0
    assert r["detector_usable_pct"] == 0.0
    assert r["detector_best_candidate_pct"] == 100.0  # the right pair was there, the rules dropped it


def test_frames_without_a_target_measure_false_positives():
    records = [rec([(10, 10)], "one_armor", "update", (10, 10)), rec([], "no_armor", "none", None)]
    labels = {0: {"visible": 0, "x": None, "y": None}, 1: {"visible": 0, "x": None, "y": None}}
    r = evaluate(records, labels, tol_px=25)

    assert r["frames_with_target"] == 0 and r["frames_without_target"] == 2
    assert r["false_target_when_empty_pct"] == 50.0
    assert r["tracker_position_when_empty_pct"] == 50.0


def test_labels_beyond_the_video_are_ignored():
    r = evaluate([rec([], "no_armor", "none", None)], {5: {"visible": 1, "x": 1, "y": 1}}, tol_px=25)
    assert r["labelled_frames"] == 0


def test_label_csv_round_trip(tmp_path):
    """The click tool must be able to reload exactly what it wrote (resume after a crash)."""
    from label_frames import load_labels, save_labels

    path = tmp_path / "labels.csv"
    labels = {0: {"visible": 1, "x": 123.4, "y": 56.7},
              10: {"visible": 0, "x": None, "y": None}}
    save_labels(path, labels, 1280, 1024, "video2.avi")
    assert load_labels(path) == labels
    assert load_labels(tmp_path / "missing.csv") == {}
