"""Scoring of detector and tracker output against clicked ground truth."""

import math

import numpy as np


def nearest(point, candidates):
    """Distance from ``point`` to the closest candidate, or inf if there is none."""
    if not candidates:
        return math.inf
    px, py = point
    return min(math.hypot(px - cx, py - cy) for cx, cy in candidates)


def closest_pair(points, candidates):
    """Smallest distance between any labelled point and any candidate.

    A frame can show more than one armor plate; hitting any of them counts.
    """
    if not points or not candidates:
        return math.inf
    return min(nearest(p, candidates) for p in points)


def _stats(errors):
    errors = np.array([e for e in errors if math.isfinite(e)], dtype=float)
    if errors.size == 0:
        return {"n": 0, "median_px": None, "p90_px": None}
    return {"n": int(errors.size), "median_px": round(float(np.median(errors)), 1),
            "p90_px": round(float(np.percentile(errors, 90)), 1)}


def evaluate(records, labels, tol_px):
    """Score per-frame results against labels.

    ``records[i]`` describes frame ``i`` and holds ``candidates`` (list of armor
    centres the detector proposed), ``status`` (the 2021 program's verdict),
    ``state`` (tracker state) and ``estimate`` (tracker position or None).
    ``labels`` maps a frame index to ``{"points": [(x, y), ...]}`` in the same
    pixel coordinates; a frame may hold several armor plates. A position counts
    as correct when it is within ``tol_px`` of any of them.
    """
    visible = {f: lab for f, lab in labels.items() if lab["points"] and f < len(records)}
    empty = {f: lab for f, lab in labels.items() if not lab["points"] and f < len(records)}

    det_single_err, det_best_err, track_err = [], [], []
    track_err_by_state = {"update": [], "coast": []}
    det_single_frames = track_frames = 0

    for f, lab in visible.items():
        r = records[f]
        points = lab["points"]
        det_best_err.append(closest_pair(points, r["candidates"]))
        if r["status"] == "one_armor":
            det_single_frames += 1
            det_single_err.append(closest_pair(points, r["candidates"]))
        if r["estimate"] is not None:
            track_frames += 1
            err = closest_pair(points, [r["estimate"]])
            track_err.append(err)
            if r["state"] in track_err_by_state:
                track_err_by_state[r["state"]].append(err)

    n_vis = len(visible)
    hit = lambda errs: sum(1 for e in errs if e <= tol_px)

    result = {
        "tolerance_px": round(tol_px, 1),
        "labelled_frames": len(visible) + len(empty),
        "frames_with_target": n_vis,
        "armor_plates_labelled": sum(len(lab["points"]) for lab in visible.values()),
        "frames_with_several_plates": sum(1 for lab in visible.values() if len(lab["points"]) > 1),
        "frames_without_target": len(empty),

        # the 2021 detector on its own
        "detector_single_frames": det_single_frames,
        "detector_single_correct": hit(det_single_err),
        "detector_single_precision_pct": round(100 * hit(det_single_err) / det_single_frames, 1) if det_single_frames else None,
        "detector_usable_pct": round(100 * hit(det_single_err) / n_vis, 1) if n_vis else None,
        "detector_best_candidate_pct": round(100 * hit(det_best_err) / n_vis, 1) if n_vis else None,
        "detector_error": _stats(det_single_err),

        # detector + tracker
        "tracker_frames_with_estimate": track_frames,
        "tracker_correct": hit(track_err),
        "tracker_precision_pct": round(100 * hit(track_err) / track_frames, 1) if track_frames else None,
        "tracker_usable_pct": round(100 * hit(track_err) / n_vis, 1) if n_vis else None,
        "tracker_error": _stats(track_err),
        "tracker_error_when_measuring": _stats(track_err_by_state["update"]),
        "tracker_error_when_coasting": _stats(track_err_by_state["coast"]),
    }

    if empty:
        false_detect = sum(1 for f in empty if records[f]["status"] == "one_armor")
        false_track = sum(1 for f in empty if records[f]["estimate"] is not None)
        result["false_target_when_empty_pct"] = round(100 * false_detect / len(empty), 1)
        result["tracker_position_when_empty_pct"] = round(100 * false_track / len(empty), 1)
    return result
