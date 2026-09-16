import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

YELLOW, GREEN, ORANGE, RED, BLUE = (0, 255, 255), (0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 0)


def _scale(frame):
    return max(1, round(max(frame.shape[:2]) / 640))


def label(img, text):
    s = _scale(img)
    cv2.putText(img, text, (10 * s, 30 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (0, 0, 0), 3 * s, cv2.LINE_AA)
    cv2.putText(img, text, (10 * s, 30 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, YELLOW, s, cv2.LINE_AA)
    return img


def draw_v1(frame, boxes):
    out = frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), BLUE, 2 * _scale(frame))
    return out


def draw_v2(frame, bars, armors):
    out = frame.copy()
    s = _scale(frame)
    for b in bars:
        box = cv2.boxPoints(((b.cx, b.cy), (b.width, b.height), b.angle)).astype(int)
        cv2.drawContours(out, [box], 0, GREEN, 2 * s)
    colour = RED if len(armors) == 1 else ORANGE
    for a in armors:
        cv2.rectangle(out, (int(a.cx - a.width / 2), int(a.cy - a.height / 2)),
                      (int(a.cx + a.width / 2), int(a.cy + a.height / 2)), colour, 2 * s)
    return out


def draw_track(frame, candidates, step, trail=None):
    out = frame.copy()
    s = _scale(frame)
    if trail:
        points = [p for p in trail if p is not None]
        for k in range(1, len(points)):
            cv2.line(out, tuple(int(v) for v in points[k - 1]), tuple(int(v) for v in points[k]),
                     RED, max(1, s), cv2.LINE_AA)
    for x, y in candidates:
        cv2.circle(out, (int(x), int(y)), 7 * s, ORANGE, 2 * s)
    if step.measurement is not None:
        cv2.circle(out, tuple(int(v) for v in step.measurement), 11 * s, GREEN, 3 * s)
    if step.estimate is not None:
        cv2.drawMarker(out, tuple(int(v) for v in step.estimate), RED, cv2.MARKER_CROSS, 30 * s, 3 * s)
    return out


STATE_COLOUR = {"update": GREEN, "init": GREEN, "coast": ORANGE, "lost": RED, "none": (200, 200, 200)}


def draw_hud(img, frame_id, total, status, step, n_bars):
    """Frame counter, detector status and tracker state, drawn in place."""
    s = _scale(img)
    lines = [
        (f"frame {frame_id + 1}/{total}", (255, 255, 255)),
        (f"armor_plate: {n_bars} bars -> {status}", GREEN if status == "one_armor" else ORANGE),
        (f"tracker: {step.state}", STATE_COLOUR.get(step.state, (200, 200, 200))),
    ]
    pad, line_h = 8 * s, 22 * s
    box_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, s)[0][0] for t, _ in lines) + 2 * pad
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, int(pad + line_h * len(lines))), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    for k, (text, colour) in enumerate(lines):
        cv2.putText(img, text, (pad, int(pad + line_h * (k + 0.75))), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * s, colour, s, cv2.LINE_AA)
    legend = "green bars/measurement  orange candidates  red estimate"
    font_scale = 0.4 * s
    while font_scale > 0.25:
        text_w = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, font_scale, s)[0][0]
        if text_w <= img.shape[1] - 2 * pad:
            break
        font_scale -= 0.05
    origin = (pad, img.shape[0] - pad)
    cv2.putText(img, legend, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3 * s, cv2.LINE_AA)
    cv2.putText(img, legend, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), s, cv2.LINE_AA)
    return img


def contact_sheet(rows, tile_height=320):
    """rows: list of lists of BGR images (all rows the same length)."""
    def fit(img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        return cv2.resize(img, (max(1, round(w * tile_height / h)), tile_height))

    rendered = []
    for row in rows:
        tiles = [fit(t) for t in row]
        gap = np.full((tile_height, 4, 3), 255, np.uint8)
        rendered.append(np.hstack([x for t in tiles for x in (t, gap)][:-1]))
    width = max(r.shape[1] for r in rendered)
    rendered = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rendered]
    sep = np.full((6, width, 3), 255, np.uint8)
    return np.vstack([x for r in rendered for x in (r, sep)][:-1])


def trajectory_plot(records, path, title):
    frames = np.array([r["frame"] for r in records])
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for ax, key in zip(axes, ("x", "y")):
        single = [(r["frame"], r[f"meas_{key}"]) for r in records
                  if r["meas_x"] is not None and r["v2_status"] == "one_armor"]
        chosen = [(r["frame"], r[f"meas_{key}"]) for r in records
                  if r["meas_x"] is not None and r["v2_status"] == "too_many"]
        est = np.array([np.nan if r["est_x"] is None else r[f"est_{key}"] for r in records], dtype=float)
        if single:
            ax.scatter(*zip(*single), s=10, c="tab:orange", label="single armor detection used")
        if chosen:
            ax.scatter(*zip(*chosen), s=14, c="tab:green", marker="x", label="multi-armor frame, chosen by tracker")
        ax.plot(frames, est, c="tab:blue", lw=1.2, label="Kalman estimate")
        ax.set_ylabel(f"{key} (px)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    axes[0].set_title(title, fontsize=10)
    axes[1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
