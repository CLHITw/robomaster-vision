# Finding a robot in video — my first computer-vision task, measured five years later

In 2021 I joined the vision team of HITCRT, the RoboMaster robotics team at Harbin Institute of Technology. My
first task was to find the enemy robot in footage from the team camera. My first attempt did not work, the
second one did, and at the time I had no way of saying how well. This repository contains both attempts
unchanged, and a 2026 test harness that runs them on the original footage and measures them.

![Armor detection and tracking on the 2021 footage](docs/figures/demo.gif)

*The original 2021 recording (1280×1024, 50 fps). Green: light bars found by the 2021 detector. Red: the
position estimate of the tracker added in 2026, with its recent path.*

## Attempt 1: the robot is the bright thing (wrong)

[`original_2021/carcarcar.cpp`](original_2021/carcarcar.cpp) converts the frame to grayscale, blurs it three
times, thresholds at a fixed value of 40, and draws the bounding box of every contour.

It never isolates the robot. On the original footage the threshold turns most of the image white, so the
contours it finds are the dark floor joints and shadows: a median of 52 boxes per frame, and in **every single
frame** one box covering the whole image. The idea that the target is simply "brighter than the rest" does not
survive contact with a real gym floor under stage lighting.

## Attempt 2: the robot is the blue thing, twice (right)

Competition robots carry two vertical LED light bars beside each armor plate, in the team colour. That is a
much stronger signal than brightness, and a pair of them is a distinctive geometric pattern.

[`original_2021/armor_plate/`](original_2021/armor_plate) subtracts the red channel from the blue one,
thresholds at 0.65 × the maximum value of that difference image and dilates. Contours are then filtered into
light-bar candidates by their ellipse angle and the aspect ratio of their minimum-area rectangle, and candidates
are paired into armor plates by four cascaded rules on angle difference, height difference, width difference
and vertical offset.

![v1, v2 and the tracker on the same frames](docs/figures/three_stages.jpg)

*The same four frames through all three stages. Top: attempt 1. Middle: attempt 2 — green light bars, red box
when exactly one armor plate is found, orange when several are. Bottom: attempt 2 plus the tracker.*

## What I could not see in 2021

The 2021 program prints `no armors!` or `too many armors!` and moves on, and back then I had no idea how often
that happened. Measured now over all 1086 frames of the original recording:

| Detector output | Share of frames |
|---|---|
| exactly one armor plate (a usable target) | **48.3%** |
| several candidates, program gives up | 18.5% |
| light bars found, none pair up | 30.5% |
| fewer than two light bars | 2.7% |

![Per-frame detector status and tracker state](docs/figures/timeline.png)

So the detector that felt like it "worked" produces a target in about half the frames. The gaps are mostly
short — 248 of them, a median of 1 frame, 90% shorter than 5 frames, the longest 28 frames (0.6 s) — which is
the signature of a per-frame detector flickering on a target that is still plainly there. A turret driven
directly by this signal would stutter constantly, and the fix is not a better threshold but remembering the
target between frames.

## 2026: turning intermittent detections into a continuous track

Detection per frame is the wrong abstraction — the robot does not disappear when a light bar does. So I added
what the 2021 code lacked: a constant-velocity Kalman filter on the armor centre
([`replay/tracking.py`](replay/tracking.py)). Each frame it predicts the next position, accepts the candidate
closest to that prediction if it falls inside a χ² gate, coasts when nothing matches, and gives up after 15
frames. This is the predict / gate / associate structure used by the team's later tracking code, in a minimal
2-D form.

| | Detector alone | With tracker |
|---|---|---|
| frames with a position | 48.3% | **96.9%** |
| of which an actual measurement | 48.3% | 49.0% |
| of which coasting on the prediction | — | 47.9% |
| ambiguous frames resolved | 0 / 201 | **169 / 201** |
| track losses over 1086 frames | — | 5 |

The tracker nearly doubles the frames with a position and picks a candidate in 84% of the frames where the
detector gave up because it saw several. Half of the output, though, is extrapolation rather than measurement,
and extrapolation drifts — this fills short gaps, it does not invent detections.

### The gate width is a knob I cannot honestly tune yet

The gate decides which detections are believed. Widening it accepts more measurements, but also accepts jumps
that are probably false positives:

| measurement σ (px) | measurements used | ambiguous resolved | single detections rejected | accepted jumps > 100 px |
|---|---|---|---|---|
| 15 | 42.2% | 150/201 | 217 | 6.6% |
| 25 (default: 2% of frame width) | 49.0% | 169/201 | 162 | 8.7% |
| 35 | 56.8% | 195/201 | 103 | 10.4% |
| 50 | 61.7% | 196/201 | 51 | 16.4% |

Without labelled frames, "more measurements used" is not evidence of "more correct". Rather than picking the
row with the nicest number, the default scales with the frame size (2% of the width), which keeps the gate the
same size relative to the image across cameras. Labelling a few hundred frames is the obvious next step, and it
is what would turn every percentage on this page into an accuracy.

## Honest boundaries

- The stage-1 and stage-2 algorithms are the 2021 C++ code, kept unchanged under `original_2021/` (tag
  `original-2021`). `carcarcar.cpp` is a scratch file: it also contains pasted OpenCV `groupRectangles` source
  and camera calibration notes, which are not my code.
- The measurements come from a **Python replay** of that C++ code, not from running the C++ binary. The replay
  uses the same OpenCV calls, parameters and branch conditions; each deviation is marked `NOTE` in
  [`replay/pipelines.py`](replay/pipelines.py). The 2021 machine (Ubuntu, OpenCV built from source) no longer
  exists.
- The C++ `display()` function (PnP pose and ballistics) was never finished in 2021 and does not compile, so it
  is not replayed.
- **There is no ground truth.** Every percentage here says how often a stage produced an output, not whether
  the output was correct. Sampled frames were checked by eye.
- The tracker, the harness, the tests and this README are from 2026 and were written with help from an AI
  coding assistant. All numbers come from the scripts in this repository.

## Robustness on other footage

Five phone recordings of robots (1080×1920, not included here) give 12–38% single-armor frames and 57–96%
tracked frames, with the light-bar colour detected automatically. Two things break on them: the pixel
thresholds of the 2021 detector assume the original camera, so frames have to be downscaled to roughly 720 px
wide (`--proc-width`), and in a scene with both a blue and a red robot the automatic colour choice is
meaningless and has to be set by hand.

## Running it

```bash
pip install -r requirements.txt
python -m pytest                                   # 8 tests on synthetic frames
python run_videos.py                               # every video in this folder, played back in a window
python run_videos.py video2.avi --proc-width 0
python make_figures.py video2.avi --proc-width 0   # rebuild the figures above
```

Each video is played back with the detections drawn on it and written to `outputs/<name>/annotated.mp4`;
**space** pauses, **n** skips, **q** quits, **s** saves a frame. Each run also writes `summary.json`, per-frame
`frames.csv`, contact sheets per stage and a trajectory plot.

| Option | Meaning |
|---|---|
| `--enemy auto\|blue\|red` | Light-bar colour; `auto` probes both on ~30 frames. |
| `--proc-width 720` | Downscale before detection (`0` = native). The 2021 thresholds assume light bars of 10–150 px. |
| `--meas-std`, `--max-coast`, `--accel-std` | Tracker gate, patience and process noise. |
| `--no-show`, `--no-video`, `--speed` | Batch mode, skip the mp4, playback speed. |

## Repository layout

| Path | What it is |
|---|---|
| [`original_2021/`](original_2021) | The 2021 C++ code, unchanged. |
| [`replay/pipelines.py`](replay/pipelines.py) | Python replay of both C++ programs. |
| [`replay/tracking.py`](replay/tracking.py) | The 2026 Kalman tracker. |
| [`replay/visualize.py`](replay/visualize.py) | Overlays, contact sheets, plots. |
| [`run_videos.py`](run_videos.py) | Playback and evaluation over one or more videos. |
| [`make_figures.py`](make_figures.py) | Builds the figures in this README. |
| [`tests/`](tests) | Unit tests on synthetic frames: detection, colour selection, pairing rules, tracker gating. |
