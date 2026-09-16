# RoboMaster robot detection (2021), replayed and evaluated

In 2021 I was on the vision team of HITCRT, the RoboMaster team at Harbin Institute of Technology. These are my
early attempts at finding an enemy robot in video, plus 2026 tooling that re-runs the old code on new videos and
measures how well it works.

## Repository layout

| Path | What it is |
|---|---|
| [`original_2021/`](original_2021) | The 2021 C++ code, **unchanged** (tag `original-2021`). |
| [`replay/pipelines.py`](replay/pipelines.py) | Python replays of the two C++ programs: same OpenCV calls, parameters and branch conditions. |
| [`replay/tracking.py`](replay/tracking.py) | 2026 addition: a constant-velocity Kalman tracker placed after the armor detector. |
| [`run_videos.py`](run_videos.py) | Runs all stages on one or more videos and writes statistics and figures. |
| [`tests/`](tests) | Unit tests on synthetic frames. |

## The three stages

1. **v1: grayscale threshold** (`original_2021/carcarcar.cpp`). Blur, apply a fixed threshold of 40, then
   take bounding boxes of the contours. The file is a scratch file that also contains pasted OpenCV
   `groupRectangles` source (not my code) and camera calibration notes.
2. **v2: armor plate from light bars** (`original_2021/armor_plate/`). Subtract the red channel from the blue
   one, threshold at 0.65 × maximum, and dilate. Contours are then filtered into light bars (ellipse angle and
   min-area-rect shape), and bars are paired by four cascaded geometric rules. The program reports a target only
   if exactly one pair is found. The C++ `display()` (PnP and ballistics) was never finished and does not
   compile, so it is not replayed.
3. **Tracker (2026)**. A 2-D constant-velocity Kalman filter on the armor centre. Each frame, the candidate
   nearest the prediction is used if it falls inside a χ² gate (99%). Otherwise the track coasts, and it is
   dropped after 15 frames. This follows the predict / gate / associate idea used in later HITCRT tracking code.

Why a replay: the machine used in 2021 (Ubuntu, OpenCV built from source) is gone, and the Python version makes
evaluation on arbitrary videos easy. The replay was checked function by function against the C++ source. It is
not a compiled run of the C++ code.

## Running on videos

```bash
pip install -r requirements.txt
python -m pytest
python run_videos.py                      # every video in this folder, played back in a window
python run_videos.py video0.avi          # just one video
python run_videos.py --no-show           # batch mode, no window
```

Each video is played back with the detections drawn on it, and the same annotated frames are written to
`outputs/<video name>/annotated.mp4`. While the window is open: **space** pauses, **n** skips to the next video,
**q** or **Esc** stops, **s** saves the current frame as a png.

Useful options:

| Option | Meaning |
|---|---|
| `--enemy auto\|blue\|red` | Light-bar colour. `auto` (default) runs both variants on ~30 frames and keeps the better one. |
| `--proc-width 720` | Frames are downscaled to this width before detection, because the 2021 thresholds (light bar height 10–150 px) were written for roughly this size. `0` keeps the original resolution. |
| `--speed 2` | Playback speed of the window. |
| `--no-video`, `--no-show` | Skip the mp4, or skip the window. |
| `--max-coast`, `--meas-std`, `--accel-std` | Tracker tuning. |

Besides `annotated.mp4`, each run writes `summary.json`, `frames.csv`, contact sheets for every stage
(`v1_sheet.jpg`, `v2_sheet.jpg`, `track_sheet.jpg`) and `track_plot.png`; `outputs/summary.csv` collects one row
per video.

## Results so far

Five phone videos (1080x1920, downscaled to 720x1280 for detection), 2430 frames in total. v1 never finds the
robot in any of them: it boxes floor tape, shadows and, in every frame, the whole image.

| Video | Frames | Colour | v2: one armor | v2: several | Tracker: estimate | measured | coasting | several resolved | lost |
|---|---|---|---|---|---|---|---|---|---|
| video0 | 440 | blue | 37.5% | 14.3% | 96.1% | 40.9% | 55.2% | 47/63 | 2 |
| video1 | 401 | blue* | 17.2% | 2.0% | 85.3% | 15.0% | 70.3% | 8/8 | 5 |
| video2 | 747 | blue | 17.7% | 5.6% | 77.9% | 21.2% | 56.8% | 40/42 | 7 |
| video3 | 602 | blue | 12.1% | 2.2% | 64.6% | 12.1% | 52.5% | 11/13 | 6 |
| video4 | 240 | red | 12.1% | 5.4% | 57.1% | 17.5% | 39.6% | 13/13 | 3 |

\* video1 shows a blue **and** a red robot, so the automatic colour choice is arbitrary there
(probe score blue 18.3 vs red 16.3); pass `--enemy` explicitly for such scenes.

The detector alone reports a single target in 12–38% of frames. Adding the tracker raises the frames with a
position to 57–96%, but a large part of that is coasting on the last velocity rather than a fresh measurement,
and tracks are lost and re-initialised several times per video. Where several armor candidates are found, the
tracker picks one in almost all frames (119 of 139 across the set).

Detections are not always on the robot: the light-bar filter also fires on background objects, and in video0
32 single detections were rejected by the tracker gate.

**No ground truth.** These numbers say how often each stage produces an output, not whether the output is
correct; sampled frames were checked by eye. Labelling a few hundred frames would be the next step.

## Limitations

- The pixel thresholds of v2 (bar height 10–150 px, pairing distances) were written for the 2021 camera, so
  very different resolutions or distances may need rescaling.
- `--enemy red` is an extension: the 2021 red branch never produced an output image.
- The calibration in the notes belongs to a 1280×1024 industrial camera, not to these phone videos, so PnP
  distance estimation is not meaningful on them.

## Credits

The 2026 replay, tracker, tests and this README were prepared with help from an AI coding assistant (Claude).
The numbers come from `run_videos.py`, and I checked sampled frames visually.
