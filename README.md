# RoboMaster robot detection (2021), replayed and evaluated

In 2021 I was on the vision team of HITCRT, the RoboMaster team at Harbin Institute of Technology. These are my
early attempts at finding an enemy robot in video, plus 2026 tooling that re-runs the old code on new videos and
measures how well it works.

> Work in progress: the results section will be filled in after more test videos have been evaluated.

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
python run_videos.py path/to/video.avi                  # one video
python run_videos.py videos/ --save-video               # every video in a folder, plus annotated mp4
python run_videos.py "videos/*.mp4" --enemy red         # red light bars (see limitations)
```

Outputs per video go to `outputs/<video name>/`: `summary.json`, `frames.csv`, contact sheets for each stage
(`v1_sheet.jpg`, `v2_sheet.jpg`, `track_sheet.jpg`), `track_plot.png` and, optionally, `annotated.mp4`.
`outputs/summary.csv` has one row per video.

## Results so far

| Video | Frames | v1 | v2: exactly one armor | v2: several armors | Tracker: estimate available | of which measured | of which coasting |
|---|---|---|---|---|---|---|---|
| video1 (blue robot, handheld phone, 720×1280) | 601 | boxes floor tape and shadows, not the robot | 22.5% | 10.0% | 91.0% | 31.4% | 59.6% |

On video1 the tracker picked a candidate in 59 of the 60 frames with several armors, and 5 single detections
were rejected by the gate. Coasting frames extrapolate the last velocity and can overshoot.

**No ground truth yet.** These numbers describe how often each stage produces an output, not whether the output
is correct. Sampled frames were checked by eye.

## Limitations

- The pixel thresholds of v2 (bar height 10–150 px, pairing distances) were written for the 2021 camera, so
  very different resolutions or distances may need rescaling.
- `--enemy red` is an extension: the 2021 red branch never produced an output image.
- The calibration in the notes belongs to a 1280×1024 industrial camera, not to these phone videos, so PnP
  distance estimation is not meaningful on them.

## Credits

The 2026 replay, tracker, tests and this README were prepared with help from an AI coding assistant (Claude).
The numbers come from `run_videos.py`, and I checked sampled frames visually.
