# Calibrating a second camera, and measuring how far off it is

The 2021 armor-plate work ends with an honest gap. `solve_pose.py` recovers a
plate's position in metres, but the README has to say this:

> No ground-truth distance exists for this recording: the two checks test whether
> the solution is self-consistent, not whether the metres are right.

This folder closes that gap on a different camera, where a ruler can reach.
A cheap dash camera is calibrated from scratch, the undistortion is checked
against something other than its own residual, and the resulting distances are
compared with positions measured on the floor.

Nothing here is novel. The point is that every number has a stated way of being
wrong, and each check tests something the previous one could not see.

---

## The camera

A TIESFONG dash camera, reachable only over its own Wi-Fi access point. A port
scan finds exactly one open port, 8554, serving a single RTSP stream from a
live555 server (`s=Session streamed by "testH264VideoStreamer"`). There is no
HTTP API and no second channel: the paths `/ch00` … `/ch08` were tried and only
`/ch01` answers.

| | |
|---|---|
| Stream | `rtsp://192.168.10.1:8554/ch01`, H.264 + AAC |
| Frame size | 640 × 360 |
| Frame rate | **31.07 fps measured** over 20 s, 623 frames, no duplicates |
| Exposure / gain | automatic, not controllable |

The frame rate needs the qualifier. Measured by wall clock the stream is a clean
30 fps, but the *arrival times* are bimodal: the 10th percentile gap is 0.7 ms
and the 90th is 65.6 ms, with 6.1% of gaps longer than twice the median. Frames
arrive in bursts over Wi-Fi. **Arrival time is not capture time**, which is why
none of the tools here timestamp a frame by when it was read.

A first attempt at measuring frame rate reported 4608 fps. It counted frames
instead of seconds and drained a decoder buffer that had filled during connect.
`measure.py` and `capture.py` therefore run the reader in a background thread
that keeps only the newest frame, so a keypress records what is in front of the
camera and not what was in front of it two seconds ago.

---

## Intrinsics

`make_targets.py` prints the target, `capture.py` collects, `calibrate.py`
solves. 40 images, 36 kept, 8 × 5 inner corners, squares measured at 24.5 mm.

```
fx 339.32   fy 339.27      (0.01% apart)
cx 340.26   cy 192.50      (image centre 320.0, 180.0)
k  [-0.3637, 0.1418, -3.6e-5, 3.4e-4, -0.0257]
RMS reprojection 0.398 px       field of view 86.6° x 55.9°
```

Written to `config/camera_dashcam.json` in the same schema as
`../config/camera_2021.json`, so `replay/pose.py`'s `load_camera()` and
`scale_camera()` read it unchanged.

**The first attempt was thrown away.** Ten images, all detected, all sharp — and
all clustered in the middle of the frame at 3–4% of the frame area. A wide lens
distorts at the edges, so the distortion coefficients would have been
extrapolated into the region where they matter most, with a reassuring RMS. The
capture tool now shows the board's area and a 3 × 3 occupancy grid live, and
refuses nothing but warns about everything, because this is a mistake that is
invisible after the fact.

### Pinhole or fisheye

The script fits both. RMS says fisheye (0.362 vs 0.398 px), which is not a good
enough reason: reprojection error asks whether a model *can explain the
observations*, and a model with enough freedom always can.

`straightness.py` asks a different question. Every row and every column of the
checkerboard is a straight line in the world, so after undistortion it must be
straight in the image too. Over 520 lines from 40 images, measuring each line's
largest deviation from its own best fit:

| | median | mean | p90 | max |
|---|---|---|---|---|
| distorted | 0.689 | 0.826 | 1.602 | 3.619 |
| undistorted, pinhole | 0.327 | 0.399 | 0.730 | 1.842 |
| undistorted, fisheye | 0.318 | 0.392 | 0.725 | 1.821 |

The two models differ by 0.01 px, which is nothing. **Pinhole is used**, for
reasons the numbers cannot supply: the field of view is 87°, well inside the
range a pinhole model handles; `cv2.solvePnP` takes pinhole intrinsics directly
while a fisheye model needs its points undistorted first; and the 2021 pose code
assumes pinhole, so it can be reused as it stands.

The residual 0.33 px is worth a sentence. It is close to what corner
localisation can achieve on a 640 × 360 H.264 stream, so both models have
removed the distortion that this stream can resolve. The limit here is the
data, not the model.

### Two ways this measurement was wrong before it was right

Both were found by the test suite rather than by looking at the output, and
neither changed the conclusion — which is the point. A measurement that happens
to be right is not the same as one that is known to be right.

**`cv2.undistortPoints` does not converge by default.** It runs a fixed handful
of iterations of the inverse distortion model and stops. On a synthetic line
bent by 11.9 px, the default leaves 0.18 px of bow behind; asking for 100
iterations leaves 0.0000 px. That 0.18 px would have been read here as the lens
model failing to fit, when it is the solver giving up. `straightness.py` now
uses `undistortPointsIter`.

**The distortion model is not invertible near the corners.** Switching to the
converged solver made the maximum deviation jump from 1.8 px to 60 px, because
for some points the iteration diverges instead of converging: the radial
polynomial stops being monotonic at large radius, so the inverse has no unique
solution. Comparing the two solvers over all 1600 detected corners, 96% agree to
within a pixel and 4% do not, by up to 178 px, and those all sit far from the
principal point — median radius 276 px against 121 px for the set as a whole.

`straightness.py` now round-trips every point (undistort, redistort, compare)
and drops frames containing a point that does not come back. One frame in forty
is dropped under the pinhole model. **None are dropped under fisheye**, which is
a real argument in its favour that RMS never surfaced: the two models fit the
data equally well, and one of them inverts cleanly across the whole frame while
the other does not. Pinhole is still used here, for the compatibility reasons
above and because the marker work stays near the centre — but any use of the
image border, which ground-plane work would be, should revisit this.

---

## Distance, against a measured floor

`measure.py` detects an ArUco marker, solves its pose with `SOLVEPNP_IPPE_SQUARE`
and records the distance. The marker was moved back in steps of one A4 sheet
(297.0 mm), 40 frames recorded at each station.

Steps rather than absolute distances, deliberately. The optical centre sits
somewhere inside the lens and cannot be reached with a ruler, so an absolute
measurement carries an unknown offset. Fitting `solved = a x position + b`
puts the scale error in `a`, which is what the calibration is being judged on,
and lets `b` absorb the unknown origin.

```
solved = 1.0391 x position + 8.3 mm        largest residual 6.3 mm
```

Linearity is good — 6.3 mm of residual across 1.2 m — so the model is right and
only a constant is wrong. A 3.91% scale error is exactly what a marker 163.6 mm
wide would produce when the solver is told it is 170 mm. Distance from
`solvePnP` is proportional to the assumed object size, so the data can be
rescaled without recollecting:

| position (mm) | n | solved (mm) | s.d. | error | reprojection |
|---:|---:|---:|---:|---:|---:|
| 297 | 40 | 310.0 | 0.4 | +4.4% | 0.26 px |
| 594 | 40 | 595.9 | 0.2 | +0.3% | 0.22 px |
| 891 | 40 | 895.8 | 0.9 | +0.5% | 0.22 px |
| 1188 | 40 | 1200.0 | 0.0 | +1.0% | 0.12 px |

```
solved = 1.0000 x position + 7.9 mm        largest residual 6.0 mm
```

**This last table is conditional and says so.** 163.6 mm is inferred from the
fit, not measured. If the printed marker turns out to be 170 mm after all, the
3.9% is in the focal length instead and this section is wrong. The raw data is
in `measurements_raw.csv`; `measurements.csv` holds the four runs used above.

---

## What these numbers do not show

- The intrinsics are bound to the 640 × 360 preview stream. Recordings pulled
  from the camera's SD card are a different size and, possibly, a different crop.
  `scale_camera()` handles a pure rescale and nothing else.
- Distances were checked between 0.3 m and 1.2 m. A 170 mm marker spans about
  20 px at 2.5 m, which is roughly where ArUco stops detecting it; nothing here
  says what happens beyond that.
- Exposure and gain are automatic and cannot be fixed, so the calibration images
  were taken in daylight and the camera was not moved between stations.
- The distortion model cannot be inverted near the image corners (above).
  Everything measured here stays away from them.
- **Reprojection error cannot detect a corner-ordering mistake on a square
  marker.** Rotating the four corners by one position leaves the distance and the
  reprojection error unchanged — 1200.5 mm and 0.0000 px either way — and only
  the orientation is wrong. Distance work is unaffected; anything using the
  rotation is not, and no self-check here would notice.
- One camera. Nothing in this folder addresses synchronising two.

---

## Files

| | |
|---|---|
| `make_targets.py` | Prints the checkerboard and ArUco sheets at exact millimetre sizes, verifies each fits inside A4 with margin, and runs the detector over its own output |
| `make_synthetic.py` | Renders checkerboards from known intrinsics, so the calibration code can be tested against an answer |
| `capture.py` | Collects calibration frames; shows board area and 3 × 3 coverage live |
| `calibrate.py` | Fits pinhole and fisheye, reports per-image error, runs four sanity checks, writes the config |
| `straightness.py` | Collinearity test on the board's own rows and columns |
| `measure.py` | ArUco pose, distance, comparison against measured positions |
| `../tests/test_dashcam_metrology.py` | Synthetic checks for all of the above, including one that pins the corner non-invertibility |
| `targets/` | The printable sheets, each with a 100 mm rule for verifying print scale |
| `captures/` | The 40 calibration frames, so the calibration can be re-run |

## Reproducing

```bash
python make_targets.py                                   # print at 100%, measure the rule
python capture.py                                        # 25-30 frames, fill the 3x3 grid
python calibrate.py --square 24.5 --drop-worst 4
python straightness.py
python measure.py --truth 297 --sizes "2=163.6"          # repeat per station
python measure.py --report
```

`calibrate.py --square` has no default. Filling in the nominal square size
instead of the measured one leaves the intrinsics untouched and scales every
distance that follows by the same factor, which is the kind of error that
survives every check in this folder.
