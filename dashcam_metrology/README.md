# Calibrating a second camera, and checking the metres against a ruler

The 2021 armor-plate work ends by admitting something. `solve_pose.py` recovers a
plate's position in millimetres, and the README has to say this:

> No ground-truth distance exists for this recording: the two checks test whether
> the solution is self-consistent, not whether the metres are right.

This folder closes that gap on a camera where a ruler can reach. A cheap dash
camera is calibrated from scratch, the undistortion is checked against something
other than its own residual, and the distances it produces are compared against
positions stepped out on the floor.

**Scale comes out correct to 0.13% between 0.3 m and 1.2 m**, with 6 mm of
residual about a straight line, and the marker size it depends on is confirmed by
two independent routes — one optical, one with a ruler — agreeing to 0.21 mm.

Nothing here is novel. What it is, is measured, and every number states what it
cannot show.

---

## The camera

A TIESFONG dash camera, reachable only over its own Wi-Fi. A port scan finds one
open port, 8554, serving a single RTSP stream from a live555 server. There is no
HTTP API and no second channel: `/ch00` … `/ch08` were tried and only `/ch01`
answers.

| | |
|---|---|
| Stream | `rtsp://192.168.10.1:8554/ch01`, H.264 + AAC |
| Frame size | 640 × 360 |
| Frame rate | 31.07 fps measured over 20 s, 623 frames, no duplicates |
| Exposure, gain | automatic, not controllable |

The frame rate needs its qualifier. By wall clock the stream is a clean 30 fps,
but *arrival* times are bimodal: the 10th-percentile gap between frames is 0.7 ms
and the 90th is 65.6 ms, with 6.1% of gaps longer than twice the median. Frames
arrive in bursts over Wi-Fi. **Arrival time is not capture time**, so nothing here
timestamps a frame by when it was read, and the tools keep only the newest frame
from a background reader rather than working through a queue.

---

## Intrinsics

40 images, 36 kept, 8 × 5 inner corners, squares measured at 24.5 mm.

```
fx 339.32   fy 339.27       0.01% apart
cx 340.26   cy 192.50       image centre is 320.0, 180.0
k  [-0.3637, 0.1418, -3.6e-5, 3.4e-4, -0.0257]
RMS reprojection 0.398 px        field of view 86.6° x 55.9°
```

Written to `config/camera_dashcam.json` in the same schema as
`../config/camera_2021.json`, so `replay/pose.py`'s `load_camera()` and
`scale_camera()` read it unchanged — the 2021 industrial camera and this one run
through the same pose code.

### Which lens model

Both are fitted. RMS prefers fisheye, 0.362 px against 0.398, which is not a good
enough reason to choose it: reprojection error asks whether a model *can explain
the observations*, and a model with enough freedom always can.

A different question is more useful. Every row and column of the checkerboard is
a straight line in the world, so after undistortion it has to come out straight
in the image. Measuring each line's largest deviation from its own best fit, over
520 lines from 40 frames:

| | median | mean | p90 | max |
|---|---|---|---|---|
| distorted | 0.689 | 0.826 | 1.602 | 3.619 |
| undistorted, pinhole | 0.327 | 0.399 | 0.730 | 1.842 |
| undistorted, fisheye | 0.318 | 0.392 | 0.725 | 1.821 |

The models differ by 0.01 px, which is nothing, so the choice has to rest on
something the numbers do not contain. **Pinhole is used**: the field of view is
87°, comfortably inside its range; `cv2.solvePnP` takes pinhole intrinsics
directly while a fisheye model needs its points undistorted first; and the 2021
pose code already assumes pinhole.

The residual 0.33 px is close to what corner localisation can achieve on a
640 × 360 H.264 stream. Both models have removed the distortion this stream can
resolve; what is left is the data, not the model.

---

## Distance, against a measured floor

An ArUco marker is detected, its pose solved with `SOLVEPNP_IPPE_SQUARE`, and the
marker stepped back one A4 sheet at a time — 297.0 mm, a standard length and a
more trustworthy ruler than most rulers. 40 frames at each station.

Steps rather than absolute distances, on purpose. The optical centre sits inside
the lens where no ruler reaches, so any absolute measurement carries an unknown
offset. Fitting `solved = a x position + b` puts the scale error in `a`, which is
what the calibration is being judged on, and lets `b` absorb the unknown origin.

The first fit came out at `a = 1.0391`, linear to within 6.3 mm across 1.2 m — so
the model was right and a constant was wrong. A 3.91% scale error is what a
marker 163.6 mm wide produces when the solver has been told 170 mm.

That inference was then checked by a route that does not involve the camera at
all. The target sheets carry a printed 100 mm rule; measuring an A4 sheet with it
reads 309 mm where the true long edge is 297.0, so the print came out at 0.9612
of nominal and a marker drawn at 170 mm is physically **163.40 mm**.

| | marker side |
|---|---|
| inferred from the distance fit | 163.61 mm |
| measured, via print scale against A4 | 163.40 mm |
| difference | 0.21 mm, 0.13% |

Recomputed with the measured size:

| position (mm) | n | solved (mm) | s.d. | error | reprojection |
|---:|---:|---:|---:|---:|---:|
| 297 | 40 | 309.6 | 0.4 | +12.6 mm | 0.26 px |
| 594 | 40 | 595.2 | 0.2 | +1.2 mm | 0.22 px |
| 891 | 40 | 894.7 | 0.9 | +3.7 mm | 0.22 px |
| 1188 | 40 | 1198.5 | 0.0 | +10.5 mm | 0.12 px |

```
solved = 0.9987 x position + 7.9 mm        largest residual 6.0 mm
```

Scale is right to 0.13%. What remains is a 7.9 mm offset — the distance from the
ruler's zero to the optical centre, which is the quantity this design was built
to set aside — and 6 mm of residual across 1.2 m.

### The square size could not have caused this

The checkerboard came off the same printer, so its squares are not 25.0 mm
either, and the 24.5 mm handed to the calibration may be wrong too. It makes no
difference. Scaling every object point by k is absorbed entirely by scaling every
estimated board distance by k, leaving the projection unchanged, so the focal
length comes out the same. Calibrating at 23.55, 24.5 and 25.0 mm returns
bit-identical intrinsics:

```
fx 339.32   fy 339.27   cx 340.26   cy 192.50   RMS 0.3982        all three
```

A wrong square size ruins the calibration's own extrinsics and nothing else. The
marker is the opposite case — its size scales the answer directly and linearly —
which is why that one had to be measured and this one did not.

---

## Three things that were wrong first

None of them changed the conclusion, and none were visible in the output. They
are recorded because a result that happens to be right is not the same as one
that is known to be right.

**The first ten calibration frames were unusable and looked fine.** All sharp,
all detected, and all in the middle of the image at 3–4% of frame area. A wide
lens distorts at its edges, so the coefficients would have been extrapolated into
the region where they matter most, and the RMS would have been reassuring.
`capture.py` now shows board area and a 3 × 3 coverage grid while shooting,
because this cannot be seen afterwards.

**`cv2.undistortPoints` does not converge by default.** It runs a fixed handful
of iterations and stops. On a synthetic line bent by 11.9 px it leaves 0.18 px
behind; asked for 100 iterations it leaves 0.0000 px. That 0.18 px would have
been read here as the lens model failing to fit.

**The distortion model is not invertible near the corners.** Switching to the
converged solver made the largest deviation jump from 1.8 px to 60 px, because
for some points the iteration diverges rather than converging: the radial
polynomial stops being monotonic at large radius, so the inverse has no unique
solution. Across all 1600 detected corners the two solvers agree within a pixel
for 96% and disagree by up to 178 px for the rest, all far from the principal
point — median radius 276 px against 121 px overall. Every point is now
round-tripped (undistort, redistort, compare) and frames holding a point that
does not come back are dropped. One frame in forty goes under pinhole; **none
under fisheye**, which is a genuine argument in fisheye's favour that RMS never
surfaced.

---

## What these numbers do not show

- The intrinsics belong to the 640 × 360 preview stream. Footage from the
  camera's SD card is a different size and possibly a different crop;
  `scale_camera()` handles a pure rescale and nothing more.
- Distances were checked between 0.3 m and 1.2 m. A 170 mm marker spans about
  20 px at 2.5 m, roughly where ArUco stops finding it. Nothing here says what
  happens beyond that.
- Exposure and gain cannot be fixed, so the calibration frames were shot in
  daylight and the camera was not moved between stations.
- The distortion model cannot be inverted near the image corners. Everything
  measured here stays away from them; anything reaching the image border should
  revisit the choice of lens model on the evidence above.
- **Reprojection error cannot catch a corner-ordering mistake on a square
  marker.** Rotating the four corners by one position leaves both the distance
  and the reprojection error unchanged — 1200.5 mm and 0.0000 px either way — and
  only the orientation moves. Distance work is unaffected; anything using the
  rotation is not, and no self-check here would notice. A test pins this.
- One camera. Nothing here addresses synchronising two.

---

## Files

| | |
|---|---|
| `make_targets.py` | Prints the checkerboard and ArUco sheets at exact millimetre sizes, checks each fits inside A4, and runs the detector over its own output |
| `make_synthetic.py` | Renders checkerboards from known intrinsics, so the calibration can be tested against an answer |
| `capture.py` | Collects calibration frames, showing board area and coverage live |
| `calibrate.py` | Fits both lens models, reports per-image error, runs four sanity checks, writes the config |
| `straightness.py` | The collinearity test, with round-trip validation |
| `measure.py` | ArUco pose and distance, against measured positions |
| `targets/`, `captures/` | The printable sheets and the 40 frames, so the calibration can be re-run |
| `../tests/test_dashcam_metrology.py` | Ten synthetic checks, including ones that pin the corner non-invertibility and the focal-length invariance |

## Reproducing

```bash
python make_targets.py                                # print at 100%, then measure the rule
python capture.py                                     # 25-30 frames, fill the 3x3 grid
python calibrate.py --square 24.5 --drop-worst 4
python straightness.py
python measure.py --truth 297 --sizes "2=163.4"       # once per station
python measure.py --report
```

`calibrate.py --square` has no default, deliberately. Entering the nominal square
size instead of the measured one leaves the intrinsics untouched and changes
nothing that any check in this folder would notice.
