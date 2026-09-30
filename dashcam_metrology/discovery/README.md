# Finding out what the camera actually is

Before anything could be calibrated, the camera had to be reachable and its
behaviour known. These are the throwaway scripts that established the facts the
main README quotes, kept because those facts are otherwise unsupported claims.

| | |
|---|---|
| `dashcam_probe.py` | Scans the access point: which host, which ports, what answers on HTTP |
| `dashcam_rtsp.py` | RTSP `OPTIONS` and `DESCRIBE` against 56 candidate stream paths |
| `dashcam_ch.py` | Enumerates `/ch00` … `/ch08` looking for further channels |
| `dashcam_timing.py` | Frame rate by wall clock, after draining the decoder buffer |

What they established:

- The camera serves **one** open port, 8554, and no HTTP interface at all. Of the
  56 RTSP paths tried, only `/ch01` returns 200; the session description names a
  live555 server. A multi-channel dash camera, but only one channel is offered
  over Wi-Fi, so nothing here can be a multi-camera experiment.
- The stream runs at **31.07 fps** measured over 20 s and 623 frames, with no
  duplicate frames. An earlier attempt reported 4608 fps: it counted frames
  rather than seconds and emptied a decoder buffer that had filled during
  connection. Every tool in the parent folder drains that buffer in a background
  thread for this reason.
- Frame *arrival* is bimodal — 0.7 ms at the 10th percentile, 65.6 ms at the
  90th, 6.1% of gaps longer than twice the median. Frames come in bursts over
  Wi-Fi, so arrival time cannot stand in for capture time.
