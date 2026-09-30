# -*- coding: utf-8 -*-
"""正确测量 RTSP 实际帧率：按墙钟时间连续读，先排空缓冲。"""
import cv2, os, time, io, sys, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

URL = "rtsp://192.168.10.1:8554/ch01"
OUT = r"E:\DESK\dashcam_timing.txt"
f = io.open(OUT, "w", encoding="utf-8")
def log(s=""):
    print(s); f.write(str(s) + "\n"); f.flush()

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
log("RTSP 计时测量  %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
log(URL); log("=" * 58)

cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    log("打不开，检查是否连着记录仪 Wi-Fi"); raise SystemExit

# --- 阶段 1：排空缓冲 ---
# 缓冲里的帧是瞬间返回的；一旦开始按真实帧率到达，单帧间隔会明显变长
log("\n[1] 排空缓冲")
drained = 0
while drained < 400:
    t0 = time.perf_counter()
    ok, _ = cap.read()
    dt = time.perf_counter() - t0
    drained += 1
    if dt > 0.010:                 # 超过 10ms 说明是真在等网络包
        break
log("   丢弃 %d 帧缓冲，最后一帧耗时 %.1f ms" % (drained, dt * 1000))

# --- 阶段 2：按墙钟连续读 20 秒 ---
log("\n[2] 连续读 20 秒")
DUR = 20.0
gaps, n, dup = [], 0, 0
prev_small = None
t_start = time.perf_counter()
last = t_start
while time.perf_counter() - t_start < DUR:
    ok, fr = cap.read()
    if not ok:
        continue
    now = time.perf_counter()
    gaps.append(now - last); last = now; n += 1
    small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (32, 18))
    if prev_small is not None and np.array_equal(small, prev_small):
        dup += 1                    # 完全相同 = 重复帧
    prev_small = small
    frame = fr
elapsed = time.perf_counter() - t_start

g = np.array(gaps[1:]) * 1000
log("   收到 %d 帧 / %.1f 秒  ->  实际 %.2f fps" % (n, elapsed, n / elapsed))
log("   完全重复的帧: %d (%.1f%%)" % (dup, 100.0 * dup / max(n, 1)))
log("   帧间隔 ms: 中位 %.1f  均值 %.1f  最小 %.1f  最大 %.1f  标准差 %.1f"
    % (np.median(g), g.mean(), g.min(), g.max(), g.std()))
log("   分位 10/25/75/90: %.1f / %.1f / %.1f / %.1f"
    % tuple(np.percentile(g, [10, 25, 75, 90])))
big = (g > np.median(g) * 2).sum()
log("   卡顿(间隔>2倍中位数): %d 次 (%.1f%%)" % (big, 100.0 * big / len(g)))

h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
log("\n[3] 画面 %dx%d  亮度均值 %.1f 标准差 %.1f" % (w, h, gray.mean(), gray.std()))
cv2.imwrite(r"E:\DESK\dash_timing_last.jpg", frame)
cap.release()
log("\n完成。报告 %s" % OUT)
f.close()
