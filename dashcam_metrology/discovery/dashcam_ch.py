# -*- coding: utf-8 -*-
"""TIESFONG 多通道枚举 + 预热取帧。连上 TIESFONG_Cam_xxxx 后运行。"""
import socket, io, os, time

HOST, PORT = "192.168.10.1", 8554
OUT = r"E:\DESK\dashcam_channels.txt"
f = io.open(OUT, "w", encoding="utf-8")
def log(s=""):
    print(s); f.write(str(s) + "\n"); f.flush()

def describe(path):
    url = "rtsp://%s:%d%s" % (HOST, PORT, path)
    try:
        s = socket.socket(); s.settimeout(1.5); s.connect((HOST, PORT))
        s.send(("DESCRIBE %s RTSP/1.0\r\nCSeq: 2\r\nAccept: application/sdp\r\n\r\n" % url).encode())
        d = b""; t0 = time.time()
        while time.time() - t0 < 1.5:
            try: c = s.recv(4096)
            except socket.timeout: break
            if not c: break
            d += c
            if len(d) > 300: break
        s.close(); return d.decode("utf-8", "replace")
    except Exception as e:
        return "ERR %s" % e

log("TIESFONG 通道枚举  %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
log("=" * 58)

# 1) 按 ch01 的命名规律扫全部通道，并试主/子码流变体
cands = []
for i in range(0, 9):
    cands += ["/ch%02d" % i, "/ch%d" % i]
for suf in ["_0", "_1", "_sub", "_main", "/0", "/1", "-1", "-2"]:
    cands += ["/ch01" + suf, "/ch02" + suf]
cands += ["/main", "/sub", "/hd", "/sd", "/big", "/small"]

log("\n[1] 通道扫描（只列 200 OK）")
hits = []
seen = set()
for p in cands:
    if p in seen: continue
    seen.add(p)
    r = describe(p)
    if r.startswith("ERR"): continue
    if "200 OK" in r.split("\r\n")[0]:
        info = [l for l in r.splitlines() if l.startswith(("i=", "m=", "a=rtpmap", "a=x-dimensions", "a=framerate"))]
        log("   \u2713 rtsp://%s:%d%-10s   %s" % (HOST, PORT, p, " | ".join(info)))
        hits.append(p)
log("   共 %d 个可用通道: %s" % (len(hits), hits))

# 2) 每个通道预热后取帧，实测分辨率与帧率
log("\n[2] 预热取帧（丢掉前 30 帧，再测 60 帧）")
try:
    import cv2, numpy as np
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
    for p in hits:
        url = "rtsp://%s:%d%s" % (HOST, PORT, p)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log("   \u00d7 %s 打不开" % url); continue
        for _ in range(30): cap.read()              # 预热
        t0 = time.perf_counter(); got = 0; last = None
        for _ in range(60):
            ok, fr = cap.read()
            if ok and fr is not None: got += 1; last = fr
        dt = time.perf_counter() - t0
        if last is not None:
            h, w = last.shape[:2]
            g = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
            name = p.strip("/").replace("/", "_")
            jpg = r"E:\DESK\dash_%s.jpg" % name
            cv2.imwrite(jpg, last)
            log("   %-28s %4dx%-4d  实测 %4.1f fps  亮度均值 %5.1f 标准差 %5.1f  -> %s"
                % (url, w, h, got/dt, g.mean(), g.std(), jpg))
        else:
            log("   %-28s 预热后仍无帧" % url)
        cap.release()
except Exception as e:
    log("   OpenCV 失败: %s" % e)

log("\n完成。报告 %s" % OUT)
f.close()
