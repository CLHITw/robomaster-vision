# -*- coding: utf-8 -*-
"""TIESFONG 行车记录仪 RTSP 路径探测。连上 TIESFONG_Cam_xxxx 后运行。
结果边跑边写 E:\DESK\dashcam_rtsp.txt"""
import socket, io, os, time

HOST, PORT = "192.168.10.1", 8554
OUT = r"E:\DESK\dashcam_rtsp.txt"
f = io.open(OUT, "w", encoding="utf-8")

def log(s=""):
    print(s)
    f.write(str(s) + "\n")
    f.flush()

log("TIESFONG RTSP 探测  %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
log("目标 rtsp://%s:%d" % (HOST, PORT))
log("=" * 58)

def rtsp(method, path, cseq):
    url = "rtsp://%s:%d%s" % (HOST, PORT, path)
    try:
        s = socket.socket(); s.settimeout(1.5)
        s.connect((HOST, PORT))
        req = ("%s %s RTSP/1.0\r\nCSeq: %d\r\nAccept: application/sdp\r\n"
               "User-Agent: probe\r\n\r\n" % (method, url, cseq))
        s.send(req.encode())
        data = b""
        t0 = time.time()
        while time.time() - t0 < 1.5:
            try:
                chunk = s.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            data += chunk
            if b"\r\n\r\n" in data and (b"Content-Length" not in data or len(data) > 200):
                break
        s.close()
        return data.decode("utf-8", "replace")
    except Exception as e:
        return "ERR %s" % e

# 1) OPTIONS：看设备支持哪些方法、有没有 Server 标识
log("\n[1] OPTIONS")
r = rtsp("OPTIONS", "/", 1)
for line in r.splitlines():
    if line.strip():
        log("   " + line.strip())

# 2) 遍历候选路径
PATHS = [
 "", "/", "/live", "/live0", "/live1", "/live2", "/live3",
 "/live/0", "/live/1", "/live/2",
 "/live0.264", "/live1.264", "/live2.264",
 "/ch0", "/ch1", "/ch2", "/ch00", "/ch01",
 "/channel0", "/channel1", "/channel2",
 "/stream0", "/stream1", "/stream2", "/stream",
 "/video0", "/video1", "/video2", "/video",
 "/0", "/1", "/2", "/11", "/12",
 "/cam0", "/cam1", "/cam2", "/front", "/rear", "/inside",
 "/media/video1", "/media/video2",
 "/h264", "/H264", "/mpeg4", "/av0_0", "/av0_1",
 "/cam/realmonitor?channel=1&subtype=0",
 "/user=admin&password=&channel=1&stream=0.sdp",
 "/profile1", "/profile2", "/main", "/sub",
 "/xxx.mov", "/record", "/rtsp",
]

log("\n[2] DESCRIBE  (只列非 404 的)")
hits = []
for i, p in enumerate(PATHS):
    r = rtsp("DESCRIBE", p, 100 + i)
    if r.startswith("ERR"):
        continue
    first = r.split("\r\n")[0].strip()
    if not first:
        continue
    if "200" in first:
        log("   \u2713 %-42s %s" % ("rtsp://%s:%d%s" % (HOST, PORT, p), first))
        hits.append(p)
        for l in r.splitlines():
            if l.startswith(("m=", "a=rtpmap", "a=framerate", "a=x-dimensions",
                             "a=control", "a=fmtp", "s=", "i=")):
                log("        %s" % l)
    elif "404" not in first:
        log("     %-42s %s" % ("rtsp://%s:%d%s" % (HOST, PORT, p), first))

if not hits:
    log("\n   没有路径返回 200。上面若出现 401 = 需要用户名密码；")
    log("   若全是 404 = 路径命名不在候选表里。")

# 3) OpenCV 实际取帧
log("\n[3] OpenCV 取帧")
try:
    import cv2
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
    targets = hits if hits else ["/live", "/0", "/ch0", "/stream0"]
    for n, p in enumerate(targets[:6]):
        url = "rtsp://%s:%d%s" % (HOST, PORT, p)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        ok = cap.isOpened()
        frame = None
        if ok:
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
        if ok and frame is not None:
            h, w = frame.shape[:2]
            jpg = r"E:\DESK\dashcam_ch%d.jpg" % n
            cv2.imwrite(jpg, frame)
            log("   \u2713 %-38s %dx%d  %.1f fps  -> %s"
                % (url, w, h, cap.get(cv2.CAP_PROP_FPS), jpg))
        else:
            log("   \u00d7 %-38s 打不开" % url)
        cap.release()
except Exception as e:
    log("   OpenCV 失败: %s" % e)

log("\n完成。报告 %s" % OUT)
f.close()
