# -*- coding: utf-8 -*-
"""连上行车记录仪 Wi-Fi 后运行，把探测结果写到 E:\DESK\dashcam_report.txt"""
import socket, subprocess, sys, io, os, re, time

OUT = r"E:\DESK\dashcam_report.txt"
log_lines = []
def log(s=""):
    print(s)
    log_lines.append(str(s))

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, timeout=25,
                              encoding="utf-8", errors="replace").stdout
    except Exception as e:
        return "ERR %s" % e

log("=" * 60)
log("行车记录仪探测报告  %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
log("=" * 60)

# ---------- 1. 网络身份 ----------
log("\n[1] 当前 Wi-Fi 连接")
for line in sh("netsh wlan show interfaces").splitlines():
    if re.search(r"SSID|BSSID|信号|Signal|状态|State|Radio|Channel|频道", line):
        log("   " + line.strip())

log("\n[2] IP 配置")
ipcfg = sh("ipconfig")
block, keep = [], False
for line in ipcfg.splitlines():
    if re.search(r"适配器|adapter", line, re.I):
        keep = "WLAN" in line or "Wi-Fi" in line or "无线" in line
    if keep and line.strip():
        log("   " + line.rstrip())

# 网关
gw = None
for line in ipcfg.splitlines():
    m = re.search(r"(?:默认网关|Default Gateway)[^\d]*(\d+\.\d+\.\d+\.\d+)", line)
    if m:
        gw = m.group(1)
log("\n   -> 网关 = %s" % gw)
if not gw:
    log("   没拿到网关，可能没连上记录仪的热点。以下用常见默认地址试。")

# ---------- 2. 候选主机 ----------
cands = []
if gw:
    cands.append(gw)
for ip in ["192.168.1.254", "192.168.42.1", "192.168.0.1", "192.168.1.1",
           "192.168.10.1", "10.0.0.1", "192.168.100.1"]:
    if ip not in cands:
        cands.append(ip)

# ---------- 3. 端口扫描 ----------
PORTS = [80, 81, 88, 443, 554, 1935, 5000, 7878, 8000, 8080, 8081,
         8090, 8554, 34567, 9527]
log("\n[3] 端口扫描")
live = {}
for host in cands:
    openp = []
    for p in PORTS:
        s = socket.socket(); s.settimeout(0.4)
        try:
            if s.connect_ex((host, p)) == 0:
                openp.append(p)
        except Exception:
            pass
        finally:
            s.close()
    if openp:
        live[host] = openp
        log("   %-16s 开放端口: %s" % (host, openp))
if not live:
    log("   没有主机响应。确认已经连上记录仪的 Wi-Fi 热点。")

# ---------- 4. HTTP 探测 ----------
import urllib.request
PATHS = ["/", "/index.html", "/?custom=1&cmd=3001", "/cgi-bin/hi3510/param.cgi",
         "/app/api/status", "/api/status", "/cgi-bin/Config.cgi", "/dev/info",
         "/vlc.html", "/live", "/status.json"]
log("\n[4] HTTP 探测")
for host, ports in live.items():
    for p in [x for x in ports if x in (80, 81, 88, 8000, 8080, 8081, 8090, 5000)]:
        for path in PATHS:
            url = "http://%s:%d%s" % (host, p, path)
            try:
                r = urllib.request.urlopen(url, timeout=3)
                body = r.read(400)
                log("   %s  -> %s  %s" % (url, r.status, dict(r.headers).get("Server", "")))
                txt = body.decode("utf-8", "replace").replace("\n", " ")[:250]
                log("        %s" % txt)
            except Exception as e:
                msg = str(e)
                if "timed out" not in msg and "refused" not in msg:
                    log("   %s  -> %s" % (url, msg[:80]))

# ---------- 5. RTSP 握手 ----------
RTSP_PATHS = ["/live", "/live0", "/live_ch0", "/ch0", "/ch1", "/stream0", "/stream1",
              "/video0", "/0", "/11", "/cam/realmonitor?channel=1&subtype=0",
              "/xxx.mov", "/media/video1", "/h264", ""]
log("\n[5] RTSP 握手 (DESCRIBE)")
found_rtsp = []
for host, ports in live.items():
    for p in [x for x in ports if x in (554, 8554, 1935)]:
        for path in RTSP_PATHS:
            url = "rtsp://%s:%d%s" % (host, p, path)
            try:
                s = socket.socket(); s.settimeout(2.5)
                s.connect((host, p))
                req = ("DESCRIBE %s RTSP/1.0\r\nCSeq: 1\r\n"
                       "Accept: application/sdp\r\nUser-Agent: probe\r\n\r\n" % url)
                s.send(req.encode())
                resp = s.recv(2048).decode("utf-8", "replace")
                s.close()
                first = resp.split("\r\n")[0]
                if "200" in first:
                    log("   ✓ %s  -> %s" % (url, first))
                    found_rtsp.append(url)
                    for l in resp.splitlines():
                        if l.startswith(("m=", "a=rtpmap", "a=framerate", "a=x-dimensions")):
                            log("        %s" % l)
                elif "401" in first or "404" not in first:
                    log("     %s  -> %s" % (url, first))
            except Exception:
                pass
if not found_rtsp:
    log("   没有 RTSP 流握手成功。")

# ---------- 6. 用 OpenCV 实际取一帧 ----------
log("\n[6] OpenCV 取帧")
try:
    import cv2
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    for url in found_rtsp[:4]:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        ok, frame = (cap.isOpened(), None)
        if ok:
            ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            log("   ✓ %s  ->  %dx%d  %.1f fps" % (url, w, h, fps))
            cv2.imwrite(r"E:\DESK\dashcam_frame_%d.jpg" % found_rtsp.index(url), frame)
        else:
            log("   × %s  打不开" % url)
        cap.release()
    if not found_rtsp:
        log("   没有可用的 RTSP 地址，跳过。")
except Exception as e:
    log("   OpenCV 失败: %s" % e)

# ---------- 写文件 ----------
with io.open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")
print("\n报告已写入 %s" % OUT)
