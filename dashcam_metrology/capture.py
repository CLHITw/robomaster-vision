# -*- coding: utf-8 -*-
"""标定图像采集，带实时覆盖反馈。

按键：  空格 = 保存    d = 删除上一张    q = 退出

窗口上会显示：
  · 棋盘格是否检出（绿框 / 红框）
  · 标定板占画幅的面积比 —— 低于 10% 会变红警告，太远的板子标不出畸变
  · 3x3 覆盖格 —— 已经拍到的格子变绿，提醒你把板子推到还没去过的区域

为什么要盯这两个指标：广角镜头的畸变集中在画面边缘，边缘没有数据，
畸变系数就是外推出来的；板子占比太小则角点像素太少，定位精度不够。
"""
import cv2, os, sys, io, time, threading, argparse
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="rtsp://192.168.10.1:8554/ch01")
ap.add_argument("--out", default=os.path.join(HERE, "captures"),
                help="默认写在脚本旁边，不是当前工作目录")
ap.add_argument("--cols", type=int, default=8, help="棋盘格内角点列数")
ap.add_argument("--rows", type=int, default=5, help="棋盘格内角点行数")
ap.add_argument("--min-area", type=float, default=10.0, help="低于这个面积占比就警告(%%)")
ap.add_argument("--no-board", action="store_true", help="不做棋盘检测，纯采集")
a = ap.parse_args()

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
os.makedirs(a.out, exist_ok=True)

cap = cv2.VideoCapture(a.url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("打不开 %s —— 确认已连上记录仪 Wi-Fi" % a.url)
    raise SystemExit(1)

# 后台线程只负责把缓冲抽干，主循环永远拿到最新帧。
# 不这么做的话，按下空格存的是几秒前的画面（RTSP 会缓冲）。
latest, lock, stop = [None], threading.Lock(), [False]
def reader():
    while not stop[0]:
        ok, fr = cap.read()
        if ok:
            with lock:
                latest[0] = fr
        else:
            time.sleep(0.01)
_reader = threading.Thread(target=reader, daemon=True)
_reader.start()
while latest[0] is None:
    time.sleep(0.05)

PAT = (a.cols, a.rows)
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK


def board_stats(gray, shape):
    """返回 (是否检出, 角点, 板心归一化坐标, 面积占比%)"""
    ok, corners = cv2.findChessboardCorners(gray, PAT, FLAGS)
    if not ok:
        return False, None, None, 0.0
    pts = corners.reshape(-1, 2)
    h, w = shape
    area = cv2.contourArea(cv2.convexHull(pts.astype(np.float32))) / (w * h) * 100.0
    return True, corners, (pts[:, 0].mean() / w, pts[:, 1].mean() / h), area


def cell_of(center):
    cx, cy = center
    return min(int(cy * 3), 2), min(int(cx * 3), 2)


# 已保存图片的覆盖情况：重启程序时重新统计，接着上次拍
grid = np.zeros((3, 3), int)
saved = sorted(f for f in os.listdir(a.out) if f.lower().endswith(".png"))
for f in saved:
    g = cv2.cvtColor(cv2.imread(os.path.join(a.out, f)), cv2.COLOR_BGR2GRAY)
    ok, _, c, _ = board_stats(g, g.shape)
    if ok:
        r, col = cell_of(c)
        grid[r, col] += 1
n = len(saved)

print("输出目录 %s （已有 %d 张）" % (a.out, n))
print("空格=保存   d=删除上一张   q=退出")
print("目标：每个格子至少 2 张，面积占比 >= %.0f%%，总共 25~30 张\n" % a.min_area)

cv2.namedWindow("capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("capture", 1280, 720)
last_saved = None

while True:
    with lock:
        frame = latest[0].copy()
    view = frame.copy()
    h, w = frame.shape[:2]

    found, corners, center, area = False, None, None, 0.0
    if not a.no_board:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners, center, area = board_stats(gray, (h, w))
        if found:
            cv2.drawChessboardCorners(view, PAT, corners, found)

    # 3x3 覆盖格：拍过的格子填绿，没拍过的填灰
    for r in range(3):
        for c in range(3):
            x0, y0 = int(c * w / 3), int(r * h / 3)
            x1, y1 = int((c + 1) * w / 3), int((r + 1) * h / 3)
            col = (0, 150, 0) if grid[r, c] >= 2 else ((0, 110, 160) if grid[r, c] == 1 else (90, 90, 90))
            cv2.rectangle(view, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), col, 1)
            cv2.putText(view, str(grid[r, c]), (x0 + 8, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    if found:
        r, c = cell_of(center)
        x0, y0 = int(c * w / 3), int(r * h / 3)
        x1, y1 = int((c + 1) * w / 3), int((r + 1) * h / 3)
        cv2.rectangle(view, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (0, 255, 255), 2)

    edge = (0, 200, 0) if found else (0, 0, 220)
    cv2.rectangle(view, (0, 0), (w - 1, h - 1), edge, 3)

    if found:
        acol = (0, 200, 0) if area >= a.min_area else (0, 165, 255)
        msg = "area %.1f%%%s" % (area, "" if area >= a.min_area else "  TOO FAR - move closer")
        cv2.putText(view, msg, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, acol, 2)
    txt = "BOARD FOUND" if found else ("no board" if not a.no_board else "raw capture")
    cv2.putText(view, "%s   saved=%d" % (txt, n), (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, edge, 2)
    cv2.imshow("capture", view)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord(" "):
        n += 1
        last_saved = os.path.join(a.out, "img_%03d.png" % n)
        cv2.imwrite(last_saved, frame)              # 存原始帧，不存标注版
        if found:
            r, c = cell_of(center)
            grid[r, c] += 1
        flag = "角点 ✓  面积 %.1f%%%s" % (area, "" if area >= a.min_area else "  ← 太远") if found else "未检出 ✗"
        print("  保存 %s   %s" % (os.path.basename(last_saved), flag))
    if k == ord("d") and last_saved and os.path.exists(last_saved):
        os.remove(last_saved)
        print("  删除 %s" % os.path.basename(last_saved))
        n -= 1
        last_saved = None

# 先等读取线程退出再 release，否则 FFmpeg 会在解码中途被抽走上下文并 abort
stop[0] = True
_reader.join(timeout=3.0)
cap.release()
cv2.destroyAllWindows()
print("\n共 %d 张在 %s" % (n, a.out))
print("覆盖格:")
for row in grid:
    print("   " + "  ".join("%2d" % v for v in row))
if (grid < 2).any():
    print("⚠ 还有格子不足 2 张，畸变系数在那些区域是外推的")
