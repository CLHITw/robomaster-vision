# -*- coding: utf-8 -*-
"""合成标定图：给定真值内参与畸变，渲染多姿态棋盘格，用于验证 calibrate.py"""
import numpy as np, cv2, os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

W, H = 640, 360
COLS, ROWS = 8, 5            # 内角点
SQ = 24.25                   # mm
OUT = r"E:\DESK\dashcam_metrology\synth_test"
os.makedirs(OUT, exist_ok=True)

# ---- 真值 ----
K_gt = np.array([[300.0, 0, 322.0],
                 [0, 301.0, 178.0],
                 [0, 0, 1.0]])
D_gt = np.array([-0.32, 0.11, 0.0008, -0.0006, -0.02])   # 桶形畸变
print("真值 fx=%.1f fy=%.1f cx=%.1f cy=%.1f" % (K_gt[0,0], K_gt[1,1], K_gt[0,2], K_gt[1,2]))

# ---- 棋盘格母版（板坐标系，单位 mm -> 像素，10 px/mm）----
PPM = 10
bw, bh = (COLS + 1) * SQ, (ROWS + 1) * SQ            # 9x6 方格
board = np.full((int(bh * PPM), int(bw * PPM)), 255, np.uint8)
for r in range(ROWS + 1):
    for c in range(COLS + 1):
        if (r + c) % 2 == 0:
            y0, x0 = int(r * SQ * PPM), int(c * SQ * PPM)
            board[y0:y0 + int(SQ * PPM), x0:x0 + int(SQ * PPM)] = 0
board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)

# ---- 畸变映射：对每个输出像素，找它在理想图上的来源 ----
uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
pts = np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2)
ideal = cv2.undistortPoints(pts, K_gt, D_gt)                  # 归一化理想坐标
ideal = ideal.reshape(-1, 2)
mx = (ideal[:, 0] * K_gt[0, 0] + K_gt[0, 2]).reshape(H, W).astype(np.float32)
my = (ideal[:, 1] * K_gt[1, 1] + K_gt[1, 2]).reshape(H, W).astype(np.float32)

rng = np.random.default_rng(7)
n = 0
for i in range(26):
    # 随机姿态：绕三轴小角度旋转 + 平移，保证板子在画面里且覆盖不同区域
    ang = rng.uniform(-0.45, 0.45, 3)          # rad
    R, _ = cv2.Rodrigues(ang.astype(np.float64))
    tx = rng.uniform(-90, 90)
    ty = rng.uniform(-60, 60)
    tz = rng.uniform(320, 620)
    t = np.array([[tx], [ty], [tz]], np.float64)
    # 板中心移到原点
    corners_mm = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], np.float64)
    corners_mm[:, 0] -= bw / 2.0
    corners_mm[:, 1] -= bh / 2.0
    proj, _ = cv2.projectPoints(corners_mm, R, t, K_gt, np.zeros(5))   # 先不加畸变
    dst = proj.reshape(-1, 2).astype(np.float32)
    if dst.min() < -40 or dst[:, 0].max() > W + 40 or dst[:, 1].max() > H + 40:
        continue
    src = np.array([[0, 0], [board.shape[1], 0],
                    [board.shape[1], board.shape[0]], [0, board.shape[0]]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    ideal_img = cv2.warpPerspective(board, Hm, (W, H), borderValue=(255, 255, 255))
    distorted = cv2.remap(ideal_img, mx, my, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    n += 1
    cv2.imwrite(os.path.join(OUT, "img_%03d.png" % n), distorted)

print("生成 %d 张合成图 -> %s" % (n, OUT))
