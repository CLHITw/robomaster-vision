# -*- coding: utf-8 -*-
"""生成按毫米精确的标定板与 ArUco 标记 PDF。
打印时必须选「实际大小 / 100%」，不要「适合页面」。
棋盘格用 A4 横向，图案 225 x 150 mm，四周留 >30 mm 余量。"""
import cv2, numpy as np, io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from PIL import Image, ImageDraw

DPI = 300
MM = DPI / 25.4
A4_P = (int(210 * MM), int(297 * MM))      # 纵向
A4_L = (int(297 * MM), int(210 * MM))      # 横向
os.makedirs("targets", exist_ok=True)

def mm(v):
    return int(round(v * MM))

def ruler(d, x, y, length_mm=100):
    d.line([(x, y), (x + mm(length_mm), y)], fill=0, width=3)
    for i in range(length_mm + 1):
        h = mm(4) if i % 10 == 0 else (mm(2.5) if i % 5 == 0 else mm(1.5))
        d.line([(x + mm(i), y), (x + mm(i), y - h)], fill=0, width=2)

def check_fits(name, page, x0, y0, w, h, min_margin_mm=8):
    """打印安全检查：图案必须完全落在纸内且四周留够余量"""
    m = [x0 / MM, y0 / MM, (page[0] - (x0 + w)) / MM, (page[1] - (y0 + h)) / MM]
    ok = all(v >= min_margin_mm for v in m)
    print("   %-22s 图案 %.0fx%.0f mm   余量 左%.1f 上%.1f 右%.1f 下%.1f  %s"
          % (name, w / MM, h / MM, m[0], m[1], m[2], m[3], "OK" if ok else "!! 超出"))
    return ok

# ---------- 1. 棋盘格：A4 横向，9x6 方格 => 8x5 内角点，25.0 mm ----------
COLS, ROWS, SQ = 9, 6, 25.0
INNER = (COLS - 1, ROWS - 1)                       # 8 x 5 内角点
bw, bh = mm(COLS * SQ), mm(ROWS * SQ)
img = Image.new("L", A4_L, 255)
d = ImageDraw.Draw(img)
ox, oy = (A4_L[0] - bw) // 2, mm(26)
for r in range(ROWS):
    for c in range(COLS):
        if (r + c) % 2 == 0:
            x0, y0 = ox + mm(c * SQ), oy + mm(r * SQ)
            d.rectangle([x0, y0, x0 + mm(SQ) - 1, y0 + mm(SQ) - 1], fill=0)
d.rectangle([ox - 2, oy - 2, ox + bw + 1, oy + bh + 1], outline=0, width=2)
d.text((ox, mm(10)), "Checkerboard  %dx%d squares  =>  %dx%d INNER CORNERS   square = %.1f mm"
       % (COLS, ROWS, INNER[0], INNER[1], SQ), fill=0)
d.text((ox, mm(16)), "Print at 100% / actual size (NOT 'fit to page'). Then measure a square and use the measured value.", fill=0)
ty = oy + bh + mm(10)
ruler(d, ox, ty + mm(8))
d.text((ox, ty + mm(10)), "100 mm reference ruler - measure it to verify print scale", fill=0)
fits = check_fits("checkerboard", A4_L, ox, oy, bw, bh)
img.save("targets/checkerboard_8x5inner_25mm_A4_landscape.pdf", resolution=DPI)

# 自检：把生成的图案直接喂给检测器，确认内角点数目正确
gray = np.array(img)
found, corners = cv2.findChessboardCorners(gray, INNER,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
print("   自检: findChessboardCorners(%dx%d) -> %s  角点 %d 个"
      % (INNER[0], INNER[1], "检出" if found else "未检出", len(corners) if found else 0))

# ---------- 2. ArUco：A4 纵向 ----------
adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for side, ids in [(100, [0, 1]), (170, [2])]:
    for i in ids:
        px = mm(side)
        sheet = Image.new("L", A4_P, 255)
        dd = ImageDraw.Draw(sheet)
        mx, my = (A4_P[0] - px) // 2, mm(34)
        sheet.paste(Image.fromarray(cv2.aruco.generateImageMarker(adict, i, px)), (mx, my))
        dd.rectangle([mx - 1, my - 1, mx + px, my + px], outline=0, width=1)
        dd.text((mx, mm(14)), "ArUco DICT_4X4_50   id = %d   side = %.1f mm (black square only)" % (i, side), fill=0)
        dd.text((mx, mm(20)), "Print at 100%. Measure the black square edge and use the measured value.", fill=0)
        ty2 = my + px + mm(14)
        ruler(dd, mx, ty2 + mm(8))
        dd.text((mx, ty2 + mm(10)), "100 mm reference ruler", fill=0)
        name = "targets/aruco_id%d_%dmm.pdf" % (i, side)
        fits &= check_fits("aruco id%d %dmm" % (i, side), A4_P, mx, my, px, px)
        sheet.save(name, resolution=DPI)
        det, dids, _ = cv2.aruco.ArucoDetector(adict).detectMarkers(np.array(sheet))
        print("   自检: detectMarkers -> %s" % ("检出 id=%d" % dids[0][0] if dids is not None else "未检出"))

print("\n全部图案适配纸张:", "是" if fits else "否")
