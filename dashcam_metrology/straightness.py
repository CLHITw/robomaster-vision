# -*- coding: utf-8 -*-
"""用棋盘格自身检验去畸变质量：每行/每列角点在世界里共线，去畸变后必须也共线。

重投影误差（RMS）衡量的是"模型能不能解释观测"，它对两个模型都可以很低，
因为畸变系数有足够自由度去拟合。共线性检验问的是另一个问题：
"去畸变之后，本来直的东西直了吗" —— 这才是后面做测距要依赖的性质。

用法
    python straightness.py --square 24.5
"""
import argparse, glob, io, json, os, sys
import numpy as np
import cv2

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def max_deviation(pts):
    """一组点到其最佳拟合直线的最大垂直距离（像素）。"""
    p = pts - pts.mean(0)
    # 主成分方向即最佳拟合直线方向，垂直分量就是偏离
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    normal = vt[1]
    return float(np.abs(p @ normal).max())


def line_errors(corners, cols, rows):
    """对 rows 行 + cols 列分别算最大偏离，返回全部数值。"""
    g = corners.reshape(rows, cols, 2)
    errs = []
    for r in range(rows):
        errs.append(max_deviation(g[r, :, :]))
    for c in range(cols):
        errs.append(max_deviation(g[:, c, :]))
    return errs


def undistort_pixels(corners, K, D, model):
    """把像素点去畸变，仍然用同一个 K 投回像素，这样三组数值可比。"""
    pts = corners.reshape(-1, 1, 2).astype(np.float64)
    # ★ 必须用迭代版并给足次数。cv2.undistortPoints 默认只迭代 5 次，在这颗镜头的
    # 畸变强度下，逆映射本身就残留约 0.18 px —— 那会被误读成"模型没校正干净"。
    crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-10)
    if model == "fisheye":
        norm = cv2.fisheye.undistortPoints(pts, K, np.asarray(D).reshape(4, 1),
                                           criteria=crit)
    else:
        norm = cv2.undistortPointsIter(pts, K, np.asarray(D).reshape(1, -1),
                                       None, None, crit)
    norm = norm.reshape(-1, 2)
    out = np.empty_like(norm)
    out[:, 0] = norm[:, 0] * K[0, 0] + K[0, 2]
    out[:, 1] = norm[:, 1] * K[1, 1] + K[1, 2]

    # 往返校验：把去畸变的点重新加畸变，应该回到原处。回不去的点说明畸变模型
    # 在那个半径上不可逆（径向多项式非单调），迭代解会跑飞。默认迭代次数少的
    # undistortPoints 不会跑飞，但那只是提前停住，答案同样是错的 —— 区别在于
    # 错得有界、看不出来。
    obj = np.hstack([norm, np.zeros((len(norm), 1))]).astype(np.float64)
    if model == "fisheye":
        back, _ = cv2.fisheye.projectPoints(obj.reshape(1, -1, 3), np.zeros(3),
                                            np.zeros(3), K, np.asarray(D).reshape(4, 1))
    else:
        back, _ = cv2.projectPoints(obj, np.zeros(3), np.array([0.0, 0.0, 1.0]),
                                    K, np.asarray(D).reshape(1, -1))
    resid = np.linalg.norm(back.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    return out, resid


def load(path):
    with io.open(path, encoding="utf-8") as f:
        d = json.load(f)
    return (np.array(d["camera_matrix"], dtype=np.float64),
            np.array(d["distortion"], dtype=np.float64),
            d.get("model", "pinhole"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="captures")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--pinhole", default="config/camera_dashcam.json")
    ap.add_argument("--fisheye", default="config/camera_dashcam_fisheye_variant.json")
    ap.add_argument("--roundtrip-px", type=float, default=0.5,
                    help="去畸变再加畸变回不到原点超过这个值，就认为该点不可逆")
    a = ap.parse_args()

    models = []
    for tag, path in (("pinhole", a.pinhole), ("fisheye", a.fisheye)):
        if os.path.exists(path):
            K, D, m = load(path)
            models.append((tag, K, D, m))
        else:
            print("跳过 %s（%s 不存在）" % (tag, path))

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    raw_all, per_model = [], {t: [] for t, _, _, _ in models}
    bad = {t: 0 for t, _, _, _ in models}
    n = 0
    for p in sorted(glob.glob(os.path.join(a.dir, "*.png"))):
        g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
        ok, c = cv2.findChessboardCorners(g, (a.cols, a.rows), flags)
        if not ok:
            continue
        cv2.cornerSubPix(g, c, (5, 5), (-1, -1), CRIT)
        n += 1
        raw_all += line_errors(c.reshape(-1, 2), a.cols, a.rows)
        for tag, K, D, m in models:
            und, resid = undistort_pixels(c, K, D, m)
            if resid.max() > a.roundtrip_px:
                bad[tag] += 1                      # 整张丢掉，一条线上缺点就没法拟合
                continue
            per_model[tag] += line_errors(und, a.cols, a.rows)

    if not n:
        print("没有可用图片")
        return 1

    print("共 %d 张，每张 %d 条线（%d 行 + %d 列），合计 %d 条\n"
          % (n, a.rows + a.cols, a.rows, a.cols, len(raw_all)))
    print("%-12s %8s %8s %8s %8s" % ("", "中位", "均值", "p90", "最大"))

    def show(tag, v):
        v = np.array(v)
        print("%-12s %8.3f %8.3f %8.3f %8.3f"
              % (tag, np.median(v), v.mean(), np.percentile(v, 90), v.max()))

    show("原始(畸变)", raw_all)
    for tag, _, _, _ in models:
        show("去畸变 " + tag, per_model[tag])
    for tag, _, _, _ in models:
        if bad[tag]:
            print("  %s: %d/%d 张因为存在不可逆点被排除（往返误差 > %.2f px）"
                  % (tag, bad[tag], n, a.roundtrip_px))

    print("\n单位：像素。数值是每条线上角点到拟合直线的最大偏离。")
    if len(models) == 2:
        a_, b_ = np.array(per_model["pinhole"]), np.array(per_model["fisheye"])
        win = "pinhole" if np.median(a_) < np.median(b_) else "fisheye"
        print("按共线性中位数：%s 更好（%.3f vs %.3f px）"
              % (win, min(np.median(a_), np.median(b_)), max(np.median(a_), np.median(b_))))
        print("★ 如果两者差别很小（< 0.05 px），就该按别的理由选："
              "视场角、以及 solvePnP 是否能直接吃这套参数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
