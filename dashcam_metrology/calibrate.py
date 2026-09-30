# -*- coding: utf-8 -*-
"""棋盘格相机标定：针孔模型与 fisheye 模型并排跑，让数据决定用哪个。

用法
    python calibrate.py --square 24.25
    python calibrate.py --square 24.25 --drop-worst 3
    python calibrate.py --square 24.25 --undistort captures/img_007.png

为什么两个模型都跑：广角镜头超过约 120° 视场时，针孔模型的 5 个畸变系数
拟合不住边缘，重投影误差降不下来或边缘反向弯曲。但"超过 120°"要标定完
才知道，所以先都跑，再比。
"""
import argparse, glob, io, json, os, sys, math
import numpy as np
import cv2

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 亚像素精化的迭代终止条件：30 次或移动小于 0.001 像素
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


# ---------------------------------------------------------------- 角点检测
def find_corners(paths, pattern, subpix_win):
    """在每张图上找内角点。返回 (图像点列表, 用上的文件名, 图像尺寸)。"""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    img_pts, used, size = [], [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print("   跳过（读不出）: %s" % p)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]              # (w, h)
        elif gray.shape[::-1] != size:
            print("   跳过（尺寸不一致）: %s" % p)
            continue

        ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if not ok:
            print("   未检出: %s" % os.path.basename(p))
            continue
        # 亚像素精化。窗口要小于半个方格，否则会把相邻角点吸进来；
        # 640x360 下方格只有十几像素，所以默认 (5,5) 而不是常见的 (11,11)。
        cv2.cornerSubPix(gray, corners, (subpix_win, subpix_win), (-1, -1), SUBPIX_CRITERIA)
        img_pts.append(corners)
        used.append(p)
    return img_pts, used, size


def object_grid(pattern, square_mm):
    """一张棋盘格在自身坐标系下的角点：z=0 平面上的规则网格，单位毫米。

    所有距离结果都按 square_mm 等比例缩放 —— 这个数填错，
    标定内参不受影响，但之后解算出来的每一个距离都会整体错同一个倍数。
    """
    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * float(square_mm)


# ---------------------------------------------------------------- 逐张误差
def per_image_error(obj_pts, img_pts, rvecs, tvecs, K, D, fisheye=False):
    """每张图的重投影 RMS。用来找出模糊或靶板变形的那几张。"""
    errs = []
    for i in range(len(obj_pts)):
        if fisheye:
            proj, _ = cv2.fisheye.projectPoints(
                obj_pts[i].reshape(-1, 1, 3), rvecs[i], tvecs[i], K, D)
        else:
            proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, D)
        diff = proj.reshape(-1, 2) - img_pts[i].reshape(-1, 2)
        errs.append(float(np.sqrt((diff ** 2).sum() / len(diff))))
    return np.array(errs)


# ---------------------------------------------------------------- 两个模型
def calibrate_pinhole(obj_pts, img_pts, size):
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    errs = per_image_error(obj_pts, img_pts, rvecs, tvecs, K, D, fisheye=False)
    return {"model": "pinhole", "rms": float(rms), "K": K, "D": D.ravel(), "errs": errs}


def calibrate_fisheye(obj_pts, img_pts, size):
    """fisheye 的输入形状和针孔不同：物点要 (1,N,3)，像点要 (1,N,2)。"""
    n = len(obj_pts)
    obj_f = [o.reshape(1, -1, 3).astype(np.float64) for o in obj_pts]
    img_f = [p.reshape(1, -1, 2).astype(np.float64) for p in img_pts]
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), np.float64) for _ in range(n)]
    tvecs = [np.zeros((1, 1, 3), np.float64) for _ in range(n)]
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_FIX_SKEW)
    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_f, img_f, size, K, D, rvecs, tvecs, flags, SUBPIX_CRITERIA)
    except cv2.error as e:
        return {"model": "fisheye", "failed": str(e).strip().splitlines()[-1]}
    errs = per_image_error(obj_f, img_f, rvecs, tvecs, K, D, fisheye=True)
    return {"model": "fisheye", "rms": float(rms), "K": K, "D": D.ravel(), "errs": errs}


# ---------------------------------------------------------------- 体检
def report(res, size, names):
    w, h = size
    K, D = res["K"], res["D"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fov_h = 2 * math.degrees(math.atan(w / (2 * fx)))
    fov_v = 2 * math.degrees(math.atan(h / (2 * fy)))

    print("\n--- %s ---" % res["model"])
    print("  整体 RMS      %.4f px" % res["rms"])
    print("  fx, fy        %.2f, %.2f      (相差 %.2f%%)"
          % (fx, fy, 100 * abs(fx - fy) / fx))
    print("  cx, cy        %.2f, %.2f      (画面中心 %.1f, %.1f)" % (cx, cy, w / 2, h / 2))
    print("  主点偏移      %.1f%% 宽, %.1f%% 高"
          % (100 * abs(cx - w / 2) / w, 100 * abs(cy - h / 2) / h))
    print("  视场角        水平 %.1f°   垂直 %.1f°" % (fov_h, fov_v))
    print("  畸变系数      %s" % np.array2string(D, precision=4, suppress_small=True))

    errs = res["errs"]
    print("  逐张误差      中位 %.3f  最大 %.3f  px" % (np.median(errs), errs.max()))
    order = np.argsort(errs)[::-1]
    print("  误差最大的几张:")
    for i in order[:5]:
        print("      %-22s %.3f px" % (os.path.basename(names[i]), errs[i]))

    # 体检：这几条不过，标定结果不可信
    warn = []
    if res["rms"] > 1.0:
        warn.append("RMS > 1.0 px：多半有模糊/靶板变形的图，或方格尺寸填错了")
    if 100 * abs(fx - fy) / fx > 2:
        warn.append("fx 与 fy 相差 > 2%：拍摄姿态多样性不够")
    if 100 * abs(cx - w / 2) / w > 10 or 100 * abs(cy - h / 2) / h > 10:
        warn.append("主点离画面中心 > 10%：流可能被裁切过，或数据有问题")
    for m in warn:
        print("  ⚠ %s" % m)
    return fov_h


# ---------------------------------------------------------------- 去畸变对比
def undistort_preview(path, res, size, out_path):
    img = cv2.imread(path)
    K, D = res["K"], res["D"]
    if res["model"] == "fisheye":
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D.reshape(4, 1), size, np.eye(3), balance=0.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D.reshape(4, 1), np.eye(3), newK, size, cv2.CV_16SC2)
        und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    else:
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, size, 0)
        und = cv2.undistort(img, K, D, None, newK)
    pair = np.hstack([img, und])
    cv2.line(pair, (size[0], 0), (size[0], size[1]), (0, 0, 255), 1)
    cv2.putText(pair, "original", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(pair, "undistorted (%s)" % res["model"], (size[0] + 8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(out_path, pair)
    print("\n去畸变对比图 -> %s" % out_path)
    print("  ★ 看画面里本来笔直的东西（门框、路沿、瓷砖缝）在右半边是不是直的。")
    print("    针孔模型在超广角上的典型失败是中间直了、靠近边缘反向弯曲，RMS 看不出来。")


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="captures", help="标定图目录")
    ap.add_argument("--cols", type=int, default=8, help="内角点列数")
    ap.add_argument("--rows", type=int, default=5, help="内角点行数")
    ap.add_argument("--square", type=float, required=True,
                    help="实测的方格边长（毫米）—— 量出来的，不是标称的 25.0")
    ap.add_argument("--subpix-win", type=int, default=5, help="亚像素搜索窗口半宽")
    ap.add_argument("--drop-worst", type=int, default=0,
                    help="剔掉重投影误差最大的 N 张后重标定")
    ap.add_argument("--undistort", default=None, help="用这张图做去畸变前后对比")
    ap.add_argument("--model", choices=["auto", "pinhole", "fisheye"], default="auto",
                    help="强制用某个模型写入结果；auto 按 RMS 选（但 RMS 不是唯一判据）")
    ap.add_argument("--out", default="config/camera_dashcam.json")
    ap.add_argument("--name", default="TIESFONG dash camera, RTSP preview stream")
    a = ap.parse_args()

    pattern = (a.cols, a.rows)
    paths = sorted(glob.glob(os.path.join(a.dir, "*.png")) +
                   glob.glob(os.path.join(a.dir, "*.jpg")))
    if not paths:
        print("目录 %s 里没有图。先跑 capture.py 采集。" % os.path.abspath(a.dir))
        return 1

    print("方格边长 %.2f mm   图案 %dx%d 内角点   共 %d 张候选\n"
          % (a.square, a.cols, a.rows, len(paths)))
    img_pts, names, size = find_corners(paths, pattern, a.subpix_win)
    print("\n检出 %d / %d 张   图像尺寸 %dx%d" % (len(img_pts), len(paths), size[0], size[1]))
    if len(img_pts) < 8:
        print("可用图太少（< 8 张），标定不可信。回去多拍几张，especially 画面四角和边缘。")
        return 1

    objp = object_grid(pattern, a.square)
    obj_pts = [objp.copy() for _ in img_pts]

    pin = calibrate_pinhole(obj_pts, img_pts, size)
    fov_h = report(pin, size, names)

    fish = calibrate_fisheye(obj_pts, img_pts, size)
    if "failed" in fish:
        print("\n--- fisheye ---\n  标定失败: %s" % fish["failed"])
        fish = None
    else:
        report(fish, size, names)

    # 剔除离群图后重跑
    if a.drop_worst > 0:
        drop = set(np.argsort(pin["errs"])[::-1][:a.drop_worst].tolist())
        print("\n剔除误差最大的 %d 张后重标定: %s"
              % (a.drop_worst, ", ".join(os.path.basename(names[i]) for i in sorted(drop))))
        keep = [i for i in range(len(img_pts)) if i not in drop]
        img_pts = [img_pts[i] for i in keep]
        names = [names[i] for i in keep]
        obj_pts = [objp.copy() for _ in img_pts]
        pin = calibrate_pinhole(obj_pts, img_pts, size)
        fov_h = report(pin, size, names)
        fish = calibrate_fisheye(obj_pts, img_pts, size)
        if "failed" not in fish:
            report(fish, size, names)
        else:
            fish = None

    # 选模型：视场角给出先验，重投影误差做裁决
    print("\n" + "=" * 62)
    if fov_h > 120:
        prior = "水平视场 %.0f° > 120°，先验上应当用 fisheye" % fov_h
    elif fov_h < 100:
        prior = "水平视场 %.0f° < 100°，针孔模型通常够用" % fov_h
    else:
        prior = "水平视场 %.0f° 处于两可区间，看误差" % fov_h
    print(prior)

    chosen = pin
    if fish is not None:
        better = "fisheye" if fish["rms"] < pin["rms"] else "pinhole"
        print("RMS: pinhole %.4f px   fisheye %.4f px   ->  %s 低 %.1f%%"
              % (pin["rms"], fish["rms"], better,
                 100 * abs(pin["rms"] - fish["rms"]) / max(pin["rms"], fish["rms"])))
        chosen = fish if better == "fisheye" else pin
    if a.model != "auto":
        chosen = fish if (a.model == "fisheye" and fish is not None) else pin
        print("手动指定采用: %s" % chosen["model"])
    else:
        print("按 RMS 采用: %s" % chosen["model"])
    print("★ 这只是初判。最终判据是去畸变后直线是否真的变直 —— 用 --undistort 看一眼。")
    print("=" * 62)

    if a.undistort:
        os.makedirs("outputs", exist_ok=True)
        undistort_preview(a.undistort, chosen, size,
                          os.path.join("outputs", "undistort_%s.png" % chosen["model"]))

    # 存成与 robomaster-vision 的 config/camera_2021.json 相同的 schema，
    # 这样 replay/pose.py 的 load_camera() 和 scale_camera() 可以直接复用。
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    data = {
        "name": a.name,
        "image_size": [int(size[0]), int(size[1])],
        "source": "checkerboard calibration, %d images, %dx%d inner corners, square %.2f mm"
                  % (len(names), a.cols, a.rows, a.square),
        "camera_matrix": chosen["K"].tolist(),
        "distortion": chosen["D"].tolist(),
        "model": chosen["model"],
        "square_size_mm": a.square,
        "rms_reprojection_px": round(chosen["rms"], 4),
        "n_images": len(names),
        "fov_deg": [round(fov_h, 1),
                    round(2 * math.degrees(math.atan(size[1] / (2 * chosen["K"][1, 1]))), 1)],
        "note": ("Bound to the %dx%d RTSP preview stream. Rescale fx, fy, cx, cy with "
                 "scale_camera() before using at any other resolution." % (size[0], size[1])),
    }
    with io.open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("\n内参写入 %s" % os.path.abspath(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
