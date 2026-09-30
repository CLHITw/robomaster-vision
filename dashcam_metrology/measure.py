# -*- coding: utf-8 -*-
"""ArUco + PnP 单目测距，并与卷尺真值对比。

这个脚本存在的理由：robomaster 那个项目里没有真值距离，只能报"自洽性"
（重投影误差、独立针孔估计），不能报精度。这里有卷尺，所以可以报真正的
误差，以及误差随距离怎么变、到多远开始不能用。

用法
    # 先探一下：不记录，只看当前检测和解算结果
    python measure.py --probe

    # 每摆一个距离跑一次，--truth 填卷尺量出来的毫米数
    python measure.py --truth 500
    python measure.py --truth 1000
    python measure.py --truth 2000

    # 全部测完出报告
    python measure.py --report

真值怎么量：卷尺从镜头玻璃面量到标记板平面。光心在镜头内部若干毫米处，
量不到，所以会有一个固定偏移 —— 报告里用线性拟合把它和比例误差分开。
"""
import argparse, csv, io, os, sys, time, threading
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ 标定参数
def load_camera(path):
    import json
    with io.open(path, encoding="utf-8") as f:
        d = json.load(f)
    K = np.array(d["camera_matrix"], dtype=np.float64)
    D = np.array(d["distortion"], dtype=np.float64).reshape(1, -1)
    if d.get("model") != "pinhole":
        print("⚠ 标定文件是 %s 模型，本脚本假定针孔（solvePnP 直接吃针孔参数）" % d.get("model"))
    return K, D, d["image_size"]


def marker_object_points(side_mm):
    """正方形标记的四角，顺序与 aruco 输出一致：左上、右上、右下、左下。
    原点在标记中心，z=0 平面，单位毫米 —— 所以解出来的 tvec 也是毫米。"""
    s = side_mm / 2.0
    return np.array([[-s,  s, 0.0],
                     [ s,  s, 0.0],
                     [ s, -s, 0.0],
                     [-s, -s, 0.0]], dtype=np.float64)


def solve_marker(corners, side_mm, K, D):
    """四角 -> 位姿。返回 (距离mm, 深度mm, 重投影误差px, rvec, tvec)。"""
    objp = marker_object_points(side_mm)
    imgp = corners.reshape(4, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    err = float(np.sqrt(((proj.reshape(4, 2) - imgp) ** 2).sum(1)).mean())
    t = tvec.ravel()
    return float(np.linalg.norm(t)), float(t[2]), err, rvec, tvec


# ------------------------------------------------------------------ 取流
class Stream:
    """后台线程把缓冲抽干，主循环永远拿最新帧。RTSP 不这么做会读到几秒前的画面。"""
    def __init__(self, url):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise SystemExit("打不开 %s —— 确认已连上记录仪 Wi-Fi" % url)
        self._latest, self._lock, self._stop = None, threading.Lock(), False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        while self._latest is None:
            time.sleep(0.05)
        # 自动曝光要爬升，开头十几帧偏暗，丢掉
        for _ in range(25):
            self.read()
            time.sleep(0.03)

    def _run(self):
        while not self._stop:
            ok, fr = self.cap.read()
            if ok:
                with self._lock:
                    self._latest = fr
            else:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def close(self):
        # 必须先等读取线程退出再 release。直接 release 会在 FFmpeg 解码中途
        # 抽走上下文，触发 libavcodec 的 async_lock 断言并 abort 整个进程。
        self._stop = True
        self._thread.join(timeout=3.0)
        self.cap.release()


# ------------------------------------------------------------------ 主流程
def parse_sizes(s):
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[int(k)] = float(v)
    return out


def collect(a, K, D, sizes):
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    st = Stream(a.url)
    rows, seen = [], {}
    t0 = time.time()
    print("采集中……需要 %d 个有效样本，按 q 提前退出" % a.samples)
    cv2.namedWindow("measure", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("measure", 1280, 720)
    while len(rows) < a.samples and time.time() - t0 < a.timeout:
        frame = st.read()
        if frame is None:
            continue
        view = frame.copy()
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(view, corners, ids)
            for c, i in zip(corners, ids.ravel()):
                if i not in sizes:
                    continue
                r = solve_marker(c, sizes[i], K, D)
                if r is None:
                    continue
                dist, depth, err, rvec, tvec = r
                seen[i] = (dist, err)
                if a.truth is not None and err <= a.max_reproj:
                    rows.append({"truth_mm": a.truth, "id": int(i),
                                 "side_mm": sizes[i], "dist_mm": round(dist, 1),
                                 "depth_mm": round(depth, 1), "reproj_px": round(err, 3)})
                cv2.putText(view, "id%d  %.0f mm  reproj %.2f px" % (i, dist, err),
                            (10, 26 + 24 * int(np.where(ids.ravel() == i)[0][0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        else:
            cv2.putText(view, "no marker", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)
        if a.truth is not None:
            cv2.putText(view, "truth %.0f mm   collected %d/%d" % (a.truth, len(rows), a.samples),
                        (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow("measure", view)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    st.close()
    cv2.destroyAllWindows()
    return rows, seen


def write_rows(path, rows):
    exists = os.path.exists(path)
    with io.open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["truth_mm", "id", "side_mm", "dist_mm", "depth_mm", "reproj_px"])
        if not exists:
            w.writeheader()
        w.writerows(rows)


def report(path, assume_side=None):
    """assume_side: 假设标记实际边长是这个值重新换算。
    solvePnP 解出的距离与假定的物体尺寸严格成正比，所以换算只是乘一个比例，
    不需要重新采集数据。"""
    if not os.path.exists(path):
        print("还没有数据：%s" % path)
        return 1
    import collections
    data = collections.defaultdict(list)
    with io.open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = float(r["dist_mm"])
            if assume_side is not None:
                d *= assume_side / float(r["side_mm"])
            data[float(r["truth_mm"])].append((d, float(r["reproj_px"])))
    if assume_side is not None:
        print("assumed marker side %.2f mm, distances rescaled" % assume_side)
        print("")
    if not data:
        print("文件是空的")
        return 1

    print("%8s %6s %10s %10s %9s %9s %9s"
          % ("真值mm", "样本", "解算中位", "标准差", "误差mm", "误差%", "重投影px"))
    xs, ys = [], []
    for t in sorted(data):
        v = np.array([d for d, _ in data[t]])
        e = np.array([p for _, p in data[t]])
        med = np.median(v)
        xs.append(t); ys.append(med)
        print("%8.0f %6d %10.1f %10.1f %9.1f %9.2f %9.3f"
              % (t, len(v), med, v.std(), med - t, 100 * (med - t) / t, np.median(e)))

    if len(xs) >= 2:
        # 解算 = a x 真值 + b。a 偏离 1 = 尺度误差（多半是标记尺寸量错或内参偏差），
        # b = 固定偏移（光心位置量不到，卷尺是从镜头玻璃面起量的）
        a_, b_ = np.polyfit(xs, ys, 1)
        resid = np.array(ys) - (a_ * np.array(xs) + b_)
        print("\n线性拟合  解算 = %.4f x 真值 %+.1f mm" % (a_, b_))
        print("  尺度误差 %+.2f%%   固定偏移 %+.1f mm   拟合残差最大 %.1f mm"
              % (100 * (a_ - 1), b_, np.abs(resid).max()))
        print("  尺度误差主要来自标记实际边长与填入值的差；固定偏移主要是光心位置。")
        sides = set()
        with io.open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sides.add(float(r["side_mm"]))
        if len(sides) == 1 and abs(a_ - 1) > 1e-6:
            s0 = sides.pop() if assume_side is None else assume_side
            print("  若尺度误差全部来自边长：标记实际边长应为 %.2f mm（当前填的是 %.2f）"
                  % (s0 / a_, s0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtsp://192.168.10.1:8554/ch01")
    ap.add_argument("--camera", default=os.path.join(HERE, "config", "camera_dashcam.json"))
    ap.add_argument("--sizes", default="0=100.0,1=100.0,2=170.0",
                    help="标记边长映射 id=毫米，用实测值，不是标称值")
    ap.add_argument("--truth", type=float, default=None, help="卷尺量出的真实距离（毫米）")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--max-reproj", type=float, default=2.0, help="重投影误差超过这个的样本丢弃")
    ap.add_argument("--csv", default=os.path.join(HERE, "measurements.csv"))
    ap.add_argument("--probe", action="store_true", help="只看检测和解算，不记录")
    ap.add_argument("--report", action="store_true", help="对已有数据出报告")
    ap.add_argument("--assume-side", type=float, default=None,
                    help="出报告时假定标记实际边长为该值重新换算（距离与边长成正比）")
    ap.add_argument("--drop-truth", type=float, default=None,
                    help="从 csv 中删掉某个 truth 的全部记录")
    a = ap.parse_args()

    if a.drop_truth is not None:
        rows = [r for r in csv.DictReader(io.open(a.csv, encoding="utf-8"))
                if float(r["truth_mm"]) != a.drop_truth]
        with io.open(a.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["truth_mm", "id", "side_mm", "dist_mm", "depth_mm", "reproj_px"])
            w.writeheader(); w.writerows(rows)
        print("已删除 truth=%.0f 的记录，剩余 %d 行" % (a.drop_truth, len(rows)))
        return 0
    if a.report:
        return report(a.csv, a.assume_side)

    K, D, size = load_camera(a.camera)
    sizes = parse_sizes(a.sizes)
    print("内参 %s  (%dx%d)" % (os.path.basename(a.camera), size[0], size[1]))
    print("标记边长 %s" % sizes)
    if a.probe:
        a.truth = None
        a.samples = 1 << 30
        a.timeout = 30.0
        print("探测模式：30 秒，只显示不记录\n")

    rows, seen = collect(a, K, D, sizes)
    for i, (dist, err) in sorted(seen.items()):
        print("  最后一次看到 id%d：%.0f mm，重投影 %.2f px" % (i, dist, err))
    if a.truth is not None and rows:
        v = np.array([r["dist_mm"] for r in rows])
        print("\n真值 %.0f mm   解算中位 %.1f mm   误差 %+.1f mm (%+.2f%%)   标准差 %.1f mm"
              % (a.truth, np.median(v), np.median(v) - a.truth,
                 100 * (np.median(v) - a.truth) / a.truth, v.std()))
        write_rows(a.csv, rows)
        print("%d 条写入 %s" % (len(rows), a.csv))
    elif a.truth is not None:
        print("没有采到有效样本 —— 标记是否在画面里、重投影是否超过 %.1f px" % a.max_reproj)
    return 0


if __name__ == "__main__":
    # 只在直接运行时重绑 stdout；被当作库 import 时不该动调用方的输出流
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.exit(main())
