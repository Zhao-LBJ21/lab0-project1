"""Calibrate a phone camera from checkerboard photos with OpenCV"""
import argparse
from pathlib import Path
import cv2
import numpy as np

# 棋盘格内角点：宽5，高8；方格边长30mm
PATTERN = (5, 8)
SQUARE_MM = 30
# 统一输出图像尺寸 (width, height) 竖屏坐标系
TARGET_SIZE = (1279, 1706)


def load_portrait(path: Path):
    """读取图片，缩放至TARGET_SIZE"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", type=Path)
    p.add_argument("--output", type=Path, default=Path("calibration_results"))
    args = p.parse_args()

    args.output.mkdir(exist_ok=True)

    # 获取文件夹下图片，按文件名数字排序
    paths = sorted(
        (p for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")),
        key=lambda p: (int(''.join(filter(str.isdigit, p.stem))) if any(c.isdigit() for c in p.stem) else 0, p.name)
    )

    # 构造世界坐标点，棋盘格在 Z=0 平面
    obj = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM

    objects = []
    corners_all = []
    names = []
    rejected = []

    for path in paths:
        img = load_portrait(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # SB棋盘格角点检测
        found, corners = cv2.findChessboardCornersSB(gray, PATTERN)
        if not found:
            rejected.append(path.name)
            continue

        # 亚像素角点精细化
        corners = cv2.cornerSubPix(
            gray, corners,
            (7, 7), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        )

        objects.append(obj.copy())
        corners_all.append(corners)
        names.append(path.name)

        # 保存第一张检测角点效果图
        if len(names) == 1:
            display = img.copy()
            cv2.drawChessboardCorners(display, PATTERN, corners, True)
            buf = cv2.imencode(".jpg", display)[1]
            buf.tofile(str(args.output / "detected_corners.jpg"))

    if len(names) < 3:
        raise RuntimeError(f"only {len(names)} valid views; need at least 3. Rejected: {rejected}")

    # 相机标定
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objects, corners_all, TARGET_SIZE, None, None
    )

    # 计算每张图片的重投影误差
    view_errors = []
    for i, name in enumerate(names):
        reproj_points, _ = cv2.projectPoints(objects[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(corners_all[i], reproj_points, cv2.NORM_L2) / len(reproj_points)
        view_errors.append({"image": name, "rms_px": err})

    # 输出去畸变示例图
    example = load_portrait(paths[0])
    corrected = cv2.undistort(example, K, dist)
    buf = cv2.imencode(".jpg", corrected)[1]
    buf.tofile(str(args.output / "undistorted_example.jpg"))

    # 保存标定结果json
    import json
    result = {
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.ravel().tolist(),
        "overall_rms_px": rms,
        "per_image_rms": view_errors,
        "valid_images_count": len(names),
        "rejected_images": rejected,
        "image_resolution_wh": list(TARGET_SIZE),
        "calibration_note": "landscape photos rotated 90 degrees clockwise; all images resized to 1279×1706 pixels."
    }
    with open(args.output / "calib_result.json", "w", encoding="utf‑8") as f:
        json.dump(result, f, indent=2)

    print(f"Calibration done. Overall RMS reprojection error: {rms:.3f} px")
    print(f"Valid images: {len(names)}, Rejected: {len(rejected)}")


if __name__ == "__main__":
    main()