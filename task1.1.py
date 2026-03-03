import os
import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET

# TASK 1

def ask_paths_task1():
    data_dir = input('Enter path to your data directory (contains cam1..cam4): ').strip().strip('"').strip("'")
    if not data_dir:
        data_dir = "./data"
    checkerboard_xml = input('Enter path to checkerboard.xml (press Enter if in data_dir): ').strip().strip('"').strip("'")
    if not checkerboard_xml:
        checkerboard_xml = os.path.join(data_dir, "checkerboard.xml")
    return data_dir, checkerboard_xml


def load_checkerboard_xml(xml_path: str):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"checkerboard.xml not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    def get_text(tag_names):
        for t in tag_names:
            node = root.find(f".//{t}")
            if node is not None and node.text is not None:
                s = node.text.strip()
                if s:
                    return s
        return None

    w = get_text(["CheckerBoardWidth", "CheckerboardWidth", "board_width", "width", "cols", "Columns", "col"])
    h = get_text(["CheckerBoardHeight", "CheckerboardHeight", "board_height", "height", "rows", "Rows", "row"])
    sq = get_text(["CheckerBoardSquareSize", "CheckerboardSquareSize", "square_size", "tile_size", "SquareSize"])

    if w is None:
        w = root.attrib.get("cols") or root.attrib.get("width")
    if h is None:
        h = root.attrib.get("rows") or root.attrib.get("height")
    if sq is None:
        sq = root.attrib.get("tile_size") or root.attrib.get("square_size")

    if w is None or h is None or sq is None:
        raise ValueError(
            "checkerboard.xml missing required fields.\n"
            f"Need width/height/square size (inner corners + square size). File: {xml_path}"
        )

    cols = int(float(w))
    rows = int(float(h))
    square_size = float(sq)

    # if xml uses mm, convert to meters
    if square_size > 1.0:
        square_size = square_size / 1000.0

    return cols, rows, square_size


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_intrinsics_txt(out_path: str, image_size_hw, rms, K, dist):
    h, w = image_size_hw
    dist = dist.reshape(-1)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Intrinsics\n")
        f.write(f"image_width {w}\n")
        f.write(f"image_height {h}\n")
        f.write(f"rms {float(rms)}\n")
        f.write("K\n")
        for r in range(3):
            f.write(" ".join(f"{K[r, c]:.10f}" for c in range(3)) + "\n")
        f.write("dist\n")
        f.write(" ".join(f"{d:.10f}" for d in dist.tolist()) + "\n")


def save_extrinsics_txt(out_path: str, rvec, tvec, R):
    rvec = rvec.reshape(3)
    tvec = tvec.reshape(3)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Extrinsics\n")
        f.write("rvec\n")
        f.write(" ".join(f"{x:.10f}" for x in rvec.tolist()) + "\n")
        f.write("tvec\n")
        f.write(" ".join(f"{x:.10f}" for x in tvec.tolist()) + "\n")
        f.write("R\n")
        for r in range(3):
            f.write(" ".join(f"{R[r, c]:.10f}" for c in range(3)) + "\n")


def save_corners4_txt(out_path: str, corners4_xy):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Choice 1: 4 outer corners (x y) in image pixels\n")
        for (x, y) in corners4_xy:
            f.write(f"{float(x):.3f} {float(y):.3f}\n")


def build_objp(cols, rows, square_size_m):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_m)
    return objp


def find_auto(gray, pattern_size):
    flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv.findChessboardCorners(gray, pattern_size, flags)
    if not ret:
        return None
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def calibrate_intrinsics_from_video(cam_dir: str, cols: int, rows: int, square_size_m: float,
                                    max_frames_used: int = 30, frame_stride: int = 10):
    video_path = os.path.join(cam_dir, "intrinsics.avi")
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    pattern_size = (cols, rows)
    objp = build_objp(cols, rows, square_size_m)

    objpoints, imgpoints = [], []
    image_size_hw = None

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        idx += 1
        if idx % int(frame_stride) != 0:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        image_size_hw = gray.shape  # (h, w)
        corners = find_auto(gray, pattern_size)

        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)

        if len(objpoints) >= int(max_frames_used):
            break

    cap.release()

    if len(objpoints) < 8:
        raise ValueError(
            f"Not enough valid chessboard detections for intrinsics in {video_path}. "
            f"Got {len(objpoints)} frames; need ~8+."
        )

    rms, K, dist, _, _ = cv.calibrateCamera(
        objpoints, imgpoints, (image_size_hw[1], image_size_hw[0]), None, None
    )
    return (image_size_hw, rms, K, dist)


# CHOICE 1: outer corners automatic + extrinsics

def extract_background_median(cam_dir: str, num_frames: int = 120):
    path = os.path.join(cam_dir, "background.avi")
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {path}")

    frames = []
    count = 0
    while count < int(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(gray)
        count += 1

    cap.release()

    if len(frames) < 10:
        raise ValueError(f"Too few frames to build background from {path}")

    bg = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return bg


def best_checkerboard_frame(cam_dir: str, bg_gray: np.ndarray, num_frames: int = 240, stride: int = 2):
    """
    Picks ONE frame from checkerboard.avi that maximizes difference vs background.
    This avoids blur/ghosting from averaging, which often breaks cam4 quad detection.
    """
    path = os.path.join(cam_dir, "checkerboard.avi")
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {path}")

    best_frame = None
    best_score = -1.0

    idx = 0
    grabbed = 0
    while grabbed < int(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % int(stride) != 0:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        diff = cv.absdiff(bg_gray, gray)
        score = float(np.mean(cv.GaussianBlur(diff, (5, 5), 0)))

        if score > best_score:
            best_score = score
            best_frame = frame.copy()

        grabbed += 1

    cap.release()

    if best_frame is None:
        raise ValueError(f"Could not pick a checkerboard frame from {path}")

    return best_frame


def order_quad_points(pts4):
    """
    Order as: TL, TR, BL, BR (stable)
    """
    pts = np.array(pts4, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, bl, br], dtype=np.float32)


def _quad_score(quad_pts: np.ndarray, contour_area: float) -> float:
    """
    Higher is better. Prefer large + rectangular-ish + not too skinny.
    """
    pts = quad_pts.reshape(4, 2).astype(np.float32)

    hull = cv.convexHull(pts.astype(np.float32))
    hull_area = float(cv.contourArea(hull))
    if hull_area <= 1e-6:
        return -1e18

    rectangularity = float(contour_area) / hull_area  # <= 1
    dists = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        dists.append(float(np.linalg.norm(a - b)))
    min_side = min(dists)

    return (contour_area * 0.001) + (rectangularity * 10.0) + (min_side * 0.01)


def detect_outer_corners_choice1(bg_gray, chess_bgr):
    """
    Robust CHOICE 1:
      absdiff(bg, frame) -> preprocess -> contours -> approxPolyDP(quad)
    Returns ordered 4 corners in pixels: TL, TR, BL, BR
    """
    gray = cv.cvtColor(chess_bgr, cv.COLOR_BGR2GRAY)
    diff0 = cv.absdiff(bg_gray, gray)

    configs = [
        # (blur_ksize, use_threshold, canny1, canny2, morph_ksize, morph_iters, approx_eps_frac)
        ((5, 5), False, 40, 120, (5, 5), 2, 0.02),
        ((5, 5), False, 60, 180, (7, 7), 2, 0.02),
        ((7, 7), False, 30, 100, (7, 7), 3, 0.025),
        ((5, 5), True,  0,  0,   (7, 7), 2, 0.02),
        ((7, 7), True,  0,  0,   (9, 9), 3, 0.025),
    ]

    best_quad = None
    best_score = -1e18

    H, W = diff0.shape[:2]
    min_area = 0.01 * (H * W)

    for (blur_k, use_thresh, c1, c2, mk, iters, eps_frac) in configs:
        diff = cv.GaussianBlur(diff0, blur_k, 0)

        if use_thresh:
            _, edges = cv.threshold(diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        else:
            edges = cv.Canny(diff, c1, c2)

        kernel = np.ones(mk, np.uint8)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=int(iters))
        edges = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = float(cv.contourArea(c))
            if area < min_area:
                continue

            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, float(eps_frac) * peri, True)

            if len(approx) != 4:
                continue
            if not cv.isContourConvex(approx):
                continue

            pts4 = approx.reshape(4, 2).astype(np.float32)

            margin = 3
            if (np.any(pts4[:, 0] < margin) or np.any(pts4[:, 0] > (W - 1 - margin)) or
                np.any(pts4[:, 1] < margin) or np.any(pts4[:, 1] > (H - 1 - margin))):
                continue

            sc = _quad_score(pts4, area)
            if sc > best_score:
                best_score = sc
                best_quad = pts4.copy()

    if best_quad is None:
        raise ValueError("Choice 1 failed: could not find a quadrilateral contour for the chessboard.")

    return order_quad_points(best_quad)


def interpolate_inner_corners_from_outer(outer_TL_TR_BL_BR, cols, rows):
    """
    Produces a full (rows*cols) grid from the 4 outer corners.
    Output shape: (rows*cols, 1, 2) float32, consistent with OpenCV.
    """
    tl, tr, bl, br = outer_TL_TR_BL_BR.astype(np.float32)
    first_col = np.linspace(tl, bl, rows)
    last_col = np.linspace(tr, br, rows)
    all_points = np.vstack([np.linspace(first_col[i], last_col[i], cols) for i in range(rows)])

    corners = all_points.reshape(-1, 1, 2).astype(np.float32)
    return corners


def draw_axes(img, rvec, tvec, K, dist, square_size_m):
    axis_len = 3.0 * float(square_size_m)
    axis = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, -axis_len],
    ])

    imgpts, _ = cv.projectPoints(axis, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2)

    o = tuple(map(int, imgpts[0]))
    xpt = tuple(map(int, imgpts[1]))
    ypt = tuple(map(int, imgpts[2]))
    zpt = tuple(map(int, imgpts[3]))

    cv.circle(img, o, 5, (0, 255, 255), -1)
    cv.arrowedLine(img, o, xpt, (0, 0, 255), 2, tipLength=0.2)
    cv.arrowedLine(img, o, ypt, (0, 255, 0), 2, tipLength=0.2)
    cv.arrowedLine(img, o, zpt, (255, 0, 0), 2, tipLength=0.2)
    cv.putText(img, "X", xpt, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, "Y", ypt, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, "Z", zpt, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
    return img


def reprojection_error(objp: np.ndarray, imgp: np.ndarray, K: np.ndarray, dist: np.ndarray,
                       rvec: np.ndarray, tvec: np.ndarray) -> float:
    proj, _ = cv.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    imgp2 = imgp.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - imgp2, axis=1)))


def rotate_corners4_180(corners4_TL_TR_BL_BR: np.ndarray) -> np.ndarray:
    """
    corners4 is TL,TR,BL,BR. 180° rotation swaps TL<->BR and TR<->BL.
    New TL = old BR, New TR = old BL, New BL = old TR, New BR = old TL
    """
    tl, tr, bl, br = corners4_TL_TR_BL_BR.reshape(4, 2)
    return np.array([br, bl, tr, tl], dtype=np.float32)


def rotate_inner_corners_180(corners_inner: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """
    corners_inner shape: (rows*cols, 1, 2).
    180° rotation corresponds to reversing both row and col indices.
    """
    grid = corners_inner.reshape(rows, cols, 1, 2)
    grid_rot = grid[::-1, ::-1, :, :]
    return grid_rot.reshape(rows * cols, 1, 2).astype(np.float32)


def axis_direction_penalty(rvec: np.ndarray, tvec: np.ndarray,
                           K: np.ndarray, dist: np.ndarray,
                           square_size_m: float) -> float:
    """
    Prefer X axis to go right in the image and Y axis to go down.
    Image coordinates: +x right, +y down.
    """
    axis_len = 3.0 * float(square_size_m)
    axis = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],   # X
        [0, axis_len, 0],   # Y
    ])
    imgpts, _ = cv.projectPoints(axis, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2)
    o = imgpts[0]
    xpt = imgpts[1]
    ypt = imgpts[2]

    pen = 0.0
    if (xpt[0] - o[0]) < 0:  # X should go right
        pen += 0.5
    if (ypt[1] - o[1]) < 0:  # Y should go down
        pen += 0.5
    return pen


def solve_extrinsics_choice1(cam_dir: str, cols: int, rows: int, square_size_m: float, K, dist,
                            show_debug: bool = True):
    bg = extract_background_median(cam_dir)

    chess = best_checkerboard_frame(cam_dir, bg_gray=bg, num_frames=240, stride=2)

    corners4_a = detect_outer_corners_choice1(bg, chess)  # TL, TR, BL, BR
    corners_inner_a = interpolate_inner_corners_from_outer(corners4_a, cols, rows)

    objp = build_objp(cols, rows, square_size_m)

    # Candidate B: 180° rotated orientation (fixes upside-down cases)
    corners4_b = rotate_corners4_180(corners4_a)
    corners_inner_b = rotate_inner_corners_180(corners_inner_a, cols, rows)

    ok_a, rvec_a, tvec_a = cv.solvePnP(objp, corners_inner_a, K, dist, flags=cv.SOLVEPNP_ITERATIVE)
    ok_b, rvec_b, tvec_b = cv.solvePnP(objp, corners_inner_b, K, dist, flags=cv.SOLVEPNP_ITERATIVE)

    if not ok_a and not ok_b:
        raise ValueError("solvePnP failed for both orientations.")

    best = None
    best_score = 1e18

    def score_pose(rvec, tvec, corners_inner):
        err = reprojection_error(objp, corners_inner, K, dist, rvec, tvec)
        pen_axes = axis_direction_penalty(rvec, tvec, K, dist, square_size_m)
        pen_z = 1000.0 if float(tvec.reshape(3)[2]) <= 0 else 0.0
        return err + pen_axes + pen_z, err

    if ok_a:
        s_a, err_a = score_pose(rvec_a, tvec_a, corners_inner_a)
        if s_a < best_score:
            best_score = s_a
            best = ("A", corners4_a, corners_inner_a, rvec_a, tvec_a, err_a)

    if ok_b:
        s_b, err_b = score_pose(rvec_b, tvec_b, corners_inner_b)
        if s_b < best_score:
            best_score = s_b
            best = ("B", corners4_b, corners_inner_b, rvec_b, tvec_b, err_b)

    if best is None:
        raise ValueError("Could not choose a valid pose.")

    which, corners4, corners_inner, rvec, tvec, err = best
    R, _ = cv.Rodrigues(rvec)

    if show_debug:
        vis = chess.copy()
        for i, (x, y) in enumerate(corners4):
            cv.circle(vis, (int(x), int(y)), 6, (0, 255, 255), -1)
            cv.putText(vis, f"{i}", (int(x) + 6, int(y) - 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv.drawChessboardCorners(vis, (cols, rows), corners_inner, True)
        vis = draw_axes(vis, rvec, tvec, K, dist, square_size_m)

        cv.putText(vis, f"picked={which} reproj={err:.3f}px",
                   (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv.imshow("Choice1 corners + axes", vis)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return corners4, corners_inner, rvec, tvec, R


def run_task1_choice1(data_dir: str, checkerboard_xml: str, cameras=(1, 2, 3, 4),
                      max_frames_used: int = 30, frame_stride: int = 10, show_debug: bool = True):
    cols, rows, square_size_m = load_checkerboard_xml(checkerboard_xml)
    print(f"[info] Checkerboard: cols={cols}, rows={rows}, square={square_size_m} m")

    for cam_id in cameras:
        cam_dir = os.path.join(data_dir, f"cam{cam_id}")
        ensure_dir(cam_dir)

        print(f"\n=== Camera {cam_id} ===")

        image_size_hw, rms, K, dist = calibrate_intrinsics_from_video(
            cam_dir, cols, rows, square_size_m, max_frames_used=max_frames_used, frame_stride=frame_stride
        )
        intr_txt = os.path.join(cam_dir, "intrinsics.txt")
        save_intrinsics_txt(intr_txt, image_size_hw, rms, K, dist)
        print(f"[ok] Saved intrinsics -> {intr_txt}")

        corners4, _, rvec, tvec, R = solve_extrinsics_choice1(
            cam_dir, cols, rows, square_size_m, K, dist, show_debug=show_debug
        )

        extr_txt = os.path.join(cam_dir, "extrinsics.txt")
        save_extrinsics_txt(extr_txt, rvec, tvec, R)
        print(f"[ok] Saved extrinsics -> {extr_txt}")

        c4_txt = os.path.join(cam_dir, "corners4.txt")
        save_corners4_txt(c4_txt, corners4)
        print(f"[ok] Saved Choice1 corners -> {c4_txt}")


def Background_subtraction(video_path, reference_path, output_path, k_h=1, k_s=1, k_v=3):
    """
    Split video into frames
    Split every image into foreground and background
    """
    ref_cap = cv.VideoCapture(reference_path)
    ref_frames = []
    hsv_frames = []
    while True:
        ret, frame = ref_cap.read()
        if not ret:
            break
        ref_frames.append(frame)
        hsv_frames.append(cv.cvtColor(frame, cv.COLOR_BGR2HSV))
    ref_cap.release()

    ref_hsv = np.mean(hsv_frames, axis=0).astype(np.uint8)
    ref_var = np.var(hsv_frames, axis=0).astype(np.float32)

    cap = cv.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.GaussianBlur(frame, (5, 5), 0)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        fgmask = np.zeros(frame.shape[:2], dtype=np.uint8)
        std = np.sqrt(ref_var + 1e-6)

        dh = np.abs(frame[..., 0].astype(np.float32) - ref_hsv[..., 0].astype(np.float32))
        dh = np.minimum(dh, 180 - dh)
        ds = np.abs(frame[..., 1].astype(np.float32) - ref_hsv[..., 1].astype(np.float32))
        dv = np.abs(frame[..., 2].astype(np.float32) - ref_hsv[..., 2].astype(np.float32))

        _, fg_h = cv.threshold(dh.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        fg_h = fg_h.astype(bool) & (std[..., 0] < k_h)

        _, fg_s = cv.threshold(ds.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        fg_s = fg_s.astype(bool) & (std[..., 1] < k_s)

        _, fg_v = cv.threshold(dv.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        fg_v = (fg_v > 0) & (std[..., 2] < k_v)

        fg = fg_h | fg_s | fg_v
        fgmask = (fg.astype(np.uint8) * 255)

        contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            mask = np.zeros_like(fgmask)
            cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)
            fgmask = cv.bitwise_and(fgmask, mask)

        Hc, Sc, Vc = frame[..., 0], frame[..., 1], frame[..., 2]
        Hr, Sr, Vr = ref_hsv[..., 0], ref_hsv[..., 1], ref_hsv[..., 2]
        v_ratio = Vc / (Vr + 1e-6)
        th_s_shadow = 45.0
        v_ratio_low = 0.50
        v_ratio_high = 0.95
        shadow = (np.abs(Sc - Sr) < th_s_shadow) & (v_ratio >= v_ratio_low) & (v_ratio <= v_ratio_high)
        fgmask[shadow] = 0

        fgmask = cv.morphologyEx(
            fgmask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=1
        )
        fgmask = cv.morphologyEx(
            fgmask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=1
        )

        out_name = f'frame_{int(cap.get(cv.CAP_PROP_POS_FRAMES))}.jpg'
        cv.imwrite(os.path.join(output_path, out_name), fgmask)

    cap.release()
    print("Background subtraction completed. Foreground masks saved in:", output_path)


def _read_intrinsics_txt(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]

    try:
        k_idx = lines.index("K")
        d_idx = lines.index("dist")
    except ValueError:
        raise ValueError(f"Invalid intrinsics.txt format (missing K/dist markers): {path}")

    K_lines = lines[k_idx + 1:k_idx + 4]
    if len(K_lines) != 3:
        raise ValueError(f"Invalid K block in {path}")
    K = np.array([[float(x) for x in row.split()] for row in K_lines], dtype=np.float64)

    dist_vals = [float(x) for x in lines[d_idx + 1].split()]
    dist = np.array(dist_vals, dtype=np.float64).reshape(-1, 1)
    return K, dist


def _read_extrinsics_txt(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]

    try:
        r_idx = lines.index("rvec")
        t_idx = lines.index("tvec")
        R_idx = lines.index("R")
    except ValueError:
        raise ValueError(f"Invalid extrinsics.txt format (missing rvec/tvec/R markers): {path}")

    rvec = np.array([float(x) for x in lines[r_idx + 1].split()], dtype=np.float64).reshape(3, 1)
    tvec = np.array([float(x) for x in lines[t_idx + 1].split()], dtype=np.float64).reshape(3, 1)

    R_lines = lines[R_idx + 1:R_idx + 4]
    if len(R_lines) != 3:
        raise ValueError(f"Invalid R block in {path}")
    R = np.array([[float(x) for x in row.split()] for row in R_lines], dtype=np.float64)

    return rvec, tvec, R


def build_lookuptable(data_dir: str, cameras=(1, 2, 3, 4),
                      grid_min=-10.0, grid_max=10.0, grid_step=0.5,
                      z_min=0.0, z_max=1.0, z_step=0.1):
    data_dir = str(data_dir)

    for cam_id in cameras:
        cam_dir = os.path.join(data_dir, f"cam{cam_id}")
        print(f"Building lookup table for camera {cam_id}")

        intr_path = os.path.join(cam_dir, "intrinsics.txt")
        extr_path = os.path.join(cam_dir, "extrinsics.txt")

        if not os.path.exists(intr_path) or not os.path.exists(extr_path):
            print(f"  Missing {intr_path} or {extr_path}, skipping.")
            continue

        K, dist = _read_intrinsics_txt(intr_path)
        rvec, tvec, _ = _read_extrinsics_txt(extr_path)

        xs = np.arange(grid_min, grid_max + 1e-9, grid_step, dtype=np.float32)
        ys = np.arange(grid_min, grid_max + 1e-9, grid_step, dtype=np.float32)
        zs = np.arange(z_min, z_max + 1e-9, z_step, dtype=np.float32)

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
        grid = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1).astype(np.float32)

        proj_points = cv.projectPoints(grid, rvec, tvec, K, dist)[0].reshape(-1, 2).astype(np.float32)

        lookup_table_path = os.path.join(cam_dir, "lookup_table.npz")
        np.savez(lookup_table_path, grid=grid, proj_points=proj_points)
        print(f"  Lookup table saved to {lookup_table_path}")


def task3_stub(data_dir: str, cameras=(1, 2, 3, 4)):
    data_dir = str(data_dir)

    for cam_id in cameras:
        cam_dir = os.path.join(data_dir, f"cam{cam_id}")
        print(f"Processing camera {cam_id} for 3D reconstruction")

        lookup_table_path = os.path.join(cam_dir, "lookup_table.npz")
        masks_path = os.path.join(cam_dir, "masks")

        if not os.path.exists(lookup_table_path):
            print(f"  Missing lookup table {lookup_table_path}, skipping.")
            continue
        if not os.path.isdir(masks_path):
            print(f"  Missing masks directory {masks_path}, skipping.")
            continue

        data = np.load(lookup_table_path)
        grid = data["grid"]
        proj_points = data["proj_points"]

        mask_files = sorted([fn for fn in os.listdir(masks_path)
                             if fn.lower().endswith(".jpg") and fn.startswith("frame_")])
        print(f"  Found {len(mask_files)} masks")

        for mf in mask_files[:1]:
            mask = cv.imread(os.path.join(masks_path, mf), cv.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            fg_points = np.column_stack(np.where(mask > 0))
            if fg_points.shape[0] == 0:
                print(f"  {mf}: no foreground")
                continue

            fg_xy = fg_points[:, ::-1].astype(np.float32)

            sample = fg_xy[:: max(1, fg_xy.shape[0] // 5000)]
            pts3d = []
            for p in sample:
                d = np.linalg.norm(proj_points - p[None, :], axis=1)
                j = int(np.argmin(d))
                if float(d[j]) < 5.0:
                    pts3d.append(grid[j])
            print(f"  {mf}: mapped ~{len(pts3d)} points (demo)")


# Main

if __name__ == "__main__":
    print("Select mode:")
    print("  1) TASK 1 (Intrinsics + CHOICE 1 + Extrinsics)  [writes intrinsics.txt/extrinsics.txt]")
    print("  2) Background subtraction (writes masks)")
    print("  3) Build lookup tables (reads intrinsics.txt/extrinsics.txt)")
    print("  4) Task3 stub (reads lookup_table.npz + masks)")
    mode = input("Enter 1/2/3/4: ").strip()

    if mode == "1":
        data_dir, checkerboard_xml = ask_paths_task1()
        cams = (1, 2, 3, 4)
        run_task1_choice1(data_dir, checkerboard_xml, cameras=cams, max_frames_used=30, frame_stride=10, show_debug=True)

    elif mode == "2":
        folder_path = input("Enter the path to the folder containing videos: ").strip().strip('"').strip("'")
        video_path = os.path.join(folder_path, "video.avi")
        reference_path = os.path.join(folder_path, "background.avi")
        output_path = os.path.join(folder_path, "masks")
        os.makedirs(output_path, exist_ok=True)
        Background_subtraction(video_path, reference_path, output_path)

    elif mode == "3":
        data_dir = input('Enter path to your data directory (contains cam1..cam4): ').strip().strip('"').strip("'")
        if not data_dir:
            data_dir = "./data"
        cams = (1, 2, 3, 4)
        build_lookuptable(data_dir, cameras=cams)

    elif mode == "4":
        data_dir = input('Enter path to your data directory (contains cam1..cam4): ').strip().strip('"').strip("'")
        if not data_dir:
            data_dir = "./data"
        cams = (1, 2, 3, 4)
        task3_stub(data_dir, cameras=cams)

    else:
        print("Unknown mode. Exiting.")