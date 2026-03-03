import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
import importlib.util
from pathlib import Path
from typing import Optional, List

# Import run_task1_choice1 from task1.1_v2.py using importlib
spec = importlib.util.spec_from_file_location("task1.1", "task1.1.py")
task1_1_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task1_1_v2)
run_task1_choice1 = task1_1_v2.run_task1_choice1

def _termcrit():
    return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def reprojection_error(
    objp: np.ndarray,
    imgp: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray
) -> float:
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    imgp2 = imgp.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - imgp2, axis=1)))


def _parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.strip().split() if x.strip()]


def order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """
    Robustly orders 4 image points into TL, TR, BR, BL using sum/diff heuristics.
    pts: (4,2)
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    s = pts.sum(axis=1)          # x+y
    d = pts[:, 0] - pts[:, 1]    # x-y

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]

    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

def build_world_corner_mapping(
    four_img_pts: np.ndarray,
    origin_click: np.ndarray,
    x_click: np.ndarray
) -> np.ndarray:
    """
    Build an ordered 4-corner list that matches a FIXED world corner layout:
    Returns image corners ordered as:
        [origin, +X corner, opposite, +Y corner]

    This removes dependence on "TL in the image" and makes world consistent across cameras.
    """
    pts = np.asarray(four_img_pts, dtype=np.float32).reshape(4, 2)
    origin_click = np.asarray(origin_click, dtype=np.float32).reshape(1, 2)
    x_click = np.asarray(x_click, dtype=np.float32).reshape(1, 2)

    # Find which corner is origin
    d0 = np.linalg.norm(pts - origin_click, axis=1)
    i0 = int(np.argmin(d0))

    # Find which corner is +X
    dX = np.linalg.norm(pts - x_click, axis=1)
    iX = int(np.argmin(dX))
    if iX == i0:
        raise ValueError("Origin and +X corner are the same. Click a different corner for +X.")

    # Remaining two corners
    remaining = [i for i in range(4) if i not in (i0, iX)]

    # The remaining corners are the +Y corner and the opposite corner
    # The +Y corner should be adjacent to origin (forms a right angle with origin and +X)
    # The opposite corner should be diagonal from origin
    
    # Calculate vectors from origin to the two remaining corners
    vec_to_rem = [pts[i] - pts[i0] for i in remaining]
    vec_to_x = pts[iX] - pts[i0]
    
    # The +Y corner should be roughly perpendicular to the +X direction
    # Calculate dot product (measures alignment): lower value means more perpendicular
    dots = [np.dot(vec_to_rem[j], vec_to_x) / (np.linalg.norm(vec_to_rem[j]) * np.linalg.norm(vec_to_x) + 1e-8) 
            for j in range(2)]
    
    # The corner with smaller dot product is more perpendicular to +X, so it's +Y
    iY = remaining[int(np.argmin(np.abs(dots)))]
    iOpp = remaining[int(np.argmax(np.abs(dots)))]

    return np.stack([pts[i0], pts[iX], pts[iOpp], pts[iY]], axis=0).astype(np.float32)


def auto_select_closest_corner(four_pts: np.ndarray, reference_corner_pos: np.ndarray) -> int:
    """
    Automatically select the corner closest to a reference position.
    Returns the index (0=TL, 1=TR, 2=BR, 3=BL) of the closest corner.
    """
    ordered_pts = order_quad_tl_tr_br_bl(four_pts)
    distances = [np.linalg.norm(ordered_pts[i] - reference_corner_pos) for i in range(4)]
    return int(np.argmin(distances))

def board_normal_camera_z(rvec: np.ndarray) -> float:
    """
    Board normal in object frame is +Z = (0,0,1).
    Transform to camera: n_c = R * n_o.
    If board faces the camera, typically n_c.z should be NEGATIVE (since camera looks along +Z).
    """
    R, _ = cv2.Rodrigues(rvec)
    n_c = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return float(n_c[2])


def pick_best_pose_from_two_orientations(
    gray: np.ndarray,
    four_pts_any_order: np.ndarray,
    origin_click: np.ndarray,
    x_click: np.ndarray,
    spec: "CheckerboardSpec",
    K: np.ndarray,
    dist: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Uses 4 selected corners + (origin click) + (+X click) to enforce the SAME world frame across cameras.
    Tries normal + mirrored corner mapping and picks best.
    """
    # Build world-consistent ordered corners: [origin, +X, opposite, +Y]
    four_world = build_world_corner_mapping(four_pts_any_order, origin_click, x_click)

    # Fixed object points
    objp = spec.object_points()

    candidates = [
        four_world,
        # mirrored candidate: swap +X and +Y (handles user accidentally clicking the other adjacent corner)
        np.array([four_world[0], four_world[3], four_world[2], four_world[1]], dtype=np.float32),
    ]

    best = None
    for quad in candidates:
        corners0 = generate_grid_from_4_corners(quad, spec)
        corners = refine_corners_locally(gray, corners0)

        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        if tvec[2, 0] <= 0:
            continue

        err = reprojection_error(objp, corners, K, dist, rvec, tvec)
        ncz = board_normal_camera_z(rvec)

        record = (corners, rvec, tvec, err, ncz)

        if best is None:
            best = record
            continue

        _, _, _, best_err, best_ncz = best
        cur_is_good = (ncz < 0)
        best_is_good = (best_ncz < 0)

        if (cur_is_good and not best_is_good) or (cur_is_good == best_is_good and err < best_err):
            best = record

    if best is None:
        raise ValueError("No valid pose from the selected corners (PnP failed / behind camera).")

    return best


@dataclass(frozen=True)
class CheckerboardSpec:
    rows: int           # inner-corner rows
    cols: int           # inner-corner cols
    tile_size: float    # meters (or your unit)

    @property
    def pattern_size(self) -> Tuple[int, int]:
        # OpenCV wants (cols, rows)
        return (self.cols, self.rows)

    def object_points(self) -> np.ndarray:
        """
        Standard checkerboard object points in a fixed world frame:
        origin at (0,0,0), x along columns, y along rows.
        """
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        grid = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2).astype(np.float32)
        objp[:, :2] = grid * float(self.tile_size)
        return objp


def load_checkerboard_xml(xml_path: Path, assume_tiles_not_corners: bool = False) -> CheckerboardSpec:

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    def strip_ns(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    def find_any(tag_name: str) -> Optional[ET.Element]:
        for e in root.iter():
            if strip_ns(e.tag) == tag_name:
                return e
        return None

    def read_value(elem: Optional[ET.Element]) -> Optional[str]:
        if elem is None:
            return None
        if "value" in elem.attrib and str(elem.attrib["value"]).strip() != "":
            return str(elem.attrib["value"]).strip()
        if elem.text is not None and elem.text.strip() != "":
            return elem.text.strip()
        return None

    rows_txt = read_value(find_any("rows"))
    cols_txt = read_value(find_any("cols"))
    tile_txt = read_value(find_any("tile_size"))

    if rows_txt is None and cols_txt is None and tile_txt is None:
        w_txt = read_value(find_any("CheckerBoardWidth"))
        h_txt = read_value(find_any("CheckerBoardHeight"))
        s_txt = read_value(find_any("CheckerBoardSquareSize"))

        cols_txt = w_txt
        rows_txt = h_txt
        tile_txt = s_txt

    if rows_txt is None or cols_txt is None or tile_txt is None:
        child_tags = [strip_ns(c.tag) for c in list(root)]
        all_tags = sorted({strip_ns(e.tag) for e in root.iter()})
        raise ValueError(
            f"checkerboard.xml missing required fields.\n"
            f"Expected rows/cols/tile_size OR CheckerBoardWidth/CheckerBoardHeight/CheckerBoardSquareSize.\n"
            f"File: {xml_path}\n"
            f"Root tag: {strip_ns(root.tag)}\n"
            f"Root attributes: {root.attrib}\n"
            f"Direct child tags: {child_tags}\n"
            f"All tags found: {all_tags}\n"
            f"Parsed: rows={rows_txt}, cols={cols_txt}, tile_size={tile_txt}\n"
        )

    rows = int(float(rows_txt))
    cols = int(float(cols_txt))
    tile_size = float(tile_txt)

    if assume_tiles_not_corners:
        rows -= 1
        cols -= 1
        if rows <= 1 or cols <= 1:
            raise ValueError("After tiles->corners conversion, rows/cols became invalid. Check checkerboard.xml.")

    return CheckerboardSpec(rows=rows, cols=cols, tile_size=tile_size)


class FourCornerGridSelector:
    """
    Click 4 extreme corners in any order.
    If need_origin_and_xdir=True:
        - After 4 corners: click ORIGIN corner (world 0,0)
        - Then click +X direction corner (adjacent to origin)
    """

    def __init__(self, window_name: str, need_origin_and_xdir: bool = False):
        self.window_name = window_name
        self.need_origin_and_xdir = need_origin_and_xdir

        self.points = []
        self.origin_click = None
        self.x_click = None

        self.stage = 0  # 0=select 4 corners, 1=select origin, 2=select +X
        self.done = False
        self.cancelled = False
        self._img0 = None
        self._disp = None

        if need_origin_and_xdir:
            self._help = [
                "STEP 1: CLICK 4 extreme inner corners (any order)",
                "STEP 2: CLICK the CORNER that is WORLD ORIGIN (0,0)",
                "STEP 3: CLICK the adjacent CORNER that defines +X direction",
                "Keys: [r]=reset, [q]=quit, [ENTER]=accept after Step 3"
            ]
        else:
            self._help = [
                "CLICK 4 extreme inner corners (any order)",
                "Keys: [r]=reset, [q]=quit, [ENTER]=accept after 4 points"
            ]

    def _mouse(self, event, x, y, flags, param):
        if self.done or self.cancelled:
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.stage == 0:
            if len(self.points) < 4:
                self.points.append((x, y))
                if len(self.points) == 4:
                    self.stage = 1 if self.need_origin_and_xdir else 3
                self._redraw()

        elif self.stage == 1:
            self.origin_click = np.array([x, y], dtype=np.float32)
            self.stage = 2
            self._redraw()

        elif self.stage == 2:
            self.x_click = np.array([x, y], dtype=np.float32)
            self.stage = 3
            self._redraw()

    def _redraw(self):
        self._disp = self._img0.copy()

        # draw 4 selected corners
        for i, (px, py) in enumerate(self.points):
            cv2.circle(self._disp, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(self._disp, str(i + 1), (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # draw origin click
        if self.origin_click is not None:
            ox, oy = int(self.origin_click[0]), int(self.origin_click[1])
            cv2.circle(self._disp, (ox, oy), 10, (0, 0, 255), -1)
            cv2.putText(self._disp, "ORIGIN", (ox + 10, oy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # draw +X click
        if self.x_click is not None:
            xx, xy = int(self.x_click[0]), int(self.x_click[1])
            cv2.circle(self._disp, (xx, xy), 10, (255, 0, 0), -1)
            cv2.putText(self._disp, "+X", (xx + 10, xy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # stage hint
        if self.need_origin_and_xdir:
            if self.stage == 1:
                cv2.putText(self._disp, "Now click ORIGIN corner (world 0,0)", (10, self._disp.shape[0] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif self.stage == 2:
                cv2.putText(self._disp, "Now click +X corner (adjacent to origin)", (10, self._disp.shape[0] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # help
        y0 = 24
        for line in self._help:
            cv2.putText(self._disp, line, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y0 += 20

    def select(self, bgr: np.ndarray):
        self.points = []
        self.origin_click = None
        self.x_click = None
        self.stage = 0
        self.done = False
        self.cancelled = False
        self._img0 = bgr.copy()
        self._redraw()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse)

        while True:
            cv2.imshow(self.window_name, self._disp)
            k = cv2.waitKey(20) & 0xFF

            if k == ord('r'):
                self.points = []
                self.origin_click = None
                self.x_click = None
                self.stage = 0
                self._redraw()
            elif k == ord('q') or k == 27:
                self.cancelled = True
                break
            elif k in (13, 10):  # Enter
                if not self.need_origin_and_xdir and len(self.points) == 4:
                    self.done = True
                    break
                if self.need_origin_and_xdir and len(self.points) == 4 and self.origin_click is not None and self.x_click is not None:
                    self.done = True
                    break

        cv2.destroyWindow(self.window_name)

        if self.cancelled or not self.done:
            return None

        four_pts = np.array(self.points, dtype=np.float32)  # any order
        if self.need_origin_and_xdir:
            return four_pts, self.origin_click, self.x_click
        else:
            return four_pts

def generate_grid_from_4_corners(four_img_pts_world_ordered: np.ndarray, spec: CheckerboardSpec) -> np.ndarray:
    """
    Homography-based full grid generation from 4 corners, using a FIXED world layout.

    four_img_pts_world_ordered must be ordered as:
        [origin, +X corner, opposite, +Y corner]
    corresponding to object corners:
        [(0,0), (cols-1,0), (cols-1,rows-1), (0,rows-1)]
    """
    cols, rows = spec.cols, spec.rows

    src_obj = np.array([
        [0, 0],                 # origin
        [cols - 1, 0],          # +X
        [cols - 1, rows - 1],   # opposite
        [0, rows - 1],          # +Y
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_obj, four_img_pts_world_ordered, method=0)
    if H is None:
        raise RuntimeError("Homography could not be estimated from the 4 selected corners.")

    grid_obj = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    grid_obj_h = cv2.convertPointsToHomogeneous(grid_obj).reshape(-1, 3)

    proj = (H @ grid_obj_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    
    # Check if any projected corners are outside reasonable bounds
    corners_2d = proj.reshape(-1, 2)
    if np.any(corners_2d < -100) or np.any(corners_2d > 10000):
        raise RuntimeError("Generated corners are outside reasonable image bounds. Check corner selection.")
    
    return proj.reshape(-1, 1, 2).astype(np.float32)



def refine_corners_locally(gray: np.ndarray, corners_init: np.ndarray) -> np.ndarray:
    corners = corners_init.astype(np.float32).copy()
    
    # Check if all corners are within image boundaries
    h, w = gray.shape[:2]
    corners_2d = corners.reshape(-1, 2)
    
    # Check bounds with margin for cornerSubPix window
    margin = 10
    valid_mask = ((corners_2d[:, 0] >= margin) & (corners_2d[:, 0] < w - margin) &
                  (corners_2d[:, 1] >= margin) & (corners_2d[:, 1] < h - margin))
    
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        raise RuntimeError(f"{invalid_count} corners are outside image boundaries. Check corner selection or homography accuracy.")
    
    # All corners are valid, apply sub-pixel refinement
    cv2.cornerSubPix(gray, corners, (9, 9), (-1, -1), _termcrit())
    return corners


def load_intrinsics_xml(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    tree = ET.parse(str(path))
    root = tree.getroot()

    # Handle both possible XML structures
    K_elem = root.find("K")  # Direct child
    if K_elem is None:
        K_elem = root.find("./intrinsics/K")  # Nested structure
    
    if K_elem is None or K_elem.text is None:
        raise ValueError(f"Missing K in {path}")
    
    rows = int(K_elem.get("rows", "3"))
    cols = int(K_elem.get("cols", "3"))
    K_vals = _parse_float_list(K_elem.text)
    K = np.array(K_vals, dtype=np.float64).reshape(rows, cols)

    # Handle both possible XML structures for distortion
    dist_elem = root.find("distortion")
    if dist_elem is None:
        dist_elem = root.find("./intrinsics/distortion")
    
    if dist_elem is None or dist_elem.text is None:
        raise ValueError(f"Missing distortion in {path}")
    
    dist_vals = _parse_float_list(dist_elem.text)
    dist = np.array(dist_vals, dtype=np.float64).reshape(-1, 1)

    return K, dist


def load_extrinsics_xml(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[int], Optional[float]]:
    tree = ET.parse(str(path))
    root = tree.getroot()

    # Handle both possible XML structures
    rvec_elem = root.find("rvec")
    if rvec_elem is None:
        rvec_elem = root.find("./extrinsics/rvec")
    
    tvec_elem = root.find("tvec")
    if tvec_elem is None:
        tvec_elem = root.find("./extrinsics/tvec")
    
    if rvec_elem is None or rvec_elem.text is None:
        raise ValueError(f"Missing rvec in {path}")
    if tvec_elem is None or tvec_elem.text is None:
        raise ValueError(f"Missing tvec in {path}")

    rvec = np.array(_parse_float_list(rvec_elem.text), dtype=np.float64).reshape(3, 1)
    tvec = np.array(_parse_float_list(tvec_elem.text), dtype=np.float64).reshape(3, 1)

    frame_idx = root.get("frame_index")
    frame_idx = int(frame_idx) if frame_idx is not None else None
    err = root.get("reprojection_error_px")
    err = float(err) if err is not None else None

    return rvec, tvec, frame_idx, err


class CameraCalibrationTask1:
    def __init__(self, camera_id: int, data_dir: Path):
        self.camera_id = camera_id
        self.data_dir = Path(data_dir)
        self.cam_dir = self.data_dir / f"cam{camera_id}"
        ensure_dir(self.cam_dir)

        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None

        self.best_rvec: Optional[np.ndarray] = None
        self.best_tvec: Optional[np.ndarray] = None
        self.best_frame_index: Optional[int] = None
        self.best_reproj_err: Optional[float] = None

    @property
    def name(self) -> str:
        return f"cam{self.camera_id}"

    def _read_frame(self, cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return frame if ok else None

    def try_load_existing_calibration(self) -> Tuple[bool, bool]:
        has_intr = False
        has_extr = False

        intr_path = self.cam_dir / "intrinsics.xml"
        extr_path = self.cam_dir / "extrinsics.xml"

        if intr_path.exists():
            try:
                self.K, self.dist = load_intrinsics_xml(intr_path)
                has_intr = True
                print(f"[cam{self.camera_id}] Loaded existing intrinsics.xml")
            except Exception as e:
                print(f"[cam{self.camera_id}] Failed loading intrinsics.xml: {e}")

        if extr_path.exists():
            try:
                rvec, tvec, frame_idx, err = load_extrinsics_xml(extr_path)
                self.best_rvec = rvec
                self.best_tvec = tvec
                self.best_frame_index = frame_idx
                self.best_reproj_err = err
                has_extr = True
                print(f"[cam{self.camera_id}] Loaded existing extrinsics.xml")
            except Exception as e:
                print(f"[cam{self.camera_id}] Failed loading extrinsics.xml: {e}")

        return has_intr, has_extr

    def calibrate_intrinsics_from_video(
        self,
        intrinsics_video: Path,
        spec: CheckerboardSpec,
        num_frames: int = 15,
        auto_first: bool = True,
        manual_fallback: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        cap = cv2.VideoCapture(str(intrinsics_video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {intrinsics_video}")

        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            raise RuntimeError(f"Video has 0 frames: {intrinsics_video}")

        frame_idxs = np.linspace(0, max(0, n - 1), num_frames, dtype=int)

        objp = spec.object_points(origin_corner_idx=0)  # Use TL origin for intrinsics (standard)
        objpoints = []
        imgpoints = []
        image_size = None

        selector = FourCornerGridSelector(window_name=f"cam{self.camera_id} intrinsics: select 4 corners", need_origin_selection=False)

        for idx in frame_idxs:
            frame = self._read_frame(cap, int(idx))
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]

            corners = None

            if auto_first:
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                found, c = cv2.findChessboardCorners(gray, spec.pattern_size, flags)
                if found:
                    c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), _termcrit())
                    corners = c

            if corners is None and manual_fallback:
                four_ord = selector.select(frame)
                if four_ord is None:
                    continue
                corners0 = generate_grid_from_4_corners(four_ord, spec, origin_corner_idx=0)
                corners = refine_corners_locally(gray, corners0)

            if corners is not None and corners.shape[0] == objp.shape[0]:
                objpoints.append(objp)
                imgpoints.append(corners)

                vis = frame.copy()
                cv2.drawChessboardCorners(vis, spec.pattern_size, corners, True)
                cv2.imshow(f"cam{self.camera_id} intrinsics preview", vis)
                cv2.waitKey(10)

        cap.release()
        cv2.destroyAllWindows()

        if len(imgpoints) < 5:
            raise ValueError(f"Not enough valid frames for intrinsics (got {len(imgpoints)}). Select more frames.")

        _, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

        self.K = K
        self.dist = dist
        return K, dist

    def calibrate_extrinsics_from_video(
        self,
        checkerboard_video: Path,
        spec: CheckerboardSpec,
        num_candidate_frames: int = 10,
        frame_stride: int = 15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.K is None or self.dist is None:
            raise ValueError("Intrinsics must be available before extrinsics.")

        cap = cv2.VideoCapture(str(checkerboard_video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {checkerboard_video}")

        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            raise RuntimeError(f"Video has 0 frames: {checkerboard_video}")

        candidates = list(range(0, n, max(1, int(frame_stride))))
        if len(candidates) > num_candidate_frames:
            pick = np.linspace(0, len(candidates) - 1, num_candidate_frames, dtype=int)
            candidates = [candidates[i] for i in pick]

        selector = FourCornerGridSelector(
            window_name=f"cam{self.camera_id} extrinsics: 4 corners + ORIGIN + +X",
            need_origin_and_xdir=True
        )

        # selector = FourCornerGridSelector(window_name=f"cam{self.camera_id} extrinsics: select 4 corners + ORIGIN", need_origin_selection=True)

        best = {"err": np.inf, "rvec": None, "tvec": None, "idx": None, "frame": None, "ncz": None}
        
        # Variables to remember origin selection from first frame
        reference_origin_corner = None
        reference_origin_idx = None

        for i, idx in enumerate(candidates):
            frame = self._read_frame(cap, int(idx))
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- NEW robust world-frame selection ---
            result = selector.select(frame)
            if result is None:
                continue

            four_pts, origin_click, x_click = result

            try:
                corners, rvec, tvec, err, ncz = pick_best_pose_from_two_orientations(
                    gray=gray,
                    four_pts_any_order=four_pts,
                    origin_click=origin_click,
                    x_click=x_click,
                    spec=spec,
                    K=self.K,
                    dist=self.dist
                )
            except Exception as e:
                print(f"[cam{self.camera_id}] pose reject at frame {idx}: {e}")
                continue

            vis = frame.copy()
            cv2.drawChessboardCorners(vis, spec.pattern_size, corners, True)
            
            # Show calibration status without origin corner index (not available in new API)
            status_text = f"idx={idx} reproj={err:.3f}px ncz={ncz:.3f}"
            if i > 0:
                status_text += " (auto)"
            cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(f"cam{self.camera_id} extrinsics preview", vis)
            cv2.waitKey(20)

            if err < best["err"]:
                best.update(err=err, rvec=rvec, tvec=tvec, idx=idx, frame=frame.copy(), ncz=ncz)

        cap.release()
        cv2.destroyAllWindows()

        if best["rvec"] is None:
            raise ValueError("Extrinsics failed: no valid solvePnP result. Select clearer frames / corners.")

        self.best_rvec = best["rvec"]
        self.best_tvec = best["tvec"]
        self.best_frame_index = int(best["idx"])
        self.best_reproj_err = float(best["err"])

        self.visualize_world_axes(best["frame"], spec, self.best_rvec, self.best_tvec)
        return self.best_rvec, self.best_tvec

    def visualize_world_axes(
        self,
        frame_bgr: np.ndarray,
        spec: CheckerboardSpec,
        rvec: np.ndarray,
        tvec: np.ndarray,
        axis_len_tiles: float = 3.0
    ) -> None:
        if self.K is None or self.dist is None:
            return

        L = float(axis_len_tiles) * float(spec.tile_size)
        axis3d = np.float32([
            [0, 0, 0],
            [L, 0, 0],
            [0, L, 0],
            [0, 0, -L],
        ])

        imgpts, _ = cv2.projectPoints(axis3d, rvec, tvec, self.K, self.dist)
        imgpts = imgpts.reshape(-1, 2).astype(int)

        o = tuple(imgpts[0]); x = tuple(imgpts[1]); y = tuple(imgpts[2]); z = tuple(imgpts[3])

        vis = frame_bgr.copy()
        cv2.line(vis, o, x, (0, 0, 255), 3)
        cv2.line(vis, o, y, (0, 255, 0), 3)
        cv2.line(vis, o, z, (255, 0, 0), 3)

        cv2.circle(vis, o, 6, (255, 255, 0), -1)
        cv2.putText(vis, "Origin (0,0,0)", (o[0] + 10, o[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        out_png = self.cam_dir / "extrinsics_axes.png"
        cv2.imwrite(str(out_png), vis)

        cv2.imshow(f"cam{self.camera_id} axes @ start corner", vis)
        cv2.waitKey(0)
        cv2.destroyWindow(f"cam{self.camera_id} axes @ start corner")

    def _write_matrix(self, parent: ET.Element, tag: str, M: np.ndarray) -> None:
        e = ET.SubElement(parent, tag)
        e.set("rows", str(M.shape[0]))
        e.set("cols", str(M.shape[1]))
        e.text = " ".join(map(str, M.astype(float).reshape(-1).tolist()))

    def _write_vector(self, parent: ET.Element, tag: str, v: np.ndarray) -> None:
        e = ET.SubElement(parent, tag)
        v = v.reshape(-1).astype(float)
        e.set("n", str(v.size))
        e.text = " ".join(map(str, v.tolist()))

    def save_intrinsics_xml(self) -> Path:
        if self.K is None or self.dist is None:
            raise ValueError("No intrinsics to save.")

        root = ET.Element("intrinsics")
        root.set("camera_id", str(self.camera_id))
        self._write_matrix(root, "K", self.K)
        self._write_vector(root, "distortion", self.dist)

        path = self.cam_dir / "intrinsics.xml"
        ET.ElementTree(root).write(str(path))
        return path

    def save_extrinsics_xml(self) -> Path:
        if self.best_rvec is None or self.best_tvec is None:
            raise ValueError("No extrinsics to save.")

        root = ET.Element("extrinsics")
        root.set("camera_id", str(self.camera_id))
        if self.best_frame_index is not None:
            root.set("frame_index", str(self.best_frame_index))
        if self.best_reproj_err is not None:
            root.set("reprojection_error_px", f"{self.best_reproj_err:.6f}")

        self._write_vector(root, "rvec", self.best_rvec)
        self._write_vector(root, "tvec", self.best_tvec)

        path = self.cam_dir / "extrinsics.xml"
        ET.ElementTree(root).write(str(path))
        return path

    def save_combined_config_xml(self) -> Path:
        if self.K is None or self.dist is None or self.best_rvec is None or self.best_tvec is None:
            raise ValueError("Need both intrinsics and extrinsics before writing combined config.")

        root = ET.Element("camera")
        root.set("id", str(self.camera_id))

        intr = ET.SubElement(root, "intrinsics")
        self._write_matrix(intr, "K", self.K)
        self._write_vector(intr, "distortion", self.dist)

        extr = ET.SubElement(root, "extrinsics")
        if self.best_frame_index is not None:
            extr.set("frame_index", str(self.best_frame_index))
        if self.best_reproj_err is not None:
            extr.set("reprojection_error_px", f"{self.best_reproj_err:.6f}")
        self._write_vector(extr, "rvec", self.best_rvec)
        self._write_vector(extr, "tvec", self.best_tvec)

        path = self.cam_dir / "calib_config.xml"
        ET.ElementTree(root).write(str(path))
        return path


def write_global_config(data_dir: Path, cams: List[CameraCalibrationTask1]) -> Path:
    root = ET.Element("cameras")
    for cam in cams:
        if cam.K is None or cam.dist is None or cam.best_rvec is None or cam.best_tvec is None:
            continue

        c = ET.SubElement(root, "camera")
        c.set("id", str(cam.camera_id))

        intr = ET.SubElement(c, "intrinsics")
        K = ET.SubElement(intr, "K")
        K.text = " ".join(map(str, cam.K.astype(float).reshape(-1).tolist()))
        dist = ET.SubElement(intr, "distortion")
        dist.text = " ".join(map(str, cam.dist.astype(float).reshape(-1).tolist()))

        extr = ET.SubElement(c, "extrinsics")
        if cam.best_frame_index is not None:
            extr.set("frame_index", str(cam.best_frame_index))
        if cam.best_reproj_err is not None:
            extr.set("reprojection_error_px", f"{cam.best_reproj_err:.6f}")

        r = ET.SubElement(extr, "rvec")
        r.text = " ".join(map(str, cam.best_rvec.astype(float).reshape(-1).tolist()))
        t = ET.SubElement(extr, "tvec")
        t.text = " ".join(map(str, cam.best_tvec.astype(float).reshape(-1).tolist()))

    out = Path(data_dir) / "cameras.xml"
    ET.ElementTree(root).write(str(out))
    return out


def run_task1(
    data_dir: str,
    checkerboard_xml: str,
    assume_tiles_not_corners: bool = False,
    intrinsics_frames: int = 15,
    extrinsics_candidates: int = 10,
    extrinsics_stride: int = 15,
    reuse_if_exists: bool = True,
) -> None:
    """
    Run Task 1:
    - For each camX/ directory, calibrate intrinsics from intrinsics.avi and extrinsics from checkerboard.avi
    - Save per-camera intrinsics.xml, extrinsics.xml, and combined calib_config.xml
    - Save global cameras.xml with all cameras' intrinsics and extrinsics
    """
    data_dir = Path(data_dir)
    spec = load_checkerboard_xml(Path(checkerboard_xml), assume_tiles_not_corners=assume_tiles_not_corners)

    cam_dirs = sorted([p for p in data_dir.glob("cam*") if p.is_dir()])
    if not cam_dirs:
        raise FileNotFoundError(f"No cam directories found in {data_dir} (expected data/camX/).")

    cams: List[CameraCalibrationTask1] = []

    for cam_path in cam_dirs:
        name = cam_path.name
        try:
            cam_id = int(name.replace("cam", ""))
        except Exception:
            continue

        intr_vid = cam_path / "intrinsics.avi"
        extr_vid = cam_path / "checkerboard.avi"

        if not intr_vid.exists():
            raise FileNotFoundError(f"Missing {intr_vid}")
        if not extr_vid.exists():
            raise FileNotFoundError(f"Missing {extr_vid}")

        cam = CameraCalibrationTask1(cam_id, data_dir)
        print(f"\n=== Camera {cam_id} ===")

        has_intr = has_extr = False
        if reuse_if_exists:
            has_intr, has_extr = cam.try_load_existing_calibration()

        if not has_intr:
            print("Calibrating intrinsics from intrinsics.avi ...")
            cam.calibrate_intrinsics_from_video(
                intrinsics_video=intr_vid,
                spec=spec,
                num_frames=intrinsics_frames,
                auto_first=True,
                manual_fallback=True,
            )
            intr_path = cam.save_intrinsics_xml()
            print(f"Saved intrinsics: {intr_path}")
        else:
            print("Using cached intrinsics.xml")

        if not has_extr:
            print("Calibrating extrinsics from checkerboard.avi (manual 4-corner + refine + orientation fix)")
            cam.calibrate_extrinsics_from_video(
                checkerboard_video=extr_vid,
                spec=spec,
                num_candidate_frames=extrinsics_candidates,
                frame_stride=extrinsics_stride,
            )
            extr_path = cam.save_extrinsics_xml()
            print(f"Saved extrinsics: {extr_path}")
        else:
            print("Using cached extrinsics.xml")

        cfg_path = cam.save_combined_config_xml()
        print(f"Saved per-camera combined config: {cfg_path}")

        cams.append(cam)

    global_path = write_global_config(data_dir, cams)
    print(f"\nSaved GLOBAL config: {global_path}")

# Task 2: Background subtraction to get foreground masks of video

def Background_subtraction(video_path, reference_path, output_path, k_h=2, k_s=2, k_v=4, th_s_shadow=45.0, v_ratio_low=0.50, v_ratio_high=0.8):
    """
    Simple background subtraction using HSV color space and Otsu's thresholding.
    - reference_path: path to a video of the static background (no foreground objects)
    - video_path: path to the video from which to extract foreground masks
    - output_path: directory to save the foreground masks as frame_XXX.jpg
    - k_h, k_s, k_v: thresholds for hue, saturation, and value channels
    """
    ref_cap = cv2.VideoCapture(reference_path)
    hsv_frames = []
    while True:
        ret, frame = ref_cap.read()
        if not ret:
            break
        hsv_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    ref_cap.release()

    ref_hsv = np.mean(hsv_frames, axis=0).astype(np.uint8)
    ref_var = np.var(hsv_frames, axis=0).astype(np.float32)

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        std = np.sqrt(ref_var + 1e-6)

        dh = np.abs(frame[..., 0].astype(np.float32) - ref_hsv[..., 0].astype(np.float32))
        dh = np.minimum(dh, 180 - dh)
        ds = np.abs(frame[..., 1].astype(np.float32) - ref_hsv[..., 1].astype(np.float32))
        dv = np.abs(frame[..., 2].astype(np.float32) - ref_hsv[..., 2].astype(np.float32))

        _, fg_h = cv2.threshold(dh.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_h = fg_h.astype(bool) & (std[..., 0] < k_h)

        _, fg_s = cv2.threshold(ds.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_s = fg_s.astype(bool) & (std[..., 1] < k_s)

        _, fg_v = cv2.threshold(dv.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_v = (fg_v > 0) & (std[..., 2] < k_v)

        fg = fg_h | fg_s | fg_v
        fgmask = (fg.astype(np.uint8) * 255)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(fgmask)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            fgmask = cv2.bitwise_and(fgmask, mask)

        Hc, Sc, Vc = frame[..., 0], frame[..., 1], frame[..., 2]
        Hr, Sr, Vr = ref_hsv[..., 0], ref_hsv[..., 1], ref_hsv[..., 2]
        v_ratio = Vc / (Vr + 1e-6)
        # th_s_shadow = 45.0
        # v_ratio_low = 0.50
        # v_ratio_high = 0.95
        shadow = (np.abs(Sc - Sr) < th_s_shadow) & (v_ratio >= v_ratio_low) & (v_ratio <= v_ratio_high)
        fgmask[shadow] = 0

        fgmask = cv2.morphologyEx(
            fgmask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1
        )
        fgmask = cv2.morphologyEx(
            fgmask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1
        )

        cv2.imwrite(
            os.path.join(output_path, f'frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg'),
            fgmask
        )
    cap.release()
    print("Background subtraction completed. Foreground masks saved in:", output_path)

def run_task2(data_dir: str):
    """
    Run Task 2:
    - For each camX/ directory, perform background subtraction on checkerboard.avi using Background_subtraction()
    - Save foreground masks in camX/masks/frame_XXX.jpg
    """
    data_dir = Path(data_dir)
    cam_dirs = sorted([p for p in data_dir.glob("cam*") if p.is_dir()])
    if not cam_dirs:
        raise FileNotFoundError(f"No cam directories found in {data_dir} (expected data/camX/).")

    for cam_path in cam_dirs:
        name = cam_path.name
        try:
            cam_id = int(name.replace("cam", ""))
        except Exception:
            continue

        video = cam_path / "video.avi"
        reference_vid = cam_path / "background.avi"
        masks_dir = cam_path / "masks"

        if not video.exists():
            raise FileNotFoundError(f"Missing {video}")
        if not reference_vid.exists():
            raise FileNotFoundError(f"Missing {reference_vid}")

        ensure_dir(masks_dir)

        print(f"\n=== Camera {cam_id} background subtraction ===")
        Background_subtraction(
            video_path=video,
            reference_path=reference_vid,
            output_path=masks_dir,
            k_h=1, k_s=1, k_v=3
        )

#### Task 3: Voxel Projection and Lookup Table Generation

def load_cams_from_data_dir(data_dir: Path) -> List[CameraCalibrationTask1]:
    """
    Utility to load existing calibrations for all cameras in the data directory.
     - Scans for camX/ directories and tries to load intrinsics.xml and extrinsics.xml for each camera.
     - Returns a list of CameraCalibrationTask1 instances with loaded calibrations where available.
     - Prints summary of loaded calibrations for each camera.
    """
    data_dir = Path(data_dir)
    cam_dirs = sorted([p for p in data_dir.glob("cam*") if p.is_dir()])

    cams: List[CameraCalibrationTask1] = []
    for cam_path in cam_dirs:
        name = cam_path.name
        try:
            cam_id = int(name.replace("cam", ""))
        except Exception:
            continue

        cam = CameraCalibrationTask1(cam_id, data_dir)
        cam.try_load_existing_calibration() #function from calib data class
        cams.append(cam)
        print(f"Loaded cam{cam_id} calibration: intrinsics={'yes' if cam.K is not None else 'no'}")
        print(f"Loaded cam{cam_id} calibration: extrinsics={'yes' if cam.best_rvec is not None and cam.best_tvec is not None else 'no'}")

    return cams


def _get_mask_size_for_cam(cam: CameraCalibrationTask1) -> Optional[Tuple[int, int]]:
    """
    Get the size of the foreground masks for a given camera.
    """
    masks_path = cam.cam_dir / "masks"
    if not masks_path.exists():
        return None
    mask_files = sorted(masks_path.glob("frame_*.jpg"))
    if not mask_files:
        return None
    m = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m.shape[0], m.shape[1])


def _infer_num_frames_from_masks(cams: List[CameraCalibrationTask1]) -> int:
    """
    Get number of frames to process based on the foreground masks available for the cameras.
    """
    counts = []
    for cam in cams:
        masks_path = cam.cam_dir / "masks"
        if not masks_path.exists():
            continue
        mask_files = sorted(masks_path.glob("frame_*.jpg"))
        if mask_files:
            counts.append(len(mask_files))
    return int(min(counts)) if counts else 0


def build_voxel_grid(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    step: float
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Build a 3D voxel grid within the specified limits and step size.
        - xlim, ylim, zlim: tuples specifying the min and max coordinates in each dimension (in mm)
        - step: the size of each voxel in mm (sapcing between points)
    """
    xs = np.arange(xlim[0], xlim[1] + 1e-9, step, dtype=np.float32)
    ys = np.arange(ylim[0], ylim[1] + 1e-9, step, dtype=np.float32)
    zs = np.arange(zlim[0], zlim[1] + 1e-9, step, dtype=np.float32)

    Nx, Ny, Nz = len(xs), len(ys), len(zs)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    vox = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1).astype(np.float32)
    return vox, (Nx, Ny, Nz)


def build_voxel_lookup_tables(
    cams: List[CameraCalibrationTask1],
    voxels_world: np.ndarray,
    image_sizes: dict,
    out_name: str = "voxel_lut.npz"
) -> None:
    """
    Based on calibrated cameras and the voxel grid
    builds lookup tables for each camera that map each voxel to its projected 2D pixel coordinates and validity.
    Look up table stored in out_name .npz
    Look up table has as keys
    """
    for cam in cams:
        if cam.K is None or cam.dist is None or cam.best_rvec is None or cam.best_tvec is None:
            print(f"[Task3] {cam.name}: missing calibration, skipping LUT.")
            continue

        size = image_sizes.get(cam.camera_id, None)
        if size is None:
            print(f"[Task3] {cam.name}: missing image size, skipping LUT.")
            continue

        H, W = size

        # For every voxel, project its 3D world coordinate to 2D pixel coordinate using the camera's intrinsics and extrinsics
        uv, _ = cv2.projectPoints(
            voxels_world.reshape(-1, 1, 3),
            cam.best_rvec, cam.best_tvec,
            cam.K, cam.dist
        )
        uv = uv.reshape(-1, 2).astype(np.float32) 

        R, _ = cv2.Rodrigues(cam.best_rvec)
        Xw = voxels_world.astype(np.float64).T
        Xc = (R @ Xw) + cam.best_tvec.astype(np.float64)
        in_front = (Xc[2, :] > 1e-6)

        uvi = np.rint(uv).astype(np.int32) #round to nearest pixel
        u = uvi[:, 0]
        v = uvi[:, 1]

        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid = inside & in_front #only pixels projected in front and inside the image are valid

        out_path = cam.cam_dir / out_name
        np.savez_compressed(out_path, uv=uv, uvi=uvi, valid=valid, H=H, W=W)
        print(f"[Task3] {cam.name}: saved voxel LUT -> {out_path}")


def _binary_volume_cleanup(volume_bool: np.ndarray) -> np.ndarray:
    """
    Morphological cleanup of a binary voxel volume. 
    - Removes isolated voxels that have fewer than 2 neighbors.
    - Fills small holes that have at least 18 neighbors (i.e. surrounded on all sides in a 3x3x3 neighborhood).
    Returns the cleaned binary volume.
    """
    V = volume_bool.astype(np.uint8)

    pad = np.pad(V, 1, mode="constant", constant_values=0)
    neigh = np.zeros_like(V, dtype=np.uint16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neigh += pad[1 + dx:1 + dx + V.shape[0],
                             1 + dy:1 + dy + V.shape[1],
                             1 + dz:1 + dz + V.shape[2]]

    V_clean = (V == 1) & (neigh >= 2)

    pad2 = np.pad(V_clean.astype(np.uint8), 1, mode="constant", constant_values=0)
    neigh2 = np.zeros_like(V_clean, dtype=np.uint16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neigh2 += pad2[1 + dx:1 + dx + V_clean.shape[0],
                               1 + dy:1 + dy + V_clean.shape[1],
                               1 + dz:1 + dz + V_clean.shape[2]]

    V_filled = V_clean | ((~V_clean) & (neigh2 >= 18))
    return V_filled.astype(bool)


def voxel_reconstruction_task3(
    cams: List[CameraCalibrationTask1],
    data_dir: Path,
    xlim: Tuple[float, float] = (-2500, 2500),
    ylim: Tuple[float, float] = (2500, 2500),
    zlim: Tuple[float, float] = (0, 2000),
    step: float = 10.0,  # Medium resolution
    lut_name: str = "voxel_lut.npz",
    out_name: str = "voxels_reconstruction.npz",
) -> None:
    """
    Task 3: Voxel Reconstruction from Multiple Cameras
    - Builds a 3D voxel grid within the specified limits and step size.
    - For each calibrated camera, builds a lookup table that maps each voxel to its projected pixels
    - For each frame (based on available foreground masks), determines which voxels are visible in all cameras' masks.
    - Applies morphological cleanup to the resulting binary voxel volume.
    - Saves the final voxel reconstruction (world coordinates of occupied voxels) and metadata to a compressed .npz file.
    """
    data_dir = Path(data_dir)

    cams_use = []
    image_sizes = {}
    for cam in cams:
        if cam.K is None or cam.dist is None or cam.best_rvec is None or cam.best_tvec is None:
            continue
        sz = _get_mask_size_for_cam(cam)
        if sz is None:
            continue
        image_sizes[cam.camera_id] = sz
        cams_use.append(cam)

    if len(cams_use) < 2:
        print("[Task3] Need at least 2 calibrated cameras with masks to reconstruct.")
        return

    voxels_world, dims = build_voxel_grid(xlim, ylim, zlim, step) #smaller for testing
    N = voxels_world.shape[0]
    print(f"[Task3] voxel grid: N={N} dims={dims} step={step}")

    build_voxel_lookup_tables(cams_use, voxels_world, image_sizes, out_name=lut_name)

    # Load LUTs for each camera
    luts = {}
    for cam in cams_use:
        lut_path = cam.cam_dir / lut_name
        if not lut_path.exists():
            print(f"[Task3] Missing LUT for {cam.name}: {lut_path}")
            return
        d = np.load(lut_path)
        luts[cam.camera_id] = {
            "uvi": d["uvi"].astype(np.int32),
            "valid": d["valid"].astype(bool),
        }

    T = _infer_num_frames_from_masks(cams_use)
    if T <= 0:
        print("[Task3] No mask frames found.")
        return
    print(f"[Task3] reconstructing for T={T} frames (using min mask count across cams)")


    # For each frame, determine which voxels are visible in all cameras' masks
    all_points = []
    all_volumes = []

    for t in range(T):
        # Start with all voxels alive, and iteratively filter down based on each camera's mask
        alive = np.ones((N,), dtype=bool)

        for cam in cams_use:
            masks_path = cam.cam_dir / "masks"

            mask_file = masks_path / f"frame_{t+1}.jpg"
            if not mask_file.exists():
                mask_files = sorted(masks_path.glob("frame_*.jpg"))
                if t < len(mask_files):
                    mask_file = mask_files[t]
                else:
                    alive[:] = False
                    break

            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                alive[:] = False
                break

            lut = luts[cam.camera_id]
            uvi = lut["uvi"]
            valid = lut["valid"]

            u = uvi[:, 0]
            v = uvi[:, 1]

            fg = np.zeros((N,), dtype=bool)
            ok = valid & alive
            if np.any(ok):
                fg_ok = (mask[v[ok], u[ok]] > 0)
                fg[ok] = fg_ok

            alive &= fg
            if not np.any(alive):
                break

        vol = alive.reshape(dims)
        vol = _binary_volume_cleanup(vol)
        all_volumes.append(vol)

        pts = voxels_world[vol.reshape(-1)]
        all_points.append(pts)
        print(f"[Task3] frame {t+1}/{T}: voxels_on={pts.shape[0]}") # Print 

    # Save voxels
    out_path = Path(data_dir) / out_name
    np.savez_compressed(
        out_path,
        xlim=np.array(xlim, dtype=np.float32),
        ylim=np.array(ylim, dtype=np.float32),
        zlim=np.array(zlim, dtype=np.float32),
        step=np.float32(step),
        dims=np.array(dims, dtype=np.int32),
        voxels_world=voxels_world.astype(np.float32),
        points_per_frame=np.array(all_points, dtype=object),
        # volume_per_frame=np.array(all_volumes, dtype=object),  # Too large for memory
        num_frames=len(all_points),
        cam_ids=np.array([c.camera_id for c in cams_use], dtype=np.int32),
    )

def visualize_3d_reconstruction(
    reconstruction_file: Path,
    frame_idx: Optional[int] = 0,
    frame_indices: Optional[List[int]] = None,
    max_points: int = 20000,
    point_size: float = 1.0,
    alpha: float = 0.6
) -> None:
    """
    Minimal 3D visualization of the reconstructed point cloud(s).

    Args:
        reconstruction_file: Path to .npz file containing 'points_per_frame'
        frame_idx: single frame to display if frame_indices is None
        frame_indices: list of frames to overlay (if provided)
        max_points: max number of points to plot total (subsampled if needed)
        point_size: marker size
        alpha: point transparency
    """
    data = np.load(reconstruction_file, allow_pickle=True)
    points_per_frame = data["points_per_frame"]

    # Decide which frames to plot
    if frame_indices is None:
        frame_indices = [int(frame_idx)]

    # Collect points
    pts_list = []
    for fi in frame_indices:
        if 0 <= fi < len(points_per_frame):
            pts = points_per_frame[fi]
            if pts is not None and len(pts) > 0:
                pts_list.append(np.asarray(pts, dtype=float))

    if not pts_list:
        print("No points found for the selected frame(s).")
        return

    pts_all = np.vstack(pts_list)

    # Subsample for performance
    if len(pts_all) > max_points:
        idx = np.random.choice(len(pts_all), max_points, replace=False)
        pts_all = pts_all[idx]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts_all[:, 0], pts_all[:, 1], pts_all[:, 2], s=point_size, alpha=alpha)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Reconstruction")

    # Optional: equal-ish scaling for nicer proportions
    x_min, x_max = pts_all[:, 0].min(), pts_all[:, 0].max()
    y_min, y_max = pts_all[:, 1].min(), pts_all[:, 1].max()
    z_min, z_max = pts_all[:, 2].min(), pts_all[:, 2].max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.5
    cx, cy, cz = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5
    ax.set_xlim(cx - max_range, cx + max_range)
    ax.set_ylim(cy - max_range, cy + max_range)
    ax.set_zlim(cz - max_range, cz + max_range)

    plt.show()

def visualize_3d_reconstruction_animation(
    reconstruction_file: Path,
    max_points: int = 20000,
    point_size: float = 1.0,
    alpha: float = 0.7,
    interval_sec: float = 1.0
) -> None:
    """
    Animate 3D reconstruction:
    - Shows one frame at a time
    - Switches every `interval_sec` seconds
    """

    data = np.load(reconstruction_file, allow_pickle=True)
    points_per_frame = data["points_per_frame"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Precompute global bounds for stable axes
    all_points = np.vstack([pts for pts in points_per_frame if len(pts) > 0])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.5
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    cz = (z_min + z_max) * 0.5

    ax.set_xlim(cx - max_range, cx + max_range)
    ax.set_ylim(cy - max_range, cy + max_range)
    ax.set_zlim(cz - max_range, cz + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.ion()
    plt.show()

    for i, pts in enumerate(points_per_frame):
        ax.cla()

        # Reset fixed bounds each frame
        ax.set_xlim(cx - max_range, cx + max_range)
        ax.set_ylim(cy - max_range, cy + max_range)
        ax.set_zlim(cz - max_range, cz + max_range)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Frame {i+1}")

        if len(pts) > 0:
            pts = np.asarray(pts, dtype=float)
            if len(pts) > max_points:
                idx = np.random.choice(len(pts), max_points, replace=False)
                pts = pts[idx]

            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=point_size, alpha=alpha)

        plt.draw()
        plt.pause(interval_sec)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    DATA_DIR = Path("/home/izujia/Desktop/3-2026/CV/Assigment2/data")
    CHECKERBOARD_XML = "/home/izujia/Desktop/3-2026/CV/Assigment2/data/checkerboard.xml"
    #DATA_DIR=r"C:\Users\abpom\Desktop\IP\ComputerVision\data"
    #CHECKERBOARD_XML= r"C:\Users\abpom\Desktop\IP\ComputerVision\data\checkerboard.xml"

    run_task1(
        data_dir=str(DATA_DIR),
        checkerboard_xml=CHECKERBOARD_XML,
        assume_tiles_not_corners=False,
        intrinsics_frames=15,
        extrinsics_candidates=10,
        extrinsics_stride=15,
        reuse_if_exists=True,
    )
    # run_task1_choice1(
    #     data_dir=str(DATA_DIR),
    #     checkerboard_xml=CHECKERBOARD_XML,
    #     cameras=(1, 2, 3, 4),
    #     max_frames_used=30,
    #     frame_stride=2,
    #     show_debug=True
    # )
    
    cams = load_cams_from_data_dir(DATA_DIR)

    # # OPTIONAL: used to do background substraction (skip if you already have masks from Task)
    # run_task2(str(DATA_DIR))

    # Exclude bad cameras (cam1 and cam4)
    bad_ids = {2,3}
    cams_filtered = [c for c in cams if c.camera_id not in bad_ids]

    # # Perform 3D reconstruction
    voxel_reconstruction_task3(cams_filtered, DATA_DIR)
    # Visualize results
    reconstruction_file = Path(DATA_DIR) / "voxels_reconstruction.npz"
    frame_asked = input("Enter frame index to visualize (0-based): ")
    visualize_3d_reconstruction(reconstruction_file, max_points=10000, frame_idx=frame_asked)
    # # visualize_3d_reconstruction_animation(reconstruction_file, max_points=10000, interval_sec=0.5)