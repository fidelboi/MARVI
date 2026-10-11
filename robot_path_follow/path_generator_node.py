#!/usr/bin/env python3
"""
path_generator_node.py
Camera-based obstacle-aware path generator (ROS 2, overhead camera + floor homography).

WHAT IT DOES
  1. Learns what the EMPTY floor looks like, either
        a) from reference image file(s)   (parameter `reference_images`), or
        b) from the first `learning_frames` camera frames (robot may be in view, see below).
  2. At runtime every camera frame is compared against that background. Anything that differs
     (an object put on the floor) becomes a candidate obstacle.
  3. The ROBOT ITSELF is never treated as an obstacle: its pose (`robot_pose`) is projected into the
     image, a parallax-shifted body mask is built and removed from the foreground, blobs that mostly
     belong to the robot are erased, and image regions under the robot are "unknown" (they neither
     create nor clear obstacles).
  4. Candidate obstacles are accumulated in a metric grid with temporal hysteresis
     (`confirm_frames` to appear, `clear_frames` to disappear), then inflated by
     robot_radius + safety_margin (C-space).
  5. When a goal arrives (`goal_pose`) an A* path from the robot to the goal is computed, shortcut
     (line-of-sight), resampled and published on `safe_world_path` (latched).
  6. While the robot drives, the remaining path is re-checked whenever the obstacle map changes;
     if it is blocked a new path is generated from the current robot position.
     No obstacle change -> the published path is never touched (stable for the path follower).

TOPICS
  Sub:  camera/image_raw   sensor_msgs/Image        (bgr8, rgb8, bgra8, rgba8, mono8)
        robot_pose         geometry_msgs/PoseStamped (x, y in the world/floor frame, metres)
        goal_pose          geometry_msgs/PoseStamped (x, y used)
        recalibrate        std_msgs/Empty            relearn the background from the camera
  Pub:  safe_world_path    nav_msgs/Path             (latched). EMPTY path = no route / E-STOP
        planner/obstacle_map nav_msgs/OccupancyGrid  (latched, inflated C-space, for RViz)
        planner/debug_image  sensor_msgs/Image       (bgr8)

CALIBRATION FILE (OpenCV YAML), same as before:
  %YAML:1.0
  homography_pixel_to_floor: !!opencv-matrix { rows: 3, cols: 3, dt: d, data: [...] }

LIMITATIONS (inherent to a single camera + floor homography)
  * The homography is valid only for points ON the floor. The top of a tall obstacle is projected
    "behind" its base, so tall objects look larger/longer in the grid. The inflation margin and the
    fact that errors are conservative (extra occupied cells) make this safe, but not exact.
  * If the robot is visible while the background is learned, the pixels under it are unknown until it
    drives away; they are then learned automatically (`bg_min_samples` frames). An object placed exactly
    there during that moment would be absorbed into the background. Use empty-floor reference images
    to avoid this completely.
"""

import math
import heapq

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy,
                       qos_profile_sensor_data)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

SQRT2 = math.sqrt(2.0)
NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]

_ENC_CHANNELS = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1, "8uc1": 1, "8uc3": 3}


# =============================================================================
# Pure helpers (no ROS) - easy to unit test
# =============================================================================
def load_calibration(cal_path):
    """Reads the pixel->floor homography from an OpenCV-style YAML file."""
    fs = cv2.FileStorage(str(cal_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {cal_path}")
    H = fs.getNode("homography_pixel_to_floor").mat()
    fs.release()
    if H is None or H.shape != (3, 3):
        raise ValueError("homography_pixel_to_floor missing or invalid in calibration file")
    return np.asarray(H, dtype=np.float64)


def project(M, x, y):
    """Applies a 3x3 homography to a single 2D point."""
    p = M @ np.array([x, y, 1.0])
    return p[:2] / p[2]


def image_to_bgr(msg):
    """sensor_msgs/Image -> contiguous BGR uint8 array (no cv_bridge needed)."""
    enc = msg.encoding.lower()
    if enc not in _ENC_CHANNELS:
        raise ValueError(f"unsupported image encoding '{msg.encoding}'")
    ch = _ENC_CHANNELS[enc]
    buf = np.frombuffer(msg.data, np.uint8)
    img = buf.reshape(msg.height, msg.step)[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
    if enc in ("bgr8", "8uc3"):
        out = img
    elif enc == "rgb8":
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif enc == "bgra8":
        out = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif enc == "rgba8":
        out = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:  # mono
        out = cv2.cvtColor(np.ascontiguousarray(img[:, :, 0]), cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(out)


def line_free(blocked, a, b):
    """True if no blocked cell lies on the grid line a->b. Cells outside the grid are ignored."""
    (x0, y0), (x1, y1) = a, b
    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.rint(np.linspace(x0, x1, n)).astype(int)
    ys = np.rint(np.linspace(y0, y1, n)).astype(int)
    h, w = blocked.shape
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    return not bool(blocked[ys[ok], xs[ok]].any())


def nearest_free(blocked, cell, max_radius):
    """Nearest free cell to `cell` (itself if free). None if nothing free within max_radius cells."""
    h, w = blocked.shape
    x = min(max(cell[0], 0), w - 1)
    y = min(max(cell[1], 0), h - 1)
    if not blocked[y, x] and (x, y) == tuple(cell):
        return (x, y)
    free = np.argwhere(~blocked)  # rows are (y, x)
    if free.size == 0:
        return None
    d2 = (free[:, 0] - cell[1]) ** 2 + (free[:, 1] - cell[0]) ** 2
    k = int(np.argmin(d2))
    if d2[k] > max_radius ** 2:
        return None
    return (int(free[k, 1]), int(free[k, 0]))


def a_star(blocked, penalty, start, goal):
    """
    8-connected A* on a boolean grid (True = blocked). `penalty` (>= 1.0 per cell) scales the cost of
    entering a cell, which keeps the path away from obstacles when there is room. No corner cutting.
    Cells are (x, y) = (col, row). Returns list of cells or [].
    """
    h, w = blocked.shape
    bl = blocked.tolist()
    pen = penalty.tolist()
    if bl[start[1]][start[0]] or bl[goal[1]][goal[0]]:
        return []

    gx, gy = goal
    open_heap = [(0.0, start)]
    g_score = {start: 0.0}
    came_from = {}
    closed = set()

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        cx, cy = cur
        g_cur = g_score[cur]
        for dx, dy, step in NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or bl[ny][nx]:
                continue
            if dx and dy and (bl[cy][nx] or bl[ny][cx]):  # do not cut corners
                continue
            nb = (nx, ny)
            if nb in closed:
                continue
            ng = g_cur + step * pen[ny][nx]
            if ng < g_score.get(nb, 1e18):
                g_score[nb] = ng
                came_from[nb] = cur
                ddx, ddy = abs(gx - nx), abs(gy - ny)
                hcost = (ddx + ddy) + (SQRT2 - 2.0) * min(ddx, ddy)  # octile, admissible (pen >= 1)
                heapq.heappush(open_heap, (ng + hcost, nb))
    return []


def shortcut(cells, blocked):
    """Greedy line-of-sight shortcutting ("string pulling") of a grid path."""
    if len(cells) < 3:
        return cells
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not line_free(blocked, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def resample(points, spacing):
    """Inserts points so consecutive waypoints are at most `spacing` apart."""
    out = [tuple(points[0])]
    for a, b in zip(points[:-1], points[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(d / spacing)))
        for k in range(1, n + 1):
            t = k / n
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


class BackgroundModel:
    """
    Per-pixel mean background built only from pixels flagged valid (i.e. not covered by the robot).
    Pixels with fewer than `min_samples` samples are 'unknown' and can never create obstacles.
    """

    def __init__(self, shape_hw, min_samples):
        h, w = shape_hw
        self.min_samples = max(1, int(min_samples))
        self.sum = np.zeros((h, w, 3), np.float32)
        self.count = np.zeros((h, w), np.uint16)
        self.lab_f = None
        self.dirty = True

    def add(self, blurred_bgr, mask):
        np.add(self.sum, blurred_bgr, out=self.sum, where=mask[..., None])
        self.count += mask.astype(np.uint16)
        self.dirty = True

    def set_from_image(self, img_bgr):
        self.sum = img_bgr.astype(np.float32) * self.min_samples
        self.count[:] = self.min_samples
        self.dirty = True

    def known(self):
        return self.count >= self.min_samples

    def lab(self):
        if self.dirty or self.lab_f is None:
            mean = (self.sum / np.maximum(self.count, 1)[..., None]).clip(0, 255).astype(np.uint8)
            self.lab_f = cv2.cvtColor(mean, cv2.COLOR_BGR2LAB).astype(np.float32)
            self.dirty = False
        return self.lab_f


# =============================================================================
# Node
# =============================================================================
class PathGeneratorNode(Node):
    def __init__(self):
        super().__init__("path_generator")

        # ---- parameters -----------------------------------------------------------------
        self.declare_parameter("cal_file", "")
        self.declare_parameter("world_frame", "world")
        # geometry / planning
        self.declare_parameter("robot_radius", 0.20)       # m
        self.declare_parameter("safety_margin", 0.10)      # m, added to robot_radius for inflation
        self.declare_parameter("grid_res", 0.05)           # m / cell
        self.declare_parameter("floor_bounds", [0.0, 0.0, 0.0, 0.0])  # xmin ymin xmax ymax; zeros = auto
        self.declare_parameter("plan_rate", 5.0)           # Hz, replan check
        self.declare_parameter("goal_tolerance", 0.10)     # m
        self.declare_parameter("waypoint_spacing", 0.10)   # m
        self.declare_parameter("clearance_dist", 0.30)     # m, prefer staying this far from inflated obstacles
        self.declare_parameter("clearance_weight", 2.0)    # 0 = pure shortest path
        self.declare_parameter("estop_on_failure", True)   # publish EMPTY path when no route exists
        # background
        self.declare_parameter("reference_images", [""])   # empty-floor image file(s)
        self.declare_parameter("learning_frames", 60)
        self.declare_parameter("bg_min_samples", 20)
        self.declare_parameter("bg_exclude_margin", 0.10)  # extra metres around robot excluded from learning
        # detection
        self.declare_parameter("blur_ksize", 7)
        self.declare_parameter("diff_threshold", 22.0)     # Lab distance
        self.declare_parameter("lightness_weight", 0.5)    # <1 reduces sensitivity to shadows
        self.declare_parameter("min_blob_px", 150)
        self.declare_parameter("confirm_frames", 8)
        self.declare_parameter("clear_frames", 30)
        # robot masking
        self.declare_parameter("robot_mask_margin", 0.12)  # m around robot body in the image mask
        self.declare_parameter("parallax", 0.15)           # = h_robot / (h_camera - h_robot)
        self.declare_parameter("owner_frac", 0.4)          # erase blob if >= this fraction is in robot zone
        self.declare_parameter("pose_timeout", 1.0)        # s; map is frozen if pose is older than this

        g = lambda n: self.get_parameter(n).value
        self.world_frame = g("world_frame")
        self.robot_radius = float(g("robot_radius"))
        self.safety_margin = float(g("safety_margin"))
        self.grid_res = float(g("grid_res"))
        self.floor_bounds = [float(v) for v in g("floor_bounds")]
        self.goal_tol = float(g("goal_tolerance"))
        self.wp_spacing = max(0.01, float(g("waypoint_spacing")))
        self.clear_dist = float(g("clearance_dist"))
        self.clear_w = float(g("clearance_weight"))
        self.estop_on_failure = bool(g("estop_on_failure"))
        self.ref_paths = [p for p in g("reference_images") if p]
        self.learning_frames = max(1, int(g("learning_frames")))
        self.bg_min_samples = min(max(1, int(g("bg_min_samples"))), self.learning_frames)
        self.bg_excl_margin = float(g("bg_exclude_margin"))
        k = int(g("blur_ksize"))
        self.blur_k = k if k % 2 == 1 else k + 1
        self.diff_thr = float(g("diff_threshold"))
        self.light_w = float(g("lightness_weight"))
        self.min_blob = int(g("min_blob_px"))
        self.confirm = max(1, int(g("confirm_frames")))
        self.clear_frames = max(self.confirm, int(g("clear_frames")))
        self.mask_margin = float(g("robot_mask_margin"))
        self.parallax = float(g("parallax"))
        self.owner_frac = float(g("owner_frac"))
        self.pose_timeout = float(g("pose_timeout"))

        self.H = load_calibration(g("cal_file"))
        self.H_inv = np.linalg.inv(self.H)

        # ---- state ----------------------------------------------------------------------
        self.grid_ready = False
        self.bg = None
        self.bg_ready = False
        self.learn_count = 0
        self.use_refs = bool(self.ref_paths)

        self.pose = None            # (x, y)
        self.pose_time = None

        self.goal = None
        self.need_plan = False
        self.path = []              # last published path (list of (x, y))
        self.path_px = []
        self.planned_version = -1
        self.map_version = 0

        self.evidence = None
        self.occupied = None        # bool grid, confirmed obstacles
        self.c_space = None         # bool grid, inflated
        self.overlay = None
        self.status = "STARTING"
        self._warned_enc = False

        self.xmin = self.ymin = 0.0
        self.grid_shape = (0, 0)
        self.M_img_to_grid = self.M_grid_to_img = None

        self.noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kr = int(math.ceil((self.robot_radius + self.safety_margin) / self.grid_res))
        self.infl_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1))

        # ---- ROS interfaces -------------------------------------------------------------
        self.path_pub = self.create_publisher(Path, "safe_world_path", LATCHED_QOS)
        self.map_pub = self.create_publisher(OccupancyGrid, "planner/obstacle_map", LATCHED_QOS)
        self.debug_pub = self.create_publisher(Image, "planner/debug_image", 1)

        self.create_subscription(Image, "camera/image_raw", self.on_image, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "robot_pose", self.on_pose, 10)
        self.create_subscription(PoseStamped, "goal_pose", self.on_goal, 10)
        self.create_subscription(Empty, "recalibrate", self.on_recalibrate, 10)
        self.create_timer(1.0 / max(0.5, float(g("plan_rate"))), self.on_timer)

        src = f"{len(self.ref_paths)} reference image(s)" if self.use_refs else f"{self.learning_frames} camera frames"
        self.get_logger().info(f"Path generator ready. Background will be learned from {src}.")

    # ------------------------------------------------------------------ small utils
    def now_s_since(self, t):
        return (self.get_clock().now() - t).nanoseconds * 1e-9

    def pose_fresh(self):
        return self.pose is not None and self.now_s_since(self.pose_time) < self.pose_timeout

    def world_to_cell(self, p):
        return (int(round((p[0] - self.xmin) / self.grid_res)),
                int(round((p[1] - self.ymin) / self.grid_res)))

    def cell_to_world(self, c):
        return (self.xmin + c[0] * self.grid_res, self.ymin + c[1] * self.grid_res)

    # ------------------------------------------------------------------ callbacks
    def on_pose(self, msg):
        self.pose = (msg.pose.position.x, msg.pose.position.y)
        self.pose_time = self.get_clock().now()

    def on_goal(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.need_plan = True
        self.get_logger().info(f"New goal ({self.goal[0]:.2f}, {self.goal[1]:.2f}).")

    def on_recalibrate(self, _msg):
        self.use_refs = False
        self.bg = None
        self.bg_ready = False
        self.learn_count = 0
        if self.grid_ready:
            self.reset_map()
        self.get_logger().info("Recalibrating: learning the background from the camera. "
                               "Keep the floor free of obstacles (the robot may stay in view).")

    # ------------------------------------------------------------------ setup from first frame
    def init_grid(self, h, w):
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        wc = np.array([project(self.H, u, v) for u, v in corners])
        xmin, ymin = wc.min(axis=0)
        xmax, ymax = wc.max(axis=0)
        fb = self.floor_bounds
        if len(fb) == 4 and fb[2] > fb[0] and fb[3] > fb[1]:
            xmin, ymin, xmax, ymax = fb
        gw = int(math.ceil((xmax - xmin) / self.grid_res)) + 1
        gh = int(math.ceil((ymax - ymin) / self.grid_res)) + 1
        if gw * gh > 4_000_000:
            raise RuntimeError(f"Grid would be {gw}x{gh} cells; set floor_bounds or increase grid_res.")
        self.xmin, self.ymin = float(xmin), float(ymin)
        self.grid_shape = (gh, gw)

        M_world_to_grid = np.array([[1 / self.grid_res, 0, -self.xmin / self.grid_res],
                                    [0, 1 / self.grid_res, -self.ymin / self.grid_res],
                                    [0, 0, 1.0]])
        self.M_img_to_grid = M_world_to_grid @ self.H
        self.M_grid_to_img = np.linalg.inv(self.M_img_to_grid)
        self.reset_map()
        self.grid_ready = True
        self.get_logger().info(f"Floor grid {gw}x{gh} cells, x[{xmin:.2f},{xmax:.2f}] y[{ymin:.2f},{ymax:.2f}] m.")

    def reset_map(self):
        self.evidence = np.zeros(self.grid_shape, np.int16)
        self.occupied = np.zeros(self.grid_shape, bool)
        self.c_space = np.zeros(self.grid_shape, bool)
        self.overlay = None
        self.map_version += 1
        self.publish_map()

    def ensure_background(self, frame):
        """Creates the background model (and loads reference images if configured)."""
        if self.bg is not None:
            return
        h, w = frame.shape[:2]
        self.bg = BackgroundModel((h, w), self.bg_min_samples)
        if not self.use_refs:
            return
        imgs = []
        for p in self.ref_paths:
            im = cv2.imread(p)
            if im is None:
                self.get_logger().error(f"Could not read reference image '{p}'.")
                continue
            if im.shape[:2] != (h, w):
                self.get_logger().warn(f"Reference '{p}' is {im.shape[1]}x{im.shape[0]}, camera is {w}x{h}; resizing.")
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
            imgs.append(cv2.GaussianBlur(im, (self.blur_k, self.blur_k), 0))
        if imgs:
            self.bg.set_from_image(np.median(np.stack(imgs), axis=0).astype(np.uint8))
            self.bg_ready = True
            self.get_logger().info(f"Background loaded from {len(imgs)} reference image(s).")
        else:
            self.get_logger().error("No usable reference images; learning from camera instead.")
            self.use_refs = False

    # ------------------------------------------------------------------ robot masking
    def build_robot_mask(self, shape, extra_margin=0.0):
        """Image-space mask of the robot: floor footprint + parallax-shifted top, swept with circles."""
        h, w = shape[:2]
        mask = np.zeros((h, w), np.uint8)
        if self.pose is None:
            return mask
        rx, ry = self.pose
        u, v = project(self.H_inv, rx, ry)
        u2, v2 = project(self.H_inv, rx + 0.1, ry)
        ppm = math.hypot(u - u2, v - v2) / 0.1  # pixels per metre at the robot
        if not all(map(math.isfinite, (u, v, ppm))) or max(abs(u), abs(v)) > 1e6:
            return mask
        radius = max(2, int((self.robot_radius + self.mask_margin + extra_margin) * ppm))
        cx, cy = w / 2.0, h / 2.0
        tu = u + (u - cx) * self.parallax
        tv = v + (v - cy) * self.parallax
        steps = max(2, int(math.hypot(tu - u, tv - v) / (0.5 * radius)) + 1)
        for t in np.linspace(0.0, 1.0, steps):
            cv2.circle(mask, (int(u + (tu - u) * t), int(v + (tv - v) * t)), radius, 255, -1)
        return mask

    # ------------------------------------------------------------------ image pipeline
    def on_image(self, msg):
        try:
            frame = image_to_bgr(msg)
        except ValueError as e:
            if not self._warned_enc:
                self.get_logger().error(str(e))
                self._warned_enc = True
            return
        h, w = frame.shape[:2]
        if not self.grid_ready:
            self.init_grid(h, w)
        self.ensure_background(frame)

        blurred = cv2.GaussianBlur(frame, (self.blur_k, self.blur_k), 0)
        fresh = self.pose_fresh()
        robot_mask = self.build_robot_mask(frame.shape) if fresh else None

        if not self.bg_ready:
            self.learn_step(blurred, frame.shape, fresh)
        elif fresh:
            self.update_obstacles(blurred, robot_mask, frame.shape)
        else:
            self.status = "POSE STALE - obstacle map frozen"

        self.publish_debug(frame, msg.header, robot_mask)

    def learn_step(self, blurred, shape, fresh):
        if not fresh:
            self.status = "WAITING FOR ROBOT POSE (needed to exclude robot from background)"
            return
        excl = self.build_robot_mask(shape, self.bg_excl_margin)
        self.bg.add(blurred, excl == 0)
        self.learn_count += 1
        self.status = f"LEARNING BACKGROUND {self.learn_count}/{self.learning_frames}"
        if self.learn_count >= self.learning_frames:
            self.bg_ready = True
            unknown = int((~self.bg.known()).sum())
            self.get_logger().info(
                f"Background learned ({unknown} px still unknown under the robot; they fill in once it moves).")

    def update_obstacles(self, blurred, robot_mask, shape):
        h, w = shape[:2]
        known = self.bg.known()

        # Fill in background pixels that were hidden by the robot while learning.
        if not known.all():
            excl = self.build_robot_mask(shape, self.bg_excl_margin)
            self.bg.add(blurred, (~known) & (excl == 0))
            known = self.bg.known()

        # --- foreground (Lab distance; lightness down-weighted to tolerate shadows) ---
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
        d = lab - self.bg.lab()
        d[..., 0] *= self.light_w
        dist = np.sqrt((d * d).sum(axis=2))
        fg = ((dist > self.diff_thr) & known).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.noise_kernel)

        # --- remove the robot: blobs mostly inside the robot zone, tiny blobs, then the zone itself ---
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n > 1:
            area = stats[:, cv2.CC_STAT_AREA]
            inside = np.bincount(labels[robot_mask > 0], minlength=n)
            drop = (inside >= self.owner_frac * area) | (area < self.min_blob)
            drop[0] = False
            fg[drop[labels]] = 0
        fg[robot_mask > 0] = 0

        # --- to metric grid; update temporal evidence only where we actually see the floor ---
        size = (self.grid_shape[1], self.grid_shape[0])
        fg_g = cv2.warpPerspective(fg, self.M_img_to_grid, size, flags=cv2.INTER_NEAREST) > 0
        visible = ((robot_mask == 0) & known).astype(np.uint8) * 255
        vis_g = cv2.warpPerspective(visible, self.M_img_to_grid, size, flags=cv2.INTER_NEAREST) > 0

        ev = self.evidence
        ev[vis_g & fg_g] += 1
        ev[vis_g & ~fg_g] -= 1
        np.clip(ev, 0, self.clear_frames, out=ev)

        occ = np.where(ev >= self.confirm, True, np.where(ev <= 0, False, self.occupied))
        self.status = "TRACKING OBSTACLES"
        if not np.array_equal(occ, self.occupied):
            self.occupied = occ
            self.c_space = cv2.dilate(occ.astype(np.uint8), self.infl_kernel) > 0
            self.map_version += 1
            self.overlay = None
            self.publish_map()
            self.get_logger().info(f"Obstacle map changed ({int(occ.sum())} occupied cells).")

    # ------------------------------------------------------------------ planning loop
    def on_timer(self):
        if self.goal is None or not self.bg_ready or self.c_space is None or not self.pose_fresh():
            return
        rx, ry = self.pose
        if math.hypot(self.goal[0] - rx, self.goal[1] - ry) < self.goal_tol:
            self.get_logger().info("Goal reached.")
            self.goal = None
            self.need_plan = False
            self.status = "GOAL REACHED"
            return

        replan = self.need_plan
        if not replan and self.map_version != self.planned_version:
            if not self.path:                 # last attempt failed: retry, the map changed
                replan = True
            elif self.path_blocked():
                self.get_logger().info("Current path is blocked by a new obstacle -> replanning.")
                replan = True
            else:
                self.planned_version = self.map_version   # still clear
        if replan:
            self.replan()

    def path_blocked(self):
        """Checks the not-yet-travelled part of the path against the inflated map."""
        pts = self.path
        if len(pts) < 2:
            return False
        rx, ry = self.pose
        i0 = int(np.argmin([math.hypot(rx - x, ry - y) for x, y in pts]))
        cells = [self.world_to_cell(p) for p in pts[i0:]]
        for a, b in zip(cells[:-1], cells[1:]):
            if not line_free(self.c_space, a, b):
                return True
        return False

    def penalty_grid(self):
        free = (~self.c_space).astype(np.uint8)
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 3)
        clear_cells = max(1.0, self.clear_dist / self.grid_res)
        return 1.0 + self.clear_w * np.clip(1.0 - dist / clear_cells, 0.0, 1.0)

    def plan_path(self, start_xy, goal_xy):
        cs = self.c_space
        h, w = cs.shape
        s = self.world_to_cell(start_xy)
        g = self.world_to_cell(goal_xy)
        for name, c in (("robot", s), ("goal", g)):
            if not (0 <= c[0] < w and 0 <= c[1] < h):
                return None, f"{name} is outside the camera floor area"

        max_r = int(math.ceil(0.5 / self.grid_res))
        s_free = nearest_free(cs, s, max_r)
        if s_free is None:
            return None, "robot is enclosed by obstacles"
        g_free = nearest_free(cs, g, max_r)
        if g_free is None:
            return None, "goal lies inside / behind an obstacle"
        goal_moved = g_free != g
        if goal_moved:
            self.get_logger().warn("Goal is inside the inflated obstacle zone; using the nearest free cell.")

        if line_free(cs, s_free, g_free):
            cells = [s_free, g_free]
        else:
            cells = a_star(cs, self.penalty_grid(), s_free, g_free)
            if not cells:
                return None, "no collision-free route"
            cells = shortcut(cells, cs)

        pts = [self.cell_to_world(c) for c in cells]
        pts[0] = tuple(start_xy)
        if not goal_moved:
            pts[-1] = tuple(goal_xy)
        return resample(pts, self.wp_spacing), "ok"

    def replan(self):
        self.need_plan = False
        self.planned_version = self.map_version
        pts, why = self.plan_path(self.pose, self.goal)
        if pts is None:
            self.status = f"NO PATH: {why}"
            self.get_logger().warn(f"Planning failed: {why}.")
            if self.estop_on_failure:
                self.publish_path([])
            return
        straight = len(pts) >= 2 and not self.occupied.any()
        self.status = "PATH CLEAR" if straight else f"PATH PLANNED ({len(pts)} pts)"
        self.publish_path(pts)
        self.get_logger().info(f"Published path with {len(pts)} points.")

    # ------------------------------------------------------------------ publishing
    def publish_path(self, pts):
        self.path = list(pts)
        self.path_px = [project(self.H_inv, x, y) for x, y in self.path]
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        for x, y in self.path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def publish_map(self):
        if self.c_space is None:
            return
        gh, gw = self.grid_shape
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.info.resolution = self.grid_res
        msg.info.width = gw
        msg.info.height = gh
        msg.info.origin.position.x = self.xmin - self.grid_res / 2.0
        msg.info.origin.position.y = self.ymin - self.grid_res / 2.0
        msg.info.origin.orientation.w = 1.0
        msg.data = (self.c_space.astype(np.int8) * 100).ravel().tolist()
        self.map_pub.publish(msg)

    def build_overlay(self, shape):
        h, w = shape[:2]
        cs = cv2.warpPerspective(self.c_space.astype(np.uint8) * 255, self.M_grid_to_img, (w, h),
                                 flags=cv2.INTER_NEAREST)
        oc = cv2.warpPerspective(self.occupied.astype(np.uint8) * 255, self.M_grid_to_img, (w, h),
                                 flags=cv2.INTER_NEAREST)
        ov = np.zeros((h, w, 3), np.uint8)
        ov[cs > 0] = (0, 0, 140)    # inflated zone: dark red
        ov[oc > 0] = (0, 0, 255)    # obstacle itself: bright red
        return ov

    def publish_debug(self, frame, header, robot_mask):
        if self.debug_pub.get_subscription_count() == 0:
            return
        dbg = frame.copy()
        if self.bg_ready and self.c_space is not None:
            if self.overlay is None or self.overlay.shape != frame.shape:
                self.overlay = self.build_overlay(frame.shape)
            dbg = cv2.addWeighted(dbg, 1.0, self.overlay, 0.5, 0)
        if robot_mask is not None:
            cnts, _ = cv2.findContours(robot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dbg, cnts, -1, (255, 0, 0), 2)
        if len(self.path_px) > 1:
            arr = np.array(self.path_px, np.int32).reshape(-1, 1, 2)
            cv2.polylines(dbg, [arr], False, (0, 255, 0), 3)
        if self.goal is not None:
            gu, gv = project(self.H_inv, *self.goal)
            cv2.circle(dbg, (int(gu), int(gv)), 10, (255, 0, 255), 3)
        cv2.putText(dbg, self.status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

        out = Image()
        out.header = header
        out.height, out.width = dbg.shape[:2]
        out.encoding = "bgr8"
        out.step = out.width * 3
        out.data = dbg.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PathGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
