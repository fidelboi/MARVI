#!/usr/bin/env python3
"""
obstacle_planner_node.py
Static-obstacle planner. Plans ONCE per GO, never continuously.

Workflow:
  1. Node starts and learns the empty floor for `learning_frames` frames, then freezes the model.
  2. You place static obstacles.
  3. You press GO. The planner takes a short burst of frames (`plan_frames`), keeps only pixels
     that are foreground in most of them (majority vote), builds the C-Space costmap, checks the
     path, and, if blocked, computes ONE A* detour.
  4. The result is published ONCE on /safe_world_path (latched). Nothing changes afterwards
     until the next GO, so the robot always follows a stable path.

What counts as GO (either one works):
  - a new message on `world_path`  (set plan_on_path:=false to disable this)
  - an std_msgs/Empty message on `plan_trigger`
Extra:
  - an std_msgs/Empty message on `recalibrate` throws the background away and learns it again.

The robot itself is removed from the foreground mask using a parallax-shifted body mask,
a metric margin, and blob ownership (blobs mostly inside the robot zone are erased).
"""

import math
import heapq
import collections
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


def load_calibration(cal_path):
    """Reads homography from the OpenCV-style YAML."""
    fs = cv2.FileStorage(str(cal_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {cal_path}")
    H = fs.getNode("homography_pixel_to_floor").mat()
    if H is None or H.shape != (3, 3):
        raise ValueError("homography_pixel_to_floor missing or invalid in calibration file")
    fs.release()
    return H


def world_to_pixel(pt, H_inv):
    """Projects a world (x,y) point to pixel (u,v)."""
    src = np.array([[[pt[0], pt[1]]]], dtype=np.float64)
    dst = cv2.perspectiveTransform(src, H_inv)
    return dst[0, 0]


def a_star(grid, start, goal):
    """8-connected A* search on a 2D numpy array (0 = free, >0 = occupied)."""
    rows, cols = grid.shape
    if not (0 <= start[1] < rows and 0 <= start[0] < cols) or grid[start[1], start[0]] > 0:
        return []
    if not (0 <= goal[1] < rows and 0 <= goal[0] < cols) or grid[goal[1], goal[0]] > 0:
        return []

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows:
                if grid[neighbor[1], neighbor[0]] > 0:
                    continue
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                    heapq.heappush(open_set, (f_score, neighbor))
    return []


def smooth_path(path, window=5):
    """Moving average smoothing for A* grid jaggedness."""
    if len(path) < window:
        return path
    path_arr = np.array(path, dtype=np.float64)
    smoothed = np.copy(path_arr)
    for i in range(1, len(path) - 1):
        start_idx = max(0, i - window // 2)
        end_idx = min(len(path), i + window // 2 + 1)
        smoothed[i] = np.mean(path_arr[start_idx:end_idx], axis=0)
    return smoothed.tolist()


class ObstaclePlannerNode(Node):
    def __init__(self):
        super().__init__("obstacle_planner")

        self.declare_parameter("cal_file", "")
        self.declare_parameter("robot_radius", 0.20)
        self.declare_parameter("safety_margin", 0.15)
        self.declare_parameter("grid_res", 0.05)
        self.declare_parameter("world_frame", "world")

        # --- Background model ---
        self.declare_parameter("learning_frames", 100)   # frames used to learn the empty floor
        self.declare_parameter("bg_learning_rate", 0.0)  # 0.0 = frozen after calibration

        # --- One-shot planning ---
        self.declare_parameter("plan_on_path", True)     # a new world_path message counts as GO
        self.declare_parameter("plan_frames", 5)         # frames captured per plan
        self.declare_parameter("vote_frac", 0.6)         # pixel must be foreground in this fraction

        # --- Robot masking ---
        self.declare_parameter("mask_scale", 2.0)    # multiplies robot radius for the image mask
        self.declare_parameter("mask_margin", 0.15)  # extra metres around the robot
        self.declare_parameter("parallax", 0.20)     # shift of robot top away from image centre (fraction)
        self.declare_parameter("owner_frac", 0.4)    # blob removed if >= this fraction lies in robot zone

        cal_file = self.get_parameter("cal_file").value
        self.robot_radius = float(self.get_parameter("robot_radius").value)
        self.safety_margin = float(self.get_parameter("safety_margin").value)
        self.grid_res = float(self.get_parameter("grid_res").value)
        self.world_frame = self.get_parameter("world_frame").value

        self.learning_frames = int(self.get_parameter("learning_frames").value)
        self.bg_learning_rate = float(self.get_parameter("bg_learning_rate").value)

        self.plan_on_path = bool(self.get_parameter("plan_on_path").value)
        self.plan_frames = max(1, int(self.get_parameter("plan_frames").value))
        self.vote_frac = float(self.get_parameter("vote_frac").value)

        self.mask_scale = float(self.get_parameter("mask_scale").value)
        self.mask_margin = float(self.get_parameter("mask_margin").value)
        self.parallax = float(self.get_parameter("parallax").value)
        self.owner_frac = float(self.get_parameter("owner_frac").value)

        self.H = load_calibration(cal_file)
        self.H_inv = np.linalg.inv(self.H)

        self.bg_subtractor = self.make_subtractor()

        # --- State ---
        self.frame_count = 0
        self.calibrated = False
        self.pending = False       # a plan has been requested and frames are being collected
        self.collected = 0
        self.vote_acc = None
        self.plan_view = None      # last plan result, kept only for the debug image

        self.pose = None      # (x, y, yaw)
        self.world_path = []  # List of (x, y)
        self.grid_bounds_computed = False

        # Grid parameters
        self.xmin = self.ymin = self.xmax = self.ymax = 0.0
        self.M_world_to_grid = None
        self.M_grid_to_world = None
        self.M_img_to_grid = None
        self.M_grid_to_img = None
        self.grid_shape = (0, 0)

        # Publishers / Subscribers
        self.safe_path_pub = self.create_publisher(Path, "safe_world_path", LATCHED_QOS)
        self.debug_pub = self.create_publisher(Image, "planner/debug_image", 1)

        self.create_subscription(PoseStamped, "robot_pose", self.on_pose, 10)
        self.create_subscription(Path, "world_path", self.on_path, LATCHED_QOS)
        self.create_subscription(Image, "camera/image_raw", self.on_image, qos_profile_sensor_data)
        self.create_subscription(Empty, "plan_trigger", self.on_plan_trigger, 10)
        self.create_subscription(Empty, "recalibrate", self.on_recalibrate, 10)

        self.get_logger().info(
            f"Obstacle Planner Initialized (one-shot mode, learning_frames={self.learning_frames}, "
            f"bg_learning_rate={self.bg_learning_rate}, plan_frames={self.plan_frames})"
        )

    # ------------------------------------------------------------------ setup helpers
    @staticmethod
    def make_subtractor():
        # detectShadows=False fixes the extreme CPU lag.
        # varThreshold=150 ensures minor lighting changes don't trigger obstacles.
        return cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=150, detectShadows=False)

    def init_grid_mapping(self, height, width):
        """Calculates metric bounds from the camera view corners via homography."""
        corners = np.array([[[0, 0]], [[width, 0]], [[width, height]], [[0, height]]], dtype=np.float64)
        world_corners = cv2.perspectiveTransform(corners, self.H).reshape(-1, 2)

        self.xmin, self.ymin = world_corners.min(axis=0)
        self.xmax, self.ymax = world_corners.max(axis=0)

        grid_w = int((self.xmax - self.xmin) / self.grid_res)
        grid_h = int((self.ymax - self.ymin) / self.grid_res)
        self.grid_shape = (grid_h, grid_w)

        self.M_world_to_grid = np.array([
            [1.0 / self.grid_res, 0.0, -self.xmin / self.grid_res],
            [0.0, 1.0 / self.grid_res, -self.ymin / self.grid_res],
            [0.0, 0.0, 1.0]
        ])
        self.M_grid_to_world = np.linalg.inv(self.M_world_to_grid)
        self.M_img_to_grid = self.M_world_to_grid @ self.H
        self.M_grid_to_img = np.linalg.inv(self.M_img_to_grid)
        self.grid_bounds_computed = True

    # ------------------------------------------------------------------ callbacks
    def on_pose(self, msg):
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.position.x, msg.pose.position.y, yaw)

    def on_path(self, msg):
        """A new world_path is treated as GO (unless plan_on_path is false)."""
        self.world_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

        if len(self.world_path) < 2:
            # Empty / cancelled path: forward it unchanged so the follower stops.
            self.pending = False
            self.publish_path(self.world_path)
            return

        if not self.plan_on_path:
            return

        if not self.calibrated:
            # Obstacles are only placed after calibration, so this cannot be a real GO yet.
            self.get_logger().info("Path received before calibration finished; waiting for GO after calibration.")
            return

        self.request_plan("new world_path")

    def on_plan_trigger(self, _msg):
        self.request_plan("plan_trigger")

    def on_recalibrate(self, _msg):
        self.bg_subtractor = self.make_subtractor()
        self.frame_count = 0
        self.calibrated = False
        self.pending = False
        self.vote_acc = None
        self.plan_view = None
        self.get_logger().info("Recalibrating background. Keep the floor free of obstacles.")

    def request_plan(self, reason):
        if len(self.world_path) < 2:
            self.get_logger().warn("Plan requested but no valid world_path has been received yet.")
            return
        if not self.calibrated:
            self.get_logger().warn("Plan requested but background calibration is not finished yet.")
            return
        self.pending = True
        self.collected = 0
        self.vote_acc = None
        self.get_logger().info(f"Planning requested ({reason}); capturing {self.plan_frames} frames.")

    # ------------------------------------------------------------------ image handling
    def on_image(self, msg):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if not self.grid_bounds_computed:
            self.init_grid_mapping(msg.height, msg.width)

        # Phase 1: learn the empty floor, then freeze.
        if not self.calibrated:
            self.learn_background(frame, msg.header)
            return

        # Phase 2: idle until GO. Only process frames while a plan is pending.
        banner = None
        if self.pending:
            if self.pose is None:
                banner = "WAITING FOR ROBOT POSE"
            else:
                self.collect_frame(frame)
                if self.pending:
                    banner = f"PLANNING {self.collected}/{self.plan_frames}"

        self.publish_live_debug(frame, msg.header, banner)

    def learn_background(self, frame, header):
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        # Automatic learning rate (-1) builds the model quickly
        self.bg_subtractor.apply(blurred)
        self.frame_count += 1

        if self.frame_count >= self.learning_frames:
            self.calibrated = True
            self.get_logger().info("Background calibration done. Place obstacles, then press GO.")

        if self.debug_pub.get_subscription_count() > 0:
            debug_frame = frame.copy()
            cv2.putText(debug_frame,
                        f"LEARNING BACKGROUND: {min(self.frame_count, self.learning_frames)}/{self.learning_frames}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
            self.publish_debug(debug_frame, header)

    def foreground_mask(self, blurred):
        """Foreground against the frozen background model."""
        fg = self.bg_subtractor.apply(blurred, learningRate=self.bg_learning_rate)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        # NOISE REMOVAL: Morphological opening to destroy tiny false-positive specks
        noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(fg, cv2.MORPH_OPEN, noise_kernel)

    def collect_frame(self, frame):
        """Adds one frame to the majority vote; plans when enough frames are collected."""
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        fg = self.foreground_mask(blurred)

        if self.vote_acc is None:
            self.vote_acc = np.zeros(fg.shape, np.uint16)
        self.vote_acc += (fg > 0).astype(np.uint16)
        self.collected += 1

        if self.collected < self.plan_frames:
            return

        need = int(math.ceil(self.vote_frac * self.plan_frames))
        fg_mask = ((self.vote_acc >= need).astype(np.uint8)) * 255
        self.vote_acc = None
        self.pending = False
        self.plan(fg_mask, frame.shape)

    def build_robot_mask(self, shape, u, v, px_per_m):
        """Image-space mask covering the robot body and its parallax shift (robot is stationary at GO)."""
        h, w = shape[:2]
        mask = np.zeros((h, w), np.uint8)
        radius = max(1, int((self.robot_radius * self.mask_scale + self.mask_margin) * px_per_m))
        cx, cy = w / 2.0, h / 2.0

        # Tall robot: its top appears displaced away from the image centre
        top_u = u + (u - cx) * self.parallax
        top_v = v + (v - cy) * self.parallax
        for t in np.linspace(0.0, 1.0, 5):
            pt = (int(u + (top_u - u) * t), int(v + (top_v - v) * t))
            cv2.circle(mask, pt, radius, 255, -1)
        return mask

    # ------------------------------------------------------------------ planning (runs once per GO)
    def plan(self, fg_mask, frame_shape):
        h, w = frame_shape[:2]
        rx, ry, _ = self.pose

        # Pixel coordinates of the robot and pixels-per-metre at its location
        u, v = world_to_pixel((rx, ry), self.H_inv)
        u_edge, v_edge = world_to_pixel((rx + self.robot_radius, ry), self.H_inv)
        px_per_m = math.hypot(u - u_edge, v - v_edge) / self.robot_radius

        # --- Robot masking ---
        robot_mask = self.build_robot_mask(frame_shape, u, v, px_per_m)

        # Remove whole blobs that mostly belong to the robot (shadow/body fringe outside the mask)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
        if n > 1:
            inside = np.bincount(labels[robot_mask > 0], minlength=n)
            area = stats[:, cv2.CC_STAT_AREA]
            owned = inside >= self.owner_frac * area
            owned[0] = False  # label 0 is background
            fg_mask[owned[labels]] = 0
        fg_mask[robot_mask > 0] = 0

        # --- Perspective correction & C-Space dilation ---
        metric_grid = cv2.warpPerspective(fg_mask, self.M_img_to_grid, (self.grid_shape[1], self.grid_shape[0]))
        kernel_r = int((self.robot_radius + self.safety_margin) / self.grid_res)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_r * 2 + 1, kernel_r * 2 + 1))
        c_space = cv2.dilate(metric_grid, kernel)

        # --- Check the original path (every segment is sampled, so thin obstacles between waypoints count) ---
        flags = self.blocked_flags(c_space, self.world_path)
        blocked_idx = -1
        safe_idx = -1
        consecutive_safe = 0
        for i, occupied in enumerate(flags):
            if occupied and blocked_idx == -1:
                blocked_idx = i
                consecutive_safe = 0
            elif not occupied and blocked_idx != -1:
                consecutive_safe += 1
                if consecutive_safe > 5:  # Require 5 safe points to prevent jumping into pockets
                    safe_idx = i - 5
                    break
            else:
                consecutive_safe = 0

        safe_world_path = self.world_path

        if blocked_idx == -1:
            state_msg = "PATH CLEAR"
        else:
            if safe_idx == -1:
                safe_idx = len(self.world_path) - 1

            start_grid = self.get_free_start(c_space, self.world_to_grid_pt((rx, ry)))
            goal_grid = self.world_to_grid_pt(self.world_path[safe_idx])
            raw_grid_path = a_star(c_space, start_grid, goal_grid)

            if not raw_grid_path:
                state_msg = "E-STOP: NO DETOUR"
                safe_world_path = []  # Trigger E-Stop
                self.get_logger().warn("No detour found around the obstacle. Publishing empty path (E-STOP).")
            else:
                state_msg = "DETOUR COMPUTED"
                smooth_grid_path = smooth_path(raw_grid_path, window=5)
                detour_world = [self.grid_to_world_pt(pt) for pt in smooth_grid_path]

                # History split (closest point behind the blockage)
                distances = [math.hypot(rx - wx, ry - wy) for wx, wy in self.world_path[:blocked_idx]]
                closest_idx = int(np.argmin(distances)) if distances else 0

                # Stitch: [History] + [A*] + [Remaining]
                safe_world_path = self.world_path[:closest_idx] + detour_world + self.world_path[safe_idx:]
                self.get_logger().info(f"Detour computed ({len(detour_world)} points).")

        if blocked_idx == -1:
            self.get_logger().info("Path is clear. Publishing original path.")

        # Publish ONCE. Latched QoS keeps it available until the next GO.
        self.publish_path(safe_world_path)

        # --- Store result for the debug image (no further processing happens until next GO) ---
        c_space_img = cv2.warpPerspective(c_space, self.M_grid_to_img, (w, h))
        overlay = np.zeros((h, w, 3), np.uint8)
        overlay[:, :, 2] = c_space_img
        contours, _ = cv2.findContours(robot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.plan_view = {
            "overlay": overlay,
            "contours": contours,
            "old_pts": [world_to_pixel(pt, self.H_inv) for pt in self.world_path] if blocked_idx != -1 else [],
            "new_pts": [world_to_pixel(pt, self.H_inv) for pt in safe_world_path],
            "state": state_msg,
        }

    def blocked_flags(self, c_space, path):
        """flags[i] is True if the segment from path[i] to path[i+1] touches an occupied cell."""
        rows, cols = c_space.shape
        pts = [self.world_to_grid_pt(p) for p in path]
        flags = []
        for i, (gx, gy) in enumerate(pts):
            nx, ny = pts[i + 1] if i + 1 < len(pts) else (gx, gy)
            n = max(abs(nx - gx), abs(ny - gy)) + 1
            xs = np.linspace(gx, nx, n).round().astype(int)
            ys = np.linspace(gy, ny, n).round().astype(int)
            ok = (xs >= 0) & (xs < cols) & (ys >= 0) & (ys < rows)
            flags.append(bool(np.any(c_space[ys[ok], xs[ok]] > 0)))
        return flags

    def world_to_grid_pt(self, pt):
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.M_world_to_grid)
        return (int(dst[0, 0, 0]), int(dst[0, 0, 1]))

    def grid_to_world_pt(self, pt):
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.M_grid_to_world)
        return (float(dst[0, 0, 0]), float(dst[0, 0, 1]))

    def get_free_start(self, grid, start):
        """BFS to find nearest unoccupied cell if the robot's cell is inside the inflated obstacle."""
        if not (0 <= start[1] < grid.shape[0] and 0 <= start[0] < grid.shape[1]):
            return start
        if grid[start[1], start[0]] == 0:
            return start

        q = collections.deque([start])
        visited = set([start])
        while q:
            curr = q.popleft()
            if grid[curr[1], curr[0]] == 0:
                return curr
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = curr[0] + dx, curr[1] + dy
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return start

    # ------------------------------------------------------------------ visualization / publishing
    def publish_live_debug(self, frame, header, banner=None):
        """Live camera frame with the LAST plan drawn on it. Purely visual, never affects the path."""
        if self.debug_pub.get_subscription_count() == 0:
            return

        pv = None if banner else self.plan_view
        if pv is not None and pv["overlay"].shape == frame.shape:
            debug = cv2.addWeighted(frame, 1.0, pv["overlay"], 0.5, 0)
            if pv["old_pts"]:
                self.draw_dashed_polyline(debug, pv["old_pts"], (0, 255, 255), 2)
            if pv["new_pts"]:
                pts_arr = np.array(pv["new_pts"], np.int32).reshape((-1, 1, 2))
                cv2.polylines(debug, [pts_arr], False, (0, 255, 0), 4)
            cv2.drawContours(debug, pv["contours"], -1, (255, 0, 0), 2)
            text = pv["state"]
        else:
            debug = frame.copy()
            text = banner or "READY: place obstacles, then press GO"

        cv2.putText(debug, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        self.publish_debug(debug, header)

    def draw_dashed_polyline(self, img, points, color, thickness=2, dash_length=15):
        """Draws a dashed line for the obstructed old path."""
        for i in range(len(points) - 1):
            p1, p2 = np.array(points[i]), np.array(points[i + 1])
            dist = np.linalg.norm(p2 - p1)
            if dist == 0:
                continue
            dashes = int(dist / dash_length)
            for j in range(dashes):
                if j % 2 == 0:
                    start = tuple((p1 + (p2 - p1) * (j / dashes)).astype(int))
                    end = tuple((p1 + (p2 - p1) * ((j + 1) / dashes)).astype(int))
                    cv2.line(img, start, end, color, thickness)

    def publish_path(self, pts):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        for wx, wy in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.safe_path_pub.publish(msg)

    def publish_debug(self, img, header):
        out = Image()
        out.header = header
        out.height, out.width = img.shape[:2]
        out.encoding = "bgr8"
        out.step = out.width * 3
        out.data = img.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ObstaclePlannerNode()
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