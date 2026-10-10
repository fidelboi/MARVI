#!/usr/bin/env python3
"""
obstacle_planner_node.py
Intercepts the camera feed and world path to perform dynamic obstacle avoidance.
Generates a metric C-Space costmap, calculates an A* detour, and publishes /safe_world_path.
"""

import math
import heapq
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image

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

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows:
                if grid[neighbor[1], neighbor[0]] > 0:
                    continue
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + math.hypot(goal[0]-neighbor[0], goal[1]-neighbor[1])
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

def smooth_path(path, window=5):
    """Moving average smoothing for A* grid jaggedness."""
    if len(path) < window: return path
    path_arr = np.array(path, dtype=np.float64)
    smoothed = np.copy(path_arr)
    for i in range(1, len(path)-1):
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

        cal_file = self.get_parameter("cal_file").value
        self.robot_radius = float(self.get_parameter("robot_radius").value)
        self.safety_margin = float(self.get_parameter("safety_margin").value)
        self.grid_res = float(self.get_parameter("grid_res").value)
        self.world_frame = self.get_parameter("world_frame").value

        self.H = load_calibration(cal_file)
        self.H_inv = np.linalg.inv(self.H)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30, detectShadows=False)

        self.pose = None      # (x, y, yaw)
        self.world_path = []  # List of (x, y)
        self.grid_bounds_computed = False
        
        # Grid parameters
        self.xmin = self.ymin = self.xmax = self.ymax = 0.0
        self.M_world_to_grid = None
        self.M_img_to_grid = None
        self.M_grid_to_img = None
        self.grid_shape = (0, 0)

        # Publishers / Subscribers
        self.safe_path_pub = self.create_publisher(Path, "safe_world_path", LATCHED_QOS)
        self.debug_pub = self.create_publisher(Image, "planner/debug_image", 1)

        self.create_subscription(PoseStamped, "robot_pose", self.on_pose, 10)
        self.create_subscription(Path, "world_path", self.on_path, LATCHED_QOS)
        self.create_subscription(Image, "camera/image_raw", self.on_image, qos_profile_sensor_data)

        self.get_logger().info("Obstacle Planner Initialized")

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
        self.M_img_to_grid = self.M_world_to_grid @ self.H
        self.M_grid_to_img = np.linalg.inv(self.M_img_to_grid)
        self.grid_bounds_computed = True

    def on_pose(self, msg):
        q = msg.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.pose = (msg.pose.position.x, msg.pose.position.y, yaw)

    def on_path(self, msg):
        self.world_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def on_image(self, msg):
        if not self.pose: return
        
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if not self.grid_bounds_computed:
            self.init_grid_mapping(msg.height, msg.width)

        rx, ry, ryaw = self.pose

        # --- STEP A: Dynamic Robot Masking ---
        u, v = world_to_pixel((rx, ry), self.H_inv)
        u_edge, v_edge = world_to_pixel((rx + self.robot_radius, ry), self.H_inv)
        px_radius = int(math.hypot(u - u_edge, v - v_edge) * 1.15)
        
        mask_frame = frame.copy()
        cv2.circle(mask_frame, (int(u), int(v)), px_radius, (0, 0, 0), -1)
        fg_mask = self.bg_subtractor.apply(mask_frame,learningRate=0.0)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # --- STEP B: Perspective Correction & C-Space Dilation ---
        metric_grid = cv2.warpPerspective(fg_mask, self.M_img_to_grid, (self.grid_shape[1], self.grid_shape[0]))
        kernel_r = int((self.robot_radius + self.safety_margin) / self.grid_res)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_r*2+1, kernel_r*2+1))
        c_space = cv2.dilate(metric_grid, kernel)

        state_msg = "NO PATH"
        safe_world_path = self.world_path
        detour_computed = False
        blocked_idx = -1
        safe_idx = -1

        if len(self.world_path) >= 2:
            # --- STEP C: Temporal Projection & Reconnection ---
            # Project robot 0.2m forward
            px = rx + 0.2 * math.cos(ryaw)
            py = ry + 0.2 * math.sin(ryaw)
            start_grid = self.world_to_grid_pt((px, py))
            start_grid = self.get_free_start(c_space, start_grid)

            # Check original path for collisions
            consecutive_safe = 0
            for i, (wx, wy) in enumerate(self.world_path):
                gx, gy = self.world_to_grid_pt((wx, wy))
                if not (0 <= gx < self.grid_shape[1] and 0 <= gy < self.grid_shape[0]): continue
                
                is_occupied = c_space[gy, gx] > 0
                if is_occupied and blocked_idx == -1:
                    blocked_idx = i
                    consecutive_safe = 0
                elif not is_occupied and blocked_idx != -1:
                    consecutive_safe += 1
                    if consecutive_safe > 5: # Require 5 safe points to prevent jumping into pockets
                        safe_idx = i - 5
                        break
                else:
                    consecutive_safe = 0

            # --- STEP D: A* Planning & Smoothing ---
            if blocked_idx != -1:
                state_msg = "PATH BLOCKED"
                if safe_idx == -1: safe_idx = len(self.world_path) - 1
                goal_grid = self.world_to_grid_pt(self.world_path[safe_idx])
                
                raw_grid_path = a_star(c_space, start_grid, goal_grid)
                if not raw_grid_path:
                    state_msg = "E-STOP: NO DETOUR"
                    safe_world_path = [] # Trigger E-Stop
                else:
                    state_msg = "DETOUR COMPUTED"
                    detour_computed = True
                    smooth_grid_path = smooth_path(raw_grid_path, window=5)
                    detour_world = [self.grid_to_world_pt(pt) for pt in smooth_grid_path]
                    
                    # Find history split (closest point behind robot)
                    distances = [math.hypot(rx-wx, ry-wy) for wx,wy in self.world_path[:blocked_idx]]
                    closest_idx = np.argmin(distances) if distances else 0
                    
                    # Stitch: [History] + [A*] + [Remaining]
                    safe_world_path = self.world_path[:closest_idx] + detour_world + self.world_path[safe_idx:]
            else:
                state_msg = "PATH CLEAR"

        self.publish_path(safe_world_path)

        # --- STEP E: Rich Visualization ---
        if self.debug_pub.get_subscription_count() > 0:
            # 1. Overlay C-Space costmap
            c_space_img = cv2.warpPerspective(c_space, self.M_grid_to_img, (frame.shape[1], frame.shape[0]))
            red_overlay = np.zeros_like(frame)
            red_overlay[:,:,2] = c_space_img
            debug = cv2.addWeighted(frame, 1.0, red_overlay, 0.5, 0)
            
            # 2. Draw Paths
            if blocked_idx != -1:
                old_pts = [world_to_pixel(pt, self.H_inv) for pt in self.world_path]
                self.draw_dashed_polyline(debug, old_pts, (0, 255, 255), 2)
            
            new_pts = [world_to_pixel(pt, self.H_inv) for pt in safe_world_path]
            if new_pts:
                pts_arr = np.array(new_pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(debug, [pts_arr], False, (0, 255, 0), 4)

            # 3. Draw Robot Mask
            cv2.circle(debug, (int(u), int(v)), px_radius, (255, 0, 0), 2)

            # 4. Text Overlay
            cv2.putText(debug, state_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            self.publish_debug(debug, msg.header)

    def world_to_grid_pt(self, pt):
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.M_world_to_grid)
        return (int(dst[0, 0, 0]), int(dst[0, 0, 1]))

    def grid_to_world_pt(self, pt):
        M_inv = np.linalg.inv(self.M_world_to_grid)
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, M_inv)
        return (float(dst[0, 0, 0]), float(dst[0, 0, 1]))

    def get_free_start(self, grid, start):
        """BFS to find nearest unoccupied cell if robot is temporally trapped."""
        if not (0 <= start[1] < grid.shape[0] and 0 <= start[0] < grid.shape[1]): return start
        if grid[start[1], start[0]] == 0: return start
        
        q = [start]
        visited = set([start])
        while q:
            curr = q.pop(0)
            if grid[curr[1], curr[0]] == 0: return curr
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = curr[0]+dx, curr[1]+dy
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return start

    def draw_dashed_polyline(self, img, points, color, thickness=2, dash_length=15):
        """Draws a dashed line for the obstructed old path."""
        for i in range(len(points)-1):
            p1, p2 = np.array(points[i]), np.array(points[i+1])
            dist = np.linalg.norm(p2 - p1)
            if dist == 0: continue
            dashes = int(dist / dash_length)
            for j in range(dashes):
                if j % 2 == 0:
                    start = tuple((p1 + (p2-p1)*(j/dashes)).astype(int))
                    end = tuple((p1 + (p2-p1)*((j+1)/dashes)).astype(int))
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
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__": main()