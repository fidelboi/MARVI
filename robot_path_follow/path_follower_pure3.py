#!/usr/bin/env python3
"""
path_follower_node.py

Pure-pursuit path follower for a differential-drive robot.
Reads the world path and the robot pose, publishes a target RPM for each wheel.
"""

import math
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float64

LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


# ----------------------------------------------------------------
# ROS-independent control logic
# ----------------------------------------------------------------

class PurePursuit:

    def __init__(self, lookahead=0.20, goal_tolerance=0.08, waypoint_tolerance=0.10,
                 max_linear=0.15, max_angular=1.0, rotate_threshold=math.radians(60.0),
                 rotate_speed=0.8, search_window=1.0, invert_turn=False):
        self.lookahead = lookahead
        self.goal_tolerance = goal_tolerance
        self.waypoint_tolerance = waypoint_tolerance
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.rotate_threshold = rotate_threshold
        self.rotate_speed = rotate_speed
        self.search_window = search_window
        self.invert_turn = invert_turn
        self.set_path(None)

    def set_path(self, pts):
        self.idx = 0
        self.done = False
        if pts is None or len(pts) < 2:
            self.pts = None
            self.s = None
            return
        self.pts = np.asarray(pts, dtype=np.float64)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])   # arc length at each point

    @property
    def active(self):
        return self.pts is not None and not self.done

    def step(self, x, y, yaw):
        """Returns (v [m/s], w [rad/s], finished_now [bool])."""
        if not self.active:
            return 0.0, 0.0, False

        pts = self.pts
        num_pts = len(pts)

        # --- 1. Sequential Index Progression (Drawn Order) ---
        while self.idx < num_pts - 1:
            p_curr = pts[self.idx]
            p_next = pts[self.idx + 1]

            d_curr = math.hypot(p_curr[0] - x, p_curr[1] - y)

            # Vector projection onto current line segment (p_curr -> p_next)
            vx, vy = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
            seg_len_sq = vx * vx + vy * vy

            if seg_len_sq > 1e-6:
                ux, uy = x - p_curr[0], y - p_curr[1]
                t = (ux * vx + uy * vy) / seg_len_sq
            else:
                t = 0.0

            # Advance index if close to waypoint OR projected past midpoint (t > 0.5)
            if d_curr < self.waypoint_tolerance or t > 0.5:
                self.idx += 1
            else:
                break

        # --- 2. Goal Arrival Check ---
        goal = pts[-1]
        dist_goal = math.hypot(goal[0] - x, goal[1] - y)

        # Only allow goal arrival if near the end of the path array (prevents loop stop)
        if self.idx >= int(num_pts * 0.8) and dist_goal < self.goal_tolerance:
            self.done = True
            return 0.0, 0.0, True

        # --- 3. Lookahead Target Selection ---
        dists = np.hypot(pts[self.idx:, 0] - x, pts[self.idx:, 1] - y)
        far = np.where(dists >= self.lookahead)[0]
        if len(far) > 0:
            target = pts[self.idx + int(far[0])]
        else:
            target = pts[-1]

        dx, dy = target[0] - x, target[1] - y
        ld = math.hypot(dx, dy)
        if ld < 1e-6:
            return 0.0, 0.0, False

        # --- 4. Transform Target to Robot Frame (x forward, y left) ---
        xr = math.cos(yaw) * dx + math.sin(yaw) * dy
        yr = -math.sin(yaw) * dx + math.cos(yaw) * dy
        alpha = math.atan2(yr, xr)

        # --- 5. Velocity and Steering Control ---
        if abs(alpha) > self.rotate_threshold:
            v = 0.0
            w = math.copysign(self.rotate_speed, alpha)
        else:
            # Smooth deceleration ramp near goal to eliminate overshooting
            decel_dist = max(2.0 * self.lookahead, 0.4)
            speed_factor = min(1.0, dist_goal / decel_dist) if self.idx >= int(num_pts * 0.8) else 1.0

            # Reduce velocity on heading misalignment
            heading_factor = max(0.1, math.cos(alpha))

            v = self.max_linear * speed_factor * heading_factor
            v = max(0.02, v)  # Minimum velocity threshold to overcome static friction

            # Curvature calculation with safe lookahead clamp
            ld_effective = max(ld, 0.05)
            w = 2.0 * v * math.sin(alpha) / ld_effective
            w = max(-self.max_angular, min(self.max_angular, w))

        if self.invert_turn:
            w = -w

        return v, w, False


def wheel_rpm(v, w, wheel_radius, track_width, max_rpm):
    """Unicycle (v, w) -> (left_rpm, right_rpm), scaled down if above max_rpm."""
    vl = v - w * track_width / 2.0
    vr = v + w * track_width / 2.0
    k = 60.0 / (2.0 * math.pi * wheel_radius)         # m/s -> rpm
    left, right = vl * k, vr * k
    biggest = max(abs(left), abs(right))
    if biggest > max_rpm > 0.0:
        scale = max_rpm / biggest
        left, right = left * scale, right * scale
    return left, right


# ----------------------------------------------------------------
# ROS 2 Node
# ----------------------------------------------------------------

class PathFollowerNode(Node):

    def __init__(self):
        super().__init__("path_follower")

        self.declare_parameter("wheel_radius", 0.0525)
        self.declare_parameter("track_width", 0.34)
        self.declare_parameter("max_rpm", 60.0)
        self.declare_parameter("max_linear", 0.10)
        self.declare_parameter("max_angular", 0.2)
        self.declare_parameter("lookahead", 0.20)
        self.declare_parameter("goal_tolerance", 0.08)
        self.declare_parameter("waypoint_tolerance", 0.10)
        self.declare_parameter("rotate_threshold_deg", 2.5)
        self.declare_parameter("rotate_speed", 0.1)
        self.declare_parameter("search_window", 1.0)
        self.declare_parameter("pose_timeout", 0.5)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("left_sign", 1)
        self.declare_parameter("right_sign", 1)
        self.declare_parameter("invert_turn", False)
        self.declare_parameter("left_topic", "left_wheel/rpm")
        self.declare_parameter("right_topic", "right_wheel/rpm")

        g = lambda n: self.get_parameter(n).value
        self.r = float(g("wheel_radius"))
        self.b = float(g("track_width"))
        self.max_rpm = float(g("max_rpm"))
        self.pose_timeout = float(g("pose_timeout"))
        self.left_sign = 1.0 if int(g("left_sign")) >= 0 else -1.0
        self.right_sign = 1.0 if int(g("right_sign")) >= 0 else -1.0

        self.ctrl = PurePursuit(
            lookahead=float(g("lookahead")),
            goal_tolerance=float(g("goal_tolerance")),
            waypoint_tolerance=float(g("waypoint_tolerance")),
            max_linear=float(g("max_linear")),
            max_angular=float(g("max_angular")),
            rotate_threshold=math.radians(float(g("rotate_threshold_deg"))),
            rotate_speed=float(g("rotate_speed")),
            search_window=float(g("search_window")),
            invert_turn=bool(g("invert_turn")),
        )

        self.pose = None          # (x, y, yaw)
        self.pose_time = None

        self.left_pub = self.create_publisher(Float64, g("left_topic"), 10)
        self.right_pub = self.create_publisher(Float64, g("right_topic"), 10)

        self.create_subscription(Path, "world_path", self.on_path, LATCHED_QOS)
        self.create_subscription(PoseStamped, "robot_pose", self.on_pose, 10)
        self.create_timer(1.0 / float(g("control_rate")), self.control)

        self.get_logger().info(
            f"Follower ready: wheel_radius={self.r} m, track_width={self.b} m, "
            f"max_rpm={self.max_rpm}")

    # ---------------- callbacks ----------------
    def on_path(self, msg):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(pts) < 2:
            self.ctrl.set_path(None)
            self.publish_rpm(0.0, 0.0)
            self.get_logger().info("Path empty -> stopped")
            return
        self.ctrl.set_path(pts)
        self.get_logger().info(f"New path with {len(pts)} points -> following")

    def on_pose(self, msg):
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.position.x, msg.pose.position.y, yaw)
        self.pose_time = self.get_clock().now()

    def control(self):
        if not self.ctrl.active:
            self.publish_rpm(0.0, 0.0)
            return

        if self.pose is None:
            self.publish_rpm(0.0, 0.0)
            self.get_logger().warn("Waiting for /robot_pose", throttle_duration_sec=2.0)
            return

        age = (self.get_clock().now() - self.pose_time).nanoseconds * 1e-9
        if age > self.pose_timeout:
            self.publish_rpm(0.0, 0.0)
            self.get_logger().warn(f"Pose is {age:.1f}s old (tag lost?) -> stopped",
                                   throttle_duration_sec=2.0)
            return

        x, y, yaw = self.pose
        v, w, finished = self.ctrl.step(x, y, yaw)

        if finished:
            self.get_logger().info("Goal reached -> stopped")

        left, right = wheel_rpm(v, w, self.r, self.b, self.max_rpm)
        self.publish_rpm(left, right)

    def publish_rpm(self, left, right):
        l, r = Float64(), Float64()
        l.data = float(left * self.left_sign)
        r.data = float(right * self.right_sign)
        self.left_pub.publish(l)
        self.right_pub.publish(r)


def main(args=None):
    rclpy.init(args=args)
    node = PathFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_rpm(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
