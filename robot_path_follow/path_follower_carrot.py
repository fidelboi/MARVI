#!/usr/bin/env python3
"""
path_follower_node.py

Strict Turn-and-Drive waypoint follower for a differential-drive robot.
Behavior: Rotates in place to face the target, then drives in a perfectly 
straight line (w=0). No curved trajectories.
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
# ROS-independent control logic (Strict Straight Line)
# ----------------------------------------------------------------

class StraightLineFollower:
    def __init__(self, waypoint_radius=0.20, goal_tolerance=0.08, max_linear=0.15,
                 rotate_threshold=math.radians(5.0), rotate_speed=0.8, invert_turn=False):
        self.waypoint_radius = waypoint_radius  
        self.goal_tolerance = goal_tolerance    
        self.max_linear = max_linear
        # Very strict threshold (e.g., 5 degrees) to ensure straight lines
        self.rotate_threshold = rotate_threshold
        self.rotate_speed = rotate_speed
        self.invert_turn = invert_turn
        self.set_path(None)

    def set_path(self, pts):
        self.idx = 0
        self.done = False
        if pts is None or len(pts) < 2:
            self.pts = None
            return
        self.pts = np.asarray(pts, dtype=np.float64)

    @property
    def active(self):
        return self.pts is not None and not self.done

    def step(self, x, y, yaw):
        """Returns (v [m/s], w [rad/s], finished_now [bool])."""
        if not self.active:
            return 0.0, 0.0, False

        pts = self.pts

        # --- 1. Advance the index if close enough to the target ---
        while self.idx < len(pts) - 1:
            target_x, target_y = pts[self.idx]
            dist_to_target = math.hypot(target_x - x, target_y - y)
            
            if dist_to_target < self.waypoint_radius:
                self.idx += 1
            else:
                break

        # --- 2. Goal check (for the very last point) ---
        goal = pts[-1]
        dist_goal = math.hypot(goal[0] - x, goal[1] - y)
        if self.idx == len(pts) - 1 and dist_goal < self.goal_tolerance:
            self.done = True
            return 0.0, 0.0, True

        # --- 3. Current Target ---
        target_x, target_y = pts[self.idx]
        #target_x = 0.0 
        #target_y = 0.0
        
        dx, dy = target_x - x, target_y - y
        ld = math.hypot(dx, dy)
        
        if ld < 1e-6:
            return 0.0, 0.0, False

        # --- 4. Transform target to robot frame to find Angle Error ---
        xr = math.cos(yaw) * dx + math.sin(yaw) * dy
        yr = -math.sin(yaw) * dx + math.cos(yaw) * dy
        
       
        alpha = math.atan2(yr, xr)  # Angle error to the target
        print(f"yaw: {yaw*180/3.14:.3f} degrees | dx: {dx:.3f}, dy: {dy:.3f} | xr: {xr:.3f}, yr: {yr:.3f}")
        print(f'AngularError {alpha*180/3.14}')
        print(f'LinearError {dist_goal}')

        # --- 5. Movement Logic (Strict Turn THEN Drive) ---
        if abs(alpha) > self.rotate_threshold:
            # PHASE 1: Stop forward motion. Rotate in place to align.
            v = 0.0
            w = math.copysign(self.rotate_speed, alpha)
            print('# PHASE 1: Stop forward motion. Rotate in place to align.')
        else:
            # PHASE 2: Aligned. Drive perfectly straight. No steering allowed.
            v = self.max_linear
            v = min(v, max(0.04, 0.8 * dist_goal)) # Slow down gently if approaching final goal
            w = 0.0  # Forces wheels to spin at exact same RPM (Straight Line)
            print('# PHASE 2: Aligned. Drive perfectly straight. No steering allowed.')

        print(f"v: {v:.3f} m/s | w: {w:.3f} rad/s")

        if self.invert_turn:
            w = -w
        
        return v, w, False


def wheel_rpm(v, w, wheel_radius, track_width, max_rpm):
    """Unicycle (v, w) -> (left_rpm, right_rpm), scaled down if above max_rpm."""
    vl = v - w * track_width / 2.0
    vr = v + w * track_width / 2.0
    k = 60.0 / (2.0 * math.pi * wheel_radius)         
    left, right = vl * k, vr * k
    biggest = max(abs(left), abs(right))
    if biggest > max_rpm > 0.0:
        scale = max_rpm / biggest
        left, right = left * scale, right * scale
    return left, right


# ----------------------------------------------------------------
# Node
# ----------------------------------------------------------------

class StraightFollowerNode(Node):

    def __init__(self):
        super().__init__("straight_follower")

        self.declare_parameter("wheel_radius", 0.0525)
        self.declare_parameter("track_width", 0.34)
        self.declare_parameter("max_rpm", 60.0)
        self.declare_parameter("max_linear", 0.10)
        self.declare_parameter("waypoint_radius", 0.20) 
        self.declare_parameter("goal_tolerance", 0.08)
        # Tightened to 5 degrees for strict alignment before driving
        self.declare_parameter("rotate_threshold_deg", 10.0) 
        self.declare_parameter("rotate_speed", 0.2)
        self.declare_parameter("pose_timeout", 0.5)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("left_sign", 1)
        self.declare_parameter("right_sign", 1)
        self.declare_parameter("invert_turn", False)
        self.declare_parameter("left_topic", "motor2/cmd_rpm")
        self.declare_parameter("right_topic", "motor1/cmd_rpm")

        g = lambda n: self.get_parameter(n).value
        self.r = float(g("wheel_radius"))
        self.b = float(g("track_width"))
        self.max_rpm = float(g("max_rpm"))
        self.pose_timeout = float(g("pose_timeout"))
        self.left_sign = 1.0 if int(g("left_sign")) >= 0 else -1.0
        self.right_sign = 1.0 if int(g("right_sign")) >= 0 else -1.0

        self.ctrl = StraightLineFollower(
            waypoint_radius=float(g("waypoint_radius")),
            goal_tolerance=float(g("goal_tolerance")),
            max_linear=float(g("max_linear")),
            rotate_threshold=math.radians(float(g("rotate_threshold_deg"))),
            rotate_speed=float(g("rotate_speed")),
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
            f"Straight Follower ready: wheel_radius={self.r} m, track_width={self.b} m, "
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
        self.get_logger().info(f"New path with {len(pts)} points -> following in straight lines")

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
    print("==========================================", flush=True)
    print("  RUNNING PATH FOLLOWER CARROT NODE       ", flush=True)
    print("==========================================", flush=True)
    rclpy.init(args=args)
    node = StraightFollowerNode()
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
