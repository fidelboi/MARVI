#!/usr/bin/env python3
"""
path_follower_node.py   (drop-in replacement for your straight_follower)

Turn-and-drive waypoint follower for a differential-drive robot.

  ROTATE : spin in place, slowing down as the heading error shrinks
  SETTLE : stop for `settle_time` so the (delayed) camera pose catches up
  DRIVE  : drive perfectly straight (w = 0); go back to ROTATE only if the
           error grows beyond `realign_tol_deg`

Why this fixes oscillation
  * The camera pose is 0.1-0.3 s late. Spinning at constant speed until the
    error is "small enough" overshoots. Here the speed drops with the error,
    and after stopping the robot WAITS before re-measuring.
  * Two thresholds (hysteresis): align at 4 deg, but only re-align above 15 deg.
  * Optional `min_wheel_rpm` compensates the motor dead-band.

Self-diagnosis (printed in the log every 0.5 s)
  WRONG DIRECTION -> error is growing while rotating  => set invert_turn: true
  NOT TURNING     -> error is not changing while rotating => dead-band / no power / tag frozen
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Float64

LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

ROTATE, SETTLE, DRIVE = "ROTATE", "SETTLE", "DRIVE"


# ----------------------------------------------------------------
# ROS-independent control logic
# ----------------------------------------------------------------

class StraightLineFollower:

    def __init__(self, waypoint_radius=0.20, goal_tolerance=0.08, max_linear=0.10,
                 align_tol=math.radians(4.0), realign_tol=math.radians(15.0),
                 max_rotate_speed=0.6, min_rotate_speed=0.15, rotate_gain=1.5,
                 settle_time=0.4, invert_turn=False):
        self.waypoint_radius = waypoint_radius
        self.goal_tolerance = goal_tolerance
        self.max_linear = max_linear
        self.align_tol = align_tol
        self.realign_tol = realign_tol
        self.max_rotate_speed = max_rotate_speed
        self.min_rotate_speed = min_rotate_speed
        self.rotate_gain = rotate_gain
        self.settle_time = settle_time
        self.invert_turn = invert_turn

        self.alpha = 0.0
        self.dist_target = 0.0
        self.dist_goal = 0.0
        self.suspect = ""
        self.set_path(None)

    def set_path(self, pts):
        self.idx = 0
        self.done = False
        self.state = None
        self.suspect = ""
        if pts is None or len(pts) < 2:
            self.pts = None
            return
        self.pts = np.asarray(pts, dtype=np.float64)

    @property
    def active(self):
        return self.pts is not None and not self.done

    def _enter(self, state, now, alpha):
        self.state = state
        self.t_state = now
        self.a_state = abs(alpha)
        self.flips = 0
        self.last_s = 0
        if state != ROTATE:
            self.suspect = ""

    def step(self, x, y, yaw, now):
        """now = time in seconds. Returns (v [m/s], w [rad/s], finished_now)."""
        if not self.active:
            return 0.0, 0.0, False
        pts = self.pts

        # 1. advance to the next waypoint when close enough
        while self.idx < len(pts) - 1:
            if math.hypot(pts[self.idx][0] - x, pts[self.idx][1] - y) < self.waypoint_radius:
                self.idx += 1
            else:
                break

        # 2. goal check
        goal = pts[-1]
        self.dist_goal = math.hypot(goal[0] - x, goal[1] - y)
        if self.idx == len(pts) - 1 and self.dist_goal < self.goal_tolerance:
            self.done = True
            return 0.0, 0.0, True

        # 3. error to the current target (angle in the robot frame, wrapped to +-pi)
        dx, dy = pts[self.idx][0] - x, pts[self.idx][1] - y
        self.dist_target = math.hypot(dx, dy)
        if self.dist_target < 1e-6:
            return 0.0, 0.0, False
        xr = math.cos(yaw) * dx + math.sin(yaw) * dy
        yr = -math.sin(yaw) * dx + math.cos(yaw) * dy
        alpha = math.atan2(yr, xr)
        self.alpha = alpha

        if self.state is None:
            self._enter(ROTATE, now, alpha)

        v = w = 0.0

        # 4. state machine
        if self.state == ROTATE:
            if abs(alpha) < self.align_tol:
                self._enter(SETTLE, now, alpha)
            else:
                speed = min(self.max_rotate_speed,
                            max(self.min_rotate_speed, self.rotate_gain * abs(alpha)))
                w = math.copysign(speed, alpha)

                # Self-diagnosis. With a wrong turn direction the robot settles
                # facing AWAY from the target, so alpha keeps flipping sign
                # around +-180 deg. Count those flips.
                sgn = 1 if alpha > 0 else -1
                if abs(alpha) > math.radians(120.0):
                    if self.last_s and sgn != self.last_s:
                        self.flips += 1
                    self.last_s = sgn

                if now - self.t_state > 2.5:
                    change = abs(alpha) - self.a_state
                    if self.flips >= 3 or change > math.radians(25.0):
                        self.suspect = "WRONG DIRECTION (spinning away / flipping near 180 deg) -> set invert_turn: true"
                    elif abs(change) < math.radians(3.0):
                        self.suspect = "NOT TURNING (error unchanged) -> dead-band / motors / tag frozen"
                    else:
                        self.suspect = ""
                    self.t_state = now
                    self.a_state = abs(alpha)
                    self.flips = 0

        elif self.state == SETTLE:
            if now - self.t_state >= self.settle_time:
                if abs(alpha) < self.align_tol * 1.5:
                    self._enter(DRIVE, now, alpha)
                else:
                    self._enter(ROTATE, now, alpha)

        elif self.state == DRIVE:
            if abs(alpha) > self.realign_tol:
                self._enter(ROTATE, now, alpha)
            else:
                v = min(self.max_linear, max(0.04, 0.8 * self.dist_goal))

        if self.invert_turn:
            w = -w
        return v, w, False


def wheel_rpm(v, w, wheel_radius, track_width, max_rpm, min_wheel_rpm=0.0):
    """Unicycle (v, w) -> (left_rpm, right_rpm)."""
    vl = v - w * track_width / 2.0
    vr = v + w * track_width / 2.0
    k = 60.0 / (2.0 * math.pi * wheel_radius)
    left, right = vl * k, vr * k

    # lift small non-zero commands above the motor dead-band
    if min_wheel_rpm > 0.0:
        if 0.0 < abs(left) < min_wheel_rpm:
            left = math.copysign(min_wheel_rpm, left)
        if 0.0 < abs(right) < min_wheel_rpm:
            right = math.copysign(min_wheel_rpm, right)

    biggest = max(abs(left), abs(right))
    if biggest > max_rpm > 0.0:
        s = max_rpm / biggest
        left, right = left * s, right * s
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
        self.declare_parameter("min_wheel_rpm", 0.0)       # set to your motors' dead-band, e.g. 10.0
        self.declare_parameter("max_linear", 0.10)
        self.declare_parameter("waypoint_radius", 0.20)
        self.declare_parameter("goal_tolerance", 0.08)
        self.declare_parameter("align_tol_deg", 4.0)
        self.declare_parameter("realign_tol_deg", 15.0)
        self.declare_parameter("max_rotate_speed", 0.6)    # rad/s
        self.declare_parameter("min_rotate_speed", 0.15)   # rad/s
        self.declare_parameter("rotate_gain", 1.5)
        self.declare_parameter("settle_time", 0.4)         # s, >= camera latency
        self.declare_parameter("pose_timeout", 0.5)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("left_sign", 1)
        self.declare_parameter("right_sign", 1)
        self.declare_parameter("invert_turn", False)
        self.declare_parameter("left_topic", "motor1/cmd_rpm")
        self.declare_parameter("right_topic", "motor2/cmd_rpm")

        g = lambda n: self.get_parameter(n).value
        self.r = float(g("wheel_radius"))
        self.b = float(g("track_width"))
        self.max_rpm = float(g("max_rpm"))
        self.min_wheel_rpm = float(g("min_wheel_rpm"))
        self.pose_timeout = float(g("pose_timeout"))
        self.left_sign = 1.0 if int(g("left_sign")) >= 0 else -1.0
        self.right_sign = 1.0 if int(g("right_sign")) >= 0 else -1.0

        self.ctrl = StraightLineFollower(
            waypoint_radius=float(g("waypoint_radius")),
            goal_tolerance=float(g("goal_tolerance")),
            max_linear=float(g("max_linear")),
            align_tol=math.radians(float(g("align_tol_deg"))),
            realign_tol=math.radians(float(g("realign_tol_deg"))),
            max_rotate_speed=float(g("max_rotate_speed")),
            min_rotate_speed=float(g("min_rotate_speed")),
            rotate_gain=float(g("rotate_gain")),
            settle_time=float(g("settle_time")),
            invert_turn=bool(g("invert_turn")),
        )

        self.pose = None
        self.pose_time = None
        self.pose_latency = 0.0

        self.left_pub = self.create_publisher(Float64, g("left_topic"), 10)
        self.right_pub = self.create_publisher(Float64, g("right_topic"), 10)

        self.create_subscription(Path, "world_path", self.on_path, LATCHED_QOS)
        self.create_subscription(PoseStamped, "robot_pose", self.on_pose, 10)
        self.create_timer(1.0 / float(g("control_rate")), self.control)

        self.get_logger().info(
            f"Follower ready: wheel_radius={self.r} track_width={self.b} "
            f"max_rpm={self.max_rpm} invert_turn={g('invert_turn')} "
            f"topics={g('left_topic')},{g('right_topic')}")

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

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
        try:
            self.pose_latency = (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9
        except Exception:
            self.pose_latency = 0.0

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
        v, w, finished = self.ctrl.step(x, y, yaw, self.now_s())
        if finished:
            self.get_logger().info("Goal reached -> stopped")

        left, right = wheel_rpm(v, w, self.r, self.b, self.max_rpm, self.min_wheel_rpm)
        self.publish_rpm(left, right)

        c = self.ctrl
        self.get_logger().info(
            f"[{c.state}] wp {c.idx} dist={c.dist_target:.2f} m "
            f"alpha={math.degrees(c.alpha):+.1f} deg yaw={math.degrees(yaw):+.1f} "
            f"pose_latency={self.pose_latency*1000:.0f} ms  L={left:+.1f} R={right:+.1f} rpm",
            throttle_duration_sec=0.5)
        if c.suspect:
            self.get_logger().error(f"DIAGNOSIS: {c.suspect}", throttle_duration_sec=2.0)

    def publish_rpm(self, left, right):
        l, r = Float64(), Float64()
        l.data = float(left * self.left_sign)
        r.data = float(right * self.right_sign)
        self.left_pub.publish(l)
        self.right_pub.publish(r)


def main(args=None):
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
