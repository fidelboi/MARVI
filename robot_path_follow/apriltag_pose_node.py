#!/usr/bin/env python3
"""
apriltag_pose_node.py

Finds the AprilTag (family tag36h11) on top of the robot in the overhead camera
image and publishes the robot's pose on the floor, in the SAME world frame as
/world_path (it uses the same homography from cal.yaml).

Position : tag centre  -> homography -> world (x, y) in meters
Heading  : tag "up" edge (the edge you chose as the robot's FRONT) -> world yaw

Subscribes
    /camera/image_raw      sensor_msgs/Image (bgr8)

Publishes
    /robot_pose            geometry_msgs/PoseStamped   frame_id = world_frame
    /apriltag/debug_image  sensor_msgs/Image           annotated image (only built if someone subscribes)

Parameters
    cal_file (str, REQUIRED)       same cal.yaml used by path_mapper
    tag_id (int, -1)               -1 = accept the first tag found, otherwise only this ID
    heading_offset_deg (float, 0)  added to the heading; use 90/180/-90 if the arrow in the
                                   debug image does not point to the robot's front
    camera_height_m (float, 0)     optional parallax correction (see README notes);
    tag_height_m (float, 0)        0 = disabled
    foot_x, foot_y (float, 0)      world coords of the floor point directly below the camera
    world_frame (str, "world")
"""

import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from robot_path_follow.path_mapper_node import load_calibration


def make_detector():
    """Return detect(gray) -> (corners, ids). Works with OpenCV 4.5 and 4.7+."""
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

    if hasattr(aruco, "DetectorParameters_create"):
        params = aruco.DetectorParameters_create()
    else:
        params = aruco.DetectorParameters()
    try:
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    except AttributeError:
        pass

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)

        def detect(gray):
            corners, ids, _ = detector.detectMarkers(gray)
            return corners, ids
    else:
        def detect(gray):
            corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
            return corners, ids

    return detect


def to_world(pts_px, H):
    src = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(src, H).reshape(-1, 2)


def compute_pose(corners4, H, heading_offset=0.0,
                 cam_h=0.0, tag_h=0.0, foot=(0.0, 0.0)):
    """
    corners4: (4,2) pixel corners in ArUco order = tag's top-left, top-right,
              bottom-right, bottom-left as the tag is printed upright.
    Returns (x, y, yaw) in the world frame.
    """
    c = np.asarray(corners4, dtype=np.float64).reshape(4, 2)
    center_px = c.mean(axis=0)
    top_mid_px = (c[0] + c[1]) / 2.0          # middle of the tag's top edge = robot FRONT

    w_center, w_top = to_world([center_px, top_mid_px], H)
    fwd = w_top - w_center
    yaw = math.atan2(fwd[1], fwd[0]) + heading_offset

    pos = w_center.copy()
    if cam_h > 0.0 and tag_h > 0.0 and tag_h < cam_h:
        foot = np.asarray(foot, dtype=np.float64)
        pos = foot + (pos - foot) * (1.0 - tag_h / cam_h)

    yaw = math.atan2(math.sin(yaw), math.cos(yaw))   # wrap to [-pi, pi]
    return float(pos[0]), float(pos[1]), yaw


class AprilTagPoseNode(Node):

    def __init__(self):
        super().__init__("apriltag_pose")

        self.declare_parameter("cal_file", "")
        self.declare_parameter("tag_id", -1)
        self.declare_parameter("heading_offset_deg", 0.0)
        self.declare_parameter("camera_height_m", 0.0)
        self.declare_parameter("tag_height_m", 0.0)
        self.declare_parameter("foot_x", 0.0)
        self.declare_parameter("foot_y", 0.0)
        self.declare_parameter("world_frame", "world")

        cal_file = self.get_parameter("cal_file").value
        if not cal_file:
            raise RuntimeError("Parameter 'cal_file' is required (path to cal.yaml)")

        self.H, _ = load_calibration(cal_file)
        self.tag_id = int(self.get_parameter("tag_id").value)
        self.heading_offset = math.radians(self.get_parameter("heading_offset_deg").value)
        self.cam_h = float(self.get_parameter("camera_height_m").value)
        self.tag_h = float(self.get_parameter("tag_height_m").value)
        self.foot = (float(self.get_parameter("foot_x").value),
                     float(self.get_parameter("foot_y").value))
        self.world_frame = self.get_parameter("world_frame").value

        self.detect = make_detector()
        self._warned_encoding = False

        self.pose_pub = self.create_publisher(PoseStamped, "robot_pose", 10)
        self.debug_pub = self.create_publisher(Image, "apriltag/debug_image", 1)
        self.create_subscription(Image, "camera/image_raw", self.on_image,
                                 qos_profile_sensor_data)

        self.get_logger().info(
            f"Looking for tag36h11 id={'any' if self.tag_id < 0 else self.tag_id}")

    def on_image(self, msg):
        if msg.encoding != "bgr8":
            if not self._warned_encoding:
                self.get_logger().error(f"Unsupported image encoding {msg.encoding}, need bgr8")
                self._warned_encoding = True
            return

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self.detect(gray)

        want_debug = self.debug_pub.get_subscription_count() > 0
        debug = frame.copy() if want_debug else None

        chosen = None
        if ids is not None and len(ids) > 0:
            flat = ids.flatten()
            if self.tag_id < 0:
                chosen = 0
            else:
                hits = np.where(flat == self.tag_id)[0]
                if len(hits):
                    chosen = int(hits[0])

        if chosen is None:
            self.get_logger().warn("Robot tag not visible", throttle_duration_sec=2.0)
            if want_debug:
                self._publish_debug(debug, msg)
            return

        c = corners[chosen].reshape(4, 2)
        try:
            x, y, yaw = compute_pose(c, self.H, self.heading_offset,
                                     self.cam_h, self.tag_h, self.foot)
        except cv2.error as e:
            self.get_logger().error(f"Homography failed: {e}")
            return

        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.world_frame
        out.pose.position.x = x
        out.pose.position.y = y
        out.pose.orientation.z = math.sin(yaw / 2.0)
        out.pose.orientation.w = math.cos(yaw / 2.0)
        self.pose_pub.publish(out)

        self.get_logger().info(
            f"x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f} deg",
            throttle_duration_sec=1.0)

        if want_debug:
            pts = c.astype(np.int32)
            cv2.polylines(debug, [pts], True, (0, 255, 0), 2)
            center = tuple(int(v) for v in c.mean(axis=0))
            front = tuple(int(v) for v in ((c[0] + c[1]) / 2.0))
            cv2.arrowedLine(debug, center, front, (0, 0, 255), 3, tipLength=0.4)
            cv2.putText(debug, f"{x:.2f},{y:.2f} {math.degrees(yaw):.0f}deg",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self._publish_debug(debug, msg)

    def _publish_debug(self, img, src_msg):
        out = Image()
        out.header = src_msg.header
        out.height, out.width = img.shape[:2]
        out.encoding = "bgr8"
        out.step = out.width * 3
        out.data = img.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagPoseNode()
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
