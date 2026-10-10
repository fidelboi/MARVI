#!/usr/bin/env python3
"""
path_mapper_node.py

Converts the pixel path from the camera server into a world path (meters, floor
plane) using the homography stored in the calibration YAML.

Subscribes
    /pixel_path          nav_msgs/Path           pixel coordinates (x = u, y = v)
    /path_clear          std_msgs/Empty          clears the world path
    /camera/camera_info  sensor_msgs/CameraInfo  only used to warn about size mismatch

Publishes
    /world_path          nav_msgs/Path           floor coordinates in meters, latched
                                                 (frame_id = world_frame)

Parameters
    cal_file (str, REQUIRED)   OpenCV YAML with homography_pixel_to_floor
    spacing  (float, 0.0)      resample spacing in meters (0 = no resampling)
    world_frame (str, "world")
"""

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Empty

LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


# ----------------------------------------------------------------
# Pure helpers (unchanged logic from path_mapper.py)
# ----------------------------------------------------------------

def load_calibration(cal_path):
    """Read homography (and image size) from the OpenCV-style YAML."""
    fs = cv2.FileStorage(str(cal_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {cal_path}")

    H = fs.getNode("homography_pixel_to_floor").mat()
    if H is None or H.shape != (3, 3):
        raise ValueError("homography_pixel_to_floor missing or not 3x3 in calibration file")

    width = fs.getNode("image_width")
    height = fs.getNode("image_height")
    size = None
    if not width.empty() and not height.empty():
        size = (int(width.real()), int(height.real()))

    fs.release()
    return H, size


def pixels_to_world(pixel_pts, H):
    src = pixel_pts.reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    if not np.all(np.isfinite(dst)):
        raise ValueError("Homography produced non-finite points (point near horizon?)")
    return dst


def drop_duplicates(pts, eps=1e-9):
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > eps:
            keep.append(i)
    return pts[keep]


def resample(pts, spacing):
    pts = drop_duplicates(pts)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < spacing:
        return pts
    n = int(np.floor(total / spacing)) + 1
    s_new = np.linspace(0.0, total, n)
    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    return np.stack([x, y], axis=1)


def path_length(pts):
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# ----------------------------------------------------------------
# Node
# ----------------------------------------------------------------

class PathMapperNode(Node):

    def __init__(self):
        super().__init__("path_mapper")

        self.declare_parameter("cal_file", "")
        self.declare_parameter("spacing", 0.0)
        self.declare_parameter("world_frame", "world")

        cal_file = self.get_parameter("cal_file").value
        if not cal_file:
            raise RuntimeError("Parameter 'cal_file' is required (path to cal.yaml)")

        self.spacing = float(self.get_parameter("spacing").value)
        self.world_frame = self.get_parameter("world_frame").value

        self.H, self.cal_size = load_calibration(cal_file)
        self.get_logger().info(f"Loaded homography from {cal_file}")
        if self.cal_size:
            self.get_logger().info(f"Calibration image size: {self.cal_size[0]}x{self.cal_size[1]}")

        self.world_pub = self.create_publisher(Path, "world_path", LATCHED_QOS)
        self.create_subscription(Path, "pixel_path", self.on_pixel_path, LATCHED_QOS)
        self.create_subscription(Empty, "path_clear", self.on_clear, 10)
        self.create_subscription(CameraInfo, "camera/camera_info", self.on_camera_info, LATCHED_QOS)

    def on_camera_info(self, msg):
        if self.cal_size and (msg.width, msg.height) != self.cal_size:
            self.get_logger().warn(
                f"Camera is {msg.width}x{msg.height} but calibration is for "
                f"{self.cal_size[0]}x{self.cal_size[1]}. World path will be wrong "
                f"unless these match.")

    def on_pixel_path(self, msg):
        if len(msg.poses) < 2:
            self.get_logger().warn("Pixel path has fewer than 2 points, ignoring")
            return

        pixel_pts = np.array(
            [(p.pose.position.x, p.pose.position.y) for p in msg.poses],
            dtype=np.float64)

        try:
            world_pts = pixels_to_world(pixel_pts, self.H)
            if self.spacing > 0.0:
                world_pts = resample(world_pts, self.spacing)
        except ValueError as e:
            self.get_logger().error(str(e))
            return

        out = Path()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.world_frame
        for x, y in world_pts:
            ps = PoseStamped()
            ps.header = out.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            out.poses.append(ps)

        self.world_pub.publish(out)
        self.get_logger().info(
            f"Mapped {len(pixel_pts)} pixel pts -> {len(world_pts)} world pts, "
            f"length {path_length(world_pts):.3f} m")

    def on_clear(self, _msg):
        out = Path()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.world_frame
        self.world_pub.publish(out)       # empty path = cleared
        self.get_logger().info("World path cleared")


def main(args=None):
    rclpy.init(args=args)
    node = PathMapperNode()
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
