#!/usr/bin/env python3
"""
apriltag_pose_node.py

Detects an AprilTag 36h11 mounted on a robot using an overhead camera
and publishes the robot pose in the calibrated world frame.

Position:
    AprilTag centre -> homography -> world coordinates (x, y)

Heading:
    Tag top edge -> world yaw

Subscribes:
    /camera/image_raw      sensor_msgs/Image

Publishes:
    /robot_pose            geometry_msgs/PoseStamped
    /apriltag/debug_image  sensor_msgs/Image

Parameters:
    cal_file              Required calibration YAML path
    tag_id                -1 = accept first detected tag
    heading_offset_deg     Heading correction in degrees
    camera_height_m        Optional camera height
    tag_height_m           Optional tag height
    foot_x, foot_y         World coordinates for parallax correction
    world_frame            World frame name

Lighting improvements:
    - CLAHE local contrast enhancement
    - Original grayscale fallback
    - Configurable AprilTag detector parameters
"""

import math

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from robot_path_follow.path_mapper_node import load_calibration


# ============================================================
# AprilTag detector
# ============================================================

def make_detector():
    """Create an OpenCV AprilTag 36h11 detector."""

    aruco = cv2.aruco

    dictionary = aruco.getPredefinedDictionary(
        aruco.DICT_APRILTAG_36h11
    )

    if hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
    else:
        params = aruco.DetectorParameters_create()

    # Improve corner localization.
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    # Adaptive thresholding for varying illumination.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 33
    params.adaptiveThreshWinSizeStep = 10

    # Permit relatively small markers.
    params.minMarkerPerimeterRate = 0.02

    # Leave other AprilTag-specific parameters at their defaults.
    # Tune them only after measuring detection performance.

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)

        def detect(gray):
            corners, ids, _ = detector.detectMarkers(gray)
            return corners, ids

    else:
        def detect(gray):
            corners, ids, _ = aruco.detectMarkers(
                gray,
                dictionary,
                parameters=params
            )
            return corners, ids

    return detect


# ============================================================
# Pixel coordinates -> world coordinates
# ============================================================

def to_world(pts_px, H):
    """Transform image pixel coordinates using a homography."""

    src = np.asarray(
        pts_px,
        dtype=np.float64
    ).reshape(-1, 1, 2)

    return cv2.perspectiveTransform(
        src,
        H
    ).reshape(-1, 2)


# ============================================================
# Robot pose calculation
# ============================================================

def compute_pose(
    corners4,
    H,
    heading_offset=0.0,
    cam_h=0.0,
    tag_h=0.0,
    foot=(0.0, 0.0)
):
    """
    Calculate robot position and heading.

    corners4:
        Four detected AprilTag corners in OpenCV ArUco order.

    Returns:
        x, y, yaw

    x, y:
        World coordinates in meters.

    yaw:
        Heading in radians.
    """

    c = np.asarray(
        corners4,
        dtype=np.float64
    ).reshape(4, 2)

    # Centre of the tag in image coordinates.
    center_px = c.mean(axis=0)

    # Midpoint of the first edge.
    # This edge defines the robot's front direction.
    top_mid_px = (c[0] + c[1]) / 2.0

    # Transform centre and front-edge midpoint into world coordinates.
    w_center, w_top = to_world(
        [center_px, top_mid_px],
        H
    )

    # Calculate world-frame heading.
    fwd = w_top - w_center

    yaw = math.atan2(
        fwd[1],
        fwd[0]
    ) + heading_offset

    # Position estimate.
    pos = w_center.copy()

    # Optional parallax correction.
    if 0.0 < tag_h < cam_h:
        foot = np.asarray(
            foot,
            dtype=np.float64
        )

        pos = foot + (
            pos - foot
        ) * (
            1.0 - tag_h / cam_h
        )

    # Normalize yaw to [-pi, pi].
    yaw = math.atan2(
        math.sin(yaw),
        math.cos(yaw)
    )

    return (
        float(pos[0]),
        float(pos[1]),
        yaw
    )


# ============================================================
# ROS 2 node
# ============================================================

class AprilTagPoseNode(Node):

    def __init__(self):

        super().__init__("apriltag_pose")

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        self.declare_parameter("cal_file", "")
        self.declare_parameter("tag_id", -1)
        self.declare_parameter("heading_offset_deg", 0.0)

        self.declare_parameter("camera_height_m", 0.0)
        self.declare_parameter("tag_height_m", 0.0)

        self.declare_parameter("foot_x", 0.0)
        self.declare_parameter("foot_y", 0.0)

        self.declare_parameter("world_frame", "world")

        cal_file = self.get_parameter(
            "cal_file"
        ).value

        if not cal_file:
            raise RuntimeError(
                "Parameter 'cal_file' is required."
            )

        # ----------------------------------------------------
        # Load calibration
        # ----------------------------------------------------

        self.H, _ = load_calibration(cal_file)

        self.tag_id = int(
            self.get_parameter("tag_id").value
        )

        self.heading_offset = math.radians(
            float(
                self.get_parameter(
                    "heading_offset_deg"
                ).value
            )
        )

        self.cam_h = float(
            self.get_parameter(
                "camera_height_m"
            ).value
        )

        self.tag_h = float(
            self.get_parameter(
                "tag_height_m"
            ).value
        )

        self.foot = (
            float(
                self.get_parameter("foot_x").value
            ),
            float(
                self.get_parameter("foot_y").value
            )
        )

        self.world_frame = str(
            self.get_parameter(
                "world_frame"
            ).value
        )

        # ----------------------------------------------------
        # AprilTag detector
        # ----------------------------------------------------

        self.detect = make_detector()

        # Image conversion.
        self.bridge = CvBridge()

        # CLAHE for uneven or dim lighting.
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        self._warned_encoding = False

        # ----------------------------------------------------
        # Publishers
        # ----------------------------------------------------

        self.pose_pub = self.create_publisher(
            PoseStamped,
            "robot_pose",
            10
        )

        self.debug_pub = self.create_publisher(
            Image,
            "apriltag/debug_image",
            1
        )

        # ----------------------------------------------------
        # Subscriber
        # ----------------------------------------------------

        self.image_sub = self.create_subscription(
            Image,
            "camera/image_raw",
            self.on_image,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            "AprilTag pose node started. "
            f"Looking for tag36h11 ID="
            f"{'any' if self.tag_id < 0 else self.tag_id}"
        )

        self.get_logger().info(
            "CLAHE and original-image fallback enabled."
        )


    # ========================================================
    # Check whether the requested robot tag was detected
    # ========================================================

    def has_target(self, ids):

        if ids is None or len(ids) == 0:
            return False

        detected_ids = ids.flatten()

        # Accept the first available tag if tag_id == -1.
        if self.tag_id < 0:
            return True

        return bool(
            np.any(detected_ids == self.tag_id)
        )


    # ========================================================
    # Select the robot tag
    # ========================================================

    def select_target(self, corners, ids):

        if ids is None or len(ids) == 0:
            return None

        flat_ids = ids.flatten()

        if self.tag_id < 0:
            return 0

        hits = np.where(
            flat_ids == self.tag_id
        )[0]

        if len(hits) == 0:
            return None

        return int(hits[0])


    # ========================================================
    # Image callback
    # ========================================================

    def on_image(self, msg):

        # ----------------------------------------------------
        # Convert ROS image
        # ----------------------------------------------------

        if msg.encoding not in (
            "bgr8",
            "rgb8",
            "mono8"
        ):

            if not self._warned_encoding:
                self.get_logger().error(
                    f"Unsupported image encoding: {msg.encoding}"
                )

                self._warned_encoding = True

            return

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except CvBridgeError as e:

            self.get_logger().error(
                f"Image conversion failed: {e}"
            )

            return

        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Enhance local contrast
        # ----------------------------------------------------

        enhanced = self.clahe.apply(gray)

        # ----------------------------------------------------
        # First attempt: enhanced grayscale
        # ----------------------------------------------------

        corners, ids = self.detect(enhanced)

        detection_method = "CLAHE"

        # ----------------------------------------------------
        # Fallback: original grayscale
        #
        # Retry when the requested robot tag is missing,
        # including when other tags were detected.
        # ----------------------------------------------------

        if not self.has_target(ids):

            corners_original, ids_original = self.detect(
                gray
            )

            if self.has_target(ids_original):

                corners = corners_original
                ids = ids_original

                detection_method = "original"

        # ----------------------------------------------------
        # Prepare debug image only when requested
        # ----------------------------------------------------

        want_debug = (
            self.debug_pub.get_subscription_count() > 0
        )

        debug = frame.copy() if want_debug else None

        # ----------------------------------------------------
        # Find robot tag
        # ----------------------------------------------------

        chosen = self.select_target(
            corners,
            ids
        )

        if chosen is None:

            self.get_logger().warn(
                "Robot tag not visible",
                throttle_duration_sec=2.0
            )

            if want_debug:

                cv2.putText(
                    debug,
                    "ROBOT TAG NOT DETECTED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                self._publish_debug(
                    debug,
                    msg
                )

            # Do not publish a fabricated or stale pose.
            return

        # ----------------------------------------------------
        # Calculate pose
        # ----------------------------------------------------

        c = corners[chosen].reshape(
            4, 2
        )

        try:

            x, y, yaw = compute_pose(
                c,
                self.H,
                self.heading_offset,
                self.cam_h,
                self.tag_h,
                self.foot
            )

        except (cv2.error, ValueError) as e:

            self.get_logger().error(
                f"Pose calculation failed: {e}"
            )

            return

        # ----------------------------------------------------
        # Publish PoseStamped
        # ----------------------------------------------------

        out = PoseStamped()

        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.world_frame

        out.pose.position.x = x
        out.pose.position.y = y
        out.pose.position.z = 0.0

        # Yaw-only quaternion.
        out.pose.orientation.x = 0.0
        out.pose.orientation.y = 0.0

        out.pose.orientation.z = math.sin(
            yaw / 2.0
        )

        out.pose.orientation.w = math.cos(
            yaw / 2.0
        )

        self.pose_pub.publish(out)

        # ----------------------------------------------------
        # Log pose
        # ----------------------------------------------------

        self.get_logger().info(
            f"x={x:.3f} "
            f"y={y:.3f} "
            f"yaw={math.degrees(yaw):.1f} deg "
            f"method={detection_method}",
            throttle_duration_sec=1.0
        )

        # ----------------------------------------------------
        # Publish annotated debug image
        # ----------------------------------------------------

        if want_debug:

            pts = c.astype(
                np.int32
            )

            # Draw tag outline.
            cv2.polylines(
                debug,
                [pts],
                True,
                (0, 255, 0),
                2
            )

            # Tag centre.
            center = tuple(
                int(v)
                for v in c.mean(axis=0)
            )

            # Robot front direction.
            front = tuple(
                int(v)
                for v in (
                    (c[0] + c[1]) / 2.0
                )
            )

            cv2.arrowedLine(
                debug,
                center,
                front,
                (0, 0, 255),
                3,
                tipLength=0.4
            )

            cv2.putText(
                debug,
                (
                    f"{x:.2f}, {y:.2f} "
                    f"{math.degrees(yaw):.0f} deg"
                ),
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                debug,
                f"Detection: {detection_method}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            self._publish_debug(
                debug,
                msg
            )


    # ========================================================
    # Publish debug image
    # ========================================================

    def _publish_debug(self, img, src_msg):

        out = self.bridge.cv2_to_imgmsg(
            img,
            encoding="bgr8"
        )

        out.header = src_msg.header

        self.debug_pub.publish(out)


# ============================================================
# Main
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = None

    try:

        node = AprilTagPoseNode()

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    except Exception as e:

        if node is not None:
            node.get_logger().fatal(
                f"Node failed: {e}"
            )
        else:
            print(f"AprilTag pose node failed: {e}")

        raise

    finally:

        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()