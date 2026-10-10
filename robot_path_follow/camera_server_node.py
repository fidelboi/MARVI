#!/usr/bin/env python3
"""
camera_server_node.py

Owns the camera, serves the phone drawing page, and bridges it to ROS 2.

Publishes
    /camera/image_raw    sensor_msgs/Image       raw BGR frames (for the AprilTag node)
    /camera/camera_info  sensor_msgs/CameraInfo  width/height only (latched), used by path_mapper
    /pixel_path          nav_msgs/Path           path drawn on the phone, in CAMERA PIXELS
                                                 (pose.position.x = u, pose.position.y = v), latched
    /path_clear          std_msgs/Empty          fired when the user presses CLEAR

Parameters
    camera_index (int, 2)   width (int, 640)   height (int, 480)
    fps (float, 30.0)       host (str, "0.0.0.0")   port (int, 5000)
    camera_frame (str, "camera")   jpeg_quality (int, 80)
"""

import threading
import numpy as np
import cv2
import rclpy
from flask import Flask, Response, jsonify, request
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty

from robot_path_follow.web_page import PAGE

LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class CameraServerNode(Node):

    def __init__(self):
        super().__init__("camera_server")

        # ---------------- parameters ----------------
        self.declare_parameter("camera_index", 2)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 5000)
        self.declare_parameter("camera_frame", "camera")
        self.declare_parameter("jpeg_quality", 80)

        index = self.get_parameter("camera_index").value
        req_w = self.get_parameter("width").value
        req_h = self.get_parameter("height").value
        fps = self.get_parameter("fps").value
        self.frame_id = self.get_parameter("camera_frame").value
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        # ---------------- camera ----------------
        self.camera = cv2.VideoCapture(index)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)

        # Use the resolution the camera ACTUALLY delivers.
        self.cam_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)) or req_w
        self.cam_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) or req_h

        self._frame = None
        self._frame_lock = threading.Lock()
        
        self._debug_frame = None
        self._debug_lock = threading.Lock()

        # ---------------- ROS interfaces ----------------
        self.image_pub = self.create_publisher(Image, "camera/image_raw", qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, "camera/camera_info", LATCHED_QOS)
        self.path_pub = self.create_publisher(Path, "pixel_path", LATCHED_QOS)
        self.clear_pub = self.create_publisher(Empty, "path_clear", 10)
        
        # Subscribe to the obstacle planner's rich visualization
        self.create_subscription(Image, "planner/debug_image", self._debug_image_callback, qos_profile_sensor_data)

        info = CameraInfo()
        info.header.frame_id = self.frame_id
        info.width = self.cam_w
        info.height = self.cam_h
        self.info_pub.publish(info)

        self.create_timer(1.0 / fps, self._grab_frame)

        # ---------------- web server (separate thread) ----------------
        self.flask_app = self._build_flask_app()
        host = self.get_parameter("host").value
        port = int(self.get_parameter("port").value)
        threading.Thread(
            target=lambda: self.flask_app.run(host=host, port=port, threaded=True,
                                              use_reloader=False),
            daemon=True,
        ).start()

        self.get_logger().info(f"Camera {self.cam_w}x{self.cam_h} on index {index}")
        self.get_logger().info(f"Open the phone browser at http://<PC-IP>:{port}")

    # ------------------------------------------------------------
    # Camera -> ROS image + shared frame for the MJPEG stream
    # ------------------------------------------------------------
    def _grab_frame(self):
        ok, frame = self.camera.read()
        if not ok:
            self.get_logger().warn("Camera read failed", throttle_duration_sec=2.0)
            return

        with self._frame_lock:
            self._frame = frame

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = frame.tobytes()
        self.image_pub.publish(msg)

    def _debug_image_callback(self, msg):
        """Receives the rich visualization from the obstacle planner."""
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        with self._debug_lock:
            self._debug_frame = frame

    def _mjpeg_generator(self):
        import time
        last = None
        while True:
            # 1. Try to get the debug frame first
            with self._debug_lock:
                frame_to_stream = self._debug_frame
            
            # 2. Fall back to raw camera frame if debug isn't available yet
            if frame_to_stream is None:
                with self._frame_lock:
                    frame_to_stream = self._frame

            if frame_to_stream is None or frame_to_stream is last:
                time.sleep(0.01)
                continue
            
            last = frame_to_stream

            ok, buf = cv2.imencode(".jpg", frame_to_stream,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    # ------------------------------------------------------------
    # Flask routes
    # ------------------------------------------------------------
    def _build_flask_app(self):
        app = Flask(__name__)

        @app.route("/")
        def index():
            return (PAGE
                    .replace("__CAM_WIDTH__", str(self.cam_w))
                    .replace("__CAM_HEIGHT__", str(self.cam_h)))

        @app.route("/video")
        def video():
            return Response(self._mjpeg_generator(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.route("/go", methods=["POST"])
        def go():
            data = request.get_json(silent=True)
            if not data or not isinstance(data.get("points"), list):
                return 'ERROR: expected {"points": [...]}', 400

            points = []
            try:
                for p in data["points"]:
                    x = min(max(float(p["x"]), 0.0), self.cam_w)
                    y = min(max(float(p["y"]), 0.0), self.cam_h)
                    points.append((x, y))
            except (KeyError, TypeError, ValueError):
                return "ERROR: bad point data", 400

            if len(points) < 2:
                return "ERROR: path needs at least 2 points", 400

            self._publish_pixel_path(points)
            return jsonify({"points": len(points)})

        @app.route("/clear", methods=["POST"])
        def clear():
            self.clear_pub.publish(Empty())
            self.get_logger().info("Path cleared")
            return "OK"

        return app

    def _publish_pixel_path(self, points):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "pixel"
        for x, y in points:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x      # u
            ps.pose.position.y = y      # v
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)
        self.get_logger().info(f"Published pixel path with {len(points)} points")

    def destroy_node(self):
        self.camera.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraServerNode()
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