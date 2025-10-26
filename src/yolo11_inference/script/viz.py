#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray
from typing import Optional
import time

class Detection2DOverlay(Node):
    def __init__(self):
        super().__init__("detection2d_overlay")

        # ---- Parameters
        self.declare_parameter("image_topic", "/zed/zed_node/rgb/image_rect_color/compressed")
        self.declare_parameter("detection_topic", "/yolo11_inference/detections")
        self.declare_parameter("output_image_topic", "detection2d_overlay")
        self.declare_parameter("show_window", False)
        self.declare_parameter("max_det_age_sec", 0.5)   # how old detections can be
        self.declare_parameter("log_every_n", 30)        # reduce log spam

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        detection_topic = self.get_parameter("detection_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_image_topic").get_parameter_value().string_value
        self.show_window = self.get_parameter("show_window").get_parameter_value().bool_value
        self.max_det_age = float(self.get_parameter("max_det_age_sec").value)
        self.log_every_n = int(self.get_parameter("log_every_n").value)

        # ---- QoS: ZED compressed images are usually BEST_EFFORT; detections often RELIABLE
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        det_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---- State
        self.bridge = CvBridge()
        self.last_det_msg: Optional[Detection2DArray] = None
        self.last_det_walltime: float = 0.0
        self.pub_count = 0
        self.drawn_last = 0

        # ---- Subscriptions
        self.img_sub = self.create_subscription(
            CompressedImage, image_topic, self.image_cb, img_qos
        )
        self.det_sub = self.create_subscription(
            Detection2DArray, detection_topic, self.det_cb, det_qos
        )

        # ---- Publisher
        self.pub_img = self.create_publisher(CompressedImage, self.output_topic, img_qos)

        self.get_logger().info(
            f"Listening:\n  image:      {image_topic}\n  detections: {detection_topic}\n"
            f"Publishing overlay: {self.output_topic}\n"
            f"show_window={self.show_window}, max_det_age_sec={self.max_det_age}"
        )

    # ---------- Callbacks
    def det_cb(self, det_msg: Detection2DArray):
        self.last_det_msg = det_msg
        self.last_det_walltime = time.time()

    def image_cb(self, img_msg: CompressedImage):
        # Convert image
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge compressed->cv2 failed: {e}")
            return

        overlay = frame.copy()

        # Draw if we have recent detections
        num_drawn = 0
        if self.last_det_msg is not None and (time.time() - self.last_det_walltime) <= self.max_det_age:
            num_drawn = self.draw_detections(overlay, self.last_det_msg)

        # Publish annotated image
        try:
            out_msg = self.bridge.cv2_to_compressed_imgmsg(overlay, dst_format="jpg")
        except Exception as e:
            self.get_logger().warn(f"cv2->compressed msg failed: {e}")
            return

        out_msg.header = img_msg.header  # keep original timestamp/frame_id
        self.pub_img.publish(out_msg)
        self.pub_count += 1
        self.drawn_last = num_drawn

        # Throttled log
        if self.pub_count % self.log_every_n == 0:
            self.get_logger().info(f"Published {self.pub_count} frames, last boxes drawn: {num_drawn}")

        # Optional preview
        if self.show_window:
            try:
                cv2.imshow("Detection2D Overlay", overlay)
                cv2.waitKey(1)
            except Exception:
                pass

    # ---------- Drawing
    def draw_detections(self, image, det_array: Detection2DArray) -> int:
        H, W = image.shape[:2]
        drawn = 0
        for det in det_array.detections:
            # bbox center/size -> corners
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y

            x1 = int(round(cx - 0.5 * w))
            y1 = int(round(cy - 0.5 * h))
            x2 = int(round(cx + 0.5 * w))
            y2 = int(round(cy + 0.5 * h))

            # Clamp
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            # Top hypothesis (if any)
            label = "unknown"
            score = None
            if det.results:
                hyp = det.results[0].hypothesis
                label = hyp.class_id if getattr(hyp, "class_id", "") else "id:?"
                score = getattr(hyp, "score", None)

            # Box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Caption
            caption = label
            if score is not None and not math.isnan(score):
                caption += f" {score:.2f}"

            (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tb_x1, tb_y1 = x1, max(0, y1 - th - baseline - 3)
            tb_x2, tb_y2 = x1 + tw + 4, y1
            cv2.rectangle(image, (tb_x1, tb_y1), (tb_x2, tb_y2), (0, 255, 0), thickness=-1)
            cv2.putText(image, caption, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            drawn += 1
        return drawn

    def destroy_node(self):
        if self.show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        return super().destroy_node()

def main():
    rclpy.init()
    node = Detection2DOverlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()