#!/usr/bin/env python3

import math
import json
from typing import List, Tuple, Optional
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from nav_msgs.msg import OccupancyGrid
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger
from nav_msgs.srv import GetMap

def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (rotation about Z) from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def world_to_map_px(x: float, y: float, origin_x: float, origin_y: float, resolution: float) -> Tuple[float, float]:
    """Map world coords (meters in map frame) to image pixels (u, v)."""
    u = (x - origin_x) / resolution
    v = (y - origin_y) / resolution
    return u, v

def oriented_rect_corners(cx: float, cy: float, yaw: float, length: float, width: float) -> np.ndarray:
    """
    Return 4x2 array of rectangle corners (meters) centered at (cx, cy) with heading yaw.
    length along local +x, width along local +y. Order: FL, FR, RR, RL.
    """
    hl = 0.5 * length
    hw = 0.5 * width
    local = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw]
    ], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)
    return (local @ R.T) + np.array([cx, cy], dtype=np.float32)

class MapObbFinalExporter(Node):
    def __init__(self):
        super().__init__('occupancy_map_exporter')

        self.declare_parameter('map_topic', '/nav2/map')
        self.declare_parameter('detections_topic', '/detections_3d')
        self.declare_parameter('png_path', '/workspaces/isaac_ros-dev/image_overlay/office/map_overlay.png')
        self.declare_parameter('json_path', '/workspaces/isaac_ros-dev/image_overlay/office/detections.json')
        self.declare_parameter('draw_labels', True)
        self.declare_parameter('line_thickness_px', 2)
        self.declare_parameter('nms_iou_thresh', 0.10)
        self.declare_parameter('min_confidence', 0.0)

        map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        det_topic = self.get_parameter('detections_topic').get_parameter_value().string_value

        self.map_msg: Optional[OccupancyGrid] = None
        self.accum: List[dict] = []  # each: {class, confidence, cx, cy, yaw, length, width, timestamp}

        # self.create_subscription(OccupancyGrid, map_topic, self.map_callback, 10)
        self.create_subscription(Detection3DArray, det_topic, self.detections_callback, 50)

        self.create_service(Trigger, 'export_final', self.on_export_srv)

    # def map_callback(self, msg: OccupancyGrid):
    #     rclpy.logging.get_logger('MapObbFinalExporter').info('Received OccupancyGrid map.')
    #     self.map_msg = msg

    def detections_callback(self, msg: Detection3DArray):
        min_conf = float(self.get_parameter('min_confidence').value)
        for det in msg.detections:
            try:
                cpose: Pose = det.bbox.center
                cx, cy = float(cpose.position.x), float(cpose.position.y)
                yaw = quat_to_yaw(cpose.orientation.x, cpose.orientation.y, cpose.orientation.z, cpose.orientation.w)
                L = float(det.bbox.size.x)
                W = float(det.bbox.size.y)

                if L <= 1e-3 or W <= 1e-3:
                    continue

                if det.results and len(det.results) > 0:
                    cls = det.results[0].hypothesis.class_id or ''
                    conf = float(det.results[0].hypothesis.score)
                else:
                    cls = ''
                    conf = 1.0

                if conf < min_conf:
                    continue

                ts = float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)

                self.accum.append({
                    'class': cls,
                    'confidence': conf,
                    'cx': cx, 'cy': cy, 'yaw': yaw,
                    'length': L, 'width': W,
                    'timestamp': ts
                })
            except Exception as e:
                self.get_logger().warn(f'Failed to parse one detection: {e}')

    def _poly_px_from_det(self, d: dict, occ: OccupancyGrid) -> np.ndarray:
        w, h = occ.info.width, occ.info.height
        res = occ.info.resolution
        ox, oy = occ.info.origin.position.x, occ.info.origin.position.y

        corners_m = oriented_rect_corners(d['cx'], d['cy'], d['yaw'], d['length'], d['width'])
        pts = []
        for x_m, y_m in corners_m:
            u, v = world_to_map_px(x_m, y_m, ox, oy, res)
            pts.append([int(round(u)), int(round(h - v))])  # flip vertical for image coords
        return np.array([pts], dtype=np.int32)  # (1,4,2)

    def oriented_iou_mapspace(self, a: dict, b: dict, occ: OccupancyGrid) -> float:
        """
        Approximate oriented IoU by rasterizing to occupancy grid resolution.
        Good enough for export-time de-duplication.
        """
        w, h = occ.info.width, occ.info.height
        pa = self._poly_px_from_det(a, occ)
        pb = self._poly_px_from_det(b, occ)

        mask_a = np.zeros((h, w), dtype=np.uint8)
        mask_b = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_a, pa, 255)
        cv2.fillPoly(mask_b, pb, 255)

        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    def nms_oriented_by_class(self, dets: List[dict], occ: OccupancyGrid, iou_thresh: float) -> List[dict]:
        final: List[dict] = []
        classes = {}
        for d in dets:
            classes.setdefault(d['class'], []).append(d)

        for cls, arr in classes.items():
            arr.sort(key=lambda x: x['confidence'], reverse=True)
            kept: List[dict] = []
            for d in arr:
                keep = True
                for k in kept:
                    if self.oriented_iou_mapspace(d, k, occ) > iou_thresh:
                        keep = False
                        break
                if keep:
                    kept.append(d)
            final.extend(kept)
        return final

    def on_export_srv(self, req, res):
        self.get_map_client = self.create_client(GetMap, 'nav2/map_server/map')
        future = self.get_map_client.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, future)

        rclpy.logging.get_logger('MapObbFinalExporter').info('Called map_server/map service.')

        if future.result() is not None:
            self.map_msg = future.result().map
            self.get_logger().info('Successfully retrieved OccupancyGrid map from map_server.')
        else:
            self.get_logger().error('Failed to call map_server/map service')
            res.success = False
            res.message = 'Failed to retrieve map from map_server.'
            return res

        if not self.accum:
            res.success = False
            res.message = 'No detections accumulated.'
            return res

        try:
            iou_th = float(self.get_parameter('nms_iou_thresh').value)
            png_path = self.get_parameter('png_path').get_parameter_value().string_value
            json_path = self.get_parameter('json_path').get_parameter_value().string_value

            final_dets = self.nms_oriented_by_class(self.accum, self.map_msg, iou_th)
            self.render_png(self.map_msg, final_dets, png_path)
            self.write_json(self.map_msg, final_dets, json_path)

            self.get_logger().info(f'Wrote PNG:  {png_path}')
            self.get_logger().info(f'Wrote JSON: {json_path}')
            res.success = True
            res.message = f'Exported {png_path} and {json_path}'
        except Exception as e:
            res.success = False
            res.message = f'Export failed: {e}'
        return res

    def render_png(self, occ: OccupancyGrid, dets: List[dict], path: str):
        w, h = occ.info.width, occ.info.height
        res = occ.info.resolution
        ox = occ.info.origin.position.x
        oy = occ.info.origin.position.y

        data = np.array(occ.data, dtype=np.int16).reshape((h, w))
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[data == 0] = (255, 255, 255)   # free
        img[data == 100] = (0, 0, 0)       # occupied
        img[data < 0] = (160, 160, 160)    # unknown

        draw_labels = bool(self.get_parameter('draw_labels').value)
        thickness = int(self.get_parameter('line_thickness_px').value)

        for d in dets:
            corners_m = oriented_rect_corners(d['cx'], d['cy'], d['yaw'], d['length'], d['width'])
            pts = []
            for x_m, y_m in corners_m:
                u, v = world_to_map_px(x_m, y_m, ox, oy, res)
                pts.append([int(round(u)), int(round(h - v))])
            pts = np.array([pts], dtype=np.int32)

            cv2.polylines(img, pts, isClosed=True, color=(0, 0, 255), thickness=thickness)

            if draw_labels:
                text = f"{d['class']} {d['length']:.2f}x{d['width']:.2f}m"
                anchor = tuple(pts[0, 0, :])
                cv2.putText(img, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
                cv2.putText(img, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if not cv2.imwrite(path, img):
            raise RuntimeError(f'cv2.imwrite failed for {path}')

    def write_json(self, occ: OccupancyGrid, dets: List[dict], path: str):
        out = {
            'frame_id': 'map',
            'resolution': occ.info.resolution,
            'origin': {
                'x': occ.info.origin.position.x,
                'y': occ.info.origin.position.y,
                'yaw': quat_to_yaw(
                    occ.info.origin.orientation.x,
                    occ.info.origin.orientation.y,
                    occ.info.origin.orientation.z,
                    occ.info.origin.orientation.w
                )
            },
            'detections': []
        }
        for i, d in enumerate(dets):
            out['detections'].append({
                'id': f'obj_{i:03d}',
                'class': d['class'],
                'pose': {'x': d['cx'], 'y': d['cy'], 'yaw': d['yaw']},
                'dimensions': {'length': d['length'], 'width': d['width']},
                'confidence': d['confidence'],
                'timestamp': d['timestamp']
            })
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)


def main():
    rclpy.init()
    node = MapObbFinalExporter()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()