#!/usr/bin/env python3

import math, os, time, threading, hashlib, json
from typing import Dict, List
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_srvs.srv import Trigger
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
from vision_msgs.msg import Detection3DArray

_PALETTE = [
    (0, 128, 255),  # orange
    (0, 200, 0),  # green
    (255, 0, 0),  # blue
    (200, 0, 200),  # purple
    (0, 220, 220),  # yellow-ish
    (220, 120, 0),  # teal
    (0, 0, 255),  # red
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (100, 100, 255),  # light red
]


def color_for_class(class_id: str) -> tuple:
    h = int(hashlib.sha1(class_id.encode("utf-8")).hexdigest(), 16)
    return _PALETTE[h % len(_PALETTE)]


class RenderTopBoxesService(Node):
    def __init__(self):
        super().__init__("render_bbox_occupancy_grid_service")

        self.declare_parameter("output_dir", "/tmp")
        self.output_dir = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.map_service_name = "/nav2/map_server/map"
        self.dets_topic = "/detections_3d"
        self.service_name = "/render_bbox_occupancy_grid"

        self.cb_srv = MutuallyExclusiveCallbackGroup()
        self.cb_cli = MutuallyExclusiveCallbackGroup()

        qos = QoSProfile(depth=50)
        self.sub = self.create_subscription(
            Detection3DArray, self.dets_topic, self._on_dets, qos
        )
        self.srv = self.create_service(
            Trigger, self.service_name, self._handle_render, callback_group=self.cb_srv
        )
        self.map_client = self.create_client(
            GetMap, self.map_service_name, callback_group=self.cb_cli
        )

        self.accum: Dict[str, List[Dict]] = {}
        self._lock = threading.Lock()

        self.get_logger().info(
            f"Subscribed to {self.dets_topic}; service {self.service_name} ready."
        )

    def _on_dets(self, msg: Detection3DArray):
        if msg.header.frame_id and msg.header.frame_id != "map":
            self.get_logger().warn_once(
                f"Detections frame_id '{msg.header.frame_id}' != 'map'; ignoring this message."
            )
            return

        stamp_ns = (int(msg.header.stamp.sec) * 1_000_000_000) + int(
            msg.header.stamp.nanosec
        )

        with self._lock:
            for det in msg.detections:
                if not det.results:
                    continue
                best = max(
                    det.results, key=lambda r: getattr(r.hypothesis, "score", 0.0)
                )
                class_id = str(getattr(best.hypothesis, "class_id", ""))
                score = float(getattr(best.hypothesis, "score", 0.0))
                if not class_id or score <= 0.0:
                    continue

                cx = float(det.bbox.center.position.x)
                cy = float(det.bbox.center.position.y)
                dx = float(det.bbox.size.x)
                dy = float(det.bbox.size.y)
                if dx <= 0.0 or dy <= 0.0:
                    continue

                bx = det.bbox.center.orientation.x
                by = det.bbox.center.orientation.y
                bz = det.bbox.center.orientation.z
                bw = det.bbox.center.orientation.w
                yaw = math.atan2(
                    2.0 * (bw * bz + bx * by), 1.0 - 2.0 * (by * by + bz * bz)
                )

                self.accum.setdefault(class_id, []).append(
                    {
                        "score": score,
                        "stamp_ns": int(stamp_ns),
                        "cx": float(cx),
                        "cy": float(cy),
                        "dx": float(dx),
                        "dy": float(dy),
                        "yaw": float(yaw),
                    }
                )

    def _handle_render(self, req, resp):
        with self._lock:
            have_any = any(self.accum.values())
        if not have_any:
            resp.success = False
            resp.message = "No detections accumulated."
            return resp

        if not self.map_client.wait_for_service(timeout_sec=2.0):
            resp.success = False
            resp.message = f"Map service '{self.map_service_name}' not available."
            return resp

        result_holder = {"grid": None}
        done_evt = threading.Event()

        def _map_done(fut):
            try:
                res = fut.result()
                result_holder["grid"] = res.map if res is not None else None
            except Exception as e:
                self.get_logger().error(f"GetMap failed: {e}")
                result_holder["grid"] = None
            finally:
                done_evt.set()

        self.map_client.call_async(GetMap.Request()).add_done_callback(_map_done)

        deadline = time.time() + 5.0
        while not done_evt.is_set() and rclpy.ok() and time.time() < deadline:
            time.sleep(0.01)

        grid = result_holder["grid"]
        if grid is None:
            resp.success = False
            resp.message = "Map service call failed or timed out."
            return resp

        h, w = int(grid.info.height), int(grid.info.width)
        resm = float(grid.info.resolution)
        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)
        qx = grid.info.origin.orientation.x
        qy = grid.info.origin.orientation.y
        qz = grid.info.origin.orientation.z
        qw = grid.info.origin.orientation.w
        oyaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        img = self._grid_to_bgr(grid)

        with self._lock:
            chosen = []
            for cid, cands in self.accum.items():
                if not cands:
                    continue
                best = max(cands, key=lambda c: (c["score"], c["stamp_ns"]))
                chosen.append((cid, best))

        if not chosen:
            resp.success = False
            resp.message = "No valid boxes after filtering."
            return resp

        polys_px, scores, class_ids = [], [], []
        best_by_class: Dict[str, Dict] = {}
        for cid, c in chosen:
            world_poly = self._top_corners_xy(
                c["cx"], c["cy"], c["dx"], c["dy"], c["yaw"]
            )
            poly_px = self._world_poly_to_img_px(world_poly, ox, oy, oyaw, resm, h)
            polys_px.append(poly_px)
            scores.append(c["score"])
            class_ids.append(cid)
            best_by_class[cid] = c

        order = np.argsort(-np.array(scores))
        accum_mask = np.zeros((h, w), dtype=np.uint8)
        overlay = img.copy()
        kept_info = []

        for idx in order:
            p = polys_px[idx]
            tmp = np.zeros_like(accum_mask)
            cv2.fillPoly(tmp, [p], 255)
            if np.any(cv2.bitwise_and(tmp, accum_mask)):
                continue
            color = color_for_class(class_ids[idx])
            cv2.fillPoly(overlay, [p], color)
            cv2.polylines(overlay, [p], True, (0, 0, 0), 2)
            accum_mask = cv2.bitwise_or(accum_mask, tmp)
            kept_info.append((p, class_ids[idx]))

        out = cv2.addWeighted(overlay, 0.65, img, 0.35, 0.0)

        H, W = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale = max(0.5, min(1.5, 0.0006 * max(W, H)))
        thickness = 1

        for poly, cid in kept_info:
            cx, cy = self._poly_centroid(poly)
            if cx is None:
                continue
            cx = int(max(0, min(W - 1, cx)))
            cy = int(max(0, min(H - 1, cy)))
            color = color_for_class(cid)
            cv2.putText(
                out,
                cid,
                (cx + 1, cy + 1),
                font,
                base_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                out, cid, (cx, cy), font, base_scale, color, thickness, cv2.LINE_AA
            )

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_png = os.path.join(self.output_dir, f"projected_boxes_{ts}.png")
        try:
            cv2.imwrite(out_png, out)
        except Exception as e:
            resp.success = False
            resp.message = f"PNG write failed: {e}"
            return resp

        render_json = {
            "timestamp_ns": int(time.time() * 1e9),
            "map": {
                "width": w,
                "height": h,
                "resolution": resm,
                "origin": {"x": ox, "y": oy, "yaw": oyaw},
            },
            "render_png": out_png,
            "boxes": [],
        }
        for _, cid in kept_info:
            c = best_by_class[cid]
            render_json["boxes"].append(
                {
                    "class_id": cid,
                    "score": float(c["score"]),
                    "cx": float(c["cx"]),
                    "cy": float(c["cy"]),
                    "dx": float(c["dx"]),
                    "dy": float(c["dy"]),
                    "yaw": float(c["yaw"]),
                }
            )

        out_json = os.path.join(self.output_dir, f"rendered_boxes_{ts}.json")
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(render_json, f, ensure_ascii=False, indent=2)
        except Exception as e:
            resp.success = False
            resp.message = f"JSON write failed: {e}"
            return resp

        resp.success = True
        kept_classes = [cid for _, cid in kept_info]
        resp.message = f"Saved: {out_png} and {out_json} (kept {len(kept_classes)} classes: {', '.join(kept_classes)})"
        return resp

    def _grid_to_bgr(self, grid: OccupancyGrid) -> np.ndarray:
        h, w = grid.info.height, grid.info.width
        dat = np.array(grid.data, dtype=np.int16).reshape((h, w))
        gray = np.where(
            dat < 0, 127, np.clip(240 - (dat.astype(np.float32) * 2.0), 40, 240)
        ).astype(np.uint8)
        gray = np.flipud(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _top_corners_xy(
        self, cx: float, cy: float, dx: float, dy: float, yaw: float
    ) -> np.ndarray:
        hx, hy = 0.5 * dx, 0.5 * dy
        local = np.array(
            [[+hx, +hy], [+hx, -hy], [-hx, -hy], [-hx, +hy]], dtype=np.float32
        )
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (local @ R.T) + np.array([cx, cy], np.float32)

    def _world_poly_to_img_px(
        self,
        poly_xy: np.ndarray,
        ox: float,
        oy: float,
        oyaw: float,
        resm: float,
        img_h: int,
    ) -> np.ndarray:
        pts = []
        c, s = math.cos(-oyaw), math.sin(-oyaw)
        for xw, yw in poly_xy:
            dx, dy = xw - ox, yw - oy
            gx = c * dx - s * dy
            gy = s * dx + c * dy
            u = int(round(gx / resm))
            v = int(round((img_h - 1) - (gy / resm)))
            pts.append([u, v])
        return np.array(pts, dtype=np.int32)

    def _poly_centroid(self, poly_px: np.ndarray):
        if poly_px is None or len(poly_px) == 0:
            return (None, None)
        M = cv2.moments(poly_px)
        if abs(M["m00"]) < 1e-6:
            cx = int(np.mean(poly_px[:, 0]))
            cy = int(np.mean(poly_px[:, 1]))
            return (cx, cy)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)


def main():
    rclpy.init()
    node = RenderTopBoxesService()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
