#!/usr/bin/env python3
import math, json, numpy as np, cv2, rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap


# ---------------- helpers ----------------
def q2yaw(x, y, z, w):
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def circ_mean(angles):
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c) if (s or c) else 0.0


def ang_diff(a, b):
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def rect_corners(cx, cy, yaw, L, W):
    hl, hw = L / 2.0, W / 2.0
    pts = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], np.float32)
    return (pts @ R.T) + np.array([cx, cy], np.float32)


def world_to_px(x, y, ox, oy, res, h):
    u = (x - ox) / res
    v = (y - oy) / res
    return int(round(u)), int(round(h - v))


def oriented_iou_mapspace(a, b, occ: OccupancyGrid):
    h, w = occ.info.height, occ.info.width
    res = occ.info.resolution
    ox, oy = occ.info.origin.position.x, occ.info.origin.position.y

    def poly(d):
        pts_m = rect_corners(d["cx"], d["cy"], d["yaw"], d["L"], d["W"])
        pts_px = np.array(
            [[world_to_px(x, y, ox, oy, res, h) for x, y in pts_m]], np.int32
        )
        return pts_px

    pa, pb = poly(a), poly(b)
    A = np.zeros((h, w), np.uint8)
    B = np.zeros((h, w), np.uint8)
    cv2.fillPoly(A, pa, 255)
    cv2.fillPoly(B, pb, 255)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return float(inter) / float(union) if union else 0.0


# ------------- main node (accumulate raw; cluster at export) -------------
class MapExporter(Node):
    def __init__(self):
        super().__init__("occupancy_map_exporter")

        # Params
        self.declare_parameter("detections_topic", "/detections_3d")
        self.declare_parameter(
            "png_path", "/workspaces/isaac_ros-dev/image_overlay/office/map_overlay.png"
        )
        self.declare_parameter(
            "json_path",
            "/workspaces/isaac_ros-dev/image_overlay/office/detections.json",
        )
        self.declare_parameter("draw_labels", True)
        self.declare_parameter("line_thickness_px", 2)
        self.declare_parameter("min_confidence", 0.0)

        self.declare_parameter("iou_thresh", 0.20)  # merge if IoU >= 0.20
        self.declare_parameter(
            "dist_thresh_m", 0.8
        )  # OR merge if center distance < 0.8 m
        self.declare_parameter("yaw_thresh_deg", 35.0)  # with yaw close
        self.declare_parameter(
            "size_ratio_thresh", 0.6
        )  # and similar size (<=60% rel diff)
        self.declare_parameter(
            "min_samples", 2
        )  # require at least N samples per cluster
        self.declare_parameter("class_aware", True)  # cluster per class

        self.cg = ReentrantCallbackGroup()
        topic = self.get_parameter("detections_topic").value
        self.create_subscription(
            Detection3DArray, topic, self.cb_dets, 50, callback_group=self.cg
        )
        self.getmap_client = self.create_client(
            GetMap, "/nav2/map_server/map", callback_group=self.cg
        )
        self.create_service(
            Trigger, "export_final", self.srv_export, callback_group=self.cg
        )

        # Raw detection buffer (all frames)
        self.raw = []  # dict: {cls, conf, cx, cy, yaw, L, W}

    def cb_dets(self, msg: Detection3DArray):
        min_conf = float(self.get_parameter("min_confidence").value)
        for d in msg.detections:
            L, W = float(d.bbox.size.x), float(d.bbox.size.y)
            if L <= 1e-3 or W <= 1e-3:
                continue
            p: Pose = d.bbox.center
            cls = d.results[0].hypothesis.class_id if d.results else ""
            conf = float(d.results[0].hypothesis.score) if d.results else 1.0
            if conf < min_conf:
                continue
            self.raw.append(
                {
                    "cls": cls,
                    "conf": conf,
                    "cx": float(p.position.x),
                    "cy": float(p.position.y),
                    "yaw": q2yaw(
                        p.orientation.x,
                        p.orientation.y,
                        p.orientation.z,
                        p.orientation.w,
                    ),
                    "L": L,
                    "W": W,
                }
            )

    # --------- export: fetch map once, cluster, render one box per object ----------
    def srv_export(self, req, res):
        if not self.getmap_client.wait_for_service(timeout_sec=3.0):
            res.success = False
            res.message = "GetMap not available"
            return res
        fut = self.getmap_client.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() is None:
            res.success = False
            res.message = "GetMap failed"
            return res
        occ = fut.result().map

        if not self.raw:
            res.success = False
            res.message = "No detections accumulated"
            return res

        finals = self.cluster_greedy(self.raw, occ)

        # Render overlay
        h, w = occ.info.height, occ.info.width
        res_m = occ.info.resolution
        ox, oy = occ.info.origin.position.x, occ.info.origin.position.y
        grid = np.array(occ.data, np.int16).reshape((h, w))
        img = np.zeros((h, w, 3), np.uint8)
        img[grid == 0] = (255, 255, 255)
        img[grid == 100] = (0, 0, 0)
        img[grid < 0] = (160, 160, 160)
        draw = bool(self.get_parameter("draw_labels").value)
        thick = int(self.get_parameter("line_thickness_px").value)

        for d in finals:
            pts_m = rect_corners(d["cx"], d["cy"], d["yaw"], d["L"], d["W"])
            pts = np.array(
                [[world_to_px(x, y, ox, oy, res_m, h) for x, y in pts_m]], np.int32
            )
            cv2.polylines(img, pts, True, (0, 0, 255), thick)
            if draw:
                txt = f"{d['cls']} {d['L']:.2f}x{d['W']:.2f}m"
                cv2.putText(
                    img,
                    txt,
                    tuple(pts[0, 0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    txt,
                    tuple(pts[0, 0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        png_path = self.get_parameter("png_path").value
        json_path = self.get_parameter("json_path").value
        if not cv2.imwrite(png_path, img):
            res.success = False
            res.message = f"cv2.imwrite failed: {png_path}"
            return res

        out = {
            "frame_id": "map",
            "detections": [
                {
                    "id": f"obj_{i:03d}",
                    "class": d["cls"],
                    "pose": {"x": d["cx"], "y": d["cy"], "yaw": d["yaw"]},
                    "dimensions": {"length": d["L"], "width": d["W"]},
                    "confidence": d["conf"],
                    "count": d["count"],
                }
                for i, d in enumerate(finals)
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        res.success = True
        res.message = "exported"
        return res

    # --------- greedy IoU + distance clustering (export-time) ----------
    def cluster_greedy(self, raw, occ):
        iou_th = float(self.get_parameter("iou_thresh").value)
        dist_th = float(self.get_parameter("dist_thresh_m").value)
        yaw_th = math.radians(float(self.get_parameter("yaw_thresh_deg").value))
        size_rt = float(self.get_parameter("size_ratio_thresh").value)
        min_samples = int(self.get_parameter("min_samples").value)
        class_aware = bool(self.get_parameter("class_aware").value)

        # Optionally split per class
        groups = {}
        if class_aware:
            for d in raw:
                groups.setdefault(d["cls"], []).append(d)
        else:
            groups["__all__"] = list(raw)

        finals = []
        for cls, dets in groups.items():
            dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
            used = [False] * len(dets)

            for i, base in enumerate(dets):
                if used[i]:
                    continue
                cluster = [base]
                used[i] = True

                for j in range(i + 1, len(dets)):
                    if used[j]:
                        continue
                    cand = dets[j]

                    # quick gates
                    dxy = math.hypot(cand["cx"] - base["cx"], cand["cy"] - base["cy"])
                    yaw_ok = ang_diff(cand["yaw"], base["yaw"]) <= yaw_th
                    size_ok = (
                        abs(cand["L"] - base["L"]) / max(base["L"], 1e-6) <= size_rt
                    ) and (abs(cand["W"] - base["W"]) / max(base["W"], 1e-6) <= size_rt)

                    merge = False
                    if dxy <= dist_th and yaw_ok and size_ok:
                        merge = True
                    else:
                        # slower but robust IoU check (rasterized at map resolution)
                        try:
                            iou = oriented_iou_mapspace(
                                _mk_det(base), _mk_det(cand), occ
                            )
                            if iou >= iou_th:
                                merge = True
                        except Exception:
                            pass

                    if merge:
                        cluster.append(cand)
                        used[j] = True

                # representative from cluster
                cx = float(np.median([d["cx"] for d in cluster]))
                cy = float(np.median([d["cy"] for d in cluster]))
                L = float(np.median([d["L"] for d in cluster]))
                W = float(np.median([d["W"] for d in cluster]))
                yaw = circ_mean([d["yaw"] for d in cluster])
                conf = max(d["conf"] for d in cluster)
                count = len(cluster)

                if count >= min_samples:
                    finals.append(
                        {
                            "cls": cls if class_aware else cluster[0]["cls"],
                            "cx": cx,
                            "cy": cy,
                            "yaw": yaw,
                            "L": L,
                            "W": W,
                            "conf": conf,
                            "count": count,
                        }
                    )
                else:
                    # keep singletons too (optional). Comment out next two lines to drop small clusters.
                    finals.append(
                        {
                            "cls": cls if class_aware else cluster[0]["cls"],
                            "cx": cx,
                            "cy": cy,
                            "yaw": yaw,
                            "L": L,
                            "W": W,
                            "conf": conf,
                            "count": count,
                        }
                    )

        # final pass: oriented-NMS to be extra safe
        finals = sorted(finals, key=lambda x: x["conf"], reverse=True)
        kept = []
        for d in finals:
            ok = True
            for k in kept:
                if d["cls"] != k["cls"] and class_aware:
                    continue
                try:
                    if oriented_iou_mapspace(_mk_det(d), _mk_det(k), occ) > max(
                        0.5, iou_th
                    ):
                        ok = False
                        break
                except Exception:
                    pass
            if ok:
                kept.append(d)
        return kept


def _mk_det(d):
    # small helper to map dict fields to names expected by IoU function
    return {"cx": d["cx"], "cy": d["cy"], "yaw": d["yaw"], "L": d["L"], "W": d["W"]}


def main():
    rclpy.init()
    node = MapExporter()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
