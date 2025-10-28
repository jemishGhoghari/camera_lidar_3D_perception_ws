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

# ---------- helpers ----------
def q2yaw(x,y,z,w):
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))

def ang_diff(a,b):
    d = (a-b+math.pi)%(2*math.pi)-math.pi
    return abs(d)

def circ_mean(angles):
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c) if (s!=0 or c!=0) else 0.0

def rect_corners(cx,cy,yaw,L,W):
    hl,hw=L/2.0,W/2.0
    pts=np.array([[ hl, hw],[ hl,-hw],[-hl,-hw],[-hl, hw]],np.float32)
    c,s=math.cos(yaw),math.sin(yaw)
    R=np.array([[c,-s],[s, c]],np.float32)
    return (pts@R.T)+np.array([cx,cy],np.float32)

def world_to_px(x,y,ox,oy,res,h):
    u=(x-ox)/res; v=(y-oy)/res
    return int(round(u)), int(round(h-v))

def oriented_iou_mapspace(a, b, occ: OccupancyGrid):
    h, w = occ.info.height, occ.info.width
    res = occ.info.resolution
    ox, oy = occ.info.origin.position.x, occ.info.origin.position.y

    def poly(d):
        pts_m = rect_corners(d['cx'], d['cy'], d['yaw'], d['L'], d['W'])
        pts_px = np.array([[world_to_px(x,y,ox,oy,res,h) for x,y in pts_m]], np.int32)
        return pts_px

    pa, pb = poly(a), poly(b)
    A = np.zeros((h,w), np.uint8); B = np.zeros((h,w), np.uint8)
    cv2.fillPoly(A, pa, 255); cv2.fillPoly(B, pb, 255)
    inter = np.logical_and(A,B).sum()
    union = np.logical_or(A,B).sum()
    return float(inter)/float(union) if union else 0.0

# ---------- simple map-space tracker ----------
class Track:
    __slots__ = ('cls','samples','best_conf')
    def __init__(self, cls):
        self.cls = cls
        self.samples = []   # each: {cx,cy,yaw,L,W,conf}
        self.best_conf = 0.0
    def add(self, d):
        self.samples.append(d)
        self.best_conf = max(self.best_conf, d['conf'])
    def representative(self):
        cx = float(np.median([s['cx'] for s in self.samples]))
        cy = float(np.median([s['cy'] for s in self.samples]))
        L  = float(np.median([s['L']  for s in self.samples]))
        W  = float(np.median([s['W']  for s in self.samples]))
        yaw = circ_mean([s['yaw'] for s in self.samples])
        return {'class': self.cls, 'cx': cx, 'cy': cy, 'yaw': yaw, 'L': L, 'W': W, 'conf': self.best_conf}

class SimpleAccumulator:
    def __init__(self, dist_thresh=0.5, yaw_thresh_deg=25.0, size_ratio_thresh=0.4):
        self.tracks_by_class = {}
        self.dist_th = dist_thresh
        self.yaw_th = math.radians(yaw_thresh_deg)
        self.size_rt = size_ratio_thresh

    def _similar_size(self, aL,aW,bL,bW):
        # symmetric relative difference
        def rd(a,b): 
            m = max(a,1e-6)
            return abs(a-b)/m
        return rd(aL,bL)<=self.size_rt and rd(aW,bW)<=self.size_rt

    def update(self, det):  # det: dict with keys cls,cx,cy,yaw,L,W,conf
        cls = det['cls']
        tracks = self.tracks_by_class.setdefault(cls, [])

        # greedy assign to the nearest compatible track
        best_i, best_d = -1, 1e9
        for i, trk in enumerate(tracks):
            rep = trk.representative()
            dxy = math.hypot(det['cx']-rep['cx'], det['cy']-rep['cy'])
            if dxy > self.dist_th: 
                continue
            if ang_diff(det['yaw'], rep['yaw']) > self.yaw_th:
                continue
            if not self._similar_size(det['L'],det['W'],rep['L'],rep['W']):
                continue
            if dxy < best_d:
                best_d, best_i = dxy, i

        if best_i >= 0:
            tracks[best_i].add({'cx':det['cx'],'cy':det['cy'],'yaw':det['yaw'],
                                'L':det['L'],'W':det['W'],'conf':det['conf']})
        else:
            t = Track(cls)
            t.add({'cx':det['cx'],'cy':det['cy'],'yaw':det['yaw'],
                   'L':det['L'],'W':det['W'],'conf':det['conf']})
            tracks.append(t)

    def finals(self):
        out=[]
        for cls, tracks in self.tracks_by_class.items():
            for t in tracks:
                rep = t.representative()
                rep['class'] = cls
                out.append(rep)
        return out

# ---------- main node ----------
class MapExporter(Node):
    def __init__(self):
        super().__init__('occupancy_map_exporter')
        self.declare_parameter('detections_topic','/detections_3d')
        self.declare_parameter('png_path','/workspaces/isaac_ros-dev/image_overlay/office/map_overlay.png')
        self.declare_parameter('json_path','/workspaces/isaac_ros-dev/image_overlay/office/detections.json')
        self.declare_parameter('draw_labels', True)
        self.declare_parameter('line_thickness_px', 2)
        self.declare_parameter('min_confidence', 0.0)
        self.declare_parameter('dist_thresh_m', 0.5)
        self.declare_parameter('yaw_thresh_deg', 25.0)
        self.declare_parameter('size_ratio_thresh', 0.4)

        self.cg = ReentrantCallbackGroup()
        topic = self.get_parameter('detections_topic').value
        self.create_subscription(Detection3DArray, topic, self.cb_dets, 30, callback_group=self.cg)
        self.getmap_client = self.create_client(GetMap, '/nav2/map_server/map', callback_group=self.cg)
        self.create_service(Trigger, 'export_final', self.srv_export, callback_group=self.cg)

        self.acc = SimpleAccumulator(
            dist_thresh=float(self.get_parameter('dist_thresh_m').value),
            yaw_thresh_deg=float(self.get_parameter('yaw_thresh_deg').value),
            size_ratio_thresh=float(self.get_parameter('size_ratio_thresh').value),
        )
        self.timestamps = []  # optional: keep the latest timestamps per class/track if you want

    def cb_dets(self, msg: Detection3DArray):
        min_conf = float(self.get_parameter('min_confidence').value)
        for d in msg.detections:
            L, W = float(d.bbox.size.x), float(d.bbox.size.y)
            if L <= 1e-3 or W <= 1e-3:
                continue
            p: Pose = d.bbox.center
            det = {
                'cls': d.results[0].hypothesis.class_id if d.results else '',
                'conf': float(d.results[0].hypothesis.score) if d.results else 1.0,
                'cx': float(p.position.x),
                'cy': float(p.position.y),
                'yaw': q2yaw(p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w),
                'L': L, 'W': W
            }
            if det['conf'] < min_conf: 
                continue
            self.acc.update(det)

    def srv_export(self, req, res):
        if not self.getmap_client.wait_for_service(timeout_sec=3.0):
            res.success=False; res.message='GetMap not available'; return res
        fut = self.getmap_client.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() is None:
            res.success=False; res.message='GetMap failed'; return res
        occ = fut.result().map

        finals = self.acc.finals()  # one box per object (per class)

        # render
        h, w = occ.info.height, occ.info.width
        res_m = occ.info.resolution
        ox, oy = occ.info.origin.position.x, occ.info.origin.position.y
        grid = np.array(occ.data, np.int16).reshape((h,w))
        img = np.zeros((h,w,3), np.uint8)
        img[grid==0]=(255,255,255); img[grid==100]=(0,0,0); img[grid<0]=(160,160,160)
        draw = bool(self.get_parameter('draw_labels').value)
        thick = int(self.get_parameter('line_thickness_px').value)

        for d in finals:
            pts_m = rect_corners(d['cx'], d['cy'], d['yaw'], d['L'], d['W'])
            pts = np.array([[world_to_px(x,y,ox,oy,res_m,h) for x,y in pts_m]], np.int32)
            cv2.polylines(img, pts, True, (0,0,255), thick)
            if draw:
                txt = f"{d['class']} {d['L']:.2f}x{d['W']:.2f}m"
                cv2.putText(img, txt, tuple(pts[0,0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,20,20), 2, cv2.LINE_AA)
                cv2.putText(img, txt, tuple(pts[0,0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        png_path = self.get_parameter('png_path').value
        json_path = self.get_parameter('json_path').value
        if not cv2.imwrite(png_path, img):
            res.success=False; res.message=f'cv2.imwrite failed: {png_path}'; return res

        out = {'frame_id':'map','detections':[
            {'id':f'obj_{i:03d}','class':d['class'],
             'pose':{'x':d['cx'],'y':d['cy'],'yaw':d['yaw']},
             'dimensions':{'length':d['L'],'width':d['W']},
             'confidence':d['conf']} for i,d in enumerate(finals)
        ]}
        with open(json_path,'w',encoding='utf-8') as f: json.dump(out,f,indent=2)

        res.success=True; res.message='exported'
        return res

def main():
    rclpy.init()
    node = MapExporter()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try: ex.spin()
    finally:
        ex.shutdown(); node.destroy_node(); rclpy.shutdown()

if __name__=='__main__': main()