#!/usr/bin/env python3
import os
import json
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import tf2_ros
import tf_transformations as tf_trans
from ament_index_python.packages import get_package_share_directory
import message_filters

class CameraParams:
    def __init__(self, data, world_frame, frame_prefix, index):
        # data: dict loaded from camera-params.json
        # Intrinsics
        K_list = data.get('intrinsic_matrix')
        if K_list is None:
            raise ValueError(f"[Camera {index}] Missing intrinsic_matrix")
        self.K = np.array(K_list, dtype=float)
        if self.K.shape != (3,3):
            raise ValueError(f"[Camera {index}] intrinsic_matrix must be 3x3")
        dist = data.get('distortion_coef', [])
        self.dist_coef = np.array(dist, dtype=float).reshape(-1)
        # Precompute approximate FOV limits
        fx = self.K[0,0]; fy = self.K[1,1]
        cx = self.K[0,2]; cy = self.K[1,2]
        self.half_fov_x = cx / fx if fx != 0 else np.inf
        self.half_fov_y = cy / fy if fy != 0 else np.inf
        # Extrinsics
        # Use explicit rotation matrix if provided
        if 'R' in data and data['R'] is not None:
            R_cw = np.array(data['R'], dtype=float)
            if R_cw.shape != (3,3):
                raise ValueError(f"[Camera {index}] Provided R must be 3x3")
        else:
            # rotation_rpy or rotation field
            rpy = data.get('rotation', data.get('rotation_rpy', [0.0, 0.0, 0.0]))
            roll, pitch, yaw = rpy
            angs = np.array([roll, pitch, yaw], dtype=float)
            if np.any(np.abs(angs) > 2*np.pi):
                roll, pitch, yaw = np.deg2rad(angs)
            R_cw = tf_trans.euler_matrix(roll, pitch, yaw, axes='sxyz')[:3, :3]
        self.R_cw = R_cw
        t = data.get('t', data.get('translation'))
        if t is None:
            raise ValueError(f"[Camera {index}] Missing translation 't' or 'translation'")
        self.C = np.array(t, dtype=float).reshape(3)
        # world->camera
        self.R_wc = self.R_cw.T
        self.t_wc = -self.R_wc.dot(self.C)
        # Frame and UV topic
        self.frame_id = data.get('frame_id', f"{frame_prefix}{index}")
        # uv_topic field in JSON
        self.uv_topic = data.get('uv_topic')
        if self.uv_topic is None:
            # derive from camera name if provided
            cam_name = data.get('camera_name')
            if cam_name:
                self.uv_topic = f"/{cam_name}/point_uv"
            else:
                raise ValueError(f"[Camera {index}] uv_topic missing in JSON entry and cannot derive")
        print(f"[CameraParams] Loaded camera {index}: frame_id={self.frame_id}, uv_topic={self.uv_topic}")

    def project(self, X_w):
        X_c = self.R_wc.dot(X_w) + self.t_wc
        x, y, z = X_c
        if z <= 0:
            return np.array([np.nan, np.nan])
        u = self.K[0,0] * (x / z) + self.K[0,2]
        v = self.K[1,1] * (y / z) + self.K[1,2]
        return np.array([u, v])

    def broadcast_static(self, broadcaster, node):
        ts = TransformStamped()
        ts.header.stamp = node.get_clock().now().to_msg()
        ts.header.frame_id = node.get_parameter('world_frame').value
        ts.child_frame_id = self.frame_id
        mat = np.eye(4)
        mat[:3, :3] = self.R_cw
        mat[:3, 3] = self.C
        quat = tf_trans.quaternion_from_matrix(mat)
        ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = self.C
        ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z, ts.transform.rotation.w = quat
        broadcaster.sendTransform(ts)
        print(f"[CameraParams] Broadcast transform for frame {self.frame_id}")

class UVBuffer:
    def __init__(self): self.data = None
    def update(self, msg):
        self.data = (msg.point.x, msg.point.y,
                     msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        print(f"[UVBuffer] {msg._topic_name if hasattr(msg,'_topic_name') else ''} UV: ({self.data[0]:.1f}, {self.data[1]:.1f})")

class Triangulation:
    def __init__(self, reproj_thresh, min_rays, room_bounds=None):
        self.reproj_thresh = reproj_thresh
        self.min_rays = min_rays
        self.room_bounds = room_bounds if room_bounds is not None else [-5.0,5.0,-5.0,5.0]

    def backproject(self, cam, uv):
        uv_arr = np.array([[uv]], dtype=float)
        uv_n = cv2.undistortPoints(uv_arr, cam.K, cam.dist_coef).reshape(2)
        x_n, y_n = uv_n
        if abs(x_n) > cam.half_fov_x or abs(y_n) > cam.half_fov_y:
            print(f"[Triangulation] UV {uv} outside FOV for {cam.frame_id}")
            return None, None
        d_c = np.array([x_n, y_n, 1.0])
        d_c /= np.linalg.norm(d_c)
        o_w = cam.C
        d_w = cam.R_cw.dot(d_c)
        return o_w, d_w

    @staticmethod
    def _solve_ls(ows, dws):
        A = np.zeros((3,3)); b = np.zeros(3)
        for o, d in zip(ows, dws):
            Dm = np.eye(3) - np.outer(d, d)
            A += Dm; b += Dm.dot(o)
        cond = np.linalg.cond(A)
        if cond > 1e6:
            print(f"[Triangulation] Ill-conditioned A (cond={cond:.2e})")
        X = np.linalg.solve(A, b)
        print(f"[Triangulation] LS solution: {X}")
        return X

    def intersect_ground(self, cam, uv):
        uv_arr = np.array([[uv]], dtype=float)
        uv_n = cv2.undistortPoints(uv_arr, cam.K, cam.dist_coef).reshape(2)
        x_n, y_n = uv_n
        if abs(x_n) > cam.half_fov_x or abs(y_n) > cam.half_fov_y:
            return None
        d_c = np.array([x_n, y_n, 1.0])
        d_w = cam.R_cw.dot(d_c)
        C = cam.C
        if abs(d_w[2]) < 1e-6:
            return None
        t = -C[2] / d_w[2]
        if t <= 0:
            return None
        X = C + t * d_w
        xmin, xmax, ymin, ymax = self.room_bounds
        if not (xmin <= X[0] <= xmax and ymin <= X[1] <= ymax):
            print(f"[Triangulation] Ground intersection {X} outside bounds")
            return None
        return X

    def triangulate_ls(self, cam_buf_pairs):
        ows, dws, uvs, cams_list = [], [], [], []
        for cam, buf in cam_buf_pairs:
            if buf.data is None:
                continue
            uv = (buf.data[0], buf.data[1])
            o, d = self.backproject(cam, uv)
            if o is None:
                continue
            ows.append(o); dws.append(d); uvs.append(uv); cams_list.append(cam)
        if len(ows) < self.min_rays:
            print("[Triangulation] Not enough valid rays for LS")
            return None, []
        X = self._solve_ls(ows, dws)
        errs = []
        for uv, cam in zip(uvs, cams_list):
            uvp = cam.project(X)
            if np.any(np.isnan(uvp)):
                errs.append(np.inf)
            else:
                errs.append(np.linalg.norm(np.array(uv)-uvp))
        print(f"[Triangulation] reproj errors: {errs}")
        if len(ows) > self.min_rays and max(errs) > self.reproj_thresh:
            idx = int(np.argmax(errs))
            print(f"[Triangulation] Removing outlier ray idx {idx}")
            del ows[idx]; del dws[idx]; del uvs[idx]; del cams_list[idx]
            if len(ows) < self.min_rays:
                print("[Triangulation] Insufficient rays after removal")
                return None, []
            X = self._solve_ls(ows, dws)
            errs = []
            for uv, cam in zip(uvs, cams_list):
                uvp = cam.project(X)
                if np.any(np.isnan(uvp)):
                    errs.append(np.inf)
                else:
                    errs.append(np.linalg.norm(np.array(uv)-uvp))
            print(f"[Triangulation] reproj errors after removal: {errs}")
        return X, errs

    def trilaterate(self, cam_buf_pairs):
        pts, cams_g, uvs_g = [], [], []
        for cam, buf in cam_buf_pairs:
            if buf.data is None:
                continue
            uv = (buf.data[0], buf.data[1])
            Xg = self.intersect_ground(cam, uv)
            if Xg is not None:
                pts.append(Xg); cams_g.append(cam); uvs_g.append(uv)
        if len(pts) >= self.min_rays:
            pts_arr = np.vstack(pts)
            X_est = np.mean(pts_arr, axis=0)
            errs = []
            for cam, uv in zip(cams_g, uvs_g):
                uvp = cam.project(X_est)
                if np.any(np.isnan(uvp)):
                    errs.append(np.inf)
                else:
                    errs.append(np.linalg.norm(np.array(uv)-uvp))
            print(f"[Triangulation] ground X: {X_est}, errs: {errs}")
            if np.nanmax(errs) < self.reproj_thresh:
                return X_est, errs
            else:
                print("[Triangulation] Ground-plane error high, fallback LS")
        return self.triangulate_ls(cam_buf_pairs)

class Triangulator(Node):
    def __init__(self):
        super().__init__('triangulation_node')
        # Load JSON camera params
        json_file = self.declare_parameter('camera_params_file',
                                          os.path.join(
                                              get_package_share_directory('thermal_3d_localization'),
                                              'config', 'camera-params.json')).value
        wf = self.declare_parameter('world_frame', 'map').value
        rth = self.declare_parameter('reproj_error_thresh', 5.0).value
        sw = self.declare_parameter('sync_window', 0.05).value
        mr = self.declare_parameter('min_rays', 3).value
        room_bounds = self.declare_parameter('room_bounds', [-5.0,5.0,-5.0,5.0]).value
        pref = self.declare_parameter('camera_frame_prefix', 'thermal_cam_').value
        # Load JSON
        with open(json_file, 'r') as f:
            cams_json = json.load(f)
        if not isinstance(cams_json, list) or len(cams_json) == 0:
            raise RuntimeError('camera-params.json must be a non-empty list')
        self.cams = []
        for i, data in enumerate(cams_json):
            cam = CameraParams(data, wf, pref, i+1)
            self.cams.append(cam)
        # Broadcast transforms
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        for cam in self.cams:
            cam.broadcast_static(self.tf_broadcaster, self)
        # UVBuffers and subscriptions
        self.bufs = []
        for cam in self.cams:
            buf = UVBuffer(); self.bufs.append(buf)
            topic = cam.uv_topic
            sub = message_filters.Subscriber(self, PointStamped, topic)
            sub.registerCallback(lambda msg, idx=len(self.bufs)-1: self.bufs[idx].update(msg))
        # Triangulation
        self.tri = Triangulation(rth, mr, room_bounds)
        self.pub = self.create_publisher(PointStamped, '/thermal_target/position', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'triangulation_markers', 10)
        self.create_timer(sw, self.timer_callback)

    def timer_callback(self):
        sw = self.get_parameter('sync_window').value
        mr = self.get_parameter('min_rays').value
        wf = self.get_parameter('world_frame').value
        stamps = [b.data[2] for b in self.bufs if b.data]
        if len(stamps) < mr:
            return
        tmax = max(stamps)
        idxs = [i for i, b in enumerate(self.bufs) if b.data and abs(b.data[2] - tmax) <= sw]
        if len(idxs) < mr:
            return
        pairs = [(self.cams[i], self.bufs[i]) for i in idxs]
        X, errs = self.tri.trilaterate(pairs)
        if X is None:
            self.get_logger().warn('Not enough valid rays or out-of-bounds')
            return
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = wf
        pt.point.x, pt.point.y, pt.point.z = float(X[0]), float(X[1]), float(X[2])
        self.pub.publish(pt)
        # Markers
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()
        for idx, cam in enumerate(self.cams):
            m = Marker(); m.header.frame_id = wf; m.header.stamp = now
            m.ns='cameras'; m.id=idx; m.type=Marker.SPHERE; m.action=Marker.ADD
            m.pose.position.x=float(cam.C[0]); m.pose.position.y=float(cam.C[1]); m.pose.position.z=float(cam.C[2])
            m.scale.x=m.scale.y=m.scale.z=0.05; m.color=ColorRGBA(r=1.0,g=0.0,b=0.0,a=1.0)
            markers.markers.append(m)
        for idx in idxs:
            cam=self.cams[idx]; buf=self.bufs[idx]
            o,d=self.tri.backproject(cam,(buf.data[0],buf.data[1]))
            if o is None: continue
            m=Marker(); m.header.frame_id=wf; m.header.stamp=now
            m.ns='rays'; m.id=idx; m.type=Marker.LINE_STRIP; m.action=Marker.ADD
            p0=cam.C.tolist(); p1=(cam.C + d*2.0).tolist()
            m.points=[Point(x=float(p0[0]),y=float(p0[1]),z=float(p0[2])), Point(x=float(p1[0]),y=float(p1[1]),z=float(p1[2]))]
            m.scale.x=0.01; m.color=ColorRGBA(r=0.0,g=1.0,b=0.0,a=1.0)
            markers.markers.append(m)
        m=Marker(); m.header.frame_id=wf; m.header.stamp=now
        m.ns='triangulated'; m.id=0; m.type=Marker.SPHERE; m.action=Marker.ADD
        m.pose.position.x=float(X[0]); m.pose.position.y=float(X[1]); m.pose.position.z=float(X[2])
        m.scale.x=m.scale.y=m.scale.z=0.1; m.color=ColorRGBA(r=0.0,g=0.0,b=1.0,a=1.0)
        markers.markers.append(m)
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = Triangulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
