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
    """
    Holds intrinsics and extrinsics for one camera, and provides
    methods for backprojection, projection, and TF broadcasting.
    """
    def __init__(self, entry: dict, world_frame: str, frame_prefix: str, index: int, node: Node):
        """
        Initialize camera intrinsics and extrinsics, computing orientation via look-at so that
        the camera’s optical axis (+X in camera frame) points toward the room center (0,0,0).
        """
        self.node = node
        self.index = index

        # 1. Intrinsics
        K_list = entry.get('intrinsic_matrix')
        if K_list is None:
            raise RuntimeError(f"[Camera {index}] Missing 'intrinsic_matrix' in JSON entry")
        K = np.array(K_list, dtype=float)
        if K.shape != (3, 3):
            raise RuntimeError(f"[Camera {index}] 'intrinsic_matrix' must be 3x3, got {K.shape}")
        self.K = K
        dist_list = entry.get('distortion_coef', [])
        self.dist_coef = np.array(dist_list, dtype=float).reshape(-1)
        fx = self.K[0, 0]; fy = self.K[1, 1]
        cx = self.K[0, 2]; cy = self.K[1, 2]
        # For crude FOV check in normalized image plane:
        self.half_fov_x = (cx / fx) if fx != 0 else np.inf
        self.half_fov_y = (cy / fy) if fy != 0 else np.inf
        node.get_logger().info(f"[Camera {index}] Intrinsics loaded: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, dist={self.dist_coef}")

        # 2. Translation (camera center in world)
        t = entry.get('t', entry.get('translation'))
        if t is None or len(t) != 3:
            raise RuntimeError(f"[Camera {index}] Missing or invalid 't' (translation) field")
        C = np.array(t, dtype=float).reshape(3)
        self.C = C

        # 3. Compute extrinsic rotation via look-at to room center (0,0,0)
        L = np.array([0.0, 0.0, 0.0], dtype=float)
        f_vec = L - self.C
        norm_f = np.linalg.norm(f_vec)
        if norm_f < 1e-6:
            node.get_logger().warn(f"[Camera {index}] look-at vector too small; defaulting orientation from entry if provided")
            # Fallback: if JSON provided an 'R', use it; else identity
            if 'R' in entry and entry['R'] is not None:
                R_cw = np.array(entry['R'], dtype=float)
                if R_cw.shape != (3,3):
                    raise RuntimeError(f"[Camera {index}] Provided 'R' must be 3x3")
                node.get_logger().info(f"[Camera {index}] Using provided R since look-at is degenerate")
            else:
                node.get_logger().info(f"[Camera {index}] No valid R provided; using identity rotation")
                R_cw = np.eye(3)
        else:
            f = f_vec / norm_f
            # Choose a world-up vector:
            world_up = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(f, world_up)) > 0.999:
                world_up = np.array([0.0, 1.0, 0.0], dtype=float)
            r = np.cross(world_up, f)
            r_norm = np.linalg.norm(r)
            if r_norm < 1e-6:
                node.get_logger().warn(f"[Camera {index}] computed right vector near zero; picking alternate up")
                world_up = np.array([1.0, 0.0, 0.0], dtype=float)
                r = np.cross(world_up, f)
                r /= np.linalg.norm(r)
            else:
                r /= r_norm
            d = np.cross(f, r)
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-6:
                raise RuntimeError(f"[Camera {index}] failed to compute down vector")
            d /= d_norm
            R_cw = np.column_stack((f, r, d))
            node.get_logger().info(f"[Camera {index}] Look-at R_cw columns: f={f}, r={r}, d={d}")
            node.get_logger().info(f"[Camera {index}] optical axis +X→world={f}, z={f[2]:.3f} (should be <0 when camera above looking down)")

        self.R_cw = R_cw
        # Compute world->camera if needed
        self.R_wc = self.R_cw.T
        self.t_wc = -self.R_wc.dot(self.C)
        node.get_logger().info(f"[Camera {index}] Extrinsics loaded: C = {self.C}, R_cw=\n{self.R_cw}")

        # 4. Frame ID and UV topic
        frame_id = entry.get('frame_id')
        if frame_id:
            self.frame_id = frame_id
        else:
            self.frame_id = f"{frame_prefix}{index}"
        uv_topic = entry.get('uv_topic')
        if uv_topic is None:
            cam_name = entry.get('camera_name')
            if cam_name:
                uv_topic = f"/{cam_name}/point_uv"
                node.get_logger().info(f"[Camera {index}] Derived uv_topic '{uv_topic}' from camera_name")
            else:
                raise RuntimeError(f"[Camera {index}] 'uv_topic' missing and cannot derive")
        self.uv_topic = uv_topic
        node.get_logger().info(f"[Camera {index}] frame_id='{self.frame_id}', uv_topic='{self.uv_topic}'")


    def broadcast_static_transform(self, broadcaster: tf2_ros.StaticTransformBroadcaster):
        """
        Broadcast static transform from world_frame to this camera frame.
        """
        ts = TransformStamped()
        ts.header.stamp = rclpy.time.Time().to_msg()
        # Retrieve world_frame from the Node, not from broadcaster
        wf = self.node.get_parameter('world_frame').value
        ts.header.frame_id = wf
        ts.child_frame_id = self.frame_id
        ts.transform.translation.x = float(self.C[0])
        ts.transform.translation.y = float(self.C[1])
        ts.transform.translation.z = float(self.C[2])
        # Build 4x4 matrix for rotation->quaternion
        mat = np.eye(4)
        mat[:3, :3] = self.R_cw
        mat[:3, 3] = self.C
        quat = tf_trans.quaternion_from_matrix(mat)
        ts.transform.rotation.x = float(quat[0])
        ts.transform.rotation.y = float(quat[1])
        ts.transform.rotation.z = float(quat[2])
        ts.transform.rotation.w = float(quat[3])
        broadcaster.sendTransform(ts)
        self.node.get_logger().info(f"[Camera {self.index}] Broadcast static TF: {wf} -> {self.frame_id}")


    def backproject(self, u: float, v: float):
        """
        Given pixel (u,v), undistort and compute ray in world frame:
        Returns (o_w, d_w) where o_w is camera center, d_w is unit direction in world.
        Uses camera’s optical axis = +X in camera frame (since simulation forward axis is X).
        """
        # 1. Undistort to normalized coords
        uv = np.array([[u, v]], dtype=float).reshape(1,1,2)
        uv_n = cv2.undistortPoints(uv, self.K, self.dist_coef).reshape(2)
        x_n, y_n = uv_n
        # FOV check
        if abs(x_n) > self.half_fov_x or abs(y_n) > self.half_fov_y:
            self.node.get_logger().warn(
                f"[Camera {self.index}] Pixel ({u:.1f},{v:.1f}) undistorted to ({x_n:.3f},{y_n:.3f}) outside FOV"
            )
            return None, None
        # 2. Ray in camera frame: optical axis along -X, image plane at X=-1
        # Direction vector in camera frame:
        d_c = np.array([1.0, x_n, y_n], dtype=float)
        norm = np.linalg.norm(d_c)
        if norm == 0:
            self.node.get_logger().error(f"[Camera {self.index}] Zero-length direction vector")
            return None, None
        d_c /= norm
        # 3. Transform to world
        o_w = self.C
        d_w = self.R_cw.dot(d_c)
        self.node.get_logger().info(
            f"[Camera {self.index}] Backproject: pixel ({u:.1f},{v:.1f}) -> ray origin {o_w}, dir {d_w}, d_w.z={d_w[2]:.3f}"
        )
        return o_w, d_w



    def project(self, X_w: np.ndarray):
        """
        Project a 3D world point X_w (shape (3,)) into this camera image.
        Returns np.array([u,v]) or [nan,nan] if behind camera.
        """
        Xc = self.R_wc.dot(X_w) + self.t_wc
        x, y, z = Xc
        if z <= 0:
            return np.array([np.nan, np.nan], dtype=float)
        u = self.K[0,0] * (x / z) + self.K[0,2]
        v = self.K[1,1] * (y / z) + self.K[1,2]
        return np.array([u, v], dtype=float)

    def compute_fov_polygon(self, max_distance: float = 10.0, ground_plane: bool = True, room_bounds=None):
        """
        Compute a polygon representing the camera's FOV intersection with either:
         - a horizontal plane at z=0 (ground), if ground_plane=True, OR
         - a frustum at fixed max_distance, if ground_plane=False.
        Returns list of 3D points in world frame (shape Nx3) for polygon corners.
        room_bounds: [xmin, xmax, ymin, ymax], used to clip ground intersection within room.
        """
        # Image corners in pixel coords
        # Assumes image resolution: derive from intrinsics principal and focal: not directly known here.
        # We assume principal point cx, cy near center; need width and height.
        # For accurate FOV polygon, you should supply image width/height separately if needed.
        # Here we approximate by projecting four normalized directions corresponding to corners:
        # Corners: (0,0), (w,0), (w,h), (0,h). But K only gives fx,fy,cx,cy: we infer w=2*cx, h=2*cy if principal at center.
        fx = self.K[0,0]; fy = self.K[1,1]; cx = self.K[0,2]; cy = self.K[1,2]
        # Infer width and height
        w = int(round(2 * cx))
        h = int(round(2 * cy))
        corners = [(0.0, 0.0), (w-1.0, 0.0), (w-1.0, h-1.0), (0.0, h-1.0)]
        pts_world = []
        for (u, v) in corners:
            # Undistort corner pixel
            uv = np.array([[u, v]], dtype=float).reshape(1,1,2)
            uv_n = cv2.undistortPoints(uv, self.K, self.dist_coef).reshape(2)
            x_n, y_n = uv_n
            d_c = np.array([x_n, y_n, 1.0], dtype=float)
            norm = np.linalg.norm(d_c)
            if norm == 0:
                continue
            d_c /= norm
            d_w = self.R_cw.dot(d_c)
            o_w = self.C
            if ground_plane:
                # intersect with z=0 plane: solve C.z + t*d_w.z = 0
                if abs(d_w[2]) < 1e-6:
                    continue
                t = -o_w[2] / d_w[2]
                if t <= 0:
                    continue
                Xg = o_w + t * d_w
                # Optionally clip to room bounds
                if room_bounds:
                    xmin, xmax, ymin, ymax = room_bounds
                    if not (xmin <= Xg[0] <= xmax and ymin <= Xg[1] <= ymax):
                        # If outside room, clamp to boundary intersection?
                        # For simplicity, skip or use clipped point
                        # Here we skip adding this corner
                        continue
                pts_world.append(Xg)
            else:
                # extend to fixed max_distance
                Xf = o_w + d_w * max_distance
                pts_world.append(Xf)
        return pts_world

class Triangulation:
    """
    Provides methods for triangulating a 3D point from multiple rays.
    """
    def __init__(self, reproj_thresh: float, min_rays: int, room_bounds=None, node: Node=None):
        self.reproj_thresh = reproj_thresh
        self.min_rays = min_rays
        self.room_bounds = room_bounds  # [xmin,xmax,ymin,ymax], or None
        self.node = node

    def intersect_ground(self, cam: CameraParams, u: float, v: float):
        """
        Intersect ray from camera through pixel (u,v) with ground plane z=0.
        Returns 3D point Xg in world if valid (t>0 and within room bounds), else None.
        """
        o_w, d_w = cam.backproject(u, v)
        if o_w is None:
            return None
        # Solve o_w.z + t * d_w.z = 0
        if abs(d_w[2]) < 1e-6:
            self.node.get_logger().info(f"[Triangulation] Camera {cam.index} ray nearly parallel to ground")
            return None
        t = -o_w[2] / d_w[2]
        if t <= 0:
            self.node.get_logger().info(f"[Triangulation] Camera {cam.index} ground intersection behind camera (t={t:.3f})")
            return None
        Xg = o_w + t * d_w
        if self.room_bounds:
            xmin, xmax, ymin, ymax = self.room_bounds
            if not (xmin <= Xg[0] <= xmax and ymin <= Xg[1] <= ymax):
                self.node.get_logger().info(f"[Triangulation] Ground intersection {Xg} outside room bounds")
                return None
        self.node.get_logger().info(f"[Triangulation] Camera {cam.index} ground intersection at {Xg}")
        return Xg

    @staticmethod
    def solve_linear_ls(ows: list, dws: list, node: Node=None):
        """
        Solve for X minimizing sum of squared distances to rays (o_i, d_i).
        Returns X (np.ndarray shape (3,)).
        """
        A = np.zeros((3,3), dtype=float)
        b = np.zeros(3, dtype=float)
        for o, d in zip(ows, dws):
            Dm = np.eye(3) - np.outer(d, d)
            A += Dm
            b += Dm.dot(o)
        cond = np.linalg.cond(A)
        # self.get_logger().info(f"[Debug] LS cond={cond:.2e}, A=\n{A}, b={b}")
        if node:
            node.get_logger().info(f"[Triangulation] LS matrix condition number: {cond:.2e}")
            if cond > 1e6:
                node.get_logger().warn(f"[Triangulation] Ill-conditioned A matrix (cond={cond:.2e})")
        try:
            X = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as e:
            if node:
                node.get_logger().error(f"[Triangulation] Linear solve failed: {e}")
            return None
        if node:
            node.get_logger().info(f"[Triangulation] LS solution X = {X}")
        # self.get_logger().info(f"[Debug] LS solution X = {X}")

        return X

    def triangulate_ls(self, cam_uv_list: list):
        """
        Perform linear LS triangulation on multiple cameras.
        cam_uv_list: list of tuples (CameraParams, (u,v))
        Returns (X, reproj_errors) or (None, []) if failure or insufficient rays.
        """
        ows = []
        dws = []
        uvs = []
        cams_used = []
        # Backproject each
        for cam, (u,v) in cam_uv_list:
            o_w, d_w = cam.backproject(u, v)
            if o_w is None:
                continue
            ows.append(o_w)
            dws.append(d_w)
            uvs.append((u,v))
            cams_used.append(cam)
        if len(ows) < self.min_rays:
            self.node.get_logger().warn(f"[Triangulation] Only {len(ows)} valid rays < min_rays {self.min_rays}")
            return None, []
        # Solve LS
        X = self.solve_linear_ls(ows, dws, node=self.node)
        if X is None:
            return None, []
        # Compute reprojection errors
        reproj_errs = []
        for (u,v), cam in zip(uvs, cams_used):
            uvp = cam.project(X)
            if np.any(np.isnan(uvp)):
                err = np.inf
            else:
                err = float(np.linalg.norm(np.array([u,v]) - uvp))
            reproj_errs.append(err)
            self.node.get_logger().info(f"[Triangulation] Camera {cam.index} reproj error = {err:.2f} px")
        # Outlier removal if too large
        if len(reproj_errs) > self.min_rays and max(reproj_errs) > self.reproj_thresh:
            idx_bad = int(np.argmax(reproj_errs))
            cam_bad = cams_used[idx_bad]
            self.node.get_logger().warn(f"[Triangulation] Removing outlier ray from Camera {cam_bad.index} with error {reproj_errs[idx_bad]:.2f}")
            # Remove and re-solve
            del ows[idx_bad]; del dws[idx_bad]; del uvs[idx_bad]; del cams_used[idx_bad]
            if len(ows) < self.min_rays:
                self.node.get_logger().warn("[Triangulation] Not enough rays after outlier removal")
                return None, []
            X2 = self.solve_linear_ls(ows, dws, node=self.node)
            if X2 is None:
                return None, []
            # Recompute reproj errs
            reproj_errs2 = []
            for (u,v), cam in zip(uvs, cams_used):
                uvp = cam.project(X2)
                if np.any(np.isnan(uvp)):
                    err2 = np.inf
                else:
                    err2 = float(np.linalg.norm(np.array([u,v]) - uvp))
                reproj_errs2.append(err2)
                self.node.get_logger().info(f"[Triangulation] After removal, Camera {cam.index} reproj error = {err2:.2f} px")
            return X2, reproj_errs2
        # Otherwise accept X
        return X, reproj_errs

    def trilaterate(self, cam_uv_list: list):
        """
        Try ground-plane intersection approach first; if fails or high reproj error, fallback to LS triangulation.
        cam_uv_list: list of (CameraParams, (u,v))
        Returns (X, reproj_errors) or (None, []).
        """
        # 1. Ground-plane intersections
        intersections = []
        cams_int = []
        uvs_int = []
        for cam, (u,v) in cam_uv_list:
            Xg = self.intersect_ground(cam, u, v)
            if Xg is not None:
                intersections.append(Xg)
                cams_int.append(cam)
                uvs_int.append((u,v))
        if len(intersections) >= self.min_rays:
            # Average intersection points
            pts = np.vstack(intersections)
            X_est = np.mean(pts, axis=0)
            self.node.get_logger().info(f"[Triangulation] Averaged ground-plane estimate: {X_est}")
            # Compute reprojection errors
            reproj_errs = []
            for cam, (u,v) in zip(cams_int, uvs_int):
                uvp = cam.project(X_est)
                if np.any(np.isnan(uvp)):
                    err = np.inf
                else:
                    err = float(np.linalg.norm(np.array([u,v]) - uvp))
                reproj_errs.append(err)
                self.node.get_logger().info(f"[Triangulation] Ground-plane reproj error Camera {cam.index} = {err:.2f} px")
            if max(reproj_errs) <= self.reproj_thresh:
                self.node.get_logger().info("[Triangulation] Accepting ground-plane triangulation")
                return X_est, reproj_errs
            else:
                self.node.get_logger().warn("[Triangulation] Ground-plane reproj error too high, fallback to LS")
        # 2. Fallback to LS triangulation
        return self.triangulate_ls(cam_uv_list)

class TriangulatorNode(Node):
    """
    Main ROS2 node: loads calibration JSON, subscribes to 2D UV topics,
    synchronizes detections, triangulates 3D position, publishes PointStamped and TF,
    and visualizes camera FOV, rays, and target in RViz via MarkerArray.
    """
    def __init__(self):
        super().__init__('triangulation_node')
        # 1. Declare parameters
        default_pkg = None
        try:
            default_pkg = get_package_share_directory('thermal_3d_localization')
        except:
            default_pkg = os.getcwd()
        default_json = os.path.join(default_pkg, 'config', 'camera-params.json')
        self.declare_parameter('camera_params_file', default_json)
        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('reproj_error_thresh', 5.0)
        self.declare_parameter('min_rays', 3)
        self.declare_parameter('sync_window', 0.05)
        # Room bounds for ground-plane FOV clipping
        # Format: [xmin, xmax, ymin, ymax]
        self.declare_parameter('room_bounds', [-5.0, 5.0, -5.0, 5.0])
        self.declare_parameter('fov_visual_ground', True)  # visualize FOV intersection with ground
        self.declare_parameter('fov_visual_distance', 10.0)  # if not ground, distance to extend frustum
        # 2. Read parameters
        camera_params_file = self.get_parameter('camera_params_file').value
        self.world_frame = self.get_parameter('world_frame').value
        self.reproj_error_thresh = self.get_parameter('reproj_error_thresh').value
        self.min_rays = int(self.get_parameter('min_rays').value)
        self.sync_window = float(self.get_parameter('sync_window').value)
        self.room_bounds = self.get_parameter('room_bounds').value
        self.fov_visual_ground = bool(self.get_parameter('fov_visual_ground').value)
        self.fov_visual_distance = float(self.get_parameter('fov_visual_distance').value)
        self.get_logger().info(f"[Node] Params: camera_params_file={camera_params_file}, world_frame={self.world_frame}, reproj_thresh={self.reproj_error_thresh}, min_rays={self.min_rays}, sync_window={self.sync_window}, room_bounds={self.room_bounds}, fov_visual_ground={self.fov_visual_ground}, fov_visual_distance={self.fov_visual_distance}")
        # 3. Load calibration JSON
        if not os.path.isabs(camera_params_file):
            # try relative to package
            path = os.path.join(default_pkg, 'config', camera_params_file)
            if os.path.exists(path):
                camera_params_file = path
        if not os.path.exists(camera_params_file):
            self.get_logger().error(f"[Node] Camera params file not found: {camera_params_file}")
            raise RuntimeError("Camera params file not found")
        with open(camera_params_file, 'r') as f:
            try:
                cams_json = json.load(f)
            except Exception as e:
                self.get_logger().error(f"[Node] Failed to parse JSON: {e}")
                raise
        if not isinstance(cams_json, list) or len(cams_json) == 0:
            self.get_logger().error("[Node] camera-params.json must be a non-empty list")
            raise RuntimeError("Invalid camera-params.json")
        # 4. Create CameraParams objects
        self.cams = []
        for i, entry in enumerate(cams_json):
            try:
                cam = CameraParams(entry, self.world_frame, 'thermal_cam_', i+1, node=self)
                self.cams.append(cam)
            except Exception as e:
                self.get_logger().error(f"[Node] Failed to load camera {i+1}: {e}")
        if len(self.cams) < self.min_rays:
            self.get_logger().error(f"[Node] Only {len(self.cams)} cameras loaded < min_rays {self.min_rays}")
            # we continue, but triangulation will fail
        # 5. Broadcast static transforms for cameras
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        for cam in self.cams:
            cam.broadcast_static_transform(self.tf_static_broadcaster)
        # 6. Prepare buffers for UV detections
        # Each buffer holds latest (u, v, timestamp)
        self.uv_buffers = [None] * len(self.cams)
        # 7. Subscribe to UV topics
        for idx, cam in enumerate(self.cams):
            topic = cam.uv_topic
            # Use message_filters Subscriber if planning approximate sync; but we'll buffer manually
            sub = message_filters.Subscriber(self, PointStamped, topic)
            # Register callback capturing idx
            sub.registerCallback(self._make_uv_callback(idx))
            self.get_logger().info(f"[Node] Subscribed to UV topic: {topic}")
        # 8. Setup triangulation helper
        self.triangulator = Triangulation(self.reproj_error_thresh, self.min_rays, self.room_bounds, node=self)
        # 9. Publisher for 3D point
        self.pub_point = self.create_publisher(PointStamped, '/thermal_target/position', 10)
        # 10. TF broadcaster for dynamic target
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # 11. Publisher for visualization markers
        self.pub_markers = self.create_publisher(MarkerArray, 'triangulation_markers', 10)
        # 12. Timer for synchronization & triangulation
        timer_period = 1.0 / 30.0  # 30 Hz
        self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info("[Node] TriangulatorNode initialized, waiting for UV detections...")

    def _make_uv_callback(self, idx):
        def uv_callback(msg: PointStamped):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            u = msg.point.x; v = msg.point.y
            self.uv_buffers[idx] = (u, v, t)
            self.get_logger().info(f"[UV] Camera {idx+1} received UV ({u:.1f},{v:.1f}) at t={t:.6f}")
        return uv_callback

    def timer_callback(self):
        """
        Periodically attempt to synchronize UV detections and triangulate.
        """
        # 1. Collect timestamps of available buffers
        times = []
        for buf in self.uv_buffers:
            if buf is not None:
                times.append(buf[2])
        if len(times) < self.min_rays:
            # Not enough detections yet
            return
        tmax = max(times)
        # 2. Select buffers within sync_window of tmax
        idxs = [i for i, buf in enumerate(self.uv_buffers) if buf is not None and abs(buf[2] - tmax) <= self.sync_window]
        if len(idxs) < self.min_rays:
            # Not enough synchronized detections
            return
        # 3. Build list of (CameraParams, (u,v))
        cam_uv_list = []
        for i in idxs:
            u, v, t = self.uv_buffers[i]
            cam_uv_list.append((self.cams[i], (u, v)))
        self.get_logger().info(f"[Triangulation] Attempting triangulation with cameras: { [i+1 for i in idxs] }")
        # 4. Triangulate (ground-plane first, then LS)
        X, reproj_errs = self.triangulator.trilaterate(cam_uv_list)
        if X is None:
            self.get_logger().warn("[Triangulation] Triangulation failed or insufficient valid rays")
            # Still visualize FOV even if no triangulation
            self.publish_visualization(None, cam_uv_list)
            return
        # 5. Publish 3D point
        pt_msg = PointStamped()
        now = self.get_clock().now().to_msg()
        pt_msg.header.stamp = now
        pt_msg.header.frame_id = self.world_frame
        pt_msg.point.x = float(X[0]); pt_msg.point.y = float(X[1]); pt_msg.point.z = float(X[2])
        self.pub_point.publish(pt_msg)
        self.get_logger().info(f"[Triangulation] Published 3D point: ({X[0]:.3f}, {X[1]:.3f}, {X[2]:.3f})")

        # 6. Broadcast TF for target
        ts = TransformStamped()
        ts.header.stamp = now
        ts.header.frame_id = self.world_frame
        ts.child_frame_id = 'thermal_target'
        ts.transform.translation.x = float(X[0])
        ts.transform.translation.y = float(X[1])
        ts.transform.translation.z = float(X[2])
        # Identity orientation
        ts.transform.rotation.x = 0.0
        ts.transform.rotation.y = 0.0
        ts.transform.rotation.z = 0.0
        ts.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(ts)
        self.get_logger().info("[Triangulation] Broadcast TF for 'thermal_target'")

        # 7. Publish visualization markers (cameras, FOV, rays, target)
        self.publish_visualization(X, cam_uv_list)

    def publish_visualization(self, X_est, cam_uv_list):
        """
        Publish MarkerArray showing:
         - Camera centers (spheres)
         - Camera FOV polygons on ground or at fixed distance
         - Rays from cameras for current detections
         - Triangulated target point
        """
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        # 1. Camera centers
        for idx, cam in enumerate(self.cams):
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = now
            m.ns = 'camera_centers'
            m.id = idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(cam.C[0])
            m.pose.position.y = float(cam.C[1])
            m.pose.position.z = float(cam.C[2])
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = 0.1
            m.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
            markers.markers.append(m)
        # 2. Camera FOV on ground or at distance
        for idx, cam in enumerate(self.cams):
            pts_world = cam.compute_fov_polygon(
                max_distance=self.fov_visual_distance,
                ground_plane=self.fov_visual_ground,
                room_bounds=self.room_bounds
            )
            if len(pts_world) >= 2:
                # Create LINE_STRIP or LINE_LIST to draw polygon
                m = Marker()
                m.header.frame_id = self.world_frame
                m.header.stamp = now
                m.ns = 'camera_fov'
                m.id = idx
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                # Points: from camera center to each polygon vertex, and polygon edges
                # First draw frustum edges:
                for Xg in pts_world:
                    # line from camera center to intersection point
                    # Use POINTS: two points per segment
                    # Instead, we add as separate markers or use LINE_LIST: but here LINE_STRIP loops vertices
                    # We'll draw polygon boundary plus lines to camera center
                    pass
                # To draw polygon boundary: add vertices in order and close loop
                # Determine order: assume pts_world correspond to corners but possibly missing some; we simply connect in given order
                m.points = []
                for p in pts_world:
                    pt = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                    m.points.append(pt)
                # Close loop
                if pts_world:
                    first = pts_world[0]
                    m.points.append(Point(x=float(first[0]), y=float(first[1]), z=float(first[2])))
                m.scale.x = 0.02
                m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5)
                markers.markers.append(m)
                # Also draw rays from camera center to each polygon vertex
                # Use LINE_LIST: pairs of points
                m2 = Marker()
                m2.header.frame_id = self.world_frame
                m2.header.stamp = now
                m2.ns = 'camera_fov_rays'
                m2.id = idx
                m2.type = Marker.LINE_LIST
                m2.action = Marker.ADD
                m2.points = []
                for p in pts_world:
                    # camera center
                    m2.points.append(Point(x=float(cam.C[0]), y=float(cam.C[1]), z=float(cam.C[2])))
                    # intersection point
                    m2.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
                m2.scale.x = 0.01
                m2.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.5)
                markers.markers.append(m2)
        # 3. Rays for current detections
        for idx, (cam, (u,v)) in enumerate(cam_uv_list):
            o_w, d_w = cam.backproject(u, v)
            if o_w is None:
                continue
            # Draw a short segment along the ray
            length = 2.0  # meters
            p0 = o_w
            p1 = o_w + d_w * length
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = now
            m.ns = 'detection_rays'
            m.id = idx
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.points = [
                Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
            ]
            m.scale.x = 0.02
            m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            markers.markers.append(m)
            # Also draw a small sphere at the projected 2D backproject to some depth or ground intersection
            # If ground-plane intersection exists:
            Xg = self.triangulator.intersect_ground(cam, u, v)
            if Xg is not None:
                ms = Marker()
                ms.header.frame_id = self.world_frame
                ms.header.stamp = now
                ms.ns = 'detection_ground'
                ms.id = idx
                ms.type = Marker.SPHERE
                ms.action = Marker.ADD
                ms.pose.position.x = float(Xg[0]); ms.pose.position.y = float(Xg[1]); ms.pose.position.z = float(Xg[2])
                ms.scale.x = ms.scale.y = ms.scale.z = 0.1
                ms.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
                markers.markers.append(ms)
        # 4. Target position marker
        if X_est is not None:
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = now
            m.ns = 'triangulated_target'
            m.id = 0
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(X_est[0])
            m.pose.position.y = float(X_est[1])
            m.pose.position.z = float(X_est[2])
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
            markers.markers.append(m)
        # Publish markers
        self.pub_markers.publish(markers)
        self.get_logger().info("[Visualization] Published MarkerArray for FOV, rays, and target")

def main(args=None):
    rclpy.init(args=args)
    node = TriangulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
