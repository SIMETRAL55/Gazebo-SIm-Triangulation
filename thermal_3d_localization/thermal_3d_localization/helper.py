import numpy as np
import cv2
import json, os
from scipy import linalg, optimize
from scipy.spatial.transform import Rotation

def make_square(img):
    h, w = img.shape[:2]
    size = max(h, w)
    new = np.zeros((size, size, 3), dtype=img.dtype)
    dy, dx = (size - h) // 2, (size - w) // 2
    new[dy:dy+h, dx:dx+w] = img
    return new

def triangulate_point_DLT(Ps, uv):
    # uv: list of [u,v]
    A = []
    for P, (u, v) in zip(Ps, uv):
        A.append(v*P[2,:] - P[1,:])
        A.append(P[0,:] - u*P[2,:])
    A = np.vstack(A)
    U, S, Vt = linalg.svd(A)
    X = Vt[-1]
    return X[:3]/X[3]

def triangulate_points(image_points, camera_poses):
    # image_points: list of lists of [u,v]
    object_points = []
    Ps = []
    for cam in camera_poses:
        K = np.array(cam['intrinsic_matrix'])
        R = np.array(cam['R'])
        t = np.array(cam['t']).reshape(3,1)
        P = K @ np.hstack((R, t))
        Ps.append(P)
    for uv in image_points:
        if any(p[0] is None for p in uv):
            object_points.append(np.array([None,None,None]))
            continue
        X = triangulate_point_DLT(Ps, uv)
        object_points.append(X)
    return np.array(object_points)

def calculate_reprojection_error(image_points, object_point, camera_poses):
    errs = []
    for (u, v), cam in zip(image_points, camera_poses):
        if u is None: continue
        K = np.array(cam['intrinsic_matrix'])
        R = np.array(cam['R'])
        t = np.array(cam['t']).reshape(3,1)
        P = K @ np.hstack((R, t))
        Xh = np.array([*object_point, 1.0])
        proj = P @ Xh
        u2, v2 = proj[0]/proj[2], proj[1]/proj[2]
        errs.append((u-u2)**2 + (v-v2)**2)
    return np.mean(errs) if errs else None

def find_point_correspondance_and_object_points(image_points, camera_poses, frames=None):
    # simple greedy grouping: use all combinations
    n = len(image_points[0])
    errors=[]; obj_pts=[]
    for i in range(n):
        # gather i-th point from each cam
        uv = []
        for pts in image_points:
            uv.append(pts[i] if i < len(pts) else [None,None])
        if any(p[0] is None for p in uv): continue
        X = triangulate_points([uv], camera_poses)[0]
        err = calculate_reprojection_error(uv, X, camera_poses)
        errors.append(err)
        obj_pts.append(X)
    return np.array(errors), np.array(obj_pts), frames

def bundle_adjustment(image_points, camera_poses, socketio=None):
    # placeholder: return original triangulated points
    errors, pts, frames = find_point_correspondance_and_object_points(image_points, camera_poses, None)
    return pts

def locate_objects(object_points, errors):
    # stub grouping by proximity
    objs=[]
    if len(object_points)>0:
        objs.append({'pos':object_points[0], 'error':errors[0]})
    return objs

def camera_pose_to_serializable(camera_poses):
    return [{'R':cam['R'], 't':cam['t']} for cam in camera_poses]
