import open3d as o3d
import numpy as np
from helper import *
from time import time
import cv2

CRATE_FRONT_PCD = load_crate_pc_front()
CRATE_FRONT_PCD.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

def estimate_plane(pcd):
    # Step 3: Estimate Normals (needed for accurate RANSAC)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=50)
    )

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.0125,  # Lower = more precise
        ransac_n=5,
        num_iterations=2500,  # Higher = more robust
    )
    plane_cloud = pcd.select_by_index(inliers)
    return plane_model, plane_cloud


def smooth_plane_cloud(pcd, model):
    normal = np.array([model[0], model[1], model[2]])
    rot_matrix = plane_to_rotation_matrix(normal, np.array([0, 1, 0]))
    pcd.rotate(rot_matrix, (0, 0, 0))
    points = np.asarray(pcd.points)
    points[:, 1] = np.mean(points[:, 1])
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.rotate(np.linalg.inv(rot_matrix), (0, 0, 0))
    return pcd

def ICP(pcd):

    # Perform ICP with rigid transformation (translation and rotation)
    threshold = 3  # Maximum distance threshold for matching points
    trans_init = np.eye(4)  # Initial transformation (identity matrix)

    icp_result = o3d.pipelines.registration.registration_icp(
        pcd,
        CRATE_FRONT_PCD,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # More accurate
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200
        ),  # Max iterations
    )
    #o3d.visualization.draw_geometries([pcd, CRATE_FRONT_PCD])
    # Get the transformation matrix
    return icp_result.transformation

def estimate_pose(transform):
    inv_transform = np.linalg.inv(transform)

    zero_point = np.append((0, 0, 0), 1)  # (x, y, z, 1)
    pose = (inv_transform @ zero_point)
    pose[0] *= -1

    rot = transform[:3, :3]
    rot_vec, _ = cv2.Rodrigues(rot)
    pose[3] = rot_vec[2][0]

    return pose
