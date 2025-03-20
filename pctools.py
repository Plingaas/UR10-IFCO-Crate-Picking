import open3d as o3d
import numpy as np
from helper import *
from time import time

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
        num_iterations=250,  # Higher = more robust
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
            max_iteration=20
        ),  # Max iterations
    )

    # Get the transformation matrix
    icp_transform = icp_result.transformation
    icp_rotation = icp_transform[:3, :3]
    inv_icp_transform = np.linalg.inv(icp_transform)


    # UNDO ICP TRANSFORM
    zero_point = np.append((0, 0, 0), 1)  # (x, y, z, 1)

    original_point_h = inv_icp_transform @ zero_point
    pos = original_point_h[:3] * -1e3
    pos[1] *= -1
    pos[2] *= -1

    rot_vec, _ = cv2.Rodrigues(icp_rotation)
    yaw = -rot_vec[2][0]

    #coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #    size=0.5
    #)  # Adjust size if needed

    #o3d.visualization.draw_geometries([pcd])
    return pos, yaw
