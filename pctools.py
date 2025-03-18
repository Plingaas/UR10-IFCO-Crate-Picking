import open3d as o3d
import numpy as np
from helper import *

def estimate_plane(pcd):
    # Step 3: Estimate Normals (needed for accurate RANSAC)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=50))

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.0125,  # Lower = more precise
        ransac_n=12,
        num_iterations=25000  # Higher = more robust
    )
    
    plane_cloud = pcd.select_by_index(inliers)
    return plane_model, plane_cloud

def smooth_plane_cloud(pcd, model):
    normal = np.array([model[0], model[1],model[2]])
    rot_matrix = plane_to_rotation_matrix(normal, np.array([0, 1, 0]))
    pcd.rotate(rot_matrix, (0,0,0))
    points = np.asarray(pcd.points)
    points[:, 1] = np.mean(points[:, 1])
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.rotate(np.linalg.inv(rot_matrix), (0, 0, 0))
    return pcd