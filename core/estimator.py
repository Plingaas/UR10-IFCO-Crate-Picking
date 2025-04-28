from concurrent.futures import ThreadPoolExecutor
from objects.crate import Crate
from objects.pallet import Pallet
from utils.helper import (
    pcd_remove_outliers,
    camera_to_robot_transform,
    estimate_plane,
    flatten_plane_cloud,
    ICP_crate,
    ICP_pallet,
    estimate_pose
)


class PoseEstimator:
    def __init__(self) -> None:
        pass

    def estimate_poses(self, point_clouds):
        objects = []
        with ThreadPoolExecutor() as pool:
            futures = []
            for data in point_clouds:
                futures.append(pool.submit(self.estimate_pose, data))

            for f in futures:
                objects.append(f.result())  # noqa: PERF401

        # Get top object, will never be pallet.
        objects_sorted = sorted(objects, key=lambda x: x.pose[2], reverse=True)

        return objects_sorted

    def estimate_pose(self, data):
        pcd, _ = pcd_remove_outliers(data["pcd"], 50, 2.0)
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        plane_model, plane_cloud = estimate_plane(pcd)
        plane_cloud = flatten_plane_cloud(plane_cloud, plane_model)

        if data["id"] == Crate.ID:
            return self.estimate_crate(plane_cloud)
        else:
            return self.estimate_pallet(plane_cloud)
    
    def estimate_crate(self, plane_cloud):
        icp_transform = ICP_crate(plane_cloud)
        pose = estimate_pose(icp_transform)
        pose_robot_frame = camera_to_robot_transform(pose)
        return Crate(pose_robot_frame)
    
    def estimate_pallet(self, plane_cloud):
        icp_transform = ICP_pallet(plane_cloud)
        pose = estimate_pose(icp_transform)
        pose_robot_frame = camera_to_robot_transform(pose)
        return Pallet(pose_robot_frame)