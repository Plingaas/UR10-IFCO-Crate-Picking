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

    def estimate_poses(self, data):
        crates = []
        # pallets = []
        with ThreadPoolExecutor() as pool:
            futures = []
            for item in data:
                if item["id"] == 1:  # Pallet
                    pass
                    # futures.append(pool.submit(self.estimate_pallet, item["pcd"]))
                else:  # Crate
                    futures.append(pool.submit(self.estimate_crate, item["pcd"]))

            for f in futures:
                crates.append(f.result())  # noqa: PERF401

        # Get top closest, top left crate first.
        crates_sorted = sorted(crates, key=lambda x: x.pose[2], reverse=True)

        return crates_sorted

    def estimate_crate(self, pcd) -> Crate:
        pcd, _ = pcd_remove_outliers(pcd, 50, 2.0)
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        plane_model, plane_cloud = estimate_plane(pcd)
        plane_cloud = flatten_plane_cloud(plane_cloud, plane_model)
        icp_transform = ICP_crate(plane_cloud)
        pose = estimate_pose(icp_transform)
        pose = camera_to_robot_transform(pose)
        return Crate(pose)

    def estimate_pallet(self, pcd) -> Pallet:
        pcd, _ = pcd_remove_outliers(pcd, 50, 2.0)
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        plane_model, plane_cloud = estimate_plane(pcd)
        plane_cloud = flatten_plane_cloud(plane_cloud, plane_model)
        icp_transform = ICP_pallet(plane_cloud)
        pose = estimate_pose(icp_transform)
        pose = camera_to_robot_transform(pose)
        return Pallet(pose)
