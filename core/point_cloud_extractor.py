import cupy as cp
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor
from core.camera import L515Intrinsics
from utils.helper import pcd_transform_L515_data


class PointCloudExtractor:
    def __init__(self) -> None:
        pass

    def process(self, data, depth_frame):
        depth_frame_gpu = cp.asarray(depth_frame)
        point_clouds = []
        with ThreadPoolExecutor() as pool:
            futures = []
            for detection in data:
                if detection["class_id"] == 1:  # Pallet
                    continue
                mask_gpu = cp.asarray(detection["mask"])
                futures.append(pool.submit(self.extract, mask_gpu, depth_frame_gpu, detection["class_id"]))

            for f in futures:
                point_clouds.append(f.result())  # noqa: PERF401

        if len(point_clouds) == 0:
            return None
        return point_clouds

    def extract(self, mask_gpu, depth_gpu, class_id_gpu):
        depth_gpu = cp.rot90(depth_gpu, k=-1)  # Rotate back to horizontal for masking
        x, y, nz_dep = self.mask_frame(mask_gpu, depth_gpu)
        points = self.compute_world_points(x, y, nz_dep, L515Intrinsics())
        pcd = self.create_pc(points)

        data = {"pcd": pcd, "id": class_id_gpu}

        return data

    def mask_frame(self, mask_gpu, depth_gpu):
        mask_gpu = cp.rot90(mask_gpu, k=-1)  # Rotate back
        mask_binary_gpu = (mask_gpu > 0).astype(cp.uint16)
        depth_masked_gpu = depth_gpu * mask_binary_gpu

        nonzero_indices = cp.nonzero(depth_masked_gpu)  # (row, col) indices
        nonzero_depths = depth_masked_gpu[nonzero_indices] * 1.0e-3  # Convert mm to meters

        # Extract x and y pixel coordinates
        x_pixels = nonzero_indices[1]  # Column indices correspond to x-coordinates
        y_pixels = nonzero_indices[0]  # Row indices correspond to y-coordinates

        return x_pixels, y_pixels, nonzero_depths

    def compute_world_points(self, x, y, nonzero_depths, intrinsics):
        intrinsics = L515Intrinsics()

        ppx = cp.float32(intrinsics.ppx)
        ppy = cp.float32(intrinsics.ppy)
        fx = cp.float32(intrinsics.fx)
        fy = cp.float32(intrinsics.fy)

        # Compute world coordinates on GPU
        x_world = ((x - ppx) / fx) * nonzero_depths
        y_world = ((y - ppy) / fy) * nonzero_depths
        z_world = nonzero_depths  # Depth in meters

        points_gpu = cp.column_stack((x_world, y_world, z_world))

        points_cpu = cp.asnumpy(points_gpu)
        return points_cpu

    def create_pc(self, points):
        points = o3d.utility.Vector3dVector(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = points
        pcd = pcd_transform_L515_data(pcd)

        return pcd

    def shutdown(self):
        self.shutdown()
