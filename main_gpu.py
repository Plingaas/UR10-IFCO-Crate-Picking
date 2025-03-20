import cv2
import cupy as cp
from time import time
from helper import get_L515_intrinsics, get_world_points
from pctools import (
    ICP,
    pcd_transform_L515_data,
    pcd_remove_outliers,
    estimate_plane,
    smooth_plane_cloud,
    load_crate_pc,
    pcd_move_center_to
)

import open3d as o3d
from ultralytics import YOLO
import numpy as np

try:
    model = YOLO("C:/Documents/yolo/venv3.11/runs/segment/train12/weights/best.pt").to(
        "cuda"
    )
    print("YOLO model loaded to GPU successfully")
except Exception:
    model = YOLO("C:/Documents/yolo/venv3.11/runs/segment/train12/weights/best.pt")
    print("YOLO model loaded to CPU successfully")

color_image = cv2.imread("devfolder/color.png", cv2.IMREAD_UNCHANGED)
depth_image = cv2.imread("devfolder/depth.png", cv2.IMREAD_UNCHANGED)

t = time()
n = 0
fps = 0

intrinsics = get_L515_intrinsics()
# Create a visualization window
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5
)  # Adjust size if needed
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(coordinate_frame)
crates = [load_crate_pc(), load_crate_pc()]
vis.add_geometry(crates[0])
vis.add_geometry(crates[1])

if __name__ == "__main__":
    while True:
        objects = 0
        color_image_rotated = cv2.rotate(color_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth_image_gpu = cp.asarray(depth_image)

        yolo_t0 = time()
        results = model.predict(
            color_image_rotated,
            show=False,
            retina_masks=True,
            verbose=False,
            conf=0.6,
            device=0,
        )
        yolo_t1 = time()
        t_start = time()

        for result in results:
            if result.masks:
                masks = cp.asarray(result.masks.data)
                class_ids = cp.asarray(result.boxes.cls)

                for i, mask in enumerate(masks):
                    objects += 1
                    mask = cp.rot90(mask, k=-1)
                    mask_binary_gpu = (mask > 0).astype(cp.uint16)
                    depth_masked_gpu = depth_image_gpu * mask_binary_gpu

                    nonzero_indices = cp.nonzero(depth_masked_gpu)  # (row, col) indices
                    nonzero_depths = (
                        depth_masked_gpu[nonzero_indices] * 1.0e-3
                    )  # Convert mm to meters

                    # Extract x and y pixel coordinates
                    x_pixels = nonzero_indices[
                        1
                    ]  # Column indices correspond to x-coordinates
                    y_pixels = nonzero_indices[
                        0
                    ]  # Row indices correspond to y-coordinates

                    # Compute 3D coordinates using intrinsics

                    points_3d_gpu = get_world_points(x_pixels, y_pixels, nonzero_depths)
                    points_3d_cpu = cp.asnumpy(points_3d_gpu)
                    points = o3d.utility.Vector3dVector(points_3d_cpu)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = points
                    pcd = pcd.voxel_down_sample(voxel_size=0.08)
                    pcd = pcd_transform_L515_data(pcd)
                    #pcd, _ = pcd_remove_outliers(pcd, 10, 0.01)
                    plane_model, plane_cloud = estimate_plane(pcd)
                    plane_cloud = smooth_plane_cloud(plane_cloud, plane_model)
                    pos, yaw = ICP(plane_cloud)
                    pcd_move_center_to(crates[i], pos*1e-3)
                    vis.update_geometry(crates[i])
        vis.poll_events()
        vis.update_renderer()
        t_end = time()
        print(f"Time spent analyzing frame: {round((t_end-t_start + yolo_t1-yolo_t0)*1e3, 1)}ms")
        print(f"Yolo prediction took: {round((yolo_t1-yolo_t0)*1e3, 1)}ms")
        print(f"Located {objects} objects in {round((t_end-t_start)*1e3, 1)}ms.\nTime per object: {round((t_end-t_start)/objects * 1e3, 1)}ms (avg)\n")
    vis.destroy_window()  # Close window after loop ends