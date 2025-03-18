import open3d as o3d
import numpy as np
from helper import *
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
from robot import *
from pctools import *

## SETUP
CAMERA_TALL = True
# SETUP YOLO
print("Setting up yolo")
model = YOLO("C:/Documents/yolo/venv3.11/runs/segment/train12/weights/best.pt")
setup_image = np.zeros((640, 384)).astype(np.uint8)
setup_image = cv2.cvtColor(setup_image, cv2.COLOR_GRAY2BGR)
model.predict(setup_image, conf=0.8, device=0, verbose=False)

print("Setting up Intel Realsense L515")
# SETUP REALSENSE
pipeline = rs.pipeline()

config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.receiver_gain, 18)
depth_sensor.set_option(rs.option.post_processing_sharpening, 1)

depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = 2.5
clipping_distance = clipping_distance_in_meters / depth_scale
align_to = rs.stream.color
align = rs.align(align_to)


coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5
)  # Adjust size if needed


def ICP(pcd):
    crate_pcd = load_crate_pc()
    points = np.asarray(crate_pcd.points)
    points = points[points[:, 1] < 0.002]
    crate_pcd.points = o3d.utility.Vector3dVector(points)
    crate_pcd.voxel_down_sample(voxel_size=0.01)
    pcd_set_color(crate_pcd, (0, 0, 0))

    ####################################### FIND PLANE

    # Step 1: Remove Outliers (Statistical Outlier Removal)
    pcd, _ = pcd_remove_outliers(pcd, 50, 2.0)

    plane_model, plane_cloud = estimate_plane(pcd)
    plane_cloud = smooth_plane_cloud(plane_cloud, plane_model)

    ############################################## ICP
    pcd = plane_cloud
    pcdcopy = o3d.geometry.PointCloud()
    pcdcopy.points = pcd.points
    pcdcopy.paint_uniform_color([1, 0, 0])

    pcd.paint_uniform_color([0, 0, 1])
    pcd.paint_uniform_color([0, 1, 0])

    # Perform ICP with rigid transformation (translation and rotation)
    threshold = 3  # Maximum distance threshold for matching points
    trans_init = np.eye(4)  # Initial transformation (identity matrix)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    crate_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    icp_result = o3d.pipelines.registration.registration_icp(
        pcd,
        crate_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # More accurate
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200
        ),  # Max iterations
    )

    # Get the transformation matrix
    transformation_matrix = icp_result.transformation
    # print(transformation_matrix)

    # Apply the transformation to the source point cloud
    pcd.transform(transformation_matrix)

    query_point = np.array([0, 0, 0])
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    _, index, _ = kdtree.search_knn_vector_3d(query_point, 1)
    nearest_point = pcd.points[index[0]]

    # UNDO ICP TRANSFORM
    # Convert to homogeneous coordinates
    nearest_point_h = np.append((0, 0, 0), 1)  # (x, y, z, 1)

    # Undo the transformation (T^-1 * p')
    inv_icp = np.linalg.inv(transformation_matrix)
    inv_rotation = np.linalg.inv(transformation_matrix[:3, :3])

    original_point_h = inv_icp @ nearest_point_h

    # Extract (x, y, z) after undoing ICP
    original_point = original_point_h[:3] * -1e3

    axis_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    axis_marker.rotate(inv_rotation, center=(0, 0, 0))  # Rotate to original orientation
    axis_marker.translate(original_point)

    x = original_point[0]
    y = original_point[1]
    z = original_point[2]
    # x = transformation_matrix[0, 3] * 1e3
    # y = transformation_matrix[1, 3] * 1e3
    # z = transformation_matrix[2, 3] * 1e3
    print(f"dx: {round(x, 3)}mm    dy: {round(y, 3)}mm    dz: {round(z, 3)}mm")

    # Visualize point cloud

    crate2_pcd = load_crate_pc()
    crate2_pcd.transform(inv_icp)

    posx = x
    posy = y
    posz = -z
    rot_vec, _ = cv2.Rodrigues(transformation_matrix[:3, :3])
    yaw = -rot_vec[2][0]

    # o3d.visualization.draw_geometries([crate2_pcd, crate_pcd, pcdcopy, pcd, axis_marker])
    return posx, posy, posz, yaw


# Load the point cloud
"""
top_crate = o3d.io.read_point_cloud(f"{DATA_FOLDER}/Crate_2_top_filtered.xyz")
top_crate = pcd_transform_L515_data(top_crate)
top_crate = do_crazy_math(top_crate)

middle_crate = o3d.io.read_point_cloud(f"{DATA_FOLDER}/Crate_2_middle_filtered.xyz")
middle_crate = pcd_transform_L515_data(middle_crate)
middle_crate = do_crazy_math(middle_crate)
"""
n_crate = 0
i = 0
try:
    poses = np.empty((0, 4))
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame is a 640x480 depth image
        color_frame = frames.get_color_frame()
        i += 1

        if i < 5:
            continue
        else:
            i = 0

        depth_intrinsics = (
            aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        )

        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        color_image = np.asanyarray(color_frame.get_data())

        if CAMERA_TALL:
            color_image = cv2.rotate(color_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        masks_np = None

        results = model.predict(color_image, show=False, conf=0.6, device=0)
        for result in results:
            if result.masks:
                masks = result.masks.data
                masks_np = masks.cpu().numpy()
                class_ids = result.boxes.cls  # Class IDs
                confidences = result.boxes.conf  # Confidence scores

            # Convert the result to an image and resize it
            # image_with_results = result.plot()  # Get the annotated image (BGR format)
            # resized_image = cv2.resize(image_with_results, (540, 960))  # Resize to (540, 960)

            # Show the image
            # cv2.imshow("Detection Results", resized_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        if CAMERA_TALL:
            complete_grayscale_mask = np.zeros((1920, 1080)).astype(np.uint8)
        else:
            complete_grayscale_mask = np.zeros((1080, 1920)).astype(np.uint8)

        complete_binary_mask = complete_grayscale_mask
        if masks_np is not None:
            for obj_idx, mask in enumerate(masks_np):
                mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                if not CAMERA_TALL:
                    mask = cv2.resize(
                        mask, (1080, 1920), interpolation=cv2.INTER_NEAREST
                    )
                else:
                    mask = cv2.resize(
                        mask, (1920, 1080), interpolation=cv2.INTER_NEAREST
                    )

                mask_grayscale = (mask > 0.001).astype(np.uint8) * ((obj_idx + 1) * 40)
                mask_binary = ((mask > 0.001).astype(np.uint8) * 255) // 255

                depth_masked = cv2.bitwise_and(
                    depth_image, depth_image, mask=mask_binary
                )
                # Get nonzero depth indices and values
                nonzero_indices = np.nonzero(depth_masked)  # (row, col) indices
                nonzero_depths = (
                    depth_masked[nonzero_indices] / 1000.0
                )  # Convert mm to meters

                # Extract x and y pixel coordinates
                x_pixels = nonzero_indices[
                    1
                ]  # Column indices correspond to x-coordinates
                y_pixels = nonzero_indices[0]  # Row indices correspond to y-coordinates

                # Compute 3D coordinates using intrinsics
                x_world = (
                    (x_pixels - depth_intrinsics.ppx)
                    / depth_intrinsics.fx
                    * nonzero_depths
                )
                y_world = (
                    (y_pixels - depth_intrinsics.ppy)
                    / depth_intrinsics.fy
                    * nonzero_depths
                )
                z_world = nonzero_depths  # Z is just the depth in meters

                # Stack into an (N, 3) array: (x, y, z)
                points_3d = np.array(
                    [
                        rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], z)
                        for x, y, z in zip(x_pixels, y_pixels, nonzero_depths)
                    ]
                )

                points = o3d.utility.Vector3dVector(points_3d)
                pcd = o3d.geometry.PointCloud()
                pcd.points = points
                # o3d.visualization.draw_geometries([coordinate_frame, pcd])
                pcd, _ = pcd_remove_outliers(pcd, 50, 2.0)
                pcd = pcd_transform_L515_data(pcd)
                x, y, z, yaw = ICP(pcd)
                poses = np.concatenate((poses, [np.array([x, y, z, yaw])]), axis=0)

        if poses.shape[0] == 3:
            # Compute max difference (spread) for each axis
            max_diff_x = np.max(poses[:, 0]) - np.min(poses[:, 0])
            max_diff_y = np.max(poses[:, 1]) - np.min(poses[:, 1])
            max_diff_z = np.max(poses[:, 2]) - np.min(poses[:, 2])
            max_diff_yaw = np.max(poses[:, 3]) - np.min(poses[:, 3])

            # Check if differences are within limits
            if (
                max_diff_x <= 7
                and max_diff_y <= 7
                and max_diff_z <= 7
                and max_diff_yaw <= 0.1
            ):
                # Compute the mean only if within limits
                x = np.mean(poses[:, 0])
                y = np.mean(poses[:, 1])
                z = np.mean(poses[:, 2])
                yaw = np.mean(poses[:, 3])
                controller = rtde_control.RTDEControlInterface("192.168.1.205")
                pick_crate(x, y, z, yaw)
                # moveToWithYaw(0, -400, 600, yaw-90)
                moveToWithYaw(300, -400, 600, 0, speed=1.2, acc=0.5)
                moveToWithYaw(550, 0, 600, 0, speed=1.2, acc=0.5)
                moveToWithYaw(550, 500, 600, 0, speed=1.2, acc=0.5)
                place_crate(n_crate)
                moveToWithYaw(550, 500, 600, 0, speed=1.2, acc=0.5)
                moveToWithYaw(550, 0, 600, 0, speed=1.2, acc=0.5)
                n_crate += 1
                print("Averaged pose within limits:", x, y, z, yaw)
            else:
                print("Pose difference exceeded limits, discarding values.")

            poses = np.empty((0, 4))

finally:
    pipeline.stop()
