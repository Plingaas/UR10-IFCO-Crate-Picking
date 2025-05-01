import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import cv2
import pyrealsense2 as rs
import cupy as cp
import time
DATA_FOLDER = "data"


def depth_to_colormap(depth_map, max_depth=5000):
    depth_clipped = np.clip(depth_map, 0, max_depth)
    depth_normalized = ((depth_clipped / max_depth) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    return depth_colored


def generate_color(class_id):
    rng = np.random.default_rng(class_id)
    color = tuple(rng.integers(0, 255, size=3).tolist())
    return color


def draw_yolo_detections(frame, detections, class_names, mask_alpha=0.4):
    overlay = frame.copy()

    for det in detections:
        bbox = det["bbox"]
        class_id = det["class_id"]
        confidence = det["confidence"]
        color = det.get("color", (0, 255, 0))  # Default green if no color given
        mask = det.get("mask", None)

        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Draw label with confidence
        label = f"{class_names[class_id]} {confidence:.2f}"
        cv2.putText(overlay, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2)

        # Draw mask if available
        if mask is not None:
            h, w = frame.shape[:2]
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            mask_binary = (mask > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = mask_binary * color[c]

            overlay = cv2.addWeighted(overlay, 1, colored_mask, mask_alpha, 0)

    return overlay


def print_with_time(owner, msg):
    print(f"[INFO] [{time.time()}] ({owner}) {msg}")


def screenshot_o3d(vis):
    image_o3d = vis.capture_screen_float_buffer(do_render=True)
    image_np = (np.asarray(image_o3d) * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    start_x = int(w * 0.37)
    end_x = start_x + int(w * 0.43)
    image_bgr = image_bgr[:, start_x:end_x]
    return image_bgr

def get_L515_intrinsics():
    depth_intrinsics = rs.intrinsics()
    depth_intrinsics.width = 1920
    depth_intrinsics.height = 1080
    depth_intrinsics.ppx = 982.673
    depth_intrinsics.ppy = 547.614
    depth_intrinsics.fx = 1352.92
    depth_intrinsics.fy = 1354.45
    depth_intrinsics.model = rs.distortion.brown_conrady
    depth_intrinsics.coeffs = [
        0.171722,
        -0.526011,
        -0.000589736,
        -0.000417008,
        0.486631,
    ]
    return depth_intrinsics


def pcd_rotate_x(pcd, angle):
    # Define the angle (in radians)

    angle = np.deg2rad(angle)

    # Rotation matrix for Z-axis
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )

    # Apply rotation to the point cloud
    pcd.rotate(rotation_matrix, center=pcd.get_center())


def pcd_rotate_y(pcd, angle):
    # Define the angle (in radians)

    angle = np.deg2rad(angle)

    # Rotation matrix for Z-axis
    rotation_matrix = np.array(
        [
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)],
        ]
    )

    # Apply rotation to the point cloud
    pcd.rotate(rotation_matrix, center=pcd.get_center())


def pcd_rotate_z(pcd, angle):
    # Define the angle (in radians)

    angle = np.deg2rad(angle)

    # Rotation matrix for Z-axis
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    # Apply rotation to the point cloud
    pcd.rotate(rotation_matrix, center=pcd.get_center())


def pcd_scale(pcd, scalar, center):
    pcd.scale(scalar, center=center)


def pcd_move_center_to(pcd, position):
    current_pos = pcd.get_center()
    dPos = np.array([position[0], position[1], position[2]]) - current_pos
    pcd.translate(dPos)


def load_europallet_pc():
    # Load the mesh file (replace with your file path)
    mesh = o3d.io.read_triangle_mesh(f"{DATA_FOLDER}/Europallet.obj")  # Change to .ply or .obj if needed

    # Convert mesh to a point cloud using uniform sampling
    pcd = mesh.sample_points_uniformly(number_of_points=100000)  # Adjust point count as needed
    pcd_scale(pcd, 0.001, pcd.get_center())
    pcd_move_center_to(pcd, (0, 0, 0))
    pcd_rotate_x(pcd, 180)
    pcd_rotate_z(pcd, 90)
    pcd_move_center_to(pcd, (0.4, 0.6, 0))
    return pcd


def load_europallet_pc_front():
    pallet_pcd = load_europallet_pc()
    points = np.asarray(pallet_pcd.points)
    points = points[points[:, 1] < 0.002]
    pallet_pcd.points = o3d.utility.Vector3dVector(points)
    pallet_pcd.voxel_down_sample(voxel_size=0.01)
    return pallet_pcd


def load_crate_pc():
    # Load the mesh file (replace with your file path)
    mesh = o3d.io.read_triangle_mesh(f"{DATA_FOLDER}/IFCOCrate.obj")  # Change to .ply or .obj if needed

    # Convert mesh to a point cloud using uniform sampling
    pcd = mesh.sample_points_uniformly(number_of_points=250000)  # Adjust point count as needed
    pcd_move_center_to(pcd, (0.2, 0.3, 0.075))
    return pcd


def load_crate_pc_front() -> o3d.geometry.PointCloud:
    crate_pcd = load_crate_pc()
    points = np.asarray(crate_pcd.points)
    points = points[points[:, 1] < 0.002]
    crate_pcd.points = o3d.utility.Vector3dVector(points)
    crate_pcd.voxel_down_sample(voxel_size=0.01)
    return crate_pcd


def pcd_box_size(pcd, verbose=False):
    bounding_box = pcd.get_axis_aligned_bounding_box()
    bbox_min = bounding_box.get_min_bound()
    bbox_max = bounding_box.get_max_bound()
    bbox_size = bbox_max - bbox_min
    if verbose:
        print(
            f"Box size: {bbox_size}.\
              \nx_min: {bbox_min[0]} x_max: {bbox_max[0]} \
              \ny_min: {bbox_min[1]} y_max: {bbox_max[1]} \
              \nz_min: {bbox_min[2]} z_max: {bbox_max[2]}"
        )
    print(type(bbox_size))
    return bbox_size


def pcd_remove_outliers(pcd, nn=20, std=1.0):
    pcd_clean, inliers = pcd.remove_statistical_outlier(nn, std)
    return pcd_clean, inliers


def pcd_plot_histogram(pcd, axis, limits, increment=0.01):
    if axis < 0 or axis > 2 or not limits or len(limits) != 2:
        return
    print(limits)
    points = np.asarray(pcd.points)
    points = points[:, axis]
    # Define histogram bins from -1m to +1m, with 1cm (0.01m) intervals
    bin_edges = np.arange(limits[0], limits[1], increment)  # From -1 to 1m, step 0.01m
    hist_counts, _ = np.histogram(points, bins=bin_edges)

    # Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.bar(bin_edges[:-1], hist_counts, width=increment, color="blue", edgecolor="black")

    labels = ["X", "Y", "Z"]

    # Labels and formatting
    plt.xlabel(f"{labels[axis]} Coordinate (meters)")
    plt.ylabel("Point Count")
    plt.title(f"Point Cloud {labels[axis]}-Axis Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plane_to_rotation_matrix(plane_normal, target_axis=np.array([0, 0, 1])):
    """
    Compute the rotation matrix to align the plane normal to the target axis.
    Default target is aligning normal to the Z-axis.
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Normalize normal vector
    target_axis = target_axis / np.linalg.norm(target_axis)  # Ensure target is normalized

    # Compute the rotation axis (cross product)
    rotation_axis = np.cross(plane_normal, target_axis)
    rotation_angle = np.arccos(np.clip(np.dot(plane_normal, target_axis), -1.0, 1.0))

    if np.linalg.norm(rotation_axis) < 1e-6:  # If already aligned, return identity
        return np.eye(3)

    # Normalize rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Use scipy Rotation to compute the matrix
    rotation_matrix = Rotation.from_rotvec(rotation_angle * rotation_axis).as_matrix()
    return rotation_matrix


def pcd_transform_L515_data(pcd):
    rot_matrix = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])

    pcd.rotate(rot_matrix, (0, 0, 0))
    pcd_scale(pcd, 0.25, (0, 0, 0))
    return pcd


def pcd_set_color(pcd, color):
    pcd.paint_uniform_color([color[0], color[1], color[2]])
    return pcd


def rotation_matrix_x(rad):
    R_x = np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])

    return R_x


def rotation_matrix_y(rad):
    R_y = np.array([[np.cos(rad), 0, -np.sin(rad)], [0, 1, 0], [np.sin(rad), 0, np.cos(rad)]])

    return R_y


def rotation_matrix_z(rad):
    R_z = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])

    return R_z


def rotate_ur10(x_deg=0, y_deg=0, z_deg=0):
    R_current = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Create X, Y, Z rotation matrices
    x_rad, y_rad, z_rad = np.radians([x_deg, y_deg, z_deg])

    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(x_rad), -np.sin(x_rad)],
            [0, np.sin(x_rad), np.cos(x_rad)],
        ]
    )

    R_y = np.array(
        [
            [np.cos(y_rad), 0, np.sin(y_rad)],
            [0, 1, 0],
            [-np.sin(y_rad), 0, np.cos(y_rad)],
        ]
    )

    R_z = np.array(
        [
            [np.cos(z_rad), -np.sin(z_rad), 0],
            [np.sin(z_rad), np.cos(z_rad), 0],
            [0, 0, 1],
        ]
    )

    # Apply rotations in order (Change order if necessary)
    R_new = R_z @ R_x @ R_y @ R_current  # First X, then Y, then previous rotation

    # Convert back to axis-angle
    new_rotation_vector, _ = cv2.Rodrigues(R_new)
    return tuple(new_rotation_vector.flatten())


def get_world_points(x_pixels, y_pixels, nonzero_depths, intrinsics):
    """Computes 3D world coordinates from depth image pixels on the GPU."""

    # Convert intrinsics to GPU memory
    ppx = cp.float32(intrinsics.ppx)
    ppy = cp.float32(intrinsics.ppy)
    fx = cp.float32(intrinsics.fx)
    fy = cp.float32(intrinsics.fy)

    # Compute world coordinates on GPU (No Loops 🚀)
    x_world = ((x_pixels - ppx) / fx) * nonzero_depths
    y_world = ((y_pixels - ppy) / fy) * nonzero_depths
    z_world = nonzero_depths  # Depth in meters

    # Stack into (N, 3) CuPy array
    points_3d = cp.column_stack((x_world, y_world, z_world))

    return points_3d  # Keeps everything on the GPU 🚀


def camera_to_robot_transform(pose):
    pose[0] += 0
    pose[1] += -0.2018
    pose[2] += 0.025

    return pose


CRATE_FRONT_PCD = load_crate_pc_front()
CRATE_FRONT_PCD.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

EUROPALLET_FRONT_PCD = load_europallet_pc_front()
EUROPALLET_FRONT_PCD.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))


def estimate_plane(pcd):
    # Step 3: Estimate Normals (needed for accurate RANSAC)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=50))

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.0125,  # Lower = more precise
        ransac_n=5,
        num_iterations=2500,  # Higher = more robust
    )
    plane_cloud = pcd.select_by_index(inliers)
    return plane_model, plane_cloud


def flatten_plane_cloud(pcd, model):
    normal = np.array([model[0], model[1], model[2]])
    rot_matrix = plane_to_rotation_matrix(normal, np.array([0, 1, 0]))
    pcd.rotate(rot_matrix, (0, 0, 0))
    points = np.asarray(pcd.points)
    points[:, 1] = np.mean(points[:, 1])
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.rotate(np.linalg.inv(rot_matrix), (0, 0, 0))
    return pcd


def ICP_crate(pcd):
    return ICP(pcd, CRATE_FRONT_PCD)


def ICP_pallet(pcd):
    return ICP(pcd, EUROPALLET_FRONT_PCD)

def ICP(pcd, target_pcd):
    # Perform ICP with rigid transformation (translation and rotation)
    threshold = 3  # Maximum distance threshold for matching points
    trans_init = np.eye(4)  # Initial transformation (identity matrix)

    icp_result = o3d.pipelines.registration.registration_icp(
        pcd,
        CRATE_FRONT_PCD,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # More accurate
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),  # Max iterations
    )
    return icp_result.transformation

