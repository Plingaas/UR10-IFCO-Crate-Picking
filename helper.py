import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import cv2

DATA_FOLDER = "data"


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
    mesh = o3d.io.read_triangle_mesh(
        f"{DATA_FOLDER}/Europallet.obj"
    )  # Change to .ply or .obj if needed

    # Convert mesh to a point cloud using uniform sampling
    pcd = mesh.sample_points_uniformly(
        number_of_points=100000
    )  # Adjust point count as needed
    pcd_scale(pcd, 0.001, pcd.get_center())
    pcd_move_center_to(pcd, (0, 0, 0))
    pcd_rotate_x(pcd, 180)
    pcd_rotate_z(pcd, 90)
    pcd_move_center_to(pcd, (0.4, 0.6, 0))
    return pcd


def load_crate_pc():
    # Load the mesh file (replace with your file path)
    mesh = o3d.io.read_triangle_mesh(
        f"{DATA_FOLDER}/IFCOCrate.obj"
    )  # Change to .ply or .obj if needed

    # Convert mesh to a point cloud using uniform sampling
    pcd = mesh.sample_points_uniformly(
        number_of_points=250000
    )  # Adjust point count as needed
    pcd_move_center_to(pcd, (0.2, 0.3, 0.075))
    return pcd


def pcd_box_size(pcd, verbose=False):
    bounding_box = pcd.get_axis_aligned_bounding_box()
    bbox_min = bounding_box.get_min_bound()
    bbox_max = bounding_box.get_max_bound()
    bbox_size = bbox_max - bbox_min
    if verbose:
        print(
            f"Box size: {bbox_size}.\nx_min: {bbox_min[0]} x_max: {bbox_max[0]} \ny_min: {bbox_min[1]} y_max: {bbox_max[1]} \nz_min: {bbox_min[2]} z_max: {bbox_max[2]}"
        )
    return bbox_size


def pcd_remove_outliers(pcd, nn=20, std=1.0):
    pcd_clean, inliers = pcd.remove_statistical_outlier(nb_neighbors=nn, std_ratio=std)
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
    plt.bar(
        bin_edges[:-1], hist_counts, width=increment, color="blue", edgecolor="black"
    )

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
    plane_normal = plane_normal / np.linalg.norm(
        plane_normal
    )  # Normalize normal vector
    target_axis = target_axis / np.linalg.norm(
        target_axis
    )  # Ensure target is normalized

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
    rot_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    pcd.rotate(rot_matrix, (0, 0, 0))
    pcd_scale(pcd, 0.25, (0, 0, 0))
    return pcd


def pcd_set_color(pcd, color):
    if len(color) != 3:
        assert "Tried to set color of point cloud, but length of colors did not consist of 3 values."
    pcd.paint_uniform_color([color[0], color[1], color[2]])
    return pcd


def rotation_matrix_x(rad):
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]]
    )

    return R_x


def rotation_matrix_y(rad):
    R_y = np.array(
        [[np.cos(rad), 0, -np.sin(rad)], [0, 1, 0], [np.sin(rad), 0, np.cos(rad)]]
    )

    return R_y


def rotation_matrix_z(rad):
    R_z = np.array(
        [[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]]
    )

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
