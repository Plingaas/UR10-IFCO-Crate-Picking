import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import zmq
from ports import *

context = zmq.Context()

# Send requests to YOLO layer
yolo_req_sender = context.socket(zmq.PUSH)
yolo_req_sender.bind(f"tcp://*:{YOLO_REQ_PORT}")

# Receive response from YOLO layer
yolo_req_receiver = context.socket(zmq.PULL)
yolo_req_receiver.connect(f"tcp://localhost:{YOLO_SEND_PORT}")

# Receive request from PC Processing layer
# pc_req_receiver = context.socket(zmq.PULL)
# pc_req_receiver.connect(f"tcp://localhost:{PC_REQ_PORT}")

# Send response to PC Processing layer
# pc_req_sender = context.socket(zmq.PUSH)
# pc_req_sender.bind(f"tcp://localhost:{PC_SEND_PORT}")

mask_binary = None
depth_image = None
depth_intrinsics = rs.intrinsics()
depth_intrinsics.width = 1920
depth_intrinsics.height = 1080
depth_intrinsics.ppx = 982.673
depth_intrinsics.ppy = 547.614
depth_intrinsics.fx = 1352.92
depth_intrinsics.fy = 1354.45
depth_intrinsics.model = rs.distortion.brown_conrady
depth_intrinsics.coeffs = [0.171722, -0.526011, -0.000589736, -0.000417008, 0.486631]

while True:
    yolo_req_sender.send_string("I want a yolo frame")
    yolo_req_receiver.recv_multipart()
    depth_masked = cv2.bitwise_and(depth_image, depth_image, mask=mask_binary)
    # Get nonzero depth indices and values
    nonzero_indices = np.nonzero(depth_masked)  # (row, col) indices
    nonzero_depths = depth_masked[nonzero_indices] / 1000.0  # Convert mm to meters

    # Extract x and y pixel coordinates
    x_pixels = nonzero_indices[1]  # Column indices correspond to x-coordinates
    y_pixels = nonzero_indices[0]  # Row indices correspond to y-coordinates

    # Compute 3D coordinates using intrinsics
    x_world = (x_pixels - depth_intrinsics.ppx) / depth_intrinsics.fx * nonzero_depths
    y_world = (y_pixels - depth_intrinsics.ppy) / depth_intrinsics.fy * nonzero_depths
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
