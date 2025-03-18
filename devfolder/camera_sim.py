import zmq
import cv2
import msgpack
import msgpack_numpy as m
import cupy as cp
from ports import *

# Initialize ZeroMQ context & sockets
context = zmq.Context()
yolo_req_receiver = context.socket(zmq.PULL)
yolo_req_receiver.connect(f"tcp://localhost:{CAMERA_REQ_PORT}")

yolo_rep_sender = context.socket(zmq.PUSH)
yolo_rep_sender.bind(f"tcp://*:{CAMERA_SEND_PORT}")


def pack_image(image_gpu):
    """Encodes and packs an RGB image (CuPy) for ZeroMQ"""
    image_cpu = cp.asnumpy(image_gpu)  # Move to CPU for encoding
    _, img_encoded = cv2.imencode(".jpg", image_cpu)
    return msgpack.packb(
        {"type": "image", "image": img_encoded.tobytes()}, default=m.encode
    )


def pack_depth(depth_gpu):
    """Packs a depth image (CuPy) for ZeroMQ"""
    depth_cpu = cp.asnumpy(depth_gpu)  # Move to CPU
    return msgpack.packb(
        {"type": "depth", "shape": depth_cpu.shape, "depth": depth_cpu.tobytes()},
        default=m.encode,
    )


def send_image(image_gpu, depth_gpu):
    """Sends the processed RGB & depth images via ZeroMQ"""
    print("📤 Sending frame...")
    image_data = pack_image(image_gpu)
    depth_data = pack_depth(depth_gpu)
    yolo_rep_sender.send_multipart([image_data, depth_data])


while True:
    # Load images to GPU
    color_image_gpu = cp.asarray(
        cv2.imread("devfolder/color.png", cv2.IMREAD_UNCHANGED), dtype=cp.uint8
    )
    depth_image_gpu = cp.asarray(
        cv2.imread("devfolder/depth.png", cv2.IMREAD_UNCHANGED), dtype=cp.uint16
    )

    # Rotate color image on GPU
    color_image_gpu = cp.rot90(color_image_gpu, k=-1)  # Rotate 90° counterclockwise

    try:
        req = yolo_req_receiver.recv_string(zmq.NOBLOCK)  # Non-blocking receive
        send_image(color_image_gpu, depth_image_gpu)
    except zmq.Again:
        pass
