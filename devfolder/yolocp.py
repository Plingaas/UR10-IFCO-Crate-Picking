import zmq
import cv2
import numpy as np
import cupy as cp
import msgpack
import msgpack_numpy as m
from ports import *
import cupyx.scipy.ndimage as cpx

print("🚀 Setting up YOLO")
from ultralytics import YOLO

# Load YOLO Model
try:
    model = YOLO("C:/Documents/yolo/venv3.11/runs/segment/train12/weights/best.pt")
    print("✅ YOLO model loaded successfully")
except:
    assert "❌ Failed to load YOLO model .pt file"
    exit(-1)

# Dummy image for YOLO initialization
setup_image = np.zeros((640, 384)).astype(np.uint8)
setup_image = cv2.cvtColor(setup_image, cv2.COLOR_GRAY2BGR)
model.predict(setup_image, conf=0.8, device=0, verbose=False)
print("✅ YOLO finished setup")

# ZeroMQ Setup
context = zmq.Context()
camera_req_sender = context.socket(zmq.PUSH)
camera_req_sender.bind(f"tcp://*:{CAMERA_REQ_PORT}")

camera_rep_receiver = context.socket(zmq.PULL)
camera_rep_receiver.connect(f"tcp://localhost:{CAMERA_SEND_PORT}")

pc_extract_req_receiver = context.socket(zmq.PULL)
pc_extract_req_receiver.connect(f"tcp://localhost:{YOLO_REQ_PORT}")

pc_extract_rep_sender = context.socket(zmq.PUSH)
pc_extract_rep_sender.bind(f"tcp://localhost:{YOLO_SEND_PORT}")


def pack_data(depth_gpu):
    """Packs depth image (CuPy) for ZeroMQ"""
    depth_cpu = cp.asnumpy(depth_gpu)  # Convert back to CPU
    return msgpack.packb(
        {"type": "depth", "shape": depth_cpu.shape, "depth": depth_cpu.tobytes()},
        default=m.encode,
    )


def send_data(image_gpu, depth_gpu):
    """Sends processed RGB & depth images via ZeroMQ"""
    pc_extract_rep_sender.send_multipart([])


while True:
    camera_req_sender.send_string("I want a frame")

    # Receive RGB & Depth images
    image, depth = camera_rep_receiver.recv_multipart()
    img_data = msgpack.unpackb(image, object_hook=m.decode)
    depth_data = msgpack.unpackb(depth, object_hook=m.decode)

    img_data = img_data["image"]
    depth_shape = depth_data["shape"]
    depth_data = depth_data["depth"]

    # Move images to GPU
    color_image_gpu = cp.asarray(
        cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR),
        dtype=cp.uint8,
    )
    depth_image_gpu = cp.asarray(
        np.frombuffer(depth_data, np.uint16).reshape(depth_shape), dtype=cp.uint16
    )

    # YOLO Segmentation
    results = model.predict(cp.asnumpy(color_image_gpu), show=False, conf=0.6, device=0)

    binary_masks = []
    mask_overlay_gpu = cp.zeros_like(color_image_gpu, dtype=cp.uint8)
    for result in results:
        if result.masks:
            masks_gpu = cp.asarray(
                result.masks.data
            )  # Convert mask tensor directly to GPU
            class_ids_gpu = cp.asarray(result.boxes.cls)
            confidences_gpu = cp.asarray(result.boxes.conf)
            boxes_gpu = cp.asarray(result.boxes.xyxy)

            # Resize masks using CuPy (Avoid cv2.resize CPU overhead)
            resized_masks_gpu = cpx.zoom(
                masks_gpu,
                (1, 1080 / masks_gpu.shape[1], 1920 / masks_gpu.shape[2]),
                order=0,
            )

            # Binary mask processing (on GPU)
            mask_binary_gpu = (resized_masks_gpu > 0.001).astype(cp.uint8) * 255

            # Apply depth mask (Bitwise AND on GPU)
            depth_masked_gpu = depth_image_gpu * (
                mask_binary_gpu / 255
            )  # Multiply instead of bitwise_and

    # Convert GPU image to CPU for display
    depth_masked_cpu = cp.asnumpy(depth_masked_gpu).squeeze()

    print(depth_masked_cpu.shape)
    print(depth_masked_cpu.dtype)
    # Normalize depth for visualization (0-255 range)
    depth_masked_normalized = cv2.normalize(
        depth_masked_cpu, None, 0, 255, cv2.NORM_MINMAX
    )
    depth_masked_normalized = depth_masked_normalized.astype(np.uint8)

    # Display using OpenCV
    cv2.imshow("Depth Masked Image", depth_masked_cpu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
