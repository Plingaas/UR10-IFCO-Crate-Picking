import zmq
import cv2
import numpy as np
import msgpack
import msgpack_numpy as m
from ports import *
from time import time
import cupy as cp

print("Setting up Yolo")
from ultralytics import YOLO

try:
    model = YOLO("C:/Documents/yolo/venv3.11/runs/segment/train12/weights/best.pt").to("cuda")
    print("YOLO model loaded successfully")
except:
    assert "Failed to load YOLO model .pt file"
    exit(-1)

setup_image = np.zeros((640, 384)).astype(np.uint8)
setup_image = cv2.cvtColor(setup_image, cv2.COLOR_GRAY2BGR)
model.predict(setup_image, conf=0.8, device=0, verbose=False)
print("Yolo finished setup")
context = zmq.Context()

camera_req_sender = context.socket(zmq.PUSH)
camera_req_sender.bind(f"tcp://*:{CAMERA_REQ_PORT}")

camera_rep_receiver = context.socket(zmq.PULL)
camera_rep_receiver.connect(f"tcp://localhost:{CAMERA_SEND_PORT}")

pc_extract_req_receiver = context.socket(zmq.PULL)
pc_extract_req_receiver.connect(f"tcp://localhost:{YOLO_REQ_PORT}")

pc_extract_rep_sender = context.socket(zmq.PUSH)
pc_extract_rep_sender.bind(f"tcp://localhost:{YOLO_SEND_PORT}")


def pack_data(data):
    data = msgpack.packb(
        {"type": "depth", "shape": data[0].shape, "depth": data[0].tobytes()},
        default=m.encode,
    )
    return data


def send_data(image, depth):
    pc_extract_rep_sender.send_multipart([])

t = time()
fps = 0
n = 0
while True:
    n += 1
    if (time() - t >= 1.0):
        t += 1.0
        fps = n
        n = 0
        print(fps)
    camera_req_sender.send_string("I want a frame")
    image, depth = camera_rep_receiver.recv_multipart()
    img_data = msgpack.unpackb(image, object_hook=m.decode)
    depth_data = msgpack.unpackb(depth, object_hook=m.decode)

    img_data = img_data["image"]
    depth_shape = depth_data["shape"]
    depth_data = depth_data["depth"]

    color_image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    depth_image_gpu = cp.asarray(np.frombuffer(depth_data, np.uint16).reshape(depth_shape))
    depth_image_gpu = cp.rot90(depth_image_gpu)
    color_image_gpu = cp.asarray(color_image)
    
    results = model.predict(color_image, show=False, retina_masks = True, conf=0.6, device=0, stream = True)
    for result in results:
        if result.masks:
            masks = cp.asarray(result.masks.data)
            class_ids = cp.asarray(result.boxes.cls) 

            for i, mask in enumerate(masks):
                mask_binary = (mask > 0).astype(cp.uint16)
                depth_masked = depth_image_gpu * mask_binary


    #data = pack_data()
    #pc_extract_req_receiver.recv_string()  # Wait for request
