import zmq
import cv2
import numpy as np
import msgpack
import msgpack_numpy as m
from ports import *

print("Setting up Yolo")
from ultralytics import YOLO

try:
    model = YOLO("C:/Documents/yolo/venv3.11/runs/segment/train12/weights/best.pt")
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


while True:
    camera_req_sender.send_string("I want a frame")
    image, depth = camera_rep_receiver.recv_multipart()
    img_data = msgpack.unpackb(image, object_hook=m.decode)
    depth_data = msgpack.unpackb(depth, object_hook=m.decode)

    img_data = img_data["image"]
    depth_shape = depth_data["shape"]
    depth_data = depth_data["depth"]

    color_image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    depth_image = np.frombuffer(depth_data, np.uint16).reshape(depth_shape)

    results = model.predict(color_image, show=False, conf=0.6, device=0)
    binary_masks = []
    for result in results:
        if result.masks:
            masks = result.masks.data.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            boxes = result.boxes.xyxy.cpu().numpy()

            mask_overlay = np.zeros_like(color_image, dtype=np.uint8)

            for i, mask in enumerate(masks):
                color = np.random.randint(
                    0, 255, (3,), dtype=np.uint8
                )  # Random color for each mask
                x1, y1, x2, y2 = map(int, boxes[i])
                label = f"Class {int(class_ids[i])}: {confidences[i]:.2f}"
                cv2.putText(
                    color_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color.tolist(), 2)

                mask = cv2.resize(mask, (1080, 1920), interpolation=cv2.INTER_NEAREST)

                mask_grayscale = (mask > 0.001).astype(np.uint8) * ((i + 1) * 40)
                mask_binary = ((mask > 0.001).astype(np.uint8) * 255) // 255
                mask_binary = (mask > 0.5).astype(
                    np.uint8
                ) * 255  # Convert mask to binary

                contours, _ = cv2.findContours(
                    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in contours:
                    cv2.drawContours(
                        mask_overlay, [cnt], -1, color.tolist(), thickness=cv2.FILLED
                    )

            color_image = cv2.addWeighted(color_image, 0.7, mask_overlay, 0.3, 0.0)

    color_image = cv2.resize(color_image, (540, 960))
    depth_image = cv2.resize(
        cv2.rotate(depth_image, cv2.ROTATE_90_COUNTERCLOCKWISE), (540, 960)
    )
    grayscale_8bit = cv2.convertScaleAbs(depth_image, alpha=(255.0 / 65535.0))
    depth_image = cv2.cvtColor(grayscale_8bit, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("yolo.png", color_image)
    stacked = np.hstack((color_image, depth_image))
    cv2.imshow("Yolo frame", stacked)
    cv2.waitKey(1)

    data = pack_data()
    pc_extract_req_receiver.recv_string()  # Wait for request
