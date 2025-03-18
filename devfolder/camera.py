import zmq
import pyrealsense2 as rs
import cv2
import msgpack
import numpy as np
import msgpack_numpy as m
from ports import *

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
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

context = zmq.Context()

yolo_req_receiver = context.socket(zmq.PULL)
yolo_req_receiver.connect(f"tcp://localhost:{CAMERA_REQ_PORT}") 

yolo_rep_sender = context.socket(zmq.PUSH) 
yolo_rep_sender.bind(f"tcp://*:{CAMERA_SEND_PORT}")

def pack_image(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    data = msgpack.packb({"type": "image", "image": img_encoded.tobytes()}, default=m.encode)
    return data

def pack_depth(data):
    data = msgpack.packb({"type": "depth", 
                          "shape": data[0].shape, 
                          "depth": data[0].tobytes()}, default=m.encode)
    return data

def send_image(image, depth):
    print("Sending frame")    
    image_data = pack_image(image)
    depth_data = pack_depth(depth)
    yolo_rep_sender.send_multipart([image_data, depth_data])

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.rotate(color_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    color_image = cv2.resize(color_image, (540, 960))

    cv2.imshow("Current Raw Image", depth_image)
    cv2.waitKey(1)
    try:
        req = yolo_req_receiver.recv_string(zmq.NOBLOCK)
        send_image(color_image, depth_image)
        
    except:
        continue
    