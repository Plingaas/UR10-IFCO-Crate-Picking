from RealsenseD415 import RealsenseD415
import cv2
import numpy as np

cam = RealsenseD415()
cam.enable_depth_camera()
cam.enable_rgb_camera()
cam.start_streaming()

while True:
    frames = cam.get_frames()
    aligned_depth = cam.align_frames(frames)
    color_img = cam.get_color_frame(frames)
    depth_img = cam.get_depth_frame(aligned_depth)

    color_frame_cpu = np.asarray(color_img)
    color_image_rotated = cv2.rotate(color_frame_cpu, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow("color", cv2.resize(color_image_rotated, (540, 960)))

    depth_img = np.asarray(depth_img)
    depth_img = depth_img * 5
    depth_image_rotated = cv2.rotate(depth_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow("depth", cv2.resize(depth_image_rotated, (540, 960)))
    cv2.waitKey(1)