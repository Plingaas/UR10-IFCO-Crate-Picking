# UR10 IFCO Crate Picking

Autonomous robotic system for picking IFCO crates using a UR10 robot, developed for Solwr.

## Overview

This project implements a full pipeline for detecting, localizing, and picking IFCO crates using:

- Intel RealSense L515 camera
- YOLO-based object segmentation
- 3D pose estimation
- Point cloud processing
- RTDE control of a UR10 robot arm

The system autonomously:

- Detects crates in RGBD images
- Estimates crate poses
- Plans and executes robot pick-and-place motions

## Requirements

- Python : 3.10.0
- CUDA Toolkit : v12.x
- Torch : Torch + cuda 12.x https://pytorch.org/get-started/locally/

All core dependencies are listed in `requirements.txt`. To install them:

```bash
pip install -r requirements.txt
```

- Ultralytics : ultralytics==8.3.85
- Open3D : open3d==0.19.0
- OpenCV : opencv-python==4.11.0.86
- NumPy : numpy==1.26.4
- CuPy : cupy-cuda12x==13.4.0
- PyQt5 : PyQt==0.13.7
- pyqtgraph : pyqtgraph==0.13.7
- pyrealsense2 : pyrealsense2==2.53.1.4623
- ur-rtde : ur-rtde==1.6.0

## See the robot in action

https://www.youtube.com/watch?v=1e_nvYarGSw
