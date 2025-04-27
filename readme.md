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

## Project Structure

## Requirements

# External

- Python : 3.10.0
- CUDA Toolkit : v12.5

# PIP Packages

- Ultralytics : 8.3.85
- Open3D : 0.19.0
- OpenCV : 4.11.0
- NumPy : 1.26.4
- CuPy : 13.4.0
- PyQt5 : 0.13.7
- pyrealsense2 : 2.53.1.4623
- ur-rtde

## Setup

1. Install Python requirements:

```bash
pip install -r requirements.txt
```

https://www.youtube.com/watch?v=1e_nvYarGSw
