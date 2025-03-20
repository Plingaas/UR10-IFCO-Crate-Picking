import rtde_io
import rtde_control
import numpy as np
from time import sleep
from helper import *

controller = rtde_control.RTDEControlInterface("192.168.1.205")

def moveToXYZ(x, y, z, speed=0.2, acc=0.5):
    controller.moveL([x / 1000, y / 1000, z / 1000, -2.23, 2.21, 0], speed, acc, True)


def moveTo(x, y, z, rx, ry, rz, speed=0.1, acc=0.1):
    controller.moveL([x / 1000, y / 1000, z / 1000, rx, ry, rz], speed, acc)


def moveToWithYaw(x, y, z, yaw, speed=0.1, acc=0.1):
    rx, ry, rz = rotate_ur10(180, 0, np.deg2rad(yaw))
    controller.moveL([x / 1000, y / 1000, z / 1000, rx, ry, rz], speed, acc)


def setOutput(pin, state):
    io = rtde_io.RTDEIOInterface("192.168.1.205")
    io.setStandardDigitalOut(pin, state)


def moveDownWithForce(force):
    task_frame = [0, 0, 0, 0, 0, 0]
    selection_vector = [0, 0, 1, 0, 0, 0]
    wrench_down = [0, 0, force, 0, 0, 0]
    wrench_up = [0, 0, -force, 0, 0, 0]
    limits = [2, 2, 1.5, 1, 1, 1]
    force_type = 2

    # Execute 500Hz control loop for 4 seconds, each cycle is 2ms
    for i in range(500):
        t_start = controller.initPeriod()
        # First move the robot down for 2 seconds, then up for 2 seconds
        if i < 300:
            controller.forceMode(
                task_frame, selection_vector, wrench_down, force_type, limits
            )
        controller.waitPeriod(t_start)


def pickup(x, y, z, rx, ry, rz, speed=0.1, acc=0.1):
    dx = 0
    dy = 0

    moveTo(x + dx, y + dy, z, rx, ry, rz, speed=1.2, acc=0.5)
    setOutput(7, True)
    moveTo(x + dx, y + dy, z - 75, rx, ry, rz, speed=0.2, acc=0.5)
    moveTo(x + dx, y + dy, z - 60, rx, ry, rz, speed=0.05, acc=0.5)
    setOutput(7, False)
    moveTo(x, y, z + 50, rx, ry, rz, speed, acc)


def place(x, y, z, rx, ry, rz, speed=0.1, acc=0.1):
    dx = 0
    dy = 0
    moveTo(x + dx, y + dy, z + 115, rx, ry, rz, speed, acc)
    moveTo(x + dx, y + dy, z + 50, rx, ry, rz, speed, acc)
    moveTo(x, y, z, rx, ry, rz, speed, acc)

    setOutput(7, False)
    sleep(0.5)
    moveTo(x + 1, y, z + 115, rx, ry, rz, speed, acc)


def pick_crate(dx, dy, dz, yaw):
    robot_to_cam_y = 125 - 25 + 64.3 + 16 + 26 - 4.5
    robot_to_cam = [0, -robot_to_cam_y, 25]

    # https://www.ifco.com/media/IFCO-DS-1003-BLACK-LL-DE-EN.pdf
    crate_offset_x = (-360 / 2) * np.cos(yaw) - (-590 / 2) * np.sin(yaw)
    crate_offset_y = (-360 / 2) * np.sin(yaw) + (-590 / 2) * np.cos(yaw)
    crate_offset_z = 216

    gripper_x = 0
    gripper_y = 0
    gripper_z = 100

    x = robot_to_cam[0] + dx + crate_offset_x + gripper_x
    y = robot_to_cam[1] + dy + crate_offset_y + gripper_y
    z = robot_to_cam[2] + dz + crate_offset_z + gripper_z

    rx, ry, rz = rotate_ur10(180, 0, np.rad2deg(yaw - np.pi / 2))
    pickup(x, y, z, rx, ry, rz)


def place_crate(i):
    rx, ry, rz = rotate_ur10(180, 0, np.rad2deg(0))
    x = -311
    y = 992

    if i == 1:
        x += 605
    if i == 2:
        y -= 405
    if i == 3:
        x += 605
        y -= 405

    moveTo(x, y, 500, rx, ry, rz, speed=1.2, acc=0.5)
    place(x, y, 385 - 3 * (i % 2), rx, ry, rz)
