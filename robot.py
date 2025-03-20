import rtde_io
import rtde_control
import numpy as np
from time import sleep
from helper import *

controller = rtde_control.RTDEControlInterface("192.168.1.205")
io = rtde_io.RTDEIOInterface("192.168.1.205")

def moveToXYZ(x, y, z, speed=0.2, acc=0.5):
    controller.moveL([x, y, z, -2.23, 2.21, 0], speed, acc, True)


def moveTo(x, y, z, rx, ry, rz, speed=0.1, acc=0.1):
    controller.moveL([x, y, z, rx, ry, rz], speed, acc)


def moveToWithYaw(x, y, z, yaw, speed=0.1, acc=0.1):
    rx, ry, rz = rotate_ur10(180, 0, np.deg2rad(yaw))
    controller.moveL([x, y, z, rx, ry, rz], speed, acc)

def set_payload(mass):
    controller.setPayload(mass, [0,0,0])

def gripper_force_open():
    io.setStandardDigitalOut(3, False)
    sleep(0.01)
    io.setStandardDigitalOut(7, True)
    io.setStandardDigitalOut(6, True)
    sleep(0.01)
    io.setStandardDigitalOut(3, True)

def gripper_force_close():
    io.setStandardDigitalOut(3, False)
    sleep(0.01)
    io.setStandardDigitalOut(6, False)
    io.setStandardDigitalOut(7, False)
    sleep(0.01)
    io.setStandardDigitalOut(3, True)

def gripper_off():
    io.setStandardDigitalOut(3, False)
    sleep(0.01)
    io.setStandardDigitalOut(6, False)
    io.setStandardDigitalOut(7, False)


def moveDownWithForce(force):
    set_payload(1)
    task_frame = [0, 0, 0, 0, 0, 0]
    selection_vector = [0, 0, 1, 0, 0, 0]
    wrench_down = [0, 0, force, 0, 0, 0]
    wrench_up = [0, 0, -force, 0, 0, 0]
    limits = [2, 2, 1.5, 1, 1, 1]
    force_type = 2

    # Execute 500Hz control loop for 4 seconds, each cycle is 2ms
    for i in range(100):
        t_start = controller.initPeriod()
        controller.forceMode(task_frame,
                             selection_vector,
                             wrench_down,
                             force_type,
                             limits)
        controller.waitPeriod(t_start)
    controller.forceModeStop()
    set_payload(4)


def pickup(x, y, z, rx, ry, rz, speed=0.1, acc=0.1):
    moveTo(x, y, z, rx, ry, rz, speed=1.2, acc=0.5)

    gripper_force_open()
    moveTo(x, y, z - 0.070, rx, ry, rz, speed=0.05, acc=0.5)
    moveDownWithForce(-50)
    moveTo(x, y - 0.003, z - 0.050, rx, ry, rz, speed=0.05, acc=0.5)
    gripper_off()

    moveTo(x, y, z + 0.050, rx, ry, rz, speed, acc)

def place(x, y, z, rx, ry, rz, speed=0.1, acc=0.1):
    moveTo(x, y, z + 0.115, rx, ry, rz, speed, acc)
    moveTo(x, y, z + 0.005, rx, ry, rz, speed, acc)
    gripper_force_close()
    moveDownWithForce(-50)
    moveTo(x, y, z + 0.115, rx, ry, rz, speed, acc)
    gripper_off()

def pick_crate(pose):
    dx = pose[0]
    dy = pose[1]
    dz = pose[2]
    yaw = pose[3]

    robot_to_cam_y = 125 - 25 + 64.3 + 16 + 26 - 4.5
    robot_to_cam = [0, -robot_to_cam_y, 25]

    # https://www.ifco.com/media/IFCO-DS-1003-BLACK-LL-DE-EN.pdf
    crate_offset_x = (-360 / 2) * np.cos(yaw) - (-590 / 2) * np.sin(yaw)
    crate_offset_y = (-360 / 2) * np.sin(yaw) + (-590 / 2) * np.cos(yaw)
    crate_offset_z = 216

    gripper_x = 0
    gripper_y = 0
    gripper_z = 100

    x = dx + (robot_to_cam[0] + crate_offset_x + gripper_x) * 1e-3
    y = dy + (robot_to_cam[1] + crate_offset_y + gripper_y) * 1e-3
    z = dz + (robot_to_cam[2] + crate_offset_z + gripper_z) * 1e-3

    rx, ry, rz = rotate_ur10(180, 0, np.rad2deg(yaw - np.pi / 2))
    pickup(x, y, z, rx, ry, rz)


def place_crate(i):
    rx, ry, rz = rotate_ur10(180, 0, np.rad2deg(0))
    x = -0.311
    y = 0.992
    z = 0.385 + 0.205 * (int(i/4))

    if i%4 == 1:
        x += 0.605
    if i%4 == 2:
        y -= 0.405
    if i%4 == 3:
        x += 0.605
        y -= 0.405

    moveTo(x, y, z + 0.115, rx, ry, rz, speed=1.5, acc=0.5)
    place(x, y, z, rx, ry, rz)