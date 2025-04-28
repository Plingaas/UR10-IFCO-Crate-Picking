import threading
from concurrent.futures import Future
import queue
import numpy as np
from time import sleep
from rtde_control import RTDEControlInterface
from rtde_io import RTDEIOInterface
from utils.helper import rotate_ur10, print_with_time
from motion.motion_profile import MotionProfile
from motion.trajectories.linear_move import LinearMove
from motion.trajectories.joint_move import JointMove
from motion.trajectories.waypoint import Waypoint


class RobotController(threading.Thread):
    def __init__(self, ip="192.168.1.205") -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.command_queue = queue.Queue()
        self.task_queue = queue.Queue()
        self.speed_profile = MotionProfile(slow=0.1, normal=0.7, fast=1.5)
        self.acc_profile = MotionProfile(slow=0.1, normal=0.5, fast=1.0)
        self.running = True
        self.daemon = True
        self.active = False

    def connect(self, ip):
        self.controller = RTDEControlInterface(ip)
        self.io = RTDEIOInterface(ip)
        self.start()

    def run(self):
        while self.running:
            try:
                with self.lock:
                    command = self.command_queue.get_nowait()
            except queue.Empty:
                sleep(0.01)
                continue

            self.run_command(command)

    def enqueue(self, func):
        future = Future()
        self.task_queue.put((func, future))
        return future

    def stop(self):
        self.running = False
        self.controller.stopScript()
        self.controller.disconnect()
        self.io.disconnect()

    def wait_for_idle(self):
        while not self.task_queue.empty():
            sleep(0.05)

    def reconnect(self):
        self.controller.reconnect()
        self.io.reconnect()

    def disconnect(self):
        self.controller.disconnect()
        self.io.disconnect()

    def add_command(self, command):
        with self.lock:
            self.command_queue.put(command)

    # === Robot actions ===

    def run_command(self, command):
        with self.lock:
            self.active = True
        pick_pose = command.pick_move
        approach_move = command.approach_move
        place_pose = command.place_move
        return_move = command.return_move

        # Pick up crate
        self.pick_crate(pick_pose)

        # Get ready for placing
        moves = self.replace_labels(approach_move.get_moves())  # Convert 'fast' etc to numbers
        if type(approach_move) is JointMove:
            self.controller.moveJ(moves)
        elif type(approach_move) is LinearMove:
            self.controller.moveL(moves)

        # Crate is picked from pallet an should be out of frame.
        if command.crate_picked_callback is not None:
            command.crate_picked_callback()

        self.place_crate(place_pose)

        moves = self.replace_labels(return_move.get_moves())  # Convert 'fast' etc to numbers
        if type(return_move) is JointMove:
            self.controller.moveJ(moves)
        elif type(return_move) is LinearMove:
            self.controller.moveL(moves)

        if command.crate_placed_callback is not None:
            command.crate_placed_callback()

        with self.lock:
            self.active = False

    def replace_labels(self, arr):
        for i, _ in enumerate(arr):
            arr[i][6] = self.speed_profile.dict[arr[i][6]]
            arr[i][7] = self.acc_profile.dict[arr[i][7]]
        return arr

    def move_to(self, x, y, z, rx, ry, rz, speed="normal", acc="normal"):
        v = self.speed_profile.get_value(speed)
        a = self.acc_profile.get_value(acc)
        self.controller.moveL([x, y, z, rx, ry, rz], v, a)

    def move_to_with_yaw(self, x, y, z, yaw, speed=0.1, acc=0.1):
        rx, ry, rz = rotate_ur10(180, 0, np.deg2rad(yaw))
        self.controller.moveL([x, y, z, rx, ry, rz], speed, acc)

    def set_payload(self, mass):
        self.controller.setPayload(mass, [0, 0, 0])

    def move_down_with_force(self, force):
        self.setPayload(1)
        tf = [0, 0, 0, 0, 0, 0]
        sv = [0, 0, 1, 0, 0, 0]
        wrench = [0, 0, force, 0, 0, 0]
        limits = [2, 2, 1.5, 1, 1, 1]
        for _ in range(50):
            t = self.controller.initPeriod()
            self.controller.forceMode(tf, sv, wrench, 2, limits)
            self.controller.waitPeriod(t)
        self.controller.forceModeStop()
        self.setPayload(8)

    def pick_crate(self, pose):
        x = pose[0]
        y = pose[1]
        z = pose[2]
        yaw = pose[3]

        print_with_time("Robot", f"Picking up crate at x:{x}, y:{y}, z:{z}.")
        rx, ry, rz = rotate_ur10(180, 0, np.rad2deg(yaw - np.pi / 2))
        self.move_to(x, y, z, rx, ry, rz, speed="fast", acc="fast")
        self.gripper_force_open()
        self.move_to(x, y, z - 0.070, rx, ry, rz, speed="normal", acc="slow")
        self.move_down_with_force(-20)
        self.move_to(x, y - 0.003, z - 0.050, rx, ry, rz, speed="normal", acc="slow") # y - 0.003 to ensure back finger engages properly.
        self.gripper_off()
        self.move_to(x, y, z + 0.050, rx, ry, rz, speed="normal", acc="normal")

    def place_crate(self, pose):
        x = pose[0]
        y = pose[1]
        z = pose[2]
        yaw = pose[3]

        print_with_time("Robot", f"Placing crate at x:{x}, y:{y}, z:{z}.")
        rx, ry, rz = rotate_ur10(180, 0, np.rad2deg(yaw))
        lmove = LinearMove()
        lmove.add_waypoint(Waypoint([x, y, z + 0.115, rx, ry, rz], "fast", "normal", 0.1))
        lmove.add_waypoint(Waypoint([x, y, z + 0.005, rx, ry, rz], "normal", "normal", 0.0))
        moves = self.replace_labels(lmove.get_moves())
        self.controller.moveL(moves)
        self.gripper_force_close()
        self.move_down_with_force(-50)
        self.move_to(x, y, z + 0.115, rx, ry, rz)
        self.gripper_off()

    def go_home(self, speed="slow"):
        self.move_to(0.500, 0.000, 0.500, 3.141, 0.0, 0.0, speed, speed)

    def gripper_force_open(self):
        self.io.setStandardDigitalOut(3, False)
        sleep(0.01)
        self.io.setStandardDigitalOut(7, True)
        self.io.setStandardDigitalOut(6, True)
        sleep(0.01)
        self.io.setStandardDigitalOut(3, True)

    def gripper_force_close(self):
        self.io.setStandardDigitalOut(3, False)
        sleep(0.01)
        self.io.setStandardDigitalOut(6, False)
        self.io.setStandardDigitalOut(7, False)
        sleep(0.01)
        self.io.setStandardDigitalOut(3, True)

    def gripper_off(self):
        self.io.setStandardDigitalOut(3, False)
        sleep(0.01)
        self.io.setStandardDigitalOut(6, False)
        self.io.setStandardDigitalOut(7, False)

    def shutdown(self):
        self.stop()
