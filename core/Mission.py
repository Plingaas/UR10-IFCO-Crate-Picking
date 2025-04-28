import numpy as np
import time
from motion.trajectories.joint_move import JointMove
from motion.trajectories.linear_move import LinearMove
from motion.trajectories.waypoint import Waypoint
from core.command import Command
from utils.helper import print_with_time

""" Class for any item """


class Item:
    def __init__(self, name, size, weight=1) -> None:
        self.name = name
        self.size = {"w": size[0], "l": size[1], "h": size[2]}
        self.weight = weight


""" Class for creating a customer order with an item list """


class Order:
    # Dummy API
    class API:
        def __init__(self) -> None:
            pass

        @staticmethod
        def fetch_latest_order():  # Dummy order  # noqa: ANN205
            print_with_time("Order", "Fetching customer order.")
            time.sleep(0.5)  # Simulate fetching time
            order = Order()
            ifco_crate_6420 = Item("IFCO_BLL6420", size=[0.36, 0.59, 0.216], weight=1.830)
            order.add_item(ifco_crate_6420, n=8)
            print_with_time("Order", "Customer order fetched.")
            print_with_time("Order", "vvvvvvvvvvvvvvvv Order vvvvvvvvvvvvvvvv")
            print_with_time("Item", "IFCO Crate BLL6420 (8pcs)")
            print_with_time("Order", "^^^^^^^^^^^^^^^^ Order ^^^^^^^^^^^^^^^^")
            return order

    def __init__(self) -> None:
        self.items = []
        self.picked = 0

    def add_item(self, item, n=1):
        for i in range(n):
            self.items.append(item)

    def get_size(self):
        return len(self.items)

    def get_item_at(self, index):
        return self.items[index]

    def get_remaining_picks(self):
        return self.get_size() - self.picked

    def is_finished(self):
        return self.picked == self.get_size()

    def update_picked(self):
        self.picked += 1

    def get_next_item(self, n=-1):
        if self.is_finished():
            return None
        n = self.picked if n == -1 else n  # Default value = self.picked
        return self.get_item_at(self.picked)


class MissionPlanner:
    def __init__(self, order) -> None:
        self.order = order

    def update_items_picked(self):
        self.order.update_picked()
        remaining = self.order.get_remaining_picks()
        print_with_time("MissionPlanner", f"Picked {self.order.picked}/{remaining} items.")
        return self.order.get_remaining_picks()

    def is_order_finished(self):
        return self.order.is_finished()

    def get_move_sequence(self, crate):
        pick_pose = self.get_pick_pose(crate)
        approach_move = self.get_approach_move()
        place_pose = self.get_place_pose()
        return_move = self.get_return_move()

        command = Command()
        command.set_pick_move(pick_pose)
        command.set_approach_move(approach_move)
        command.set_place_move(place_pose)
        command.set_return_move(return_move)

        return command

    def get_pick_pose(self, crate):
        yaw = crate.pose[3]
        width = -crate.size[0] * 0.5
        length = -crate.size[1] * 0.5
        ox = width * np.cos(yaw) - length * np.sin(yaw)
        oy = width * np.sin(yaw) + length * np.cos(yaw)

        pose = np.array([crate.pose[0] + ox, crate.pose[1] + oy, crate.pose[2] + crate.size[2], crate.pose[3]])
        return pose

    def get_approach_move(self):
        jmove = JointMove()
        jmove.add_waypoint(Waypoint([2.129, -1.919, 1.867, -1.518, 4.712, -2.234], "fast", "fast", 0.5))
        jmove.add_waypoint(Waypoint([3.473, -1.884, 1.640, -1.291, 4.712, -1.274], "fast", "fast", 0.0))
        return jmove

    def get_place_pose(self):
        i = self.order.picked
        x = -0.311
        y = 0.992
        z = 0.285 + 0.205 * (int(i / 4))

        if i % 4 == 1:
            x += 0.605
        if i % 4 == 2:
            y -= 0.405
        if i % 4 == 3:
            x += 0.605
            y -= 0.405

        return [x, y, z, 0]

    def get_return_move(self):
        lmove = LinearMove()
        lmove.add_waypoint(Waypoint([0.550, 0.500, 0.600, 3.141, 0, 0], "fast", "fast", 0.25))
        lmove.add_waypoint(Waypoint([0.550, 0.000, 0.600, 3.141, 0, 0], "fast", "fast", 0.0))
        return lmove

    def go_home(self):
        jmove = JointMove()
        jmove.add_waypoint(Waypoint([3.473, -1.884, 1.640, -1.291, 4.712, -1.274], "fast", "fast", 0.0))
        return jmove
