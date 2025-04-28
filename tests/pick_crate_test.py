from core.robot_controller import RobotController
from core.mission import MissionPlanner, Order
from objects.crate import Crate
import time


# Create controller
robot = RobotController()
robot.connect("192.168.1.205")

# Get latest order and create mission
order = Order.API.fetch_latest_order()
mp = MissionPlanner(order)

# Create fake crate at hardcoded position
crate = Crate([0.2, -0.6, -0.1, 0.0])
command = mp.get_move_sequence(crate)

# Add command to queue
robot.add_command(command)

# Wait for robot to finish
while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        robot.shutdown()
