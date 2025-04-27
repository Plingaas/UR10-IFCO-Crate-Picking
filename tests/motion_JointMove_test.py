import rtde_control
from motion.trajectories.JointMove import JointMove
from motion.trajectories.Waypoint import Waypoint

# Connect to robot
controller = rtde_control.RTDEControlInterface("192.168.1.205")

# Define waypoints (list of joint angles, in radians)
jmove = JointMove()
jmove.add_waypoint(Waypoint([2.129, -1.919, 1.867, -1.518, 4.712, -2.234], 0.5, 0.5, 0.1))
jmove.add_waypoint(Waypoint([2.775, -1.919, 1.884, -1.518, 4.712, -1.902], 0.5, 0.5, 0.1))
jmove.add_waypoint(Waypoint([3.473, -1.884, 1.640, -1.291, 4.712, -1.274], 0.5, 0.5, 0.0))

# Move between points
controller.moveJ(jmove.get_moves())

# Disconnect
controller.stopScript()
