import rtde_control
import numpy as np
from motion.trajectories.linear_move import LinearMove
from motion.trajectories.waypoint import Waypoint

# Connect to robot
controller = rtde_control.RTDEControlInterface("192.168.1.205")

# Define waypoints (list of joint angles, in radians)
jmove = LinearMove()
jmove.add_waypoint(Waypoint([0.500, -0.200, 0.500, np.pi, 0.0, 0.0], 0.5, 0.5, 0.1))
jmove.add_waypoint(Waypoint([0.000, -0.600, 0.500, np.pi, 0.0, 0.0], 0.5, 0.5, 0.1))
jmove.add_waypoint(Waypoint([0.250, -0.400, 0.500, np.pi, 0.0, 0.0], 0.5, 0.5, 0.1))
jmove.add_waypoint(Waypoint([0.500, -0.200, 0.500, np.pi, 0.0, 0.0], 0.5, 0.5, 0.0))

# Move between points
controller.moveJ(jmove.get_moves())

# Disconnect
controller.stopScript()
