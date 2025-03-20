
"""
rtde_control.servoJ(joint_q, velocity, acceleration, dt, lookahead_time, gain);

Remember that to allow for a fast control rate when servoing, the joint positions must be close to each other e.g. (dense trajectory).
If the robot is not reaching the target fast enough try to increase the acceleration or the gain parameter.
"""

class JointMove:

    def __init__(self, waypoints):
        self.waypoints = waypoints