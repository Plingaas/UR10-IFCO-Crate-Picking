from MotionProfile import MotionProfile
from trajectories.JointMove import JointMove
from trajectories.LinearMove import LinearMove

class MotionPlan:

    def __init__(self) -> None:
        self.motion_profile = MotionProfile()
        self.moves = []
        self.state = 0

    def add_move(self, move):
        if type(move) is LinearMove or type(move) is JointMove:
            self.moves.append(move)

    def set_speed_profile(self, slow, normal, fast):
        self.speed_profile.set_params(slow, normal, fast)

    def set_acceleration_profile(self, slow, normal, fast):
        self.accleration_profile.set_params(slow, normal, fast)

    def clear_moves(self):
        self.moves = []

