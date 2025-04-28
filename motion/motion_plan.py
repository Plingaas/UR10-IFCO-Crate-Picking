from MotionProfile import MotionProfile
from trajectories.JointMove import JointMove
from trajectories.LinearMove import LinearMove


class MotionPlan:
    def __init__(self) -> None:
        self.allowed_moves = [LinearMove, JointMove]
        self.speed_profile = MotionProfile()
        self.acceleration_profile = MotionProfile()
        self.moves = []
        self.state = 0

    def add_move(self, move):
        if type(move) in self.allowed_moves:
            self.moves.append(move)

    def set_speed_profile(self, slow, normal, fast):
        self.speed_profile.set_params(slow, normal, fast)

    def set_acceleration_profile(self, slow, normal, fast):
        self.accleration_profile.set_params(slow, normal, fast)

    def get_moves(self):
        return self.moves

    def clear(self):
        self.moves = []
