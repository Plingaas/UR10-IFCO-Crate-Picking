import numpy as np


class Pallet:
    ID = 1
    def __init__(self, pose) -> None:
        self.pose = pose
        self.size = np.array([0.8, 1.2, 0.144])
