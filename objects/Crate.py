import numpy as np


class Crate:
    def __init__(self, pose) -> None:
        self.pose = pose
        self.size = np.array([0.36, 0.59, 0.216])
