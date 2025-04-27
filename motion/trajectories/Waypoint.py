class Waypoint:
    """self, pose"""

    def __init__(self, pose, spd, acc, blend) -> None:
        self.pose = pose
        self.speed = spd
        self.acceleration = acc
        self.blend = blend

    def to_array(self):
        return [*self.pose, self.speed, self.acceleration, self.blend]
