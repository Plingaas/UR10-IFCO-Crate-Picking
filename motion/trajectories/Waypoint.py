class Waypoint:

    def __init__(self, pose) -> None:
        self.pose = pose
        self.speed = 0
        self.acceleration = 0

    def set_speed(self, speed):
        self.speed = speed

    def set_acceleration(self, acceleration):
        self.acceleration = acceleration
