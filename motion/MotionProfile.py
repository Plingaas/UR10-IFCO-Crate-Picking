class MotionProfile:
    def __init__(self, slow=0.03, normal=0.1, fast=0.5) -> None:
        self.set_params(slow, normal, fast)

    def set_params(self, slow, normal, fast) -> None:
        self.slow = slow
        self.normal = normal
        self.fast = fast
        self.dict = {"slow": self.slow, "normal": self.normal, "fast": self.fast}

    def get_value(self, speed_type) -> float:
        return self.dict[speed_type]
