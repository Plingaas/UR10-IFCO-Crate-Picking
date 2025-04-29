from motion.trajectories.move import Move


class JointMove(Move):
    """Creates a path from waypoints with blending"""

    def __init__(self) -> None:
        """Initializes a LinearMove with waypoints"""
        super().__init__()
        self.waypoints = []

    def add_waypoint(self, waypoint):
        """Adds a waypoint to the path"""
        self.waypoints.append(waypoint)

    def get_moves(self):
        """Creates a path from waypoints"""
        return [wp.to_array() for wp in self.waypoints]
