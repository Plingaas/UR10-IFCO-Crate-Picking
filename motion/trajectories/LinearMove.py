from Waypoint import Waypoint

class LinearMove:
    """ Creates a path from waypoints with blending """

    def __init__(self, waypoints=[], blend=0.0) -> None:
        """ Initializes a LinearMove with waypoints """
        self.waypoints = waypoints
        self.blend = blend
        self.create_path()

    def add_waypoint(self, waypoint):
        """ Adds a waypoint to the path """
        self.waypoints.append(waypoint)

    def set_blend(self, blend):
        """ Sets the blend radius in meters range[0.0-2.0] """
        self.blend = max(0.0, min(2.0, blend))

    def create_path(self):
        """ Creates a path from waypoints """
        self.path = [[*waypoint.pose,
                      waypoint.speed,
                      waypoint.acceleration,
                      self.blend]
                      for waypoint in self.waypoints
                    ]

"""
example
waypoints = []
speed = 0.1
acc = 0.1
pose =[0.0, -0.500, 0.700, 3.14, 0, 0]
waypoint = Waypoint(pose)
waypoint.set_speed(speed)
waypoint.set_acceleration(acc)
waypoints.append(waypoint)
linear_move = LinearMove(waypoints, blend=50*1e-3)
"""
