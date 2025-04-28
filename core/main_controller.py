import cv2
import threading
import cupy as cp
import open3d as o3d
from utils.helper import draw_yolo_detections, depth_to_colormap, screenshot_o3d, print_with_time
from core.display_grid import DisplayGrid
from core.camera import RealsenseL515
from core.yolo_segmenter import YoloSegmenter
from core.point_cloud_extractor import PointCloudExtractor
from core.estimator import PoseEstimator
from core.robot_controller import RobotController
from core.mission import MissionPlanner, Order


class MainController:
    def __init__(self) -> None:
        self.ready_for_analyzing = True
        self.lock = threading.Lock()
        self.finished = False

    def setup(self):
        # Fetch order from dummy api
        self.customer_order = Order.API.fetch_latest_order()
        self.mission_planner = MissionPlanner(order=self.customer_order)

        # Start camera thread
        print_with_time("Main", "Configuring Realsense L515.")
        self.cam = RealsenseL515()
        self.cam.init()

        # Start robot controller thread
        print_with_time("Main", "Connecting to UR10.")
        self.robot = RobotController()
        self.robot.connect("192.168.1.205")
        self.robot.go_home(speed="normal")

        # Processing objects
        self.yolo = YoloSegmenter("best.pt", conf=0.8)
        self.pc_processor = PointCloudExtractor()
        self.estimator = PoseEstimator()

        # GUI
        print_with_time("Main", "Creating visualization windows.")
        self.grid = DisplayGrid()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

    def loop(self):
        while not self.finished:
            color, depth = self.get_frames(display=False)

            self.update_color_image(color, render=False)
            self.update_depth_image(depth, render=True)

            # Thread safety, avoid race condition with robot callback
            with self.lock:
                if not self.ready_for_analyzing:
                    continue

            # Look for crates in the image
            detections = self.yolo.predict(color)
            if detections is None:
                continue


            self.update_yolo_image(color, detections, render=True)

            # Process the yolo detections, aka extract point clouds
            point_clouds = self.pc_processor.process(detections, depth)
            if point_clouds is None:
                continue
            
            self.update_pcd_image(point_clouds, render=True)

            # Convert point clouds into poses of the objects found
            objects = self.estimator.estimate_poses(point_clouds)

            if len(objects) == 0:
                continue

            # Command robot to pick item
            crate = self.get_optimal_crate(objects)
            self.send_pick_command(crate)

    def get_optimal_crate(self, objects):
        return objects[0]
    
    def send_pick_command(self, crate):
        try:
            command = self.mission_planner.get_move_sequence(crate)
            command.set_crate_picked_callback(self.crate_picked_callback)
            self.robot.add_command(command)
            self.ready_for_analyzing = False
            print_with_time("Main", "Command sent to robot.")

        except Exception:
            print("Unable to retrieve object, restarting loop.")

    def render_visualization(self, data):
        # Add point clouds
        for pcd in data:
                self.vis.add_geometry(pcd["pcd"])

        # Hack to position and orient camera
        ctr = self.vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters("data/screenshot_visualizer.json")
        ctr.convert_from_pinhole_camera_parameters(parameters)

        # Get screenshot
        image_bgr = screenshot_o3d(self.vis)

        # Remove point clouds
        for pcd in data:
                self.vis.remove_geometry(pcd["pcd"])

        return image_bgr

    def update_color_image(self, color_image, render=False):
        self.grid.set_data(0, 0, color_image)
        if not render:
            return
        self.grid.show()
        cv2.waitKey(1)

    def update_depth_image(self, depth_image, render=False):
        self.grid.set_data(1, 0, depth_to_colormap(depth_image))
        if not render:
            return
        self.grid.show()
        cv2.waitKey(1)

    def update_yolo_image(self, color_image, data, render=False):
        self.grid.set_data(2, 0, draw_yolo_detections(color_image, data, ["Crate", "Pallet"]))
        if not render:
            return
        self.grid.show()
        cv2.waitKey(1)

    def update_pcd_image(self, data, render=False):
        image_bgr = self.render_visualization(data)
        if not render:
            return
        self.grid.set_data(3, 0, image_bgr)
        self.grid.show()
        cv2.waitKey(1)

    def crate_picked_callback(self):
        with self.lock:  # Thread safety
            self.mission_planner.update_items_picked()
            self.ready_for_analyzing = True

            if self.mission_planner.is_order_finished():
                self.finished = True
                print_with_time("Main", "Finished picking. Waiting for Robot to complete commands.")

    def get_frames(self, display=False):
        frames = self.cam.get_latest_frame()

        while frames is None:
            frames = self.cam.get_latest_frame()

        color_image = frames["color"]
        depth_image = frames["depth"]

        # Rotate for yolo model
        color_image = cv2.rotate(color_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if display:
            cv2.imshow("frame", color_image)
            cv2.waitKey(1)

        return color_image, depth_image

    def shutdown(self):
        self.cam.shutdown()
        self.robot.shutdown()
        cv2.destroyAllWindows()
