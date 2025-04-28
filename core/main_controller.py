import cv2
import threading
import cupy as cp
import open3d as o3d
from utils.helper import draw_yolo_detections, depth_to_colormap, screenshot_o3d, print_with_time
from core.DisplayGrid import DisplayGrid
from core.Camera import RealsenseL515
from core.YoloSegmenter import YoloSegmenter
from core.PointCloudExtractor import PointCloudExtractor
from core.Estimator import PoseEstimator
from core.RobotController import RobotController
from core.Mission import MissionPlanner, Order


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

            # Update display
            self.grid.set_data(0, 0, color)
            self.grid.set_data(1, 0, depth_to_colormap(depth))
            self.grid.show()
            cv2.waitKey(1)

            with self.lock:  # Thread safety, avoid race condition with robot callback
                if not self.ready_for_analyzing:
                    continue

            # Look for crates in the image
            data = self.yolo.predict(color)
            if data is None:  # Check if anything was found
                continue

            # Update display
            self.grid.set_data(2, 0, draw_yolo_detections(color, data, ["Crate", "Pallet"]))
            self.grid.show()
            cv2.waitKey(1)

            # Process the yolo detections, aka extract point clouds
            depth_gpu = cp.asarray(depth)
            data = self.pc_processor.process(data, depth_gpu)

            if data is None:  # Check if any crates were found
                continue
            for pcd in data:
                self.vis.add_geometry(pcd["pcd"])
            image_bgr = self.screenshot_visualizer()
            for pcd in data:
                self.vis.remove_geometry(pcd["pcd"])

            self.grid.set_data(3, 0, image_bgr)
            self.grid.show()

            # Convert point clouds into poses of the objects found
            objects = self.estimator.estimate_poses(data)

            # Command robot to pick item
            try:
                crate = objects[0]
                command = self.mission_planner.get_move_sequence(crate)
                command.set_crate_picked_callback(self.crate_picked_callback)
                self.robot.add_command(command)
                self.ready_for_analyzing = False
                print("Command sent to robot.")

            except Exception:
                print("Unable to retrieve object, restarting loop.")

    def screenshot_visualizer(self):
        ctr = self.vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters("data/screenshot_visualizer.json")
        ctr.convert_from_pinhole_camera_parameters(parameters)
        image_bgr = screenshot_o3d(self.vis)
        return image_bgr

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
