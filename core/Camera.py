import pyrealsense2 as rs
import threading
import numpy as np


class L515Intrinsics:
    def __init__(self) -> None:
        self.width = 1920
        self.height = 1080
        self.ppx = 982.673
        self.ppy = 547.614
        self.fx = 1352.92
        self.fy = 1354.45
        self.model = rs.distortion.brown_conrady
        self.coeffs = [
            0.171722,
            -0.526011,
            -0.000589736,
            -0.000417008,
            0.486631,
        ]


class RealsenseL515(threading.Thread):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        self.daemon = True
        self.running = False
        self.aligner = rs.align(rs.stream.color)
        self.lock = threading.Lock()
        self.frame = None
        self.processed_frame = None

    def enable_rgb_camera(self, res=(1920, 1080), fps=30) -> None:
        self.config.enable_stream(rs.stream.color, res[0], res[1], rs.format.bgr8, fps)

    def enable_depth_camera(self, res=(640, 480), fps=30) -> None:
        self.config.enable_stream(rs.stream.depth, res[0], res[1], rs.format.z16, fps)

    def start_streaming(self) -> None:
        self.running = True
        self.start()

    def run(self):
        while self.running:
            frame = self.pipeline.wait_for_frames()
            with self.lock:
                self.frame = frame

    def stop(self) -> None:
        self.running = False
        self.pipeline.stop()

    def init(self) -> None:
        self.enable_depth_camera()
        self.enable_rgb_camera()
        self.profile = self.pipeline.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.start_streaming()
        self.set_depth_receiver_gain(18)
        self.set_depth_post_processing_sharpening(1)
        self.set_clipping_distance(2.5)

    def set_depth_receiver_gain(self, value) -> None:
        value = max(1, min(18, value))
        self.depth_sensor.set_option(rs.option.receiver_gain, value)

    def set_depth_post_processing_sharpening(self, value) -> None:
        value = max(0, min(4, value))
        self.depth_sensor.set_option(rs.option.post_processing_sharpening, value)

    def set_clipping_distance(self, meters) -> None:
        self.clipping_distance_in_meters = meters
        self.clipping_distance = meters / self.depth_scale

    def align_frames(self, frames):
        return self.aligner.process(frames)

    def get_intrinsics(self, aligned_depth_frame):
        return aligned_depth_frame.profile.as_video_stream_profile().intrinsics

    def get_latest_frame(self):
        if self.frame is None:
            return None

        color = None
        depth = None

        with self.lock:
            try:
                color = self.frame.get_color_frame()
                depth = self.align_frames(self.frame).get_depth_frame()
            except Exception:
                print("No frame available")

        color = np.asarray(color.get_data())
        depth = np.asarray(depth.get_data())

        frame = {"color": color, "depth": depth}

        return frame

    def shutdown(self):
        self.running = False
        self.join()
        self.pipeline.stop()
