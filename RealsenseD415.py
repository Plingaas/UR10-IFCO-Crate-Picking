import pyrealsense2 as rs

class RealsenseD415:

    def __init__(self) -> None:
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

    def enable_rgb_camera(self, res=(1920, 1080), fps: int = 30) -> None:
        """Enable the RGB camera stream."""
        self.config.enable_stream(rs.stream.color, res[0], res[1], rs.format.bgr8, fps)

    def enable_depth_camera(self, res=(640, 480), fps: int = 30) -> None:
        """Enable the Depth camera stream."""
        self.config.enable_stream(rs.stream.depth, res[0], res[1], rs.format.z16, fps)

    def start_streaming(self) -> None:
        """Start the pipeline streaming."""
        self.profile = self.pipeline.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()

    def stop(self) -> None:
        """Stop the pipeline streaming."""
        self.pipeline.stop()

    def set_depth_receiver_gain(self, value) -> None:
        """Set the depth receiver gain, constrained between 1 and 18."""
        self.depth_sensor.set_option(rs.option.receiver_gain, max(1, min(18, value)))

    def set_depth_post_processing_sharpening(self, value) -> None:
        """Set the depth post-processing sharpening option, constrained between 0 and 4."""
        self.depth_sensor.set_option(rs.option.post_processing_sharpening, max(0, min(4, value)))

    def set_clipping_distance(self, meters) -> None:
        """Set the clipping distance for depth camera."""
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.clipping_distance_in_meters = meters
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

    def get_frames(self):
        """Wait for frames from the camera."""
        return self.pipeline.wait_for_frames()

    def align_frames(self, frames):
        """Align the depth frames with the color frames."""
        align = rs.align(rs.stream.color)
        return align.process(frames)

    def get_intrinsics(self, aligned_depth_frame):
        """Get the camera intrinsics from the aligned depth frame."""
        self.intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        return self.intrinsics

    def get_color_frame(self, frames):
        """Get the color frame from the frames."""
        return frames.get_color_frame().get_data()

    def get_depth_frame(self, frames):
        """Get the depth frame from the frames."""
        return frames.get_depth_frame().get_data()
