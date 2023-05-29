import pyrealsense2 as rs
from src.config import CAMERA_CONFIG
import open3d as o3d
import numpy as np
import time
import logging


class CamaraLidar:
    def __init__(self, camara_config: dict, level=logging.DEBUG) -> None:
        self._logger = logging.getLogger(__name__)
        self._camara_config = camara_config
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._align = rs.align(rs.stream.color)
        self._depth_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self._rgb_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self._extrinsic_depth_rgb = np.eye(4)
        self._depth_scale = 0.0002500000118743628
        self._logger.setLevel(level)

    def init(self):
        self._logger.debug("initialization of the lidar camara in process")
        self._config.enable_stream(
            rs.stream.depth,
            self._camara_config[CAMERA_CONFIG.WIDTH_DEPTH.value],
            self._camara_config[CAMERA_CONFIG.HEIGHT_DEPTH.value],
            rs.format.any,
            self._camara_config[CAMERA_CONFIG.FRAMERATE.value],
        )
        self._config.enable_stream(
            rs.stream.color,
            self._camara_config[CAMERA_CONFIG.WIDTH_COLOR.value],
            self._camara_config[CAMERA_CONFIG.HEIGHT_COLOR.value],
            rs.format.any,
            self._camara_config[CAMERA_CONFIG.FRAMERATE.value],
        )

        pipeline_profile = self._config.resolve(self._pipeline)

        self._pipeline.start(self._config)

        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        depth_sensor.set_option(rs.option.visual_preset, 5)
        depth_sensor.set_option(rs.option.confidence_threshold, 3)
        self.get_first_frames()

    def init_only_imu(self):
        conf = rs.config()
        conf.enable_stream(rs.stream.accel)
        conf.enable_stream(rs.stream.gyro)
        self._pipeline.start(conf)
        self.obtain_imu_data()
        self._pipeline.stop()

    def obtain_imu_data(self):
        f = self._pipeline.wait_for_frames()
        accel = f[0].as_motion_frame().get_motion_data()
        self._accel = np.asarray([accel.x, accel.y, accel.z])
        gyro = f[1].as_motion_frame().get_motion_data()
        self._gyro = np.asarray([gyro.x, gyro.y, gyro.z])
        timestamp = f[0].as_motion_frame().get_timestamp()
        self._logger.info(f"accelerometer: {self._accel}")
        self._logger.info(f"gyro: {gyro}")
        self._logger.info(f"timestamp: {timestamp}")

    def get_first_frames(self):
        start = time.time()
        while time.time() - start < 5:
            frames = self._pipeline.wait_for_frames()
            self.slow_processing(frames)

    # Implement two "processing" functions, each of which
    # occassionally lags and takes longer to process a frame.
    def slow_processing(self, frame):
        n = frame.get_frame_number()
        if n % 20 == 0:
            time.sleep(1 / 4)

    def get_frames_depth_and_color(self):
        frames = self._pipeline.wait_for_frames()
        (
            self.depth_intrinsic,
            self.rgb_intrinsic,
            self.extrinsic_depth_rgb,
        ) = self.extract_intrinsic_extrinsic(frames)
        # align with rs_method
        aligned_frames = self._align.process(frames)
        profile_frames = aligned_frames.get_profile()

        # frame with aligment
        # depth_frame = aligned_frames.get_depth_frame()
        # color_frame = aligned_frames.get_color_frame()
        # frame without aligment
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

    @staticmethod
    def extract_intrinsic_extrinsic(frames):
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_intrinsic = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        color_intrinsic = color_frame.profile.as_video_stream_profile().get_intrinsics()
        depth_to_color_extrinsic = depth_frame.profile.get_extrinsics_to(
            color_frame.profile
        )
        return depth_intrinsic, color_intrinsic, depth_to_color_extrinsic

    @property
    def depth_intrinsic(self):
        return self._depth_intrinsic

    @depth_intrinsic.setter
    def depth_intrinsic(self, value):
        width, height = value.width, value.height
        fx, fy = value.fx, value.fy
        ppx, ppy = value.ppx, value.ppy
        self._depth_intrinsic.set_intrinsics(width, height, fx, fy, ppx, ppy)

    @property
    def rgb_intrinsic(self):
        return self._rgb_intrinsic

    @rgb_intrinsic.setter
    def rgb_intrinsic(self, value):
        width, height = value.width, value.height
        fx, fy = value.fx, value.fy
        ppx, ppy = value.ppx, value.ppy
        self._rgb_intrinsic.set_intrinsics(width, height, fx, fy, ppx, ppy)

    @property
    def extrinsic_depth_rgb(self):
        return self._extrinsic_depth_rgb

    @extrinsic_depth_rgb.setter
    def extrinsic_depth_rgb(self, value):
        if self._extrinsic_depth_rgb is not None:
            self._extrinsic_depth_rgb[0:3, 0] = value.rotation[0:3]
            self._extrinsic_depth_rgb[0:3, 1] = value.rotation[3:6]
            self._extrinsic_depth_rgb[0:3, 2] = value.rotation[6:9]
            self._extrinsic_depth_rgb[:3, 3] = value.translation
        else:
            raise ValueError(
                f"Shape of array should be {self._extrinsic_depth_rgb.shape}"
            )

    def end_pip(self):
        self._pipeline.stop()
