import open3d as o3d
import numpy as np
from src.helper_3d import Helper3D
from src.config import SCAN_CONFIG, SAVE_CONFIG
from src.utils.file_helper import FileHelper
from src.utils.generate_jsons import PinholeCameraTrajectory
from src.camaras.camara_lidar import CamaraLidar
from src.utils.register_frames import Registration
from src.utils.camera_info_helper import CameraInfoHelper, CameraInfo,CameraInfoHelperExtrinsic
#from src.camaras.camara_seek import ThermalSeek
import logging
import copy


class Scanner3D:
    MAX_ANGLE = 360
    ANGLE_OPT = 2

    def __init__(self, config: dict, object_name: str, log_level=logging.DEBUG) -> None:
        self._camera_lidar = CamaraLidar(
            camara_config=config.get_camara_config(), level=log_level
        )
        #self._camera_thermal = ThermalSeek()
        self._helper_3d = Helper3D(log_level=log_level)
        self._save_config = config.get_save_config()
        self._object_name = object_name
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._trajectory = PinholeCameraTrajectory(
            file_name=FileHelper().get_trajectory_path(
                base_folder=self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
                object_folder_name=object_name,
                folder_sub_base=self._save_config[SAVE_CONFIG.POSE_FOLDER.value],
                trj_name=self._save_config[SAVE_CONFIG.ICP_FILE_NAME.value],
            )
        )
        self._scan_config = config.get_scan_config()
        self._point_cloud = None
        self._initial_point_cloud = None

        self._current_point_cloud = None
        self._object_cloud = None
        self._rgb_img = None
        self._current_rgb_img_path = ""
        self._depth_img = None
        self._current_depth_img_path = ""
        self._thermal_img = None
        self._current_thermal_img_path = ""

        self._point_cloud_cplt = None
        self._angle_opt = 0
        self._position = 0
        self._current_angle = 0
        self._max_number_of_positions = 0
        self._transform = None
        self._debug_rms = []
        self._debug_fitness = []
        self._camera_info_lidar = CameraInfoHelper(
            FileHelper().get_path_camera_info(
                CameraInfo.BASE_STRING,
                CameraInfo.FOLDER_NAME,
                CameraInfo.CameraLidar.FILE_NAME.value,
            )
        )
        self._camera_info_extrinsic_lidar = CameraInfoHelperExtrinsic(
            FileHelper().get_path_camera_info(
                CameraInfo.BASE_STRING,
                CameraInfo.FOLDER_NAME,
                CameraInfo.CameraExtrinsic.FILE_NAME_RGB_DEPTH.value,
            )
        )
        # initialize camara
        #self._initialize_all()

    def _initialize_all(self):
        self._camera_lidar.init()

    def finilize(self):
        self._camera_lidar.end_pip()

    def create_rgbd_from_camera(self):
        self._depth_img, self._rgb_img = self._camera_lidar.get_frames_depth_and_color()
        # self._rgb_img, self._depth_img = Registration().register_images(
        #     self._depth_img,
        #     self._rgb_img,
        #     self._camera_lidar.depth_intrinsic.intrinsic_matrix,
        #     self._camera_lidar.rgb_intrinsic.intrinsic_matrix,
        #     self._camera_lidar.extrinsic_depth_rgb,
        #     self._camera_lidar._depth_scale,
        # )
        rgbd_image = self._helper_3d.create_rgbd_from_color_and_depth_img(
            depth_img=self._depth_img, color_img=self._rgb_img
        )

        return rgbd_image

    def create_point_cloud_from_rgbd_camera(self, rgbd_image, instrinsic_param=None):
        if instrinsic_param is None:
            intrinsic = self._camera_lidar.depth_intrinsic
        else:
            intrinsic = instrinsic_param
        if intrinsic.is_valid():
            point_cloud = self._helper_3d.create_point_cloud_from_rgbd(
                rgbd_image=rgbd_image, camera_intrinsic=intrinsic
            )
            return point_cloud
        else:
            self._logger.error(
                f"Invalid intrinsic param with: matrix = {intrinsic.intrinsic_matrix}  height = {intrinsic.height}  width = {intrinsic.width}"
            )
            return None

    # todo: better namming
    def remove_not_needed_clouds(self, point_cloud):
        diff_max_bound = np.array(
            [
                [0.0],
                [-self._scan_config[SCAN_CONFIG.CROP_OUTLIER_POINTS.value]],
                [0.0],#[-0.4],
            ]
        )
        diff_min_bound = np.array([[0.0], [0.0], [0.0]])
        #diff_min_bound = np.array([[0.35], [0.0], [0.0]])
        point_cloud_cropped = self._helper_3d.crop_point_cloud(
            point_cloud=point_cloud,
            diff_max_bound=diff_max_bound,
            diff_min_bound=diff_min_bound,
        )
        return point_cloud_cropped

    def remove_turn_table(self, point_cloud):
        diff_max_bound = np.array(
            [
                [0.0],
                [-self._scan_config[SCAN_CONFIG.CROP_TURN_TABLE.value]],
                [0.0],
            ]
        )
        diff_min_bound = np.array([[0.0], [0.0], [0.0]])
        point_cloud_cropped = self._helper_3d.crop_point_cloud(
            point_cloud=point_cloud,
            diff_min_bound=diff_min_bound,
            diff_max_bound=diff_max_bound,
        )
        return point_cloud_cropped

    def create_point_cloud_of_object(self, rgbd_image=None, instrinsic=None):
        if rgbd_image is None:
            rgbd_image = self.create_rgbd_from_camera()
        point_cloud = self.create_point_cloud_from_rgbd_camera(
            rgbd_image=rgbd_image, instrinsic_param=instrinsic
        )
        #point_cloud = self._helper_3d.remove_noise_point_cloud(point_cloud,nb_neighbors=14)
        # rot_zyx = (-11 * np.pi / 180, np.pi, 46 * np.pi / 180)
        # rot_zyx = (3 * np.pi / 180, 0, -38 * np.pi / 180)
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)
        #o3d.visualization.draw_geometries([frame,point_cloud])
        rot_zyx = (
            self._helper_3d.calculate_deg_to_rad(
                self._scan_config[SCAN_CONFIG.CAMERA_ANGLE_ROT_Z.value]
            ),
            self._helper_3d.calculate_deg_to_rad(
                self._scan_config[SCAN_CONFIG.CAMERA_ANGLE_ROT_Y.value]
            ),
            self._helper_3d.calculate_deg_to_rad(
                self._scan_config[SCAN_CONFIG.CAMERA_ANGLE_ROT_X.value]
            ),
        )
        
        rotation_matrix = point_cloud.get_rotation_matrix_from_zyx(rot_zyx)
        point_cloud.rotate(rotation_matrix, center=(0, 0, 0))
        #o3d.visualization.draw_geometries([frame,point_cloud])
        point_cloud.translate(-point_cloud.get_center())
        return point_cloud, rgbd_image

    def initialize_scan_process(self, rgbd=None, instrinsic=None) -> bool:
        self._trajectory.remove_current_file()
        self._initial_point_cloud, rgbd_image = self.create_point_cloud_of_object(
            rgbd, instrinsic
        )
        self._initial_point_cloud = self.remove_not_needed_clouds(
            self._initial_point_cloud
        )
        self._initial_point_cloud = self._helper_3d.local_refiment_point_cloud(
            point_cloud=self._initial_point_cloud,
            voxel_size=self._scan_config[SCAN_CONFIG.VOXEL_SIZE_TURNTABLE.value],
        )
        if rgbd is None:
            self.set_object_path_rgb_depth("init")
            self.save_current_depth_rgb_imgs()
            # self._thermal_img = self._camera_thermal.take_and_save_image(
            #     self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
            #     self._save_config[SAVE_CONFIG.THERMAL_IMAGES_FOLDER.value],
            #     self._object_name,
            #     self._current_thermal_img_path,
            # )

        return True

    def first_process(self, rgbd_image=None, instrinsic=None) -> bool:
        if self._initial_point_cloud is not None:
            self._point_cloud, rgbd = self.create_point_cloud_of_object(
                rgbd_image, instrinsic
            )
            self._point_cloud_cplt = copy.deepcopy(self._point_cloud)
            self._point_cloud = self.remove_not_needed_clouds(self._point_cloud)

            reg_point_2_plane = self._helper_3d.registration_icp(
                source=self._point_cloud,
                target=self._initial_point_cloud,
                voxel_size=self._scan_config[SCAN_CONFIG.VOXEL_SIZE_TURNTABLE.value],
            )

            self._point_cloud.transform(reg_point_2_plane.transformation)

            self._transform = reg_point_2_plane.transformation

            object_cloud = copy.deepcopy(self._point_cloud)
            self._point_cloud = self.remove_turn_table(point_cloud=self._point_cloud)
            self._logger.debug(
                f"ICP Registration Init Point Cloud: fitness = {round(reg_point_2_plane.fitness, 5)}"
            )
            self._point_cloud_cplt.transform(reg_point_2_plane.transformation)

            self._point_cloud = self._helper_3d.remove_noise_point_cloud(
                self._point_cloud
            )

            object_cloud = self._helper_3d.remove_noise_point_cloud(object_cloud)
            # self._point_cloud_cplt = self._helper_3d.remove_noise_point_cloud(
            #     self._point_cloud_cplt
            # )

            self._object_cloud = object_cloud + self._point_cloud_cplt
            self._current_angle += self._scan_config[
                SCAN_CONFIG.ROTATION_STEP_ANGLE.value
            ]

            self._trajectory.append_parameter_to_trajectory(
                intrinsic=self._camera_info_lidar.intrinsic_parameter_open3d(
                    CameraInfo.CameraLidar.INTRINSIC_RGB.value
                ),
                extrinsic=self._transform,
            )
            self._trajectory.save_current_trajectory()
            # self._thermal_img = self._camera_thermal.take_and_save_image(
            #     self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
            #     self._save_config[SAVE_CONFIG.THERMAL_IMAGES_FOLDER.value],
            #     self._object_name,
            #     self._current_thermal_img_path,
            # )
            return True
        self._logger.error("No initial Point Cloud or Rotation center was found")
        return False

    def turn_table_step_angle_process(self, rgbd_image=None, instrinsic=None) -> bool:
        if self._point_cloud and self._initial_point_cloud is not None:
            self._current_point_cloud, rgbd_image = self.create_point_cloud_of_object(
                rgbd_image, instrinsic
            )
            self._current_point_cloud = self.remove_not_needed_clouds(
                self._current_point_cloud
            )

            reg_point_2_plane_with_init_pc = self._helper_3d.registration_icp(
                source=self._current_point_cloud,
                target=self._initial_point_cloud,
                voxel_size=self._scan_config[SCAN_CONFIG.VOXEL_SIZE_TURNTABLE.value],
            )

            self._logger.debug(
                f"ICP Registration Init Point Cloud: fitness = {round(reg_point_2_plane_with_init_pc.fitness, 5)}"
            )

            self._point_cloud_cplt = copy.deepcopy(self._current_point_cloud)

            self._point_cloud_cplt.transform(
                reg_point_2_plane_with_init_pc.transformation
            )

            self._current_point_cloud = self.remove_turn_table(
                point_cloud=self._current_point_cloud
            )

            frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)

            # rotation_pc_current_angle = (
            #     self._current_point_cloud.get_rotation_matrix_from_zyx(
            #         (0, -(self._current_angle + self._angle_opt) * np.pi / 180, 0)
            #     )
            # )
            rotation_pc_current_angle = (
                self._current_point_cloud.get_rotation_matrix_from_zyx(
                    (
                        0,
                        self._helper_3d.calculate_deg_to_rad(
                            self._current_angle + self._angle_opt
                        ),
                        0,
                    )
                )
            )

            # Check here the better solution
            trans_estim = (
                self._helper_3d.calculate_transformation_matrix_with_rot_trans(
                    rot=rotation_pc_current_angle,
                    trans=self._point_cloud.get_center(),
                )
            )

            # o3d.visualization.draw_geometries(
            #     [self._current_point_cloud, self._point_cloud, frame]
            # )
            # remove outliers
            self._current_point_cloud = self._helper_3d.remove_noise_point_cloud(
                self._current_point_cloud, nb_points=1, radius=0.5
            )
            reg_point_2_plane_with_pc = (
                self._helper_3d.registration_icp_colored(  # registration_icp_colored
                    source=self._current_point_cloud,
                    target=self._point_cloud,
                    voxel_size=self._scan_config[SCAN_CONFIG.VOLEX_SIZE_OBJECT.value],
                    trans_init=trans_estim,
                )
            )
            self._logger.debug(
                f"ICP Registration Point Cloud: fitness = {round(reg_point_2_plane_with_pc.fitness, 5)}"
            )

            self._debug_fitness.append(reg_point_2_plane_with_pc.fitness)
            self._debug_rms.append(reg_point_2_plane_with_pc.inlier_rmse)

            if (
                round(reg_point_2_plane_with_pc.fitness, 5)
                < self._scan_config[SCAN_CONFIG.MIN_FITNESS_SCORE.value]
            ):
                self._logger.debug(
                    f"The fitness score {reg_point_2_plane_with_pc.fitness} of the current icp regristation is to small, repeat the process"
                )
                self._angle_opt += 2
                return False

            self._angle_opt = 0

            self._transform = reg_point_2_plane_with_pc.transformation

            self._current_point_cloud.transform(
                reg_point_2_plane_with_pc.transformation
            )

            self._point_cloud_cplt.transform(reg_point_2_plane_with_pc.transformation)

            # o3d.visualization.draw_geometries(
            #     [self._current_point_cloud, self._point_cloud, frame]
            # )
            # self._object_cloud = self._object_cloud + self._current_point_cloud
            self._trajectory.append_parameter_to_trajectory(
                intrinsic=self._camera_info_lidar.intrinsic_parameter_open3d(
                    CameraInfo.CameraLidar.INTRINSIC_RGB.value
                ),
                extrinsic=self._transform,
            )
            self._trajectory.save_current_trajectory()

            self._current_angle += self._scan_config[
                SCAN_CONFIG.ROTATION_STEP_ANGLE.value
            ]
            self._position += 1
            self._point_cloud = self._point_cloud + self._current_point_cloud
            self._point_cloud = self._helper_3d.remove_noise_point_cloud(
                self._point_cloud,
                nb_neighbors=25,
                std_ratio=3.0,
                radius_outlier=True,
                radius=1.25,
            )

            # self._thermal_img = self._camera_thermal.take_and_save_image(
            #     self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
            #     self._save_config[SAVE_CONFIG.THERMAL_IMAGES_FOLDER.value],
            #     self._object_name,
            #     self._current_thermal_img_path,
            # )
            return True

        self._logger.error("No Point Cloud or Initial Point Cloud was found")
        return False

    def visualize_main_object(self):
        if self._point_cloud is not None:
            self._point_cloud = self._helper_3d.remove_noise_point_cloud(
                self._point_cloud, nb_neighbors=25, std_ratio=1.5
            )
            self._object_cloud = self._object_cloud + self._point_cloud
            self._object_cloud = self._helper_3d.remove_noise_point_cloud(
                self._object_cloud, nb_neighbors=25, std_ratio=2.0
            )
            frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
            o3d.visualization.draw_geometries([self._point_cloud, frame])
            o3d.visualization.draw_geometries([self._object_cloud, frame])

            o3d.io.write_point_cloud(
                FileHelper().get_object_path(
                    folder_base=self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
                    folder_sub_base=self._save_config[
                        SAVE_CONFIG.POINT_CLOUD_FOLDER.value
                    ],
                    object_folder_name=self._object_name,
                    object_name=self._object_name,
                    object_type=self._save_config[SAVE_CONFIG.POINT_CLOUD_FORMAT.value],
                ),
                self._point_cloud,
            )
            o3d.io.write_point_cloud(
                FileHelper().get_object_path(
                    folder_base=self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
                    folder_sub_base=self._save_config[
                        SAVE_CONFIG.POINT_CLOUD_FOLDER.value
                    ],
                    object_folder_name=self._object_name,
                    object_name=self._object_name + "_with_table",
                    object_type=self._save_config[SAVE_CONFIG.POINT_CLOUD_FORMAT.value],
                ),
                self._object_cloud,
            )
            np.savez(
                "debug.npz",
                fitness=np.vstack(self._debug_fitness),
                rms=np.vstack(self._debug_rms),
            )
        else:
            self._logger.error(
                "There is None object cloud, check if initialize obejct and first process was successful"
            )

    def ignore_point_cloud(self):
        self._logger.debug(
            f"step was ignore moving to position : {self._position} and angle: {self._current_angle}"
        )
        self._current_angle += self._scan_config[SCAN_CONFIG.ROTATION_STEP_ANGLE.value]
        self._position += 1

    def restart_scan_process_values(self):
        self._point_cloud = None
        self._current_point_cloud = None
        self._object_cloud = None
        self._current_angle = 0
        self._position = 0

    def set_object_path_rgb_depth(self, name="0"):
        img_name = self._object_name + "_" + name
        self._current_rgb_img_path = FileHelper().get_object_path(
            folder_base=self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
            folder_sub_base=self._save_config[SAVE_CONFIG.RGB_IMAGES_FOLDER.value],
            object_folder_name=self._object_name,
            object_name=img_name,
            object_type=self._save_config[SAVE_CONFIG.RGB_FORMAT.value],
        )

        self._current_depth_img_path = FileHelper().get_object_path(
            folder_base=self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
            folder_sub_base=self._save_config[SAVE_CONFIG.DEPTH_IMAGES_FOLDER.value],
            object_folder_name=self._object_name,
            object_name=img_name,
            object_type=self._save_config[SAVE_CONFIG.DEPTH_FORMAT.value],
        )
        self._current_thermal_img_path = (
            img_name + self._save_config[SAVE_CONFIG.DEPTH_FORMAT.value]
        )
        # self._current_thermal_img_path = FileHelper().get_object_path(
        #     folder_base=self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
        #     folder_sub_base=self._save_config[SAVE_CONFIG.DEPTH_IMAGES_FOLDER.value],
        #     object_folder_name=self._object_name,
        #     object_name=img_name,
        #     object_type=self._save_config[SAVE_CONFIG.DEPTH_FORMAT.value],
        # )

    def save_current_depth_rgb_imgs(self):
        self._helper_3d.save_image_cv2(
            image=self._rgb_img, path=self._current_rgb_img_path
        )
        self._helper_3d.save_image_cv2(
            image=self._depth_img, path=self._current_depth_img_path
        )
        # self._helper_3d.save_image_cv2(
        #     image=self._thermal_img, path=self._current_thermal_img_path
        # )

    def create_point_cloud_from_files(self):
        self.set_object_path_rgb_depth("init")
        (
            self._rgb_img,
            self._depth_img,
            rgbd_image,
        ) = self._helper_3d.create_rgbd_from_color_and_depth(
            depth_path=self._current_depth_img_path,
            color_path=self._current_rgb_img_path,
            depth_intrinsic=self._camera_info_lidar.intrinsic_parameter_as_array(
                CameraInfo.CameraLidar.INTRINSIC_DEPTH.value
            ),
            rgb_intrinsic=self._camera_info_lidar.intrinsic_parameter_as_array(
                CameraInfo.CameraLidar.INTRINSIC_RGB.value
            ),
            extrinsic_depth_rgb=self._camera_info_extrinsic_lidar.extrinsic_parameter_as_array(
                CameraInfo.CameraExtrinsic.ROTATION.value,
                CameraInfo.CameraExtrinsic.TRANSLATION.value,
            ),
            depth_scale=self._camera_lidar._depth_scale,
        )
        self.initialize_scan_process(
            rgbd_image,
            self._camera_info_lidar.intrinsic_parameter_open3d(
                CameraInfo.CameraLidar.INTRINSIC_RGB.value
            ),
        )
        for pos in range(int(self.max_number_of_positions)):
            if pos < 4:
                counter = 0
                self.set_object_path_rgb_depth(str(pos))
                self._rgb_img, self._depth_img,rgbd_image = self._helper_3d.create_rgbd_from_color_and_depth(
                    depth_path=self._current_depth_img_path, color_path=self._current_rgb_img_path,
                    depth_intrinsic=self._camera_info_lidar.intrinsic_parameter_as_array(
                    CameraInfo.CameraLidar.INTRINSIC_DEPTH.value
                    ),
                    rgb_intrinsic=self._camera_info_lidar.intrinsic_parameter_as_array(
                        CameraInfo.CameraLidar.INTRINSIC_RGB.value
                    ),
                    extrinsic_depth_rgb=self._camera_info_extrinsic_lidar.extrinsic_parameter_as_array(
                        CameraInfo.CameraExtrinsic.ROTATION.value,
                        CameraInfo.CameraExtrinsic.TRANSLATION.value,
                    ),
                    depth_scale=0.0002500000118743628,#self._camera_lidar._depth_scale,
                )
                if pos == 0:
                    success = self.first_process(
                        rgbd_image,
                        self._camera_info_lidar.intrinsic_parameter_open3d(
                            CameraInfo.CameraLidar.INTRINSIC_RGB.value
                        ),
                    )
                else:
                    success = self.turn_table_step_angle_process(
                        rgbd_image,
                        self._camera_info_lidar.intrinsic_parameter_open3d(
                            CameraInfo.CameraLidar.INTRINSIC_RGB.value
                        ),
                    )
                    while not (success) and counter < 1:
                        self._logger.debug("repeating icp")
                        success = self.turn_table_step_angle_process(
                            rgbd_image,
                            self._camera_info_lidar.intrinsic_parameter_open3d(
                                CameraInfo.CameraLidar.INTRINSIC_RGB.value
                            ),
                        )
                        counter += 1

        return success

    def auto_scanning(self) -> bool:
        rgbd = self.create_rgbd_from_camera()
        if rgbd is not None:
            self.set_object_path_rgb_depth(str(self.current_turn_position))
            # self._thermal_img = self._camera_thermal.take_and_save_image(
            #     self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
            #     self._save_config[SAVE_CONFIG.THERMAL_IMAGES_FOLDER.value],
            #     self._object_name,
            #     self._current_thermal_img_path,
            # )
            self.save_current_depth_rgb_imgs()
            self._current_angle += self._scan_config[
                SCAN_CONFIG.ROTATION_STEP_ANGLE.value
            ]
            self._position += 1
            return True
        return False

    @property
    def current_turn_position(self):
        return self._position

    @current_turn_position.setter
    def current_turn_position(self, value):
        self._position = value

    @property
    def current_turn_angle(self):
        return self._current_angle

    @current_turn_angle.setter
    def current_turn_angle(self, value):
        self._current_angle = value

    @property
    def max_number_of_positions(self):
        return self.MAX_ANGLE / self._scan_config[SCAN_CONFIG.ROTATION_STEP_ANGLE.value]
