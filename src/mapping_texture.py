import argparse
import re
import cv2
import numpy as np
from src.utils.file_helper import FileHelper
from src.config import Config, SAVE_CONFIG
import open3d as o3d
import os
from src.utils.generate_jsons import PinholeCameraTrajectory
from src.utils.register_frames import Registration
from src.utils.camera_info_helper import (
    CameraInfoHelper,
    CameraInfoHelperExtrinsic,
    CameraInfo,
)

def calculate_deg_to_rad(angle=0.0):
        return angle * np.pi / 180

class MappingTexture:
    def __init__(
        self,
        object_name,
        save_config,
        rot_z,
        rot_y,
        rot_x,
        debug=True,
    ) -> None:
        self._trajectory = PinholeCameraTrajectory(
            file_name=FileHelper().get_trajectory_path(
                base_folder=save_config[SAVE_CONFIG.BASE_FOLDER.value],
                object_folder_name=object_name,
                folder_sub_base=save_config[SAVE_CONFIG.POSE_FOLDER.value],
                trj_name=save_config[SAVE_CONFIG.TRAJECTORY_FILE_NAME.value],
            )
        )
        self._intrinsic_lidar = o3d.camera.PinholeCameraIntrinsic(
            width=640, height=480, fx=594.6319, fy=594.5186, cx=326.8489, cy=236.473
        )
        self._save_config = save_config
        self._object_name = object_name
        self._rot_zyx= (calculate_deg_to_rad(rot_z), calculate_deg_to_rad(rot_y), calculate_deg_to_rad(rot_x))
        self._mesh = None
        self._point_cloud = None
        self._rgbd_images = []
        self._transform_icp = None
        self._debug = debug
        self._camera_info_lidar = CameraInfoHelper(
            FileHelper().get_path_camera_info(
                CameraInfo.BASE_STRING,
                CameraInfo.FOLDER_NAME,
                CameraInfo.CameraLidar.FILE_NAME.value,
            )
        )
        self._camera_info_thermal = CameraInfoHelper(
            FileHelper().get_path_camera_info(
                CameraInfo.BASE_STRING,
                CameraInfo.FOLDER_NAME,
                CameraInfo.CameraThermal.FILE_NAME.value,
            )
        )
        self._camera_info_thermal_lidar = CameraInfoHelperExtrinsic(
            FileHelper().get_path_camera_info(
                CameraInfo.BASE_STRING,
                CameraInfo.FOLDER_NAME,
                CameraInfo.CameraExtrinsic.FILE_NAME_RGB_THERMAL.value,
            )
        )
        self._camera_info_extrinsic_lidar = CameraInfoHelperExtrinsic(
            FileHelper().get_path_camera_info(
                CameraInfo.BASE_STRING,
                CameraInfo.FOLDER_NAME,
                CameraInfo.CameraExtrinsic.FILE_NAME_RGB_DEPTH.value,
            )
        )

    @staticmethod
    def read_trayectory_camera(path):
        # get pose of camera lidar w.r.t object of each image
        if os.path.exists(path):
            trajectory = o3d.io.read_pinhole_camera_trajectory(path)
            return trajectory
        return None

    @staticmethod
    def read_mesh(path):
        if os.path.exists(path):
            mesh = o3d.io.read_triangle_mesh(path)
            return mesh
        return None

    def read_mesh_pcl_trajectory(self):
        self._transform_icp = self.read_trayectory_camera(
            FileHelper().get_trajectory_path(
                base_folder=self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
                object_folder_name=self._object_name,
                folder_sub_base=self._save_config[SAVE_CONFIG.POSE_FOLDER.value],
                trj_name=self._save_config[SAVE_CONFIG.ICP_FILE_NAME.value],
            )
        )
        self._mesh = self.read_mesh(
            FileHelper().get_object_path(
                folder_base=self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
                folder_sub_base=self._save_config[SAVE_CONFIG.MESH_FOLDER.value],
                object_folder_name=self._object_name,
                object_name=self._object_name + "_final",
                object_type=self._save_config[SAVE_CONFIG.MESH_FORMAT.value],
            )
        )
        self._point_cloud = o3d.io.read_point_cloud(
            FileHelper().get_object_path(
                folder_base=self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
                folder_sub_base=self._save_config[SAVE_CONFIG.POINT_CLOUD_FOLDER.value],
                object_folder_name=self._object_name,
                object_name=self._object_name + "_with_table",
                object_type=self._save_config[SAVE_CONFIG.POINT_CLOUD_FORMAT.value],
            )
        )

    @staticmethod
    def color_mapping(mesh, rgbd_images, trajectory):
        """http://www.open3d.org/docs/latest/tutorial/pipelines/color_map_optimization.html?highlight=colormapoptimizationoption"""
        maximum_iteration = 1000
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            mesh_textured, trajectory = o3d.pipelines.color_map.run_rigid_optimizer(
                mesh,
                rgbd_images,
                trajectory,
                o3d.pipelines.color_map.RigidOptimizerOption(
                    maximum_iteration=maximum_iteration
                ),
            )
        # maximum_iteration = 300
        # with o3d.utility.VerbosityContextManager(
        #     o3d.utility.VerbosityLevel.Debug
        # ) as cm:
        #     mesh_textured, camera_trajectory = o3d.pipelines.color_map.run_non_rigid_optimizer(
        #         mesh,
        #         rgbd_images,
        #         trajectory,
        #         o3d.pipelines.color_map.NonRigidOptimizerOption(
        #             maximum_iteration=maximum_iteration
        #         ),
        #     )
        o3d.visualization.draw_geometries([mesh_textured])

    @staticmethod
    def sorted_alphanum(file_list_ordered):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        return sorted(file_list_ordered, key=alphanum_key)

    def get_rgb_depth_paths(self):
        rgb_img_path = FileHelper().get_parent_of_object_path(
            folder_base=self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
            folder_sub_base=self._save_config[SAVE_CONFIG.RGB_IMAGES_FOLDER.value],
            object_name=self._object_name,
        )
        depth_img_path = FileHelper().get_parent_of_object_path(
            folder_base=self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
            folder_sub_base=self._save_config[SAVE_CONFIG.DEPTH_IMAGES_FOLDER.value],
            object_name=self._object_name,
        )
        thermal_img_path = FileHelper().get_parent_of_object_path(
            folder_base=self._save_config[SAVE_CONFIG.BASE_FOLDER.value],
            folder_sub_base=self._save_config[SAVE_CONFIG.THERMAL_IMAGES_FOLDER.value],
            object_name=self._object_name,
        )
        return rgb_img_path, depth_img_path, thermal_img_path

    def create_rgbd_from_thermal_and_depth(self, color_path, depth_path, thermal_path):
        color_img = cv2.imread(color_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        thermal_img = cv2.imread(thermal_path)
        color_img_r, depth_img_r = Registration().register_images(
            depth_img,
            color_img,
            self._camera_info_lidar.intrinsic_parameter_as_array(
                CameraInfo.CameraLidar.INTRINSIC_DEPTH.value
            ),
            self._camera_info_lidar.intrinsic_parameter_as_array(
                CameraInfo.CameraLidar.INTRINSIC_RGB.value
            ),
            self._camera_info_extrinsic_lidar.extrinsic_parameter_as_array(
                CameraInfo.CameraExtrinsic.ROTATION.value,
                CameraInfo.CameraExtrinsic.TRANSLATION.value,
            ),
            self._camera_info_lidar.depth_scale(
                CameraInfo.CameraLidar.DEPTH_SCALE.value
            ),
            25
        )
        depth, thermal_reg = Registration().register_thermal_rgb(
            color_img,
            thermal_img,
            depth_img,
            self._camera_info_lidar.intrinsic_parameter_as_array(
                CameraInfo.CameraLidar.INTRINSIC_RGB.value
            ),
            self._camera_info_thermal.intrinsic_parameter_as_array(
                CameraInfo.CameraThermal.INTRINSIC.value
            ),
            self._camera_info_lidar.intrinsic_parameter_as_array(
                CameraInfo.CameraLidar.INTRINSIC_DEPTH.value
            ),
            self._camera_info_thermal_lidar.extrinsic_parameter_as_array(
                CameraInfo.CameraExtrinsic.ROTATION.value,
                CameraInfo.CameraExtrinsic.TRANSLATION.value,
            ),
            self._camera_info_extrinsic_lidar.extrinsic_parameter_as_array(
                CameraInfo.CameraExtrinsic.ROTATION.value,
                CameraInfo.CameraExtrinsic.TRANSLATION.value,
            ),
            self._camera_info_lidar.depth_scale(
                CameraInfo.CameraLidar.DEPTH_SCALE.value
            ),
        )
        depth = o3d.geometry.Image(depth)
        thermal_reg = o3d.geometry.Image(thermal_reg)
        rgbd_image_thermal = o3d.geometry.RGBDImage.create_from_color_and_depth(
            thermal_reg,
            depth,
            convert_rgb_to_intensity=False,
        )
        
        color_img_open3d = o3d.geometry.Image(color_img_r)
        depth_img_open3d = o3d.geometry.Image(depth_img_r)
        rgbd_image_color = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img_open3d,
            depth_img_open3d,
            convert_rgb_to_intensity=False,
        )
        return rgbd_image_thermal , rgbd_image_color

    def append_rgbd_images_from_path(self):
        self._trajectory.remove_current_file()
        rgb_img_path, depth_img_path, thermal_img_path = self.get_rgb_depth_paths()
        rgb_list = self.sorted_alphanum(os.listdir(rgb_img_path))
        depth_list = self.sorted_alphanum(os.listdir(depth_img_path))
        thermal_list = self.sorted_alphanum(os.listdir(thermal_img_path))
        counter = 0
        assert len(rgb_list) == len(depth_list) and len(rgb_list) == len(thermal_list)
        for i in range(len(rgb_list)):
            if i < 4:
                color_img = rgb_list[i]
                depth_img = depth_list[i]
                thermal_img = thermal_list[i]
                if not (color_img.endswith("init.png") and thermal_img.endswith("init.png") and depth_img.endswith("init.png") ):
                    rgbd_thermal, rgbd_color = self.create_rgbd_from_thermal_and_depth(
                        os.path.join(rgb_img_path, color_img),
                        os.path.join(depth_img_path, depth_img),
                        os.path.join(thermal_img_path, thermal_img),
                    )
                    # depth = o3d.io.read_image(os.path.join(depth_img_path, depth_img))
                    # color = o3d.io.read_image(os.path.join(rgb_img_path, color_img))
                    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    #     color, depth, convert_rgb_to_intensity=False
                    # )
                    self.transformation_generation(
                        rgbd_thermal=rgbd_thermal,
                        rgbd_color=rgbd_color,
                        pos=counter,
                    )
                    counter += 1
                    self._rgbd_images.append(rgbd_thermal)

    def transformation_generation(self, rgbd_thermal, rgbd_color, pos):
        if self._transform_icp is not None:
            extrinsic_icp = self._transform_icp.parameters[pos].extrinsic
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_thermal, self._intrinsic_lidar
            )
            point_cloud_ref = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_color, self._intrinsic_lidar
            )
            # rot_zyx = (-11 * np.pi / 180, np.pi, 46 * np.pi / 180)
            rot_zyx = (-1.0 * np.pi / 180, 0, -41.5 * np.pi / 180)
            
            rotation_matrix = point_cloud.get_rotation_matrix_from_zyx(self._rot_zyx)
            transform_rot = self.get_transform_matrix_with_rot_trans(rotation_matrix)

            t_trans = self.get_transform_matrix_translation(
                trans=-point_cloud_ref.get_center(),
            )
            transform_1_2 = np.dot(transform_rot, t_trans)

            transform_final = np.dot(extrinsic_icp, transform_1_2)

            point_cloud.transform(transform_final)
            #o3d.visualization.draw_geometries([point_cloud, self._mesh])

            self._trajectory.append_parameter_to_trajectory(
                intrinsic=self._intrinsic_lidar, extrinsic=transform_final
            )
            self._trajectory.save_current_trajectory()

    @staticmethod
    def get_transform_matrix_translation(trans):
        transform = np.eye(4)
        transform[0, 3] = trans[0]
        transform[1, 3] = trans[1]
        transform[2, 3] = trans[2]
        return transform

    @staticmethod
    def get_transform_matrix_with_rot_trans(
        rot=np.eye(3), trans=np.array([0.0, 0.0, 0.0])
    ):
        T = np.eye(4)
        T[:3, :3] = rot
        tran_mov = np.dot(rot, trans)
        T[0, 3] = trans[0] - tran_mov[0]
        T[1, 3] = trans[1] - tran_mov[1]
        T[2, 3] = trans[2] - tran_mov[2]
        return T

    def apply_color_optimization(self):
        self.read_mesh_pcl_trajectory()
        if self._mesh and self._transform_icp is not None:
            self.append_rgbd_images_from_path()
            trajectory = self.read_trayectory_camera(self._trajectory.trajectory_path)
            frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
            self._mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([self._mesh, frame])
            self.color_mapping(
                mesh=self._mesh,
                rgbd_images=self._rgbd_images,
                trajectory=trajectory,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A test program.")
    parser.add_argument("-c", "--config", help="config file")
    args = parser.parse_args()
    config = Config(config_path=args.config)
    # trj_path = "C:\\Users\\nicor\\OneDrive\\Documentos\\ELM\\Masterarbeit\\masterarbeit\\config_files\\trajectory.json"
    # mesh_path = "C:\\Users\\nicor\\OneDrive\\Documentos\\ELM\\Masterarbeit\\masterarbeit\\data_set\\banana\\mesh\\banana_final.STL"
    # point_cloud_path = "C:\\Users\\nicor\\OneDrive\\Documentos\\ELM\\Masterarbeit\\masterarbeit\\data_set\\banana\\point_cloud\\banana_with_table.ply"
    trj_file_mane = "trajectory.json"
    mesh_path = (
        "/home/nrodrigu/Documents/masterarbeit/data_set/marker/mesh/marker_final.STL"
    )
    point_cloud_path = "/home/nrodrigu/Documents/masterarbeit/data_set/marker/point_cloud/marker_with_table.ply"
    trj_name = "trj_icp.json"
    try:

        mapping_t = MappingTexture(
            object_name="marker",
            save_config=config.get_save_config(),
            mesh_path=mesh_path,
            point_cloud_path=point_cloud_path,
            debug=True,
        )
        mapping_t.apply_color_optimization()
    except Exception as excp:
        print(excp)
