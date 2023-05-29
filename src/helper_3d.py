import logging
import open3d as o3d
import numpy as np
import copy
import cv2
from src.utils.register_frames import Registration

class Helper3D:
    """
    This Class contains methods which implements algorithms for the:
        * Registration of ICP
        * Calculation of local refiment
        * Extration geometric features
        * Visualize result of transformation given by icp
    Links:
        - http://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html
    """

    def __init__(self, log_level=logging.DEBUG) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

    def registration_icp_colored(
        self, source, target, voxel_size, trans_init=np.identity(4)
    ):
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        voxel_radius = [0.08, 0.06, 0.04]  # [0.04, 0.02, 0.01]
        max_iter = [50, 30, 14]
        for scale in range(3):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            self._logger.debug([iter, radius, scale])
            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down,
                target_down,
                radius,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter
                ),
            )
            self._logger.debug(result_icp.fitness)
            trans_init = result_icp.transformation
        self.draw_registration_result(source, target, result_icp.transformation)
        return result_icp

    def local_refiment_point_cloud(self, point_cloud, voxel_size, radius=0.0):
        """This method downsample a point cloud and estimate normals to Extract geometric feature.
        Args:
            point_cloud (open3d.geometry.PointCloud): a Point cloud
            voxel_size (float): size of voxel for point cloud to be down sampled

        Returns:
            open3d.geometry.PointCloud: Downsampled Point cloud
        """
        # downsample voxel
        point_cloud = point_cloud.voxel_down_sample(voxel_size)
        # caluclation radius normal
        if radius == 0.0:
            radius = voxel_size * 2
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        return point_cloud

    def registration_icp(self, source, target, voxel_size, trans_init=np.identity(4)):
        """Computes the registration of the icp algorithm with the estimation method of open3d Point To Plane

        Args:
            source (open3d.geometry.PointCloud): the source point cloud
            target (open3d.geometry.PointCloud): the target point cloud
            voxel_size (float): size of vozex for the calculation of the max correspondence points-pair distance

        Returns:
            open3d.pipelines.registration.RegistrationResult: result of the transformation
        """
        threshold = voxel_size * 0.4
        source_down = source.voxel_down_sample(voxel_size)
        registration_icp = o3d.pipelines.registration.registration_icp(
            source=source_down,
            target=target,
            max_correspondence_distance=threshold,
            init=trans_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        self.draw_registration_result(
            source=source, target=target, transformation=registration_icp.transformation
        )

        return registration_icp

    def global_icp_registration(self, source, target, voxel_size):
        distance_threshold = voxel_size * 1.5
        radius_feature = voxel_size * 5
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
        result = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source,
                target,
                source_fpfh,
                target_fpfh,
                True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            )
        )
        return result

    # helper visualization from: http://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html
    def draw_registration_result(self, source, target, transformation):
        """_summary_

        Args:
            source (_type_): _description_
            target (_type_): _description_
            transformation (_type_): _description_
        """
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0, 0])
        # target_temp.paint_uniform_color([0,1, 0])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def crop_point_cloud(
        self,
        point_cloud,
        diff_max_bound=np.array([[0.0], [0.0], [0.0]]),
        diff_min_bound=np.array([[0.0], [0.00], [0.0]]),
    ):
        """Crop a point cloud

        Args:
            point_cloud (_type_): _description_
            diff_max_bound (np.ndarray[np.float64[3, 1]], optional): _description_. Defaults to np.array([[0.0],[0.0], [0.0]]).
            diff_min_bound (np.ndarray[np.float64[3, 1]], optional): _description_. Defaults to np.array([[0.0],[0.00],[0.0]]).

        Returns:
            _type_: _description_
        """
        max_bound_pc = point_cloud.get_max_bound()
        min_bound_pc = point_cloud.get_min_bound()

        max_bound_pc = np.expand_dims(max_bound_pc, axis=1)
        min_bound_pc = np.expand_dims(min_bound_pc, axis=1)

        max_bound_diff = max_bound_pc + diff_max_bound
        min_bound_diff = min_bound_pc + diff_min_bound

        aligned_box_pc = point_cloud.get_axis_aligned_bounding_box()

        aligned_box_pc.color = (0, 1, 0)

        aligned_box_axis = o3d.geometry.AxisAlignedBoundingBox(
            max_bound=max_bound_diff, min_bound=min_bound_diff
        )

        aligned_box_axis.color = (0, 0, 1)

        point_cloud_cropped = point_cloud.crop(aligned_box_axis)

        # todo: to visualize (remove)
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries(
            [frame, point_cloud_cropped, aligned_box_axis],
        )

        return point_cloud_cropped

    # def create_rgbd_from_color_and_depth(
    #     self, depth_img, color_img, convert_rgb_to_intensity=False
    # ):
    #     depth_img_open3d = o3d.geometry.Image(depth_img)
    #     color_img_open3d = o3d.geometry.Image(color_img)
    #     rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         color_img_open3d,
    #         depth_img_open3d,
    #         convert_rgb_to_intensity=convert_rgb_to_intensity,
    #     )
    #     return rgbd_image

    @staticmethod
    def get_depth_and_color_from_path_o3d(color_path, depth_path):
        color_img_open3d = o3d.io.read_image(color_path)
        depth_img_open3d = o3d.io.read_image(depth_path)
        return depth_img_open3d, color_img_open3d
    
    @staticmethod
    def get_depth_and_color_from_path_cv2(color_path, depth_path):
        color_img = cv2.imread(color_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        return depth_img, color_img
    
    def create_rgbd_from_color_and_depth_img(self, color_img, depth_img, convert_rgb_to_intensity=False):
        color_img_open3d = o3d.geometry.Image(color_img)
        depth_img_open3d = o3d.geometry.Image(depth_img)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img_open3d,
            depth_img_open3d,
            convert_rgb_to_intensity=convert_rgb_to_intensity,
        )
        return rgbd_image

    def create_rgbd_from_color_and_depth(
        self, color_path, depth_path,depth_intrinsic,rgb_intrinsic,extrinsic_depth_rgb,depth_scale, convert_rgb_to_intensity=False
    ):
        depth_img, color_img = self.get_depth_and_color_from_path_cv2(
            color_path, depth_path
        )
        color_img_r, depth_img_r = Registration().register_images(
            depth_img,
            color_img,
            depth_intrinsic,
            rgb_intrinsic,
            extrinsic_depth_rgb,
            depth_scale,
            25
        )
        color_img_open3d = o3d.geometry.Image(color_img_r)
        depth_img_open3d = o3d.geometry.Image(depth_img_r)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img_open3d,
            depth_img_open3d,
            convert_rgb_to_intensity=convert_rgb_to_intensity,
        )
        return color_img_r, depth_img_r, rgbd_image

    def create_point_cloud_from_rgbd(
        self,
        rgbd_image,
        camera_intrinsic,
        radius=0.1,
        max_nn=30,
    ):
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsic
        )
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )
        # source: https://github.com/isl-org/Open3D/issues/2046
        point_cloud.orient_normals_towards_camera_location(
            camera_location=np.array([0.0, 0.0, 0.0])
        )
        return point_cloud

    def remove_noise_point_cloud(
        self,
        point_cloud,
        nb_neighbors=50,
        std_ratio=2.0,
        nb_points=30,
        radius=0.5,
        radius_outlier=False,
    ):
        """_summary_

        Args:
            point_cloud (_type_): _description_
            nb_neighbors (int, optional): _description_. Defaults to 200.
            std_ratio (int, optional): _description_. Defaults to 1.
            nb_points (int, optional): _description_. Defaults to 100.
            radius (float, optional): _description_. Defaults to 0.01.

        Returns:
            _type_: _description_
        """
        pr_pointcloud, index1 = point_cloud.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        self.display_inlier_outlier(point_cloud, index1, [0, 1, 0])
        if radius_outlier:
            pr_pointcloud, index2 = pr_pointcloud.remove_radius_outlier(
                nb_points=nb_points, radius=radius
            )
            self.display_inlier_outlier(point_cloud, index2, [0, 0, 1])

        return pr_pointcloud

    @staticmethod
    def display_inlier_outlier(cloud, ind, color=[0, 0, 0]):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        outlier_cloud.paint_uniform_color(color)
        inlier_cloud.paint_uniform_color([0, 0, 0])
        # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    def calculate_transformation_matrix_with_rot_trans(
        self, rot=np.eye(3), trans=np.array([0.0, 0.0, 0.0])
    ):
        T = np.eye(4)
        T[:3, :3] = rot
        tran_mov = np.dot(rot, trans)
        T[0, 3] = trans[0] - tran_mov[0]
        T[1, 3] = trans[1] - tran_mov[1]
        T[2, 3] = trans[2] - tran_mov[2]
        return T

    def calculate_deg_to_rad(self, angle=0.0):
        return angle * np.pi / 180

    def save_image_cv2(self, image, path):
        cv2.imwrite(filename=path, img=image)

    def save_image_open3d(self, image, path):
        o3d.io.write_image(path, image)
