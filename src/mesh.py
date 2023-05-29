import logging
import open3d as o3d
import numpy as np
from src.config import MESH_CONFIG, SAVE_CONFIG
from src.utils.file_helper import FileHelper
from src.helper_3d import Helper3D
from matplotlib import pyplot as plt
import copy


class Mesh:
    def __init__(
        self,
        mesh_config: dict,
        save_config: dict,
        object_name: str,
        log_level=logging.DEBUG,
    ) -> None:
        self._logger = logging.getLogger()
        self._logger.setLevel(log_level)
        self._config = mesh_config
        self._helper_3d = Helper3D(log_level=log_level)
        self._mesh_path_save = FileHelper().get_object_path(
            folder_base=save_config[SAVE_CONFIG.BASE_FOLDER.value],
            folder_sub_base=save_config[SAVE_CONFIG.MESH_FOLDER.value],
            object_folder_name=object_name,
            object_name=object_name,
            object_type=save_config[SAVE_CONFIG.MESH_FORMAT.value],
        )
        self._mesh_final_path_save = FileHelper().get_object_path(
            folder_base=save_config[SAVE_CONFIG.BASE_FOLDER.value],
            folder_sub_base=save_config[SAVE_CONFIG.MESH_FOLDER.value],
            object_folder_name=object_name,
            object_name=object_name + "_final",
            object_type=save_config[SAVE_CONFIG.MESH_FORMAT.value],
        )
        self._point_cloud_path = FileHelper().get_object_path(
            folder_base=save_config[SAVE_CONFIG.BASE_FOLDER.value],
            folder_sub_base=save_config[SAVE_CONFIG.POINT_CLOUD_FOLDER.value],
            object_folder_name=object_name,
            object_name=object_name + "_with_table",
            object_type=save_config[SAVE_CONFIG.POINT_CLOUD_FORMAT.value],
        )
        self._point_cloud_obj_path = FileHelper().get_object_path(
            folder_base=save_config[SAVE_CONFIG.BASE_FOLDER.value],
            folder_sub_base=save_config[SAVE_CONFIG.POINT_CLOUD_FOLDER.value],
            object_folder_name=object_name,
            object_name=object_name,
            object_type=save_config[SAVE_CONFIG.POINT_CLOUD_FORMAT.value],
        )
        self._point_cloud = None
        self._point_cloud_obj = None
        self._mesh = None
        self._densities = None
        self._density_mesh = None
        self._final_mesh = None

    def _read_point_cloud(self, path):
        self._logger.debug("reading point cloud from file with path: " + path)
        point_cloud = o3d.io.read_point_cloud(filename=path)
        return point_cloud

    def normals_estimation(self):
        self._point_cloud.normals = o3d.utility.Vector3dVector(
            np.zeros((1, 3))
        )  # invalidate existing normals
        self._point_cloud.estimate_normals()
        self._point_cloud.orient_normals_consistent_tangent_plane(
            k=self._config[MESH_CONFIG.NUMBER_NEAREST_NEIGHBORS_K.value]
        )
        o3d.visualization.draw_geometries([self._point_cloud], point_show_normal=True)

    @staticmethod
    def create_mesh_and_density(point_cloud, depth, linear):
        (
            mesh,
            density_mesh,
        ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud,
            depth=depth,
            linear_fit=linear,
        )
        densities = np.asarray(density_mesh)
        o3d.visualization.draw_geometries([mesh])
        density_colors = plt.get_cmap("plasma")(
            (densities - densities.min()) / (densities.max() - densities.min())
        )

        density_colors = density_colors[:, :3]
        density_mesh = o3d.geometry.TriangleMesh()
        density_mesh.vertices = mesh.vertices
        density_mesh.triangles = mesh.triangles
        density_mesh.triangle_normals = mesh.triangle_normals

        return mesh, densities, density_mesh, density_colors

    @staticmethod
    def combine_low_density_with_mesh(
        mesh, density, vertices_to_remove, number_of_iterations
    ):
        density.remove_vertices_by_mask(vertices_to_remove)
        recom_mesh = density + mesh
        recom_mesh.remove_duplicated_vertices()
        final_mesh = recom_mesh.filter_smooth_simple(
            number_of_iterations=number_of_iterations
        )
        final_mesh.compute_vertex_normals()
        return final_mesh

    def point_cloud_to_mesh(self):
        o3d.visualization.draw_geometries([self._point_cloud])
        # todo: check if rescale posible here
        # self._point_cloud.scale(4, center=(0, 0, 0))
        # cropped point_cloud
        point_cloud_copy = copy.deepcopy(self._point_cloud)
        # point_cloud_cropped = self._helper_3d.crop_point_cloud(
        #     point_cloud_copy,
        #     diff_max_bound=np.array([[0.0], [-0.6], [0.0]]),
        #     diff_min_bound=np.array([[0.0], [0.0], [0.0]]),
        # )

        # (
        #     mesh_cropped,
        #     densities_cropped,
        #     density_mesh_cropped,
        #     density_colors_cropped,
        # ) = self.create_mesh_and_density(
        #     point_cloud=self._point_cloud_obj,
        #     depth=self._config[MESH_CONFIG.POISSON_DEPTH.value],
        #     linear=True,
        # )
        # mesh_cropped = mesh_cropped.filter_smooth_simple(
        #     number_of_iterations=self._config[MESH_CONFIG.FILTER_ITERATIONS.value]
        # )
        # mesh_cropped.compute_vertex_normals()
        # vertices_to_remove = densities_cropped < np.quantile(
        #     densities_cropped, self._config[MESH_CONFIG.QUANTILE_VALUE.value]
        # )
        # mesh_cropped = self.combine_low_density_with_mesh(
        #     mesh_cropped,
        #     density_mesh_cropped,
        #     vertices_to_remove,
        #     self._config[MESH_CONFIG.FILTER_ITERATIONS.value],
        # )

        # o3d.visualization.draw_geometries([mesh_cropped])

        (
            self._mesh,
            densities,
            self._density_mesh,
            density_colors,
        ) = self.create_mesh_and_density(
            point_cloud=self._point_cloud,
            depth=self._config[MESH_CONFIG.POISSON_DEPTH.value],
            linear=True,
        )
        density_2 = copy.deepcopy(self._density_mesh)

        self._density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)

        o3d.visualization.draw_geometries([self._density_mesh])

        mesh = copy.deepcopy(self._mesh)
        vertices_to_remove = densities < np.quantile(
            densities, self._config[MESH_CONFIG.QUANTILE_VALUE.value]
        )
        self._mesh.remove_vertices_by_mask(vertices_to_remove)
        self._mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([self._mesh])
        self._final_mesh = copy.deepcopy(self._mesh)
        self._final_mesh = self.combine_low_density_with_mesh(
            mesh,
            density_2,
            vertices_to_remove,
            self._config[MESH_CONFIG.FILTER_ITERATIONS.value],
        )
        o3d.visualization.draw_geometries([self._final_mesh])

    def smooth_surface(self, mesh):
        # different methods for smoothing see (http://open3d.org/docs/0.12.0/python_api/open3d.geometry.TriangleMesh.html)
        # mesh = mesh.filter_smooth_simple(
        #     number_of_iterations=self._config[MESH_CONFIG.FILTER_ITERATIONS.value]
        # )
        mesh = mesh.filter_smooth_laplacian(
            number_of_iterations=self._config[MESH_CONFIG.FILTER_ITERATIONS.value],
            lamda=0.5,
        )
        # mesh = mesh.filter_smooth_taubin(number_of_iterations=self._config[MESH_CONFIG.FILTER_ITERATIONS.value])
        return mesh

    def point_cloud_to_mesh_alpha(self):
        point_cloud_copy = copy.deepcopy(self._point_cloud)
        point_cloud_cropped = self._helper_3d.crop_point_cloud(
            point_cloud_copy,
            diff_max_bound=np.array([[0.0], [-0.6], [0.0]]),
            diff_min_bound=np.array([[0.0], [0.0], [0.0]]),
        )
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(
            point_cloud_cropped
        )
        self._mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            point_cloud_cropped, 0.03, tetra_mesh, pt_map
        )
        self._mesh.compute_vertex_normals()
        self._final_mesh = copy.deepcopy(self._mesh)
        self._final_mesh.filter_smooth_simple(
            number_of_iterations=self._config[MESH_CONFIG.FILTER_ITERATIONS.value]
        )
        o3d.visualization.draw_geometries([self._mesh])
        o3d.visualization.draw_geometries([self._final_mesh])

    def point_cloud_to_mesh_ball(self):
        point_cloud_copy = copy.deepcopy(self._point_cloud)
        point_cloud_cropped = self._helper_3d.crop_point_cloud(
            point_cloud_copy,
            diff_max_bound=np.array([[0.0], [-0.6], [0.0]]),
            diff_min_bound=np.array([[0.0], [0.0], [0.0]]),
        )
        radii = [0.005, 0.01]
        self._mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            point_cloud_cropped, o3d.utility.DoubleVector(radii)
        )
        self._mesh.compute_vertex_normals()
        self._final_mesh = copy.deepcopy(self._mesh)
        o3d.visualization.draw_geometries([self._mesh])
        # self._final_mesh.filter_smooth_simple(
        #     number_of_iterations=self._config[MESH_CONFIG.FILTER_ITERATIONS.value]
        # )
        o3d.visualization.draw_geometries([self._final_mesh])

    def object_point_cloud_to_mesh(self) -> bool:
        self._point_cloud = self._read_point_cloud(self._point_cloud_path)
        self._point_cloud_obj = self._read_point_cloud(self._point_cloud_obj_path)
        if self._point_cloud is not None:
            # self._point_cloud = self._helper_3d.preprocess_point_cloud(
            #     point_cloud=self._point_cloud,
            #     voxel_size=self._config[MESH_CONFIG.VOXEL_SIZE.value],
            # )
            # todo: check if needed normal estimation
            if not (self._point_cloud.has_normals()):
                self._logger.debug("normal estimation will be calculated")
                self.normals_estimation()
            self.point_cloud_to_mesh()
            self._logger.debug("successfully reconstruction")
            o3d.io.write_triangle_mesh(filename=self._mesh_path_save, mesh=self._mesh)
            o3d.io.write_triangle_mesh(
                filename=self._mesh_final_path_save, mesh=self._final_mesh
            )
            self._logger.debug("Mesh construction of the object was successfully")
            return True
        self._logger.error(
            "Not Point Cloud Found, check that file exist or is correctly"
        )
        return False
    
    def visualize_mesh(self):
        o3d.visualization.draw_geometries([self._final_mesh])
