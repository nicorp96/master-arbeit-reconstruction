import enum
import json


class Config(object):
    def __init__(self, config_path):
        with open(config_path) as config_file:
            # todo: use a json validator!
            self._data = json.loads(config_file.read())

    def get_validated_config(self):
        return self._data

    def get_camara_config(self):
        return self._data[CAMERA_CONFIG.CAMERA.value]

    def get_scan_config(self):
        return self._data[SCAN_CONFIG.SCAN.value]

    def get_mesh_config(self):
        return self._data[MESH_CONFIG.MESH.value]

    def get_save_config(self):
        return self._data[SAVE_CONFIG.SAVE.value]


class CAMERA_CONFIG(enum.Enum):
    CAMERA = "camera"
    WIDTH_DEPTH = "width_depth"
    HEIGHT_DEPTH = "height_depth"
    WIDTH_COLOR = "width_color"
    HEIGHT_COLOR = "height_color"
    FRAMERATE = "framerate"


class SCAN_CONFIG(enum.Enum):
    SCAN = "scan"
    CAMERA_ANGLE_ROT_X = "camera_angle_rot_x_d"
    CAMERA_ANGLE_ROT_Y = "camera_angle_rot_y_d"
    CAMERA_ANGLE_ROT_Z = "camera_angle_rot_z_d"
    CROP_OUTLIER_POINTS = "crop_outlier_points"
    CROP_TURN_TABLE = "crop_turn_table"
    ROTATION_STEP_ANGLE = "rotation_step_angle"
    VOXEL_SIZE_TURNTABLE = "voxel_size_turntable"
    VOLEX_SIZE_OBJECT = "voxel_size_object"
    MIN_FITNESS_SCORE = "min_fitness_score"


class MESH_CONFIG(enum.Enum):
    MESH = "mesh"
    POISSON_DEPTH = "poisson_depth"
    VOXEL_SIZE = "voxel_size"
    QUANTILE_VALUE = "quantile_value"
    NUMBER_NEAREST_NEIGHBORS_K = "number_nearest_neighbors_k"
    FILTER_ITERATIONS = "filter_iterations"
    SAVE_FOLDER_NAME = "save_folder_name"
    SAVE_FORMAT = "save_format"


class SAVE_CONFIG(enum.Enum):
    SAVE = "save_config"
    BASE_FOLDER = "base_folder"
    POINT_CLOUD_FOLDER = "point_cloud_folder"
    POINT_CLOUD_FORMAT = "point_cloud_format"
    MESH_FOLDER = "mesh_folder"
    MESH_FORMAT = "mesh_format"
    DEPTH_IMAGES_FOLDER = "depth_images_folder"
    DEPTH_FORMAT = "depth_format"
    RGB_IMAGES_FOLDER = "rgb_images_folder"
    RGB_FORMAT = "rgb_format"
    THERMAL_IMAGES_FOLDER = "thermal_images_folder"
    THERMAL_FORMAT = "thermal_format"
    POSE_FOLDER = "pose_folder"
    ICP_FILE_NAME = "icp_file_name"
    TRAJECTORY_FILE_NAME = "trajectory_file_name"


class VISUALIZATION_CONFIG(enum.Enum):
    VISUALIZATION = "visualization"
    ICP = "icp"
    SPHERE = "sphere"
    COORDS = "coords"
