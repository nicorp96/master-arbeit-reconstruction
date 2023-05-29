from src.config import Config, SCAN_CONFIG
from src.gui import GUI, MSG_TYPE
from scanner_3d import Scanner3D
from src.mesh import Mesh
from src.mapping_texture import MappingTexture
import logging


class MainClass:
    def __init__(self, path: str, log_level, object_name="no_name", modality=1) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._config = Config(path)
        self._scan_3d = Scanner3D(
            config=self._config, log_level=log_level, object_name=object_name
        )
        self._object_mesh = Mesh(
            mesh_config=self._config.get_mesh_config(),
            save_config=self._config.get_save_config(),
            log_level=log_level,
            object_name=object_name,
        )
        self._mapping_texture = MappingTexture(
            object_name=object_name,
            save_config=self._config.get_save_config(),
            rot_z=self._config.get_scan_config()[SCAN_CONFIG.CAMERA_ANGLE_ROT_Z.value],
            rot_y=self._config.get_scan_config()[SCAN_CONFIG.CAMERA_ANGLE_ROT_Y.value],
            rot_x=self._config.get_scan_config()[SCAN_CONFIG.CAMERA_ANGLE_ROT_X.value]
        )
        self._gui = GUI(None)
        self._object_name = object_name
        self._modality = modality

    def run(self):
        if self._modality == 1:
            self._gui.create_gui_modality_1(
                self.initialize_process,
                self.first_process,
                self.turn_table_step_angle_process,
                self.create_and_save_mesh,
                self._scan_3d.visualize_main_object,
                self._mapping_texture.apply_color_optimization,
                self._scan_3d.ignore_point_cloud,
            )
        elif self._modality == 2:
            self._gui.create_gui_modality_2(
                self.initialize_process,
                self.turn_table_step_angle_process,
                self.run_create_point_cloud_from_files,
                self.create_and_save_mesh,
                self._scan_3d.visualize_main_object,
                self._mapping_texture.apply_color_optimization,
                self._object_mesh.visualize_mesh()
            )
        else:
            self._logger.error(
                f"The Modality ({self._modality}) is wrong please check with --help"
            )
        self._scan_3d.finilize()

    def initialize_process(self):
        self._gui.remove_text()
        self._logger.debug("Initilization of the scan process of " + self._object_name)
        self._gui.label("Initilization of the scan process ...")
        self._gui.label(
            "Make sure there are no objects on the turntable",
            type=MSG_TYPE.USER,
        )

        successful_init = self._scan_3d.initialize_scan_process()
        if successful_init:
            self._gui.label("Initialization process was successful")
            self._gui.label(
                "Place the object on the turntable to be scan",
                type=MSG_TYPE.USER,
            )
        else:
            self._gui.label(
                "Initialization process was not successful, check logs for more details",
                type=MSG_TYPE.ERROR,
            )

    def first_process(self):
        self._gui.remove_text()
        self._gui.label(
            "First Scan of the object (" + self._object_name + ") in Process ..."
        )
        self._gui.label(
            "Make Sure the object is in the center of the turntable", type=MSG_TYPE.USER
        )
        self._gui.label(
            text=f"Starting with scan {self._scan_3d.current_turn_position} and {self._scan_3d.current_turn_angle} grad : "
        )

        self._scan_3d.current_turn_position = 0
        self._scan_3d.current_turn_angle = 0

        self._scan_3d.set_object_path_rgb_depth(
            str(self._scan_3d.current_turn_position)
        )

        successful = self._scan_3d.first_process()

        self._scan_3d.save_current_depth_rgb_imgs()

        if successful:
            self._gui.label(
                text=f"Rotate the turn table {self._config.get_scan_config()[SCAN_CONFIG.ROTATION_STEP_ANGLE.value]} grad counter clockwise and click the button: Next Position / Angle",
                type=MSG_TYPE.USER,
            )
        else:
            self._gui.label(
                "First Scan of the object was not successful, check logs for more details ",
                type=MSG_TYPE.ERROR,
            )

    def turn_table_step_angle_process(self):
        self._gui.remove_text()
        if not (self._scan_3d.current_turn_angle == 360):
            self._gui.label(
                text=f"Make Sure that the turntable rotated {self._config.get_scan_config()[SCAN_CONFIG.ROTATION_STEP_ANGLE.value]} grad counter clockwise",
                type=MSG_TYPE.USER,
            )
            if self._modality == 1:
                self._gui.label(f"current_pos: {self._scan_3d.current_turn_position}")
                self._scan_3d.set_object_path_rgb_depth(
                    str(self._scan_3d.current_turn_position)
                )
                successful = self._scan_3d.turn_table_step_angle_process()
                self._scan_3d.save_current_depth_rgb_imgs()
            else:
                successful = self._scan_3d.auto_scanning()
            self._gui.label(
                text=f"Starting with scan {self._scan_3d.current_turn_position} and {self._scan_3d.current_turn_angle} grad : "
            )
            if successful:
                self._gui.label(
                    text=f"Rotate the turn table {self._config.get_scan_config()[SCAN_CONFIG.ROTATION_STEP_ANGLE.value]} grad counter clockwise and click the button: Next Position / Angle",
                    type=MSG_TYPE.USER,
                )
            else:
                self._gui.label(
                    "The fitness score of the current icp regristation is to small, repeat the process",
                    type=MSG_TYPE.ERROR,
                )
        else:
            self._gui.label(
                text=f"Scan finished click on Visualize and Save to see result"
            )

    def run_create_point_cloud_from_files(self):
        self._gui.remove_text()
        self._logger.debug("Surface reconstruction of the object by given point cloud ")
        self._gui.label(
            "Starting with reconstruction of surface and the creation of point object"
        )
        successful = self._scan_3d.create_point_cloud_from_files()
        if successful:
            self._gui.label(
                text=f"The surface reconstruction was succesfully, and the Point cloud of the object was saved",
                type=MSG_TYPE.USER,
            )
            self._logger.debug(
                f"Succesfully finish the 3d construction of the {self._object_name}"
            )
        else:
            self._gui.label(
                "The fitness score of the current icp regristation is to small, repeat the process",
                type=MSG_TYPE.ERROR,
            )

    def create_and_save_mesh(self):
        self._gui.remove_text()
        self._logger.debug("Surface reconstruction of the object by given point cloud ")
        self._gui.label(
            "Starting with reconstruction of surface and the creation of mesh object"
        )
        result = self._object_mesh.object_point_cloud_to_mesh()
        if result:
            self._gui.label(
                "The surface reconstruction was succesfully, and the Mesh of the object was saved"
            )
        else:
            self._gui.label(
                "There was an Error by the reconstrcution -> see logs",
                type=MSG_TYPE.ERROR,
            )

    def restart(self):
        self._scan_3d.restart_scan_process_values()
        self._gui.remove_text()
        self._gui.label("Restarting Scan Process ....")
        self.first_process()
