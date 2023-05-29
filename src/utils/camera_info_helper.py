import enum
import json
import numpy as np
import open3d as o3d


class CameraInfoHelper(object):
    def __init__(self, camara_info_path: str):
        with open(camara_info_path) as file:
            # todo: use a json validator!
            self._data = json.loads(file.read())

    def camera_info_as_dict(self):
        return self._data

    def intrinsic_parameter_as_array(self, name: str):
        intrinsic_matrix = np.eye(3)
        intrinsic_dict = self._data[name]
        intrinsic_matrix[0, 0] = intrinsic_dict[CameraInfo.Intrinsic.FX.value]
        intrinsic_matrix[1, 1] = intrinsic_dict[CameraInfo.Intrinsic.FY.value]
        intrinsic_matrix[0, 2] = intrinsic_dict[CameraInfo.Intrinsic.PPX.value]
        intrinsic_matrix[1, 2] = intrinsic_dict[CameraInfo.Intrinsic.PPY.value]

        return intrinsic_matrix

    def intrinsic_parameter_open3d(self, name: str):
        intrinsic_dict = self._data[name]
        intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
            width=640,
            height=480,
            fx=intrinsic_dict[CameraInfo.Intrinsic.FX.value],
            fy=intrinsic_dict[CameraInfo.Intrinsic.FY.value],
            cx=intrinsic_dict[CameraInfo.Intrinsic.PPX.value],
            cy=intrinsic_dict[CameraInfo.Intrinsic.PPY.value],
        )
        intrinsic_dict = self._data[name]

        return intrinsic_matrix

    def dist_parameter_as_list(self, name: str):
        return self._data[name]

    def dist_parameter_as_array(self, name: str):
        return np.asarray(self._data[name])
    
    def depth_scale(self, name: str):
        if name in self._data.keys():
            return np.float64(self._data[name])
        return None
    
    
class CameraInfoHelperExtrinsic(object):
    def __init__(self, camara_info_path: str):
        with open(camara_info_path) as file:
            # todo: use a json validator!
            self._data = json.loads(file.read())

    def camera_info_as_dict(self):
        return self._data

    def extrinsic_parameter_as_array(self,  name_rot: str, name_trans:str):
        extrinsic_matrix = np.eye(4)
        extrinsic_dict_rot = self._data[name_rot]
        extrinsic_dict_trans = self._data[name_trans]
        extrinsic_matrix[0, 0] = extrinsic_dict_rot[CameraInfo.ExtrinsicRot.R11.value]
        extrinsic_matrix[0, 1] = extrinsic_dict_rot[CameraInfo.ExtrinsicRot.R12.value]
        extrinsic_matrix[0, 2] = extrinsic_dict_rot[CameraInfo.ExtrinsicRot.R13.value]
        extrinsic_matrix[0, 3] = extrinsic_dict_trans[CameraInfo.ExtrinsicTrans.T1.value]/1000.0
        
        extrinsic_matrix[1, 0] = extrinsic_dict_rot[CameraInfo.ExtrinsicRot.R21.value]
        extrinsic_matrix[1, 1] = extrinsic_dict_rot[CameraInfo.ExtrinsicRot.R22.value]
        extrinsic_matrix[1, 2] = extrinsic_dict_rot[CameraInfo.ExtrinsicRot.R23.value]
        extrinsic_matrix[1, 3] = extrinsic_dict_trans[CameraInfo.ExtrinsicTrans.T2.value]/1000.0

        extrinsic_matrix[2, 0] = extrinsic_dict_rot[CameraInfo.ExtrinsicRot.R31.value]
        extrinsic_matrix[2, 1] = extrinsic_dict_rot[CameraInfo.ExtrinsicRot.R32.value]
        extrinsic_matrix[2, 2] = extrinsic_dict_rot[CameraInfo.ExtrinsicRot.R33.value]
        extrinsic_matrix[2, 3] = extrinsic_dict_trans[CameraInfo.ExtrinsicTrans.T3.value]/1000.0
        return extrinsic_matrix


class CameraInfo:
    BASE_STRING = "data_set"
    FOLDER_NAME = "camera_info"

    class Intrinsic(enum.Enum):
        FX = "fx"
        FY = "fy"
        PPX = "ppx"
        PPY = "ppy"
        
    class ExtrinsicRot(enum.Enum):
        R11 = "r11"
        R12 = "r12"
        R13 = "r13"
        R21 = "r21"
        R22 = "r22"
        R23 = "r23"
        R31 = "r31"
        R32 = "r32"
        R33 = "r33"

    class ExtrinsicTrans(enum.Enum):
        T1 = "t1"
        T2 = "t2"
        T3 = "t3"

    class CameraLidar(enum.Enum):
        INTRINSIC_DEPTH = "intrinsic_depth"
        INTRINSIC_RGB = "intrinsic_rgb"
        DIST_RGB = "dist_rgb"
        DEPTH_SCALE = "depth_scale"
        FILE_NAME = "camera_lidar.json"

    class CameraThermal(enum.Enum):
        INTRINSIC = "intrinsic"
        DIST = "dist"
        FILE_NAME = "camera_thermal.json"
        
    class CameraExtrinsic(enum.Enum):
        ROTATION = "rotation"
        TRANSLATION = "translation"
        ESSENTIAL = "essential"
        RET = "ret"
        FILE_NAME_RGB_THERMAL = "extrinsic_color_thermal.json"
        FILE_NAME_RGB_DEPTH = "extrinsic_lidar.json"

if __name__ == "__main__":
    camera_info = CameraInfoHelper(
        "C:\\Users\\nicor\\OneDrive\\Documentos\\ELM\\Masterarbeit\\masterarbeit\\data_set\\camera_info\\camera_lidar.json"
    )
    intrinsic_depth = camera_info.intrinsic_parameter_as_array(
        CameraInfo.CameraLidar.INTRINSIC_DEPTH.value
    )
    intrinsic_rgb = camera_info.intrinsic_parameter_as_array(
        CameraInfo.CameraLidar.INTRINSIC_RGB.value
    )
    dist = camera_info.dist_parameter_as_array(CameraInfo.CameraLidar.DIST_RGB.value)
    depth_scale = camera_info.depth_scale(CameraInfo.CameraLidar.DEPTH_SCALE.value)
    print(intrinsic_depth)
    print(intrinsic_rgb)
    print(dist)
    print(depth_scale)

    camera_info_thermal = CameraInfoHelper(
        "C:\\Users\\nicor\\OneDrive\\Documentos\\ELM\\Masterarbeit\\masterarbeit\\data_set\\camera_info\\camera_thermal.json"
    )
    intrinsic = camera_info_thermal.intrinsic_parameter_as_array(
        CameraInfo.CameraThermal.INTRINSIC.value
    )
    dist = camera_info_thermal.dist_parameter_as_array(CameraInfo.CameraThermal.DIST.value)
    depth_scale = camera_info_thermal.depth_scale(CameraInfo.CameraLidar.DEPTH_SCALE.value)
    print(intrinsic)
    print(dist)
    print(depth_scale)
    
    camera_info_extrinsic = CameraInfoHelperExtrinsic(
        "C:\\Users\\nicor\\OneDrive\\Documentos\\ELM\\Masterarbeit\\masterarbeit\\data_set\\camera_info\\extrinsic_color_thermal.json"
    )
    extrinsic = camera_info_extrinsic.extrinsic_parameter_as_array(
        CameraInfo.CameraExtrinsic.ROTATION.value, CameraInfo.CameraExtrinsic.TRANSLATION.value
    )
    print(extrinsic)
    
    camera_info_extrinsic = CameraInfoHelperExtrinsic(
        "C:\\Users\\nicor\\OneDrive\\Documentos\\ELM\\Masterarbeit\\masterarbeit\\data_set\\camera_info\\extrinsic_lidar.json"
    )
    extrinsic = camera_info_extrinsic.extrinsic_parameter_as_array(
        CameraInfo.CameraExtrinsic.ROTATION.value, CameraInfo.CameraExtrinsic.TRANSLATION.value
    )
    print(extrinsic)