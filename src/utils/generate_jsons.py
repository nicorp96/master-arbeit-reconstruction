import json
import numpy as np
import open3d as o3d
import os
from src.utils.file_helper import FileHelper


def get_array_as_list(matrix):
    result = []
    for num in np.nditer(matrix.T.copy(order="C")):
        result.append(float(num))
        # print(num, end=' ')
    return result


class IntrinsicParameters:
    def __init__(self, intrinsic_matrix, height, width):
        self.intrinsic_matrix = get_array_as_list(intrinsic_matrix)
        self.height = height
        self.width = width

    def __iter__(self):
        yield from {
            "height": self.height,
            "intrinsic_matrix": self.intrinsic_matrix,
            "width": self.width,
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return self.__str__()

    def to_dict(self):
        return dict(self)


class PinholeCameraParameters:
    CLASS_NAME = "PinholeCameraParameters"

    def __init__(self, extrinsic, intrinsic):
        self.extrinsic = get_array_as_list(extrinsic)
        self.intrinsic = intrinsic

    def __iter__(self):
        yield from {
            "class_name": self.CLASS_NAME,
            "extrinsic": self.extrinsic,
            "intrinsic": self.intrinsic,
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return self.__str__()

    def to_dict(self):
        return dict(self)


class PinholeCameraTrajectory:
    CLASS_NAME = "PinholeCameraTrajectory"

    def __init__(self, file_name="trajectory.json"):
        self._parameters = []
        self._current_intrinsic = None
        self._current_extrinsic = None
        self.save_path_trj = file_name
        self._counter_trj = 0

    def remove_current_file(self):
        if os.path.exists(self.save_path_trj):
            os.remove(self.save_path_trj)

    @property
    def trajectory_path(self):
        return self.save_path_trj

    @property
    def counter_trj(self):
        return self._counter_trj

    def append_parameter_to_trajectory(
        self, intrinsic: o3d.camera.PinholeCameraIntrinsic, extrinsic
    ):
        if intrinsic and extrinsic is not None:
            self._current_intrinsic = IntrinsicParameters(
                intrinsic_matrix=intrinsic.intrinsic_matrix,
                height=intrinsic.height,
                width=intrinsic.width,
            )
            self._current_extrinsic = extrinsic
            pin_param = PinholeCameraParameters(
                extrinsic=self._current_extrinsic,
                intrinsic=self._current_intrinsic.to_dict(),
            )
            self._parameters.append(pin_param.to_dict())
            self._counter_trj += 1

    def to_json(self):
        to_return = {"class_name": self.CLASS_NAME}
        to_return["parameters"] = self._parameters
        return to_return

    def save_current_trajectory(self):
        with open(self.save_path_trj, "w") as f:
            json.dump(self.to_json(), f)


if __name__ == "__main__":
    pinhole_trajectory = PinholeCameraTrajectory()
    for i in range(10):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=640, height=480, fx=594.6319, fy=594.5186, cx=326.8489, cy=236.473
        )
        extrinsic = np.array(
            [
                [
                    0.99875666779623684,
                    0.0042556066034366898,
                    -0.049668987758906415,
                    -2.1888509581876607,
                ],
                [
                    -0.015804533047987467,
                    0.97198214070610867,
                    -0.23452278116103262,
                    -1.181867321777915,
                ],
                [
                    0.047279332352442131,
                    0.2350161865931947,
                    0.97084090188428929,
                    0.94394657256919134,
                ],
                [0, 0, 0, 1],
            ]
        )
        # pinhole_trajectory.append_parameter_to_trajectory(
        #     intrinsic=intrinsic, extrinsic=extrinsic
        # )
    print(extrinsic)
    # pinhole_trajectory.save_current_trajectory()

    # traj = o3d.io.read_pinhole_camera_trajectory(pinhole_trajectory.save_path_trj)
    # print(traj)
