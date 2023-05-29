import json
import numpy as np


def get_array_as_list(matrix):
    result = []
    for num in np.nditer(matrix.T.copy(order="C")):
        result.append(float(num))
    return result


class Intrinsics:
    def __init__(self, intrinsic_matrix):
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.ppx = intrinsic_matrix[0, 2]
        self.ppy = intrinsic_matrix[1, 2]

    def __iter__(self):
        yield from {
            "fx": self.fx,
            "fy": self.fy,
            "ppx": self.ppx,
            "ppy": self.ppy,
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return self.__str__()

    def to_dict(self):
        return dict(self)


class IntrinsicRGB:
    DEPTH = "intrinsic_depth"
    RGB = "intrinsic_rgb"
    DIST = "dist_rgb"
    DEPTH_SCALE= "depth_scale"

    def __init__(self, k_depth:Intrinsics, k_rgb:Intrinsics, dist_rgb,depth_scale, file_path):
        self.k_depth = k_depth.to_dict()
        self.k_rgb = k_rgb.to_dict()
        self.dist_rgb = get_array_as_list(dist_rgb)
        self.depth_scale = depth_scale
        self.save_path = file_path

    def __iter__(self):
        yield from {
            self.DEPTH: self.k_depth,
            self.RGB: self.k_rgb,
            self.DEPTH_SCALE: self.depth_scale,
            self.DIST: self.dist_rgb,
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return self.__str__()

    def to_dict(self):
        return dict(self)

    def save_camera_info(self):
        with open(self.save_path, "w") as f:
            json.dump(self.to_dict(), f)


class IntrinsicThermal:
    DEPTH = "intrinsic"
    DIST = "dist"

    def __init__(self, k:Intrinsics, dist, file_path):
        self.k = k.to_dict()
        self.dist = get_array_as_list(dist)
        self.save_path = file_path

    def __iter__(self):
        yield from {
            self.DEPTH: self.k,
            self.DIST: self.dist,
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return self.__str__()

    def to_dict(self):
        return dict(self)

    def save_camera_info(self):
        with open(self.save_path, "w") as f:
            json.dump(self.to_dict(), f)

class ExtrinsicRot:
    def __init__(self, rot_matrix):
        self.r11 = rot_matrix[0, 0]
        self.r12 = rot_matrix[0, 1]
        self.r13 = rot_matrix[0, 2]
        self.r21 = rot_matrix[1, 0]
        self.r22 = rot_matrix[1, 1]
        self.r23 = rot_matrix[1, 2]
        self.r31 = rot_matrix[2, 0]
        self.r32 = rot_matrix[2, 1]
        self.r33 = rot_matrix[2, 2]

    def __iter__(self):
        yield from {
            "r11": self.r11,
            "r12": self.r12,
            "r13": self.r13,
            "r21": self.r21,
            "r22": self.r22,
            "r23": self.r23,
            "r31": self.r31,
            "r32": self.r32,
            "r33": self.r33,
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return self.__str__()

    def to_dict(self):
        return dict(self)
    
class ExtrinsicTrans:
    def __init__(self, trans_matrix):
        self.t1 = trans_matrix[0][0]
        self.t2 = trans_matrix[1][0]
        self.t3 = trans_matrix[2][0]

    def __iter__(self):
        yield from {
            "t1": self.t1,
            "t2": self.t2,
            "t3": self.t3,
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return self.__str__()

    def to_dict(self):
        return dict(self)
    

class Extrinsic:
    ROTATION = "rotation"
    TRANSLATION = "translation"
    RET = "ret"

    def __init__(self, rot:ExtrinsicRot, trans:ExtrinsicTrans,ret, file_path):
        self.rot = rot.to_dict()
        self.trans = trans.to_dict()
        self.ret = ret
        self.save_path = file_path

    def __iter__(self):
        yield from {
            self.ROTATION: self.rot,
            self.TRANSLATION: self.trans,
            self.RET: self.ret,
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return self.__str__()

    def to_dict(self):
        return dict(self)

    def save_camera_info(self):
        with open(self.save_path, "w") as f:
            json.dump(self.to_dict(), f)