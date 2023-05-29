import numpy as np
import cv2
import os
import re
import logging
from matplotlib import pyplot as plt

from utils.register_frames import Registration
from utils.file_helper import FileHelper
from utils.camera_info_helper import CameraInfoHelper, CameraInfo
from utils.camera_info_generation import (
    Intrinsics,
    IntrinsicThermal,
    Extrinsic,
    ExtrinsicRot,
    ExtrinsicTrans,
    IntrinsicRGB,
)
from scipy.spatial.transform import Rotation
"""
links:
- https://henryzh47.github.io/Thermal-Camera-Calibration/
- https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
- https://www.codefull.net/2016/03/align-depth-and-color-frames-depth-and-rgb-registration/
- https://henryzh47.github.io/Thermal-Camera-Calibration/
- https://arxiv.org/pdf/2101.05725.pdf
"""


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(file_list_ordered, key=alphanum_key)


def mean_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    # re-projection error
    #########################################
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("mean error: {}".format(mean_error / len(objpoints)))


def image_undistored(img, k=None, dist=None):
    if k is not None:
        img = cv2.undistort(img, k, dist)

    return img

def calculate_euler_angles_from_rotation(rotation_matrix):
    teta_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi
    teta_y = (
        np.arctan2(
            -rotation_matrix[2, 0],
            np.sqrt(
                (
                    rotation_matrix[2, 1] * rotation_matrix[2, 1]
                    + rotation_matrix[2, 2] * rotation_matrix[2, 2]
                )
            ),
        )
        * 180
        / np.pi
    )
    teta_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi

    print(f"angles: x= {teta_x}  y={teta_y}  z={teta_z}")

def get_simple_blon_detector(min_area=175, min_circularity=0.1, min_convexity=0.1):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area

    # # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = min_circularity

    # # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = min_convexity

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    return detector

class Calibration:
    FOLDER_BASE_NAME = "calibration"
    FORDER_DEPTH_STRING = "depth"
    FORDER_THERMAL_STRING = "thermal"
    FORDER_THERMAL_ONLY_STRING = "only_thermal"
    FORDER_LIDAR_STRING = "lidar"
    PATTERNS = {
        "circle": (
            cv2.findCirclesGrid,
            cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
        ),
        "chess": (cv2.findChessboardCorners, cv2.CALIB_CB_NORMALIZE_IMAGE),
    }
    RET_LIST = [1.2, 1.1]

    def __init__(
        self,
        pattern="chess",
        size_board=(6, 4),
        size_rec_mm=35,
        detector=None,
        debug=True,
        log_level=logging.DEBUG,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._debug = debug
        self._camera_info_lidar = CameraInfoHelper(
            FileHelper().get_path_camera_info(
                CameraInfo.BASE_STRING,
                CameraInfo.FOLDER_NAME,
                CameraInfo.CameraLidar.FILE_NAME.value,
            )
        )
        self._size_board = size_board
        self._size_rec_mm = size_rec_mm
        self._depth_path = FileHelper().get_path_calibration(
            self.FOLDER_BASE_NAME, self.FORDER_DEPTH_STRING
        )
        self._rgb_path = FileHelper().get_path_calibration(
            self.FOLDER_BASE_NAME, self.FORDER_LIDAR_STRING
        )
        self._thermal_path = FileHelper().get_path_calibration(
            self.FOLDER_BASE_NAME, self.FORDER_THERMAL_STRING
        )
        self._thermal_only_path = FileHelper().get_path_calibration(
            self.FOLDER_BASE_NAME, self.FORDER_THERMAL_ONLY_STRING
        )
        self._func_pattern, self._flag = self.PATTERNS[pattern]
        self._detector = detector

    @staticmethod
    def read_and_find_corners(
        path: str,
        logger: logging.Logger,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        size_rec_mm=35,
        size_board=(6, 4),
        pattern_func=cv2.findChessboardCorners,
        ret_value=2.0,
        flag=None,
        k=None,
        dist=None,
        detector=None,
        debug=True,
    ):

        images_list = sorted_alphanum(os.listdir(path))
        obj_p = np.zeros((size_board[0] * size_board[1], 3), np.float32)
        obj_p[:, :2] = np.mgrid[0 : size_board[0], 0 : size_board[1]].T.reshape(-1, 2)

        obj_p = obj_p * size_rec_mm

        obj_points = []
        img_points = []
        obj_points_test = []
        img_points_test = []
        for i in range(len(images_list)):
            image = os.path.join(path, images_list[i])
            img = cv2.imread(image)
            img = image_undistored(img, k, dist)
            image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if detector is not None:
                keypoints = detector.detect(img)
                blank = np.zeros((1, 1))
                blobs = cv2.drawKeypoints(
                    img,
                    keypoints,
                    blank,
                    (0, 0, 255),
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                )
                # plt.imshow(blobs)
                # plt.show()
                ret, corners = pattern_func(
                    image_gray, size_board, flags=flag, blobDetector=detector
                )
            else:
                ret, corners = pattern_func(image_gray, size_board, flags=flag)
            if ret is True:
                corners = cv2.cornerSubPix(
                    image_gray, corners, (11, 11), (-1, -1), criteria
                )
                img_points_test.append(corners)
                obj_points_test.append(obj_p)
                rt, k, d, r, t = cv2.calibrateCamera(
                    obj_points_test, img_points_test, image_gray.shape[::-1], None, None
                )
                # logger.debug(f"ret = {rt}")
                cv2.drawChessboardCorners(img, size_board, corners, ret)
                if rt < ret_value:
                    #logger.debug(f"ret = {rt}")
                    img_points.append(corners)
                    obj_points.append(obj_p)
                if debug == True:
                    plt.imshow(img)
                    plt.show()

        img_shape = image_gray.shape[::-1]
        params_3d_2d = dict(
            [
                ("obj_points", obj_points),
                ("img_points", img_points),
                ("img_shape", img_shape),
            ]
        )
        return params_3d_2d

    @staticmethod
    def read_and_find_corners_stereo(
        path_cam_1,
        path_cam_2,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        size_rec_mm=35,
        size_board=(6, 4),
        pattern_func=cv2.findChessboardCorners,
        flag=None,
        k_1=None,
        d_1=None,
        k_2=None,
        d_2=None,
        detector=None,
        debug=False,
        ret_val=1.11,
    ):
        images_list_1 = sorted_alphanum(os.listdir(path_cam_1))
        images_list_2 = sorted_alphanum(os.listdir(path_cam_2))

        if not (len(images_list_1) == len(images_list_2)):
            raise ValueError

        obj_p = np.zeros((size_board[0] * size_board[1], 3), np.float32)
        obj_p[:, :2] = np.mgrid[0 : size_board[0], 0 : size_board[1]].T.reshape(-1, 2)

        obj_p = obj_p * size_rec_mm

        obj_points = []
        img_points_1 = []
        img_points_2 = []
        obj_points_test = []
        img_points_test = []

        for i in range(len(images_list_1)):
            image_1 = os.path.join(path_cam_1, images_list_1[i])
            img_1 = cv2.imread(image_1)
            plt.subplot(4, 1, 1)
            plt.imshow(img_1)
            img_1 = image_undistored(img_1, k_1, d_1)
            plt.subplot(4, 1, 2)
            plt.imshow(img_1)
            image_2 = os.path.join(path_cam_2, images_list_2[i])
            img_2 = cv2.imread(image_2)
            plt.subplot(4, 1, 3)
            plt.imshow(img_2)
            img_2 = image_undistored(img_2, k_2, d_2)
            plt.subplot(4, 1, 4)
            plt.imshow(img_2)
            # plt.show()
            image_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            image_gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            if detector is not None:
                keypoints = detector.detect(img_1)
                blank = np.zeros((1, 1))
                blobs = cv2.drawKeypoints(
                    img_1,
                    keypoints,
                    blank,
                    (0, 0, 255),
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                )
                # plt.imshow(blobs)
                # plt.show()
                ret_1, corners_1 = pattern_func(
                    image_gray_1, size_board, flags=flag, blobDetector=detector
                )
                ret_2, corners_2 = pattern_func(
                    image_gray_2, size_board, flags=flag, blobDetector=detector
                )
            else:
                ret_1, corners_1 = pattern_func(image_gray_1, size_board, flags=flag)
                ret_2, corners_2 = pattern_func(image_gray_2, size_board, flags=flag)

            if ret_1 and ret_2 is True:
                corners_1 = cv2.cornerSubPix(
                    image_gray_1, corners_1, (11, 11), (-1, -1), criteria
                )
                corners_2 = cv2.cornerSubPix(
                    image_gray_2, corners_2, (11, 11), (-1, -1), criteria
                )

                img_points_test.append(corners_2)
                obj_points_test.append(obj_p)
                rt, k, d, r, t = cv2.calibrateCamera(
                    obj_points_test,
                    img_points_test,
                    image_gray_2.shape[::-1],
                    None,
                    None,
                )
                if rt < ret_val:
                    obj_points.append(obj_p)
                    img_points_1.append(corners_1[:,0,:])
                    img_points_2.append(corners_2[:,0,:])

                if debug == True:
                    ret_1 = cv2.drawChessboardCorners(
                        img_1, size_board, corners_1, ret_1
                    )
                    ret_2 = cv2.drawChessboardCorners(
                        img_2, size_board, corners_2, ret_2
                    )
                    plt.subplot(2, 1, 1)
                    plt.imshow(img_1)
                    plt.subplot(2, 1, 2)
                    plt.imshow(img_2)
                    plt.show()
        img_shape = image_gray_1.shape[::-1]
        params_3d_2d = dict(
            [
                ("obj_points", obj_points),
                ("img_points_1", img_points_1),
                ("img_points_2", img_points_2),
                ("img_shape", img_shape),
            ]
        )
        return params_3d_2d

    @staticmethod
    def calibrate_camera(
        obj_points, img_points, img_shape, k_init=None, criteria=False, optimal=True
    ):
        flag = None
        if criteria:
            flag = 0
            flag |= cv2.CALIB_FIX_INTRINSIC
            # flag |= cv2.CALIB_USE_INTRINSIC_GUESS
            # flag |= cv2.CALIB_FIX_ASPECT_RATIO

        rt, k, dist, rot_vec, t_vec = cv2.calibrateCamera(
            obj_points, img_points, img_shape, k_init, flag
        )
        if optimal:
            k, roi = cv2.getOptimalNewCameraMatrix(k, dist, img_shape, 1, img_shape)
        # mean_error(obj_points, img_points, r, t, M, d)

        camera_param = dict(
            [
                ("rt", rt),
                ("K", k),
                ("dist", dist),
                ("rotvec", rot_vec),
                ("tvec", t_vec),
            ]
        )
        return camera_param

    @staticmethod
    def stereo_calibrate(
        obj_points, img_points_1, img_points_2, k_1, d_1, k_2, d_2, shape
    ):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (
            cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
            100,
            1e-5,
        )
        (
            ret,
            k_1,
            d_1,
            k_2,
            d_2,
            rotation,
            translation,
            essential,
            fundamental,
        ) = cv2.stereoCalibrate(
            obj_points,
            img_points_1,
            img_points_2,
            k_1,
            d_1,
            k_2,
            d_2,
            shape,
            criteria=stereocalib_criteria,
            flags=flags,
        )

        stereo_param = dict(
            [
                ("ret", ret),
                ("K1", k_1),
                ("K2", k_2),
                ("dist1", d_1),
                ("dist2", d_2),
                ("R", rotation),
                ("T", translation),
                ("E", essential),
                ("F", fundamental),
            ]
        )
        return stereo_param

    def calibrate_thermal(self):
        params_3d_2d_thermal = self.read_and_find_corners(
            self._thermal_only_path,
            self._logger,
            size_board=self._size_board,
            size_rec_mm=self._size_rec_mm,
            pattern_func=self._func_pattern,
            flag=self._flag,
            detector=self._detector,
            debug=self._debug,
            ret_value=1.4,
        )

        if len(params_3d_2d_thermal["obj_points"]) == 0:
            self._logger.error(
                f"No pattern where found in the images provided check: *pattern type -> circle / chess and *size of board: {self._size_board}"
            )
            raise ValueError

        camera_param_thermal = self.calibrate_camera(
            params_3d_2d_thermal["obj_points"],
            params_3d_2d_thermal["img_points"],
            params_3d_2d_thermal["img_shape"],
        )

        self._logger.debug("K_thermal_init = {}".format(camera_param_thermal["K"]))

        self.mean_re_projection_error(
            params_3d_2d_thermal["obj_points"],
            params_3d_2d_thermal["img_points"],
            camera_param_thermal["rotvec"],
            camera_param_thermal["tvec"],
            camera_param_thermal["K"],
            camera_param_thermal["dist"],
        )
        for ret_i in self.RET_LIST:
            params_3d_2d_thermal = self.read_and_find_corners(
                self._thermal_only_path,
                self._logger,
                size_board=self._size_board,
                size_rec_mm=self._size_rec_mm,
                pattern_func=self._func_pattern,
                flag=self._flag,
                detector=self._detector,
                debug=self._debug,
                k=camera_param_thermal["K"],
                dist=camera_param_thermal["dist"],
                ret_value=ret_i,
            )

            camera_param_thermal = self.calibrate_camera(
                params_3d_2d_thermal["obj_points"],
                params_3d_2d_thermal["img_points"],
                params_3d_2d_thermal["img_shape"],
                camera_param_thermal["K"],
            )
            self.mean_re_projection_error(
                params_3d_2d_thermal["obj_points"],
                params_3d_2d_thermal["img_points"],
                camera_param_thermal["rotvec"],
                camera_param_thermal["tvec"],
                camera_param_thermal["K"],
                camera_param_thermal["dist"],
            )
            self._logger.debug("K_thermal_i = {}".format(camera_param_thermal["K"]))
        return camera_param_thermal

    def calibrate_thermal_lidar(self):
        camera_param_thermal = self.calibrate_thermal()
        # camera_param_thermal = {}
        # K_t = np.array(
        #     [
        #         [647.302978515625, 0.0, 252.8042473678288],
        #         [0.0, 694.7332763671875, 236.9809379074468],
        #         [0.0, 0.0, 1.0],
        #     ]
        # )
        # dist = np.asarray(
        # [
        #     0.09921718633227439,
        #     -1.1645498132794279,
        #     -0.003274072134242213,
        #     -0.009549113879067933,
        #     2.0988729157065107
        # ])
        camera_param_thermal["K"] = K_t
        camera_param_thermal["dist"] = dist
        params_3d_2d = self.read_and_find_corners_stereo(
            path_cam_1=self._rgb_path,
            path_cam_2=self._thermal_path,
            size_board=self._size_board,
            size_rec_mm=self._size_rec_mm,
            pattern_func=self._func_pattern,
            flag=self._flag,
            k_1=self._camera_info_lidar.intrinsic_parameter_as_array(
                CameraInfo.CameraLidar.INTRINSIC_RGB.value
            ),
            d_1=self._camera_info_lidar.dist_parameter_as_array(
                CameraInfo.CameraLidar.DIST_RGB.value
            ),
            k_2=camera_param_thermal["K"],
            d_2=camera_param_thermal["dist"],
            detector=self._detector,
            debug=self._debug,
        )
        rot_thermal_rgb, t_thermal_rgb = self.get_camera_displacement(
            params_3d_2d["obj_points"],
            params_3d_2d["img_points_1"],
            self._camera_info_lidar.intrinsic_parameter_as_array(
                CameraInfo.CameraLidar.INTRINSIC_RGB.value
            ),
            self._camera_info_lidar.dist_parameter_as_array(
                CameraInfo.CameraLidar.DIST_RGB.value
            ),
            params_3d_2d["img_points_2"],
            camera_param_thermal["K"],
            camera_param_thermal["dist"],
        )
        self._logger.debug("stereo rotation R = {}".format(rot_thermal_rgb))
        self._logger.debug("stereo translation T = {}".format(t_thermal_rgb))
        #calculate_euler_angles_from_rotation(rot_thermal_rgb)
        self.save_camera_info(
            camera_param_thermal["K"],
            camera_param_thermal["dist"],
            rot_thermal_rgb,
            t_thermal_rgb,
            0.0,
            FileHelper().get_path_camera_info(
                CameraInfo.BASE_STRING,
                CameraInfo.FOLDER_NAME,
                CameraInfo.CameraThermal.FILE_NAME.value,
            ),
            FileHelper().get_path_camera_info(
                CameraInfo.BASE_STRING,
                CameraInfo.FOLDER_NAME,
                CameraInfo.CameraExtrinsic.FILE_NAME_RGB_THERMAL.value,
            ),
        )
        return camera_param_thermal, rot_thermal_rgb, t_thermal_rgb

    def calibrate_thermal_lidar_with_stereo(self):
        params_3d_2d_thermal = self.read_and_find_corners(
            self._thermal_path,
            self._logger,
            size_board=self._size_board,
            size_rec_mm=self._size_rec_mm,
            pattern_func=self._func_pattern,
            flag=self._flag,
            detector=self._detector,
            debug=self._debug,
        )
        if len(params_3d_2d_thermal["obj_points"]) == 0:
            self._logger.error(
                f"No pattern where found in the images provided check: *pattern type -> circle / chess and *size of board: {self._size_board}"
            )

        else:
            camera_param_thermal = self.calibrate_camera(
                params_3d_2d_thermal["obj_points"],
                params_3d_2d_thermal["img_points"],
                params_3d_2d_thermal["img_shape"],
            )

            params_3d_2d = self.read_and_find_corners_stereo(
                path_cam_1=self._rgb_path,
                path_cam_2=self._thermal_path,
                size_board=self._size_board,
                size_rec_mm=self._size_rec_mm,
                pattern_func=self._func_pattern,
                flag=self._flag,
                k_1=self._camera_info_lidar.intrinsic_parameter_as_array(
                    CameraInfo.CameraLidar.INTRINSIC_RGB.value
                ),
                d_1=self._camera_info_lidar.dist_parameter_as_array(
                    CameraInfo.CameraLidar.DIST_RGB.value
                ),
                k_2=camera_param_thermal["K"],
                d_2=camera_param_thermal["dist"],
                detector=self._detector,
                debug=self._debug,
            )

            camera_param_thermal = self.calibrate_camera(
                params_3d_2d["obj_points"],
                params_3d_2d["img_points_2"],
                params_3d_2d["img_shape"],
                camera_param_thermal["K"],
                True,
            )
            self.mean_re_projection_error(
                params_3d_2d["obj_points"],
                params_3d_2d["img_points_2"],
                camera_param_thermal["rotvec"],
                camera_param_thermal["tvec"],
                camera_param_thermal["K"],
                camera_param_thermal["dist"],
            )
            stereo_param = self.stereo_calibrate(
                obj_points=params_3d_2d["obj_points"],
                img_points_1=params_3d_2d["img_points_1"],
                img_points_2=params_3d_2d["img_points_2"],
                k_1=self._camera_info_lidar.intrinsic_parameter_as_array(
                    CameraInfo.CameraLidar.INTRINSIC_RGB.value
                ),
                k_2=camera_param_thermal["K"],
                d_1=self._camera_info_lidar.dist_parameter_as_array(
                    CameraInfo.CameraLidar.DIST_RGB.value
                ),
                d_2=camera_param_thermal["dist"],
                shape=params_3d_2d["img_shape"],
            )
            self._logger.debug("stereo ret = {}".format(stereo_param["ret"]))
            self._logger.debug("stereo rotation R = {}".format(stereo_param["R"]))
            self._logger.debug("stereo translation T = {}".format(stereo_param["T"]/1000.0))
            #calculate_euler_angles_from_rotation(stereo_param["R"])
            self.save_camera_info(
                camera_param_thermal["K"],
                camera_param_thermal["dist"],
                stereo_param["R"],
                stereo_param["T"] / 1000.0,
                stereo_param["ret"],
                FileHelper().get_path_camera_info(
                    CameraInfo.BASE_STRING,
                    CameraInfo.FOLDER_NAME,
                    CameraInfo.CameraThermal.FILE_NAME.value,
                ),
                FileHelper().get_path_camera_info(
                    CameraInfo.BASE_STRING,
                    CameraInfo.FOLDER_NAME,
                    CameraInfo.CameraExtrinsic.FILE_NAME_RGB_THERMAL.value,
                ),
            )
            return camera_param_thermal, stereo_param

    def save_camera_info(self, k, dist, rot, trans, ret, path_1, path_2):
        json_camera_info = IntrinsicThermal(
            Intrinsics(k),
            dist,
            path_1,
        )
        json_extrinsic_thermal_color = Extrinsic(
            ExtrinsicRot(rot), ExtrinsicTrans(trans), ret, path_2
        )
        json_camera_info.save_camera_info()
        json_extrinsic_thermal_color.save_camera_info()

    def mean_re_projection_error(self, obj_points, img_points, rvecs, tvecs, k, dist):
        mean_error = 0
        for i in range(len(obj_points)):
            img_points_2, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i], k, dist
            )
            error = cv2.norm(img_points[i], img_points_2, cv2.NORM_L2) / len(
                img_points_2
            )
            mean_error += error
        self._logger.debug("mean error: {}".format(mean_error / len(obj_points)))

    def get_camera_displacement(
        self,
        obj_points,
        image_points_1,
        k_color,
        dist_color,
        image_points_2,
        k_thermal,
        dist_thermal,
    ):
        """Computes camera displacement for thermal and RGB

        Args:
            obj_points (_type_): _description_
            image_points_1 (_type_): _description_
            k_color (_type_): _description_
            dist_color (_type_): _description_
            image_points_2 (_type_): _description_
            k_thermal (_type_): _description_
            dist_thermal (_type_): _description_

        Returns:
            _type_: _description_
        """
        # https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
        ret_1, rvect_color, tvec_color = cv2.solvePnP(
            np.vstack(obj_points), np.vstack(image_points_1), k_color, dist_color 
        )
        ret_2, rvect_thermal, tvec_thermal = cv2.solvePnP(
            np.vstack(obj_points), np.vstack(image_points_2), k_thermal, dist_thermal
        )
        R_color, _ = cv2.Rodrigues(rvect_color)
        R_thermal, _ = cv2.Rodrigues(rvect_thermal)
        
        R_2to1 = np.dot(R_color, np.transpose(R_thermal))
        tvec_2to1 = (
            np.dot(R_color, (np.dot(-np.transpose(R_thermal), tvec_thermal)))
            + tvec_color
        )
        R_2to1 = self.change_x_rotation(R_2to1)
        # tvec_2to1[0] = tvec_2to1[0] - 2.5
        # tvec_2to1[1] = tvec_2to1[1] - 2.5
        
        return R_2to1, tvec_2to1
    
    @staticmethod
    def change_x_rotation(R):
        """Change rotation of the x axis

        Args:
            R (): rotation matrix

        Returns:
            R: modifided rotation matrix
        """
        rot =  Rotation.from_matrix(R)
        angles = rot.as_euler("xyz",degrees=True)
        angles[0]= -(angles[0] + 180)
        rot = Rotation.from_euler("xyz",angles,degrees=True)
        R = rot.as_matrix()
        return R


if __name__ == "__main__":
    name_obj= "banana_1.png"
    thermal_path = os.path.join(
        os.getcwd(), "examples", "test", "thermal", name_obj
    )
    lidar_path = os.path.join(os.getcwd(), "examples", "test", "color", name_obj)
    depth_path = os.path.join(os.getcwd(), "examples", "test", "depth", name_obj)
    img_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    img_right = cv2.imread(thermal_path)
    img_left = cv2.imread(lidar_path)

    extrinsic = np.array(
        [
            [9.99986172e-01, 4.60144877e-03, 2.54564988e-03, -3.25388246e-04],
            [-4.66419989e-03, 9.99671042e-01, 2.52197012e-02, 1.35215586e-02],
            [-2.42876541e-03, -2.52312254e-02, 9.99678671e-01, -6.01826468e-03],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    depth_scale = 0.0002500000118743628

    logging.basicConfig(level=logging.DEBUG)
    # calibration_obj = Calibration(
    #     pattern="chess", size_board=(5, 4), size_rec_mm=35, debug=True
    # )
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 175

    # # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.1

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    calibration_obj = Calibration(
        pattern="circle",
        size_board=(5, 4),
        size_rec_mm=25,
        detector=detector,
        debug=False,
    )
    M_L = calibration_obj._camera_info_lidar.intrinsic_parameter_as_array(
        CameraInfo.CameraLidar.INTRINSIC_RGB.value
    )
    k_depth = calibration_obj._camera_info_lidar.intrinsic_parameter_as_array(
        CameraInfo.CameraLidar.INTRINSIC_DEPTH.value
    )
    
    #calibration_obj.calibrate_thermal()
    camera_param_thermal, R, t = calibration_obj.calibrate_thermal_lidar()
    img_right = cv2.undistort(
        img_right, camera_param_thermal["K"], camera_param_thermal["dist"]
    )
    img_left = cv2.undistort(
        img_left,
        M_L,
        calibration_obj._camera_info_lidar.dist_parameter_as_array(
            CameraInfo.CameraLidar.DIST_RGB.value
        ),
    )
    
    t_thermal_rgb = np.eye(4)
    t_thermal_rgb[0:3, 0:3] = R
    t_thermal_rgb[0:3, 3] = t.T/1000.0

    Registration().register_thermal_rgb(
        img_left,
        img_right,
        img_depth,
        M_L,
        camera_param_thermal["K"],
        k_depth,
        t_thermal_rgb,
        extrinsic,
        depth_scale,
    )
