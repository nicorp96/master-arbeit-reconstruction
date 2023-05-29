import numpy as np
import cv2
import os
import re
from matplotlib import pyplot as plt
import cv2


"""
Links:
 - https://github.com/waynekyrie/Thermal_Camera_Calibration/blob/fc54751c04dec26bb10325cc4a031ee758eee27d/extract_template.py
 - https://henryzh47.github.io/Thermal-Camera-Calibration/
 - https://www.ri.cmu.edu/pub_files/2009/10/Calib.pdf
"""


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(file_list_ordered, key=alphanum_key)


def calibrate_camera(
    obj_points, img_points, img_shape, k_init=None, criteria=False, optimal=False
):
    flag = None
    if criteria:
        flag = cv2.CALIB_USE_INTRINSIC_GUESS
    rt, M, d, r, t = cv2.calibrateCamera(
        obj_points, img_points, img_shape, k_init, flag
    )
    if optimal:
        M, roi = cv2.getOptimalNewCameraMatrix(M, d, img_shape, 1, img_shape)
    print(f"rt = {rt}")
    return rt, M, d, r, t


def chess_board_detection(img, size_board, size_rec_mm):
    obj_p = np.zeros((size_board[0] * size_board[1], 3), np.float32)
    obj_p[:, :2] = np.mgrid[0 : size_board[0], 0 : size_board[1]].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    obj_p = obj_p * size_rec_mm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(image_gray, size_board, None)

    if ret is True:
        obj_points.append(obj_p)
        corners = cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners)
        # rt, M, d, r, t = cv2.calibrateCamera(
        #     obj_points, img_points, image_gray.shape[::-1], None, None
        # )
        # print(rt)

        # Draw and display the corners
        ret = cv2.drawChessboardCorners(img, size_board, corners, ret)
        # plt.imshow(img)
        # plt.show()
        # cv2.imshow("image", img)
        # cv2.waitKey(500)
    img_shape = image_gray.shape[::-1]
    return obj_points, img_points, img_shape


def draw_corners(img, corners):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = np.int0(corners)
    # Iterate over the corners and draw a circle at that location
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    # Show the frames
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.imshow(img)
    plt.show()


def read_and_extract_features(path, mod="c"):
    images_list = sorted_alphanum(os.listdir(path))
    obj_points = None
    img_points = None
    img_shape = None
    for i in range(len(images_list)):
        image = os.path.join(path, images_list[i])
        img = cv2.imread(image)
        if mod == "c":
            obj_points, img_points, img_shape = chess_board_detection(
                img, size_rec_mm=35, size_board=(5, 4)
            )
        elif mod == "v":
            image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(
                image_gray, 30, 0.01, 10, useHarrisDetector=True, k=0.04
            )
            kps = []
            if corners is not None:
                for x, y in np.float32(corners).reshape(-1, 2):
                    kps.append([(x, y)])
            draw_corners(img, corners)

    return obj_points, img_points, img_shape


def iterative_calib(path, k, d):
    images_list = sorted_alphanum(os.listdir(path))
    for i in range(len(images_list)):
        image = os.path.join(path, images_list[i])
        img = cv2.imread(image)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        img = cv2.undistort(img, k, d)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        obj_points, img_points, img_shape = chess_board_detection(
            img, size_rec_mm=35, size_board=(5, 4)
        )

    rt, M, d, r, t = calibrate_camera(
        obj_points, img_points, img_shape, k_init=None, criteria=False, optimal=False
    )
    print(M)


if __name__ == "__main__":
    base = os.path.join(os.getcwd(), "calibration")
    # read_images(base_path=base)
    path_save = os.path.join(os.getcwd(), "examples", "template")
    thermal_name = os.path.join(base, "thermal")

    k_thermal = np.array(
        [[689.08187135, 0, 316.87802919], [0, 712.12247478, 342.62316578], [0, 0, 1]]
    )
    d_thermal = np.array(
        [
            [1.51510371e-01],
            [-8.07170736e-01],
            [3.17334694e-04],
            [-2.48494564e-03],
            [1.00460207e00],
        ]
    )
    iterative_calib(thermal_name, k_thermal, d_thermal)
    # mod = "c"
    # obj_points,img_points, img_shape = read_and_extract_features(thermal_name,mod)
