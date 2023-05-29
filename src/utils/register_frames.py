import os
import sys
import platform

os_platf = platform.system()
if os_platf == "Linux":
    sys.path.append(os.path.abspath("../kinect_Smoothing"))
else:
    sys.path.append(os.path.abspath("..\\Kinect_Smoothing"))
from kinect_smoothing.depth_image_smoothing import HoleFilling_Filter, Denoising_Filter
import matplotlib.pyplot as plt
import numpy as np
import cv2
import open3d as o3d
from src.utils.depth_holes import fill_depth_colorization

"""
https://github.com/intelligent-control-lab/Kinect_Smoothing
"""


def fill_holes_depth(image_frame):
    hole_filter = HoleFilling_Filter(flag="fmi")
    image_frame = hole_filter.smooth_image(image_frame)
    # noise_filter = Denoising_Filter(flag='modeling', theta=60)
    # image_frame = noise_filter.smooth_image(image_frame)
    return image_frame


class Registration:
    def register_images(
        self,
        depth_img,
        color_img,
        k_depth,
        k_rgb,
        t_depth_rgb,
        depth_scale,
        threshold=0,
    ):
        height = depth_img.shape[0]
        width = depth_img.shape[1]

        x_y_z_rgb = np.zeros((height, width, 3))

        rgb_img_register = np.zeros((height, width, 3))
        for v in range(height):
            for u in range(width):
                # 1. Calculate (X,Y,Z) points from depth image at the coordinate (v,u)
                z = depth_img[v, u] * depth_scale
                x = ((u - k_depth[0, 2]) * z) / k_depth[0, 0]
                y = ((v - k_depth[1, 2]) * z) / k_depth[1, 1]

                # 2. Transform (X,Y,Z)  points from depth frame to color camera frame
                x_y_z_rgb[v, u, 0:3] = np.dot(t_depth_rgb, np.array([x, y, z, 1])).T[
                    0:3
                ]

                # 3. Calculate (v,u) points rgb image plane at the coordinate (X,Y,Z) rgb
                u_rgb = (
                    (x_y_z_rgb[v, u, 0] * k_rgb[0, 0] / x_y_z_rgb[v, u, 2])
                    + k_rgb[0, 2]
                    + threshold
                )
                v_rgb = (
                    (x_y_z_rgb[v, u, 1] * k_rgb[1, 1] / x_y_z_rgb[v, u, 2])
                    + k_rgb[1, 2]
                    + threshold
                )

                if u_rgb > width - 1 or v_rgb > height - 1 or u_rgb < 0 or v_rgb < 0:
                    pass

                else:
                    u_rgb = int(round(u_rgb))
                    v_rgb = int(round(v_rgb))
                    rgb_img_register[v, u, 0:3] = color_img[v_rgb, u_rgb, 0:3]

        return rgb_img_register.astype(np.uint8), depth_img

    def register_thermal_rgb(
        self,
        color_img,
        thermal_img,
        depth_img,
        k_rgb,
        k_thermal,
        k_depth,
        t_thermal_rgb,
        t_depth_rgb,
        depth_scale,
    ):
        depth_img = fill_holes_depth(depth_img)
        threshold = 25
        depth_img_f = fill_depth_colorization(color_img, depth_img)

        height = depth_img.shape[0]
        width = depth_img.shape[1]

        x_y_z_rgb = np.zeros((height, width, 3))

        x_y_z_thermal = np.zeros((height, width, 3))
        thermal_img_register = np.zeros((height, width, 3))

        for v in range(height):
            for u in range(width):
                # 1. Calculate (X,Y,Z) points from depth image at the coordinate (v,u)
                z = depth_img_f[v, u] * depth_scale
                x = ((u - k_depth[0, 2]) * z) / k_depth[0, 0]
                y = ((v - k_depth[1, 2]) * z) / k_depth[1, 1]

                # 2. Transform (X,Y,Z)  points from depth frame to color camera frame
                x_y_z_rgb[v, u, 0:3] = np.dot(t_depth_rgb, np.array([x, y, z, 1])).T[
                    0:3
                ]
                z = x_y_z_rgb[v, u, 2]
                x = ((u - k_rgb[0, 2]) * z) / k_rgb[0, 0]
                y = ((v - k_rgb[1, 2]) * z) / k_rgb[1, 1]

                x_y_z_thermal[v, u, 0:3] = np.dot(
                    t_thermal_rgb, np.array([x, y, z, 1])
                ).T[0:3]

                # 3. Calculate (v,u) points rgb image plane at the coordinate (X,Y,Z) rgb
                z_thermal = x_y_z_thermal[v, u, 2]

                u_thermal = (
                    (x_y_z_thermal[v, u, 0] * k_thermal[0, 0] / z_thermal)
                    + k_thermal[0, 2]
                    + threshold
                )
                v_thermal = (
                    (x_y_z_thermal[v, u, 1] * k_thermal[1, 1] / z_thermal)
                    + k_thermal[1, 2]
                    + threshold
                )

                if (
                    u_thermal > width - 1
                    or v_thermal > height - 1
                    or u_thermal < 0
                    or v_thermal < 0
                ):
                    pass

                else:
                    u_thermal = int(round(u_thermal))
                    v_thermal = int(round(v_thermal))
                    thermal_img_register[v, u, 0:3] = thermal_img[
                        v_thermal, u_thermal, 0:3
                    ]

        depth_img = fill_holes_depth(depth_img)
        color_registerd, depth_img = self.register_images(
            depth_img, color_img, k_depth, k_rgb, t_depth_rgb, depth_scale
        )

        thermal_registered, depth_img = self.register_images(
            depth_img,
            thermal_img_register.astype(np.uint8),
            k_depth,
            k_rgb,
            t_depth_rgb,
            1,
            threshold,
        )

        fig = plt.figure()
        fig.add_subplot(2, 2, 1)
        plt.imshow(color_img)
        fig.add_subplot(2, 2, 2)
        plt.imshow(thermal_img_register.astype(np.uint8))
        fig.add_subplot(2, 2, 3)
        plt.imshow(color_registerd)
        fig.add_subplot(2, 2, 4)
        plt.imshow(thermal_registered)
        plt.show()

        depth_img_open3d = o3d.geometry.Image(depth_img.astype(np.uint16))
        color_img_open3d = o3d.geometry.Image(thermal_registered)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img_open3d,
            depth_img_open3d,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=594.6319,
            fy=594.5186,
            cx=326.8489,
            cy=236.473,
        )

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic
        )
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries(
            [frame, point_cloud],
        )
        return depth_img.astype(np.uint16), thermal_registered


if __name__ == "__main__":
    base = os.path.join(os.getcwd(), "data_set", "banana")
    # read_images(base_path=base)
    depth_name = os.path.join(base, "depth_images")
    rgb_name = os.path.join(base, "rgb_images")
    depth_path = os.path.join(depth_name, "banana_0.png")
    rgb_path = os.path.join(rgb_name, "banana_0.png")
    img_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    img_rgb = cv2.imread(rgb_path)

    lidar_path = os.path.join(os.getcwd(), "examples", "test", "color", "image_L17.jpg")
    depth_path = os.path.join(os.getcwd(), "examples", "test", "depth", "image_L17.png")
    img_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    img_left = cv2.imread(lidar_path)

    k_depth = np.array(
        [[456.43359375, 0.0, 341.421875], [0.0, 455.796875, 251.80859375], [0, 0, 1]]
    )
    k_color = np.array(
        [[594.64318848, 0, 326.84893799], [0, 594.51861572, 236.47299194], [0, 0, 1]]
    )
    extrinsic = np.array(
        [
            [9.99986172e-01, 4.60144877e-03, 2.54564988e-03, -3.25388246e-04],
            [-4.66419989e-03, 9.99671042e-01, 2.52197012e-02, 1.35215586e-02],
            [-2.42876541e-03, -2.52312254e-02, 9.99678671e-01, -6.01826468e-03],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    depth_scale = 0.0002500000118743628
    width = 640
    height = 480
    rgb_img_register = Registration().register_images(
        depth_img=img_depth,
        color_img=img_left,
        k_depth=k_depth,
        k_rgb=k_color,
        t_depth_rgb=extrinsic,
        depth_scale=depth_scale,
    )
    depth_img_open3d = o3d.geometry.Image(img_depth)
    color_img_open3d = o3d.geometry.Image(rgb_img_register)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img_open3d,
        depth_img_open3d,
        convert_rgb_to_intensity=False,
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, fx=594.6319, fy=594.5186, cx=326.8489, cy=236.473
    )

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries(
        [frame, point_cloud],
    )
