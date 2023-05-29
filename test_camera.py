import argparse
import logging
from src.camaras.camara_lidar import CamaraLidar
from src.camaras.camara_seek import ThermalSeek
from src.config import Config
import os
import cv2

if __name__ == "__main__":
    counter = 0
    try:
        parser = argparse.ArgumentParser(description="A test program.")
        parser.add_argument("-c", "--config", help="config file")
        args = parser.parse_args()
        config = Config(config_path=args.config)
        camera = CamaraLidar(camara_config=config.get_camara_config())
        # camera_thermal = ThermalSeek()
        camera.init()
        base = os.path.join(os.getcwd(), "calib", "color")
        base2 = os.path.join(os.getcwd(), "calib", "depth")
        base3 = os.path.join(os.getcwd(), "calib", "thermal")
        # read_images(base_path=base)
        while True:
            name = "image_L{:d}".format(counter)
            name_r = "image_R{:d}".format(counter)

            depth_name = os.path.join(base2, name + ".png")
            color_name = os.path.join(base, name + ".png")
            thermal_name = name_r + ".png"
            input_s = input()
            if input_s == "c":
                depth, color = camera.get_frames_depth_and_color()
                # camera_thermal.take_and_save_image_calib(
                #     "calib", "thermal", thermal_name
                # )
                cv2.imwrite(filename=depth_name, img=depth)
                cv2.imwrite(filename=color_name, img=color)
                counter += 1

    except Exception:
        logging.exception("Exception occured")
    finally:
        camera.end_pip()
