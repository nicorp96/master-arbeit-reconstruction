import argparse
import logging
from src.camaras.camara_lidar import CamaraLidar
from src.config import Config
from matplotlib import pyplot as plt
from src.utils.register_frames import Registration

if __name__ == "__main__":
    counter = 0
    try:
        parser = argparse.ArgumentParser(description="A test program.")
        parser.add_argument("-c", "--config", help="config file")
        args = parser.parse_args()
        config = Config(config_path=args.config)
        camera = CamaraLidar(camara_config=config.get_camara_config())
        camera.init()
        while True:
            input_s = input()
            if input_s == "c":
                depth, color = camera.get_frames_depth_and_color()
                aligned_image= Registration().register_depth_rgb(depth,color,camera.depth_intrinsic.intrinsic_matrix,camera.rgb_intrinsic.intrinsic_matrix,camera.extrinsic_depth_rgb, camera._depth_scale)
                plt.imshow(aligned_image)
                plt.show()
                counter += 1

    except Exception:
        logging.exception("Exception occured")
    finally:
        camera.end_pip()
