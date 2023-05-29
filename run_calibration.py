import argparse
import logging
from src.calibration import Calibration, get_simple_blon_detector

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="STEREO CALIBRATION FOR LIDAR AND THERMAL--MASTERARBEIT--NICOLAS-RODRIGUEZ"
        )
        parser.add_argument(
            "-p", "--pattern", help="pattern used: chess or circle", default="chess"
        )
        parser.add_argument(
            "-s",
            "--size_rec",
            help="size or area of circle or rectangle",
            default=25,
            type=int,
        )
        args = parser.parse_args()
        logging.basicConfig(level=logging.DEBUG)
        detector = get_simple_blon_detector()
        calibration_obj = Calibration(
            pattern=args.pattern,
            size_board=(5, 4),
            size_rec_mm=args.size_rec,
            detector=detector,
            debug=False,
        )
        calibration_obj.calibrate_thermal_lidar()
    except Exception:
        logging.exception("Exception occured")
    finally:
        logging.debug("Program was succesfull finilized")
