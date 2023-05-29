import argparse
import logging

from src.main import MainClass

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="SCANNER FOR 3D OBJECTS --MASTERARBEIT--NICOLAS-RODRIGUEZ"
        )
        parser.add_argument(
            "-c",
            "--config",
            help="path to a json config file",
        )
        parser.add_argument(
            "-n", "--name", help="name of the object", default="no_name"
        )
        parser.add_argument(
            "-m",
            "--modality",
            help="1: debug mode \n 2: automatic scann",
            default=1,
            type=int,
        )
        args = parser.parse_args()
        logging.basicConfig(level=logging.DEBUG)
        main_program = MainClass(
            path=args.config,
            log_level=logging.DEBUG,
            object_name=args.name,
            modality=args.modality,
        )
        main_program.run()
    except Exception:
        logging.exception("Exception occured")
    finally:
        logging.debug("Program was succesfull finilized")
