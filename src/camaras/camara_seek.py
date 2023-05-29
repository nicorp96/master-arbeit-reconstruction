import cv2
import numpy as np
import os
import logging
import thermal_seek
from thermal_seek import *
import copy


class ThermalSeek(PyThermalPro):
    WIDTH = 320
    HEIGHT = 240

    def __init__(
        self,
        warmup_frame=30,
        warmup_wait=10,
        smooth_fr=100,
        smooth_wait=10,
        level=logging.DEBUG,
    ) -> None:
        super().__init__("")
        self._logger = logging.getLogger(__name__)
        self._warmup_frame = warmup_frame
        self._warmup_wait = warmup_wait
        self._smooth_fr = smooth_fr
        self._smooth_wait = smooth_wait

        self._logger.setLevel(level)

    def open_seek_init(self):
        if not (self.open()):
            self._logger.error("failed to open seek cam")

    def warmup(self, frame):
        for i in range(30):
            if not (self.grab()):
                self._logger.error("no more LWIR img")
            else:
                self.retrieve(frame)
                cv2.waitKey(10)
        self._logger.debug("warmup complete")

    def smoothing(self, frame):
        avg_frame = np.zeros((240, 320), dtype=np.float32)
        for i in range(self._smooth_fr):
            if not (self.grab()):
                self._logger.error("no more LWIR img")
            else:
                self.retrieve(frame)
                frame = np.float32(frame)
                avg_frame += frame
                cv2.waitKey(self._smooth_wait)
        avg_frame = avg_frame / self._smooth_fr
        return avg_frame

    def take_and_save_image(self, folder, folder_2, object, file_name):
        if self.open():
            frame = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint16)
            self.warmup(frame)
            avg_frame = self.smoothing(frame)
            frame = np.uint16(avg_frame)
            frame = cv2.normalize(
                frame, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX
            )
            frame = frame / 255
            frame = np.uint8(frame)
            frame = cv2.bitwise_not(frame)
            path = os.path.join(
                os.getcwd(),
                folder,
                object,
                folder_2,
                file_name,
            )
            cv2.imwrite(
                path,
                frame,
            )
            self.close()
            return True
        self._logger.error("failed to open seek cam")
        return False

    def take_and_save_image_calib(self, folder, folder_2, file_name):
        if self.open():
            frame = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint16)
            self.warmup(frame)
            avg_frame = self.smoothing(frame)
            frame = np.uint16(avg_frame)
            frame = cv2.normalize(
                frame, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX
            )
            frame = frame / 255
            frame = np.uint8(frame)
            frame = cv2.bitwise_not(frame)
            path = os.path.join(
                os.getcwd(),
                folder,
                folder_2,
                file_name,
            )
            cv2.imwrite(
                path,
                frame,
            )
            self.close()
            return True
        self._logger.error("failed to open seek cam")
        return False


if __name__ == "__main__":
    seek = ThermalSeek()

    seek.take_and_save_image()
