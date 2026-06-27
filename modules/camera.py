import cv2
import time
import numpy as np


class Camera:
    def __init__(self, device_id=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self._last_frame_time = time.time()
        self._frame_times = []

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)

        now = time.time()
        self._frame_times.append(now - self._last_frame_time)
        self._last_frame_time = now
        if len(self._frame_times) > 100:
            self._frame_times.pop(0)

        return frame

    @property
    def fps(self):
        if len(self._frame_times) < 2:
            return 0.0
        return 1.0 / (sum(self._frame_times) / len(self._frame_times))

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
