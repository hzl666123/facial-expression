import ctypes
import os
import sys

if sys.platform == "linux":
    _LIBS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"
    )
    _lib_path = os.path.join(_LIBS_DIR, "libGLESv2.so.2.1.0")
    if os.path.exists(_lib_path):
        ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)

import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceDetector,
    FaceDetectorOptions,
    RunningMode,
    FaceLandmarksConnections,
)
from mediapipe.tasks.python.core.base_options import BaseOptions


class FaceTracker:
    def __init__(self, model_path="models/face_landmarker.task",
                 max_faces=1, detection_conf=0.5, tracking_conf=0.5):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            output_face_blendshapes=True,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        self._detector_path = model_path
        self._detector = None
        self._detection_conf = detection_conf
        self._frame_counter = 0

    def process(self, frame_rgb):
        h, w = frame_rgb.shape[:2]

        self._frame_counter += 1
        timestamp_ms = int(time.time() * 1000)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None

        landmarks_raw = result.face_landmarks[0]
        landmarks_3d = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks_raw],
            dtype=np.float32,
        )
        landmarks_pixel = np.array(
            [[lm.x * w, lm.y * h] for lm in landmarks_raw],
            dtype=np.float32,
        )

        blendshapes = {}
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            for bs in result.face_blendshapes[0]:
                blendshapes[bs.category_name] = bs.score

        triangulation = [
            (conn.start, conn.end)
            for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
        ]

        return {
            "landmarks_3d": landmarks_3d,
            "landmarks_pixel": landmarks_pixel,
            "blendshapes": blendshapes,
            "triangulation": triangulation,
            "image_size": (w, h),
        }

    def _get_detector(self):
        if self._detector is None:
            det_options = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=self._detector_path),
                running_mode=RunningMode.IMAGE,
                min_detection_confidence=self._detection_conf,
            )
            self._detector = FaceDetector.create_from_options(det_options)
        return self._detector

    def detect_face_roi(self, frame_rgb):
        detector = self._get_detector()
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        result = detector.detect(mp_image)

        if not result.detections:
            return None

        detection = result.detections[0]
        h, w = frame_rgb.shape[:2]
        bbox = detection.bounding_box

        margin_x = int(bbox.width * 0.3)
        margin_y = int(bbox.height * 0.3)
        x = max(0, bbox.origin_x - margin_x)
        y = max(0, bbox.origin_y - margin_y)
        bw = min(w - x, bbox.width + 2 * margin_x)
        bh = min(h - y, bbox.height + 2 * margin_y)

        return (x, y, bw, bh)

    def close(self):
        self.landmarker.close()
        if self._detector is not None:
            self._detector.close()
