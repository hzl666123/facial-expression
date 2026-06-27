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

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay

from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.core.base_options import BaseOptions


MODEL_PATH = "models/face_landmarker.task"


class TextureProvider:
    def __init__(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def process(self, photo_path):
        img_bgr = cv2.imread(photo_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image from {photo_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb,
        )
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            raise ValueError(f"No face detected in photo: {photo_path}")

        landmarks_raw = result.face_landmarks[0]
        landmarks_2d = np.array(
            [[lm.x * w, lm.y * h] for lm in landmarks_raw],
            dtype=np.float32,
        )

        tri = Delaunay(landmarks_2d)

        x_min, y_min = landmarks_2d.min(axis=0)
        x_max, y_max = landmarks_2d.max(axis=0)
        face_size = max(x_max - x_min, y_max - y_min)
        margin = face_size * 0.15

        x_min = max(0, int(x_min - margin))
        y_min = max(0, int(y_min - margin))
        x_max = min(w, int(x_max + margin))
        y_max = min(h, int(y_max + margin))

        face_texture = img_rgb[y_min:y_max, x_min:x_max].copy()
        tex_h, tex_w = face_texture.shape[:2]

        landmarks_cropped = landmarks_2d - np.array([x_min, y_min])

        uv_coords = landmarks_cropped / np.array([tex_w, tex_h], dtype=np.float32)
        uv_coords[:, 1] = 1.0 - uv_coords[:, 1]

        return {
            "texture": face_texture,
            "uv_coords": uv_coords,
            "landmarks_2d": landmarks_2d,
            "landmarks_cropped": landmarks_cropped,
            "triangulation": tri.simplices,
            "face_bbox": (x_min, y_min, x_max, y_max),
            "image_size": (w, h),
            "texture_size": (tex_w, tex_h),
        }

    def close(self):
        self.landmarker.close()
