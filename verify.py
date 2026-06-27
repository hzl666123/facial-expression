#!/usr/bin/env python3
"""Verify all modules. Uses a generated test face or a real photo path."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LD_LIBRARY_PATH", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "libs"
))

import numpy as np
import cv2
from modules.renderer import FaceRenderer
from scipy.spatial import Delaunay


def test_renderer():
    print("[1/3] Testing FaceRenderer...")
    renderer = FaceRenderer(width=256, height=256)

    n_landmarks = 468
    tt = np.linspace(0, 2 * np.pi, n_landmarks)
    landmarks_3d = np.zeros((n_landmarks, 3), dtype=np.float32)
    landmarks_3d[:, 0] = np.sin(tt) * 0.25 + np.sin(tt * 3) * 0.08
    landmarks_3d[:, 1] = np.cos(tt * 1.5) * 0.25
    landmarks_3d[:, 2] = np.cos(tt) * 0.05

    texture = np.full((64, 64, 3), 180, dtype=np.uint8)
    texture[20:44, 20:44] = [120, 80, 60]

    uv = np.full((n_landmarks, 2), 0.5, dtype=np.float32)
    for i in range(n_landmarks):
        uv[i] = [i / n_landmarks, 0.5]

    tri = Delaunay(landmarks_3d[:, :2])
    renderer.set_texture(texture, uv, tri.simplices, landmarks_3d)
    result = renderer.render(landmarks_3d)

    assert result.shape == (256, 256, 3), f"Shape: {result.shape}"
    assert result.dtype == np.uint8, f"Dtype: {result.dtype}"
    print("  OK — render output:", result.shape, result.dtype)

    landmarks_3d_2 = landmarks_3d.copy()
    landmarks_3d_2[:100, 1] += 0.05
    result2 = renderer.render(landmarks_3d_2)
    diff = np.abs(result.astype(float) - result2.astype(float)).mean()
    assert diff > 0, "No change between two renders"
    print("  OK — expression change detected (diff=%.1f)" % diff)

    renderer.delete()
    return True


def test_texture_provider(photo_path=None):
    print("\n[2/3] Testing TextureProvider...")
    from modules.texture_provider import TextureProvider

    if photo_path and os.path.exists(photo_path):
        provider = TextureProvider()
        try:
            data = provider.process(photo_path)
            print(f"  OK — face detected in photo")
            print(f"  Texture: {data['texture'].shape}")
            print(f"  UV coords: {data['uv_coords'].shape}")
            print(f"  Triangles: {data['triangulation'].shape}")
            provider.close()
            return True
        except ValueError as e:
            print(f"  WARNING: {e}")
            provider.close()
            return False
    else:
        print("  No photo path provided — skipping (use --photo PATH)")
        return None


def test_face_tracker():
    print("\n[3/3] Testing FaceTracker...")
    from modules.face_tracker import FaceTracker

    try:
        tracker = FaceTracker()
        print("  OK — FaceTracker initialized")
        tracker.close()
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", "-p", help="Path to a face photo for texture test")
    args = parser.parse_args()

    results = []
    results.append(test_renderer())
    results.append(test_texture_provider(args.photo))
    results.append(test_face_tracker())

    passed = sum(1 for r in results if r is True)
    skipped = sum(1 for r in results if r is None)
    failed = sum(1 for r in results if r is False)

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"\nTo run the full pipeline:")
    print(f"  python run.py --photo /path/to/face_photo.jpg")
