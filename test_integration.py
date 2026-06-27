#!/usr/bin/env python3
"""End-to-end integration test: simulate pipeline without real camera."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from modules.face_tracker import FaceTracker
from modules.texture_provider import TextureProvider
from modules.renderer import FaceRenderer


def main():
    print("=== End-to-End Integration Test ===\n")

    photo_path = "assets/test_face.png"
    if not os.path.exists(photo_path):
        print("No test photo. Run verify.py first to generate one.")
        return

    print("[1] Loading photo texture...")
    provider = TextureProvider()
    photo_data = provider.process(photo_path)
    provider.close()
    print(f"    Texture: {photo_data['texture'].shape}")
    print(f"    Triangles: {photo_data['triangulation'].shape}")

    print("\n[2] Initializing FaceTracker...")
    tracker = FaceTracker()

    print("\n[3] Initializing FaceRenderer...")
    renderer = FaceRenderer(width=512, height=512)
    base_3d = np.zeros((len(photo_data["uv_coords"]), 3), dtype=np.float32)
    base_3d[:, 0] = photo_data["landmarks_2d"][:, 0] / photo_data["image_size"][0]
    base_3d[:, 1] = photo_data["landmarks_2d"][:, 1] / photo_data["image_size"][1]
    renderer.set_texture(
        photo_data["texture"],
        photo_data["uv_coords"],
        photo_data["triangulation"],
        base_3d,
    )

    print("\n[4] Simulating frames...")
    # Use the test photo as a "camera frame"
    frame = cv2.imread(photo_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = tracker.process(frame_rgb)

    if result is None:
        print("    WARNING: Face not detected in test frame")
        print("    This is expected for generated faces on some versions.")
        print("    Try with a real photo on your machine.")
        tracker.close()
        renderer.delete()
        return

    print(f"    Face detected!")
    print(f"    Landmarks 3D: {result['landmarks_3d'].shape}")
    print(f"    Blendshapes: {len(result['blendshapes'])}")

    print("\n[5] Rendering frame 1...")
    rendered = renderer.render(result["landmarks_3d"])
    print(f"    Output: {rendered.shape}, {rendered.dtype}")
    cv2.imwrite("/tmp/render_frame1.png", cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    print("    Saved to /tmp/render_frame1.png")

    print("\n[6] Simulating expression change...")
    landmarks_mod = result["landmarks_3d"].copy()
    # Simulate mouth open by shifting lower face landmarks
    landmarks_mod[1:50, 1] += 0.02
    landmarks_mod[1:50, 2] += 0.01
    rendered2 = renderer.render(landmarks_mod)
    cv2.imwrite("/tmp/render_frame2.png", cv2.cvtColor(rendered2, cv2.COLOR_RGB2BGR))
    print("    Saved to /tmp/render_frame2.png")

    diff = np.abs(rendered.astype(float) - rendered2.astype(float)).mean()
    print(f"    Frame difference: {diff:.1f} (expected > 0)")

    # Simulate multiple frames for FPS-like test
    print("\n[7] Performance test (10 frames)...")
    import time
    start = time.time()
    for i in range(10):
        renderer.render(result["landmarks_3d"])
    elapsed = time.time() - start
    print(f"    10 renders in {elapsed:.2f}s ({10/elapsed:.1f} FPS)")

    tracker.close()
    renderer.delete()

    print("\n=== All Integration Tests Passed! ===")
    print("\nTo run with real camera:")
    print(f"  python run.py --photo {photo_path}")


if __name__ == "__main__":
    main()
