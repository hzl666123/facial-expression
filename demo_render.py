#!/usr/bin/env python3
"""
Photo 3D Face Renderer — renders the reconstructed 3D face from a photo,
from multiple angles and with simulated expressions. No camera required.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import trimesh
import pyrender
from scipy.spatial import Delaunay

from modules.texture_provider import TextureProvider
from modules.renderer import FaceRenderer


def normalize_landmarks(landmarks_3d):
    pts = landmarks_3d.copy()
    pts[:, 0] = -(pts[:, 0] - 0.5)
    pts[:, 1] = -(pts[:, 1] - 0.5)
    pts[:, 2] = -(pts[:, 2] - pts[:, 2].min())
    scale = 1.0 / (pts[:, 0].max() - pts[:, 0].min())
    pts *= scale
    pts[:, 1] *= 1.0
    center = pts.mean(axis=0)
    pts -= center
    return pts


def render_at_angle(texture, uv, faces, landmarks_3d,
                    rotate_y=0.0, rotate_x=0.0, size=512, bg=(0.2, 0.2, 0.2, 1.0)):
    scene = pyrender.Scene(bg_color=bg)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 5.0, aspectRatio=1.0)
    cam_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1.8],
        [0, 0, 0, 1],
    ])
    scene.add(cam, pose=cam_pose)

    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0)
    scene.add(light, pose=cam_pose)
    light2 = pyrender.DirectionalLight(color=[0.8, 0.8, 1.0], intensity=2.0)
    light2_pose = np.array([
        [1, 0, 0, -0.5],
        [0, 1, 0, 0.3],
        [0, 0, 1, 1.5],
        [0, 0, 0, 1],
    ])
    scene.add(light2, pose=light2_pose)

    vertices = normalize_landmarks(landmarks_3d).astype(np.float64)

    # Apply rotation
    ry = np.array([
        [np.cos(rotate_y), 0, np.sin(rotate_y), 0],
        [0, 1, 0, 0],
        [-np.sin(rotate_y), 0, np.cos(rotate_y), 0],
        [0, 0, 0, 1],
    ])
    rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rotate_x), -np.sin(rotate_x), 0],
        [0, np.sin(rotate_x), np.cos(rotate_x), 0],
        [0, 0, 0, 1],
    ])
    rot = ry @ rx
    v_homo = np.hstack([vertices, np.ones((len(vertices), 1))])
    vertices_rot = (rot @ v_homo.T).T[:, :3].astype(np.float64)

    tri_mesh = trimesh.Trimesh(
        vertices=vertices_rot,
        faces=faces,
        process=False,
    )
    tri_mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv.astype(np.float64),
        image=texture,
    )
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
    scene.add(render_mesh)

    r = pyrender.OffscreenRenderer(size, size)
    color, _ = r.render(scene)
    r.delete()
    return color


def apply_expression(landmarks_3d, blendshape_name, strength=1.0):
    """Simulate expression by shifting landmark positions."""
    pts = landmarks_3d.copy()

    # Define coarse expression morphs (approximate landmark ranges)
    if blendshape_name == "jawOpen":
        # Mouth landmarks (approximate indices for lower lip/jaw)
        for i in range(0, 100):  # lower face region
            if pts[i, 1] > 0.55:  # lower half
                pts[i, 1] += 0.03 * strength
                pts[i, 2] += 0.015 * strength
    elif blendshape_name == "mouthSmile":
        for i in range(0, 100):
            if 0.45 < pts[i, 1] < 0.65:  # mouth region
                spread = (pts[i, 0] - 0.5)
                pts[i, 0] += spread * 0.04 * strength
                if pts[i, 1] > 0.55:
                    pts[i, 1] += 0.01 * strength
    elif blendshape_name == "eyeBlink":
        # Upper eyelids down
        for i in range(100, 250):
            if 0.3 < pts[i, 1] < 0.45:  # eye region
                pts[i, 1] += 0.025 * strength
    elif blendshape_name == "browInnerUp":
        for i in range(100, 250):
            if 0.2 < pts[i, 1] < 0.35:
                pts[i, 1] -= 0.02 * strength
    elif blendshape_name == "mouthPucker":
        for i in range(0, 100):
            if 0.45 < pts[i, 1] < 0.65:
                pts[i, 0] = 0.5 + (pts[i, 0] - 0.5) * 0.7
                pts[i, 1] = 0.55 + (pts[i, 1] - 0.55) * 0.7

    return pts


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Render 3D face from photo")
    parser.add_argument("--photo", "-p", default="assets/test_face.png",
                        help="Path to face photo")
    parser.add_argument("--output", "-o", default="output",
                        help="Output directory for rendered images")
    args = parser.parse_args()

    if not os.path.exists(args.photo):
        print(f"Error: Photo not found: {args.photo}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    print(f"Rendering 3D face from: {args.photo}")
    print()

    # Step 1: Load and process photo
    print("[1/4] Extracting face texture...")
    provider = TextureProvider()
    try:
        photo_data = provider.process(args.photo)
    except ValueError as e:
        print(f"  ERROR: {e}")
        print("  MediaPipe could not detect a face in the photo.")
        print("  Try a clearer front-facing photo.")
        provider.close()
        sys.exit(1)
    provider.close()

    texture = photo_data["texture"]
    uv = photo_data["uv_coords"]
    faces = photo_data["triangulation"]
    print(f"  Texture: {texture.shape[1]}x{texture.shape[0]}")
    print(f"  Landmarks: {len(uv)} points")
    print(f"  Triangles: {len(faces)} faces")

    # We need the photo's 3D landmarks for rendering
    # Use the 2D landmarks as XY and estimate Z from MediaPipe's 3D output
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
    )
    with FaceLandmarker.create_from_options(options) as landmarker:
        img_bgr = cv2.imread(args.photo)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

    if result.face_landmarks and len(result.face_landmarks) > 0:
        lm = result.face_landmarks[0]
        landmarks_3d = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    else:
        # Fallback: use 2D only
        h, w = photo_data["image_size"]
        landmarks_3d = np.zeros((len(photo_data["landmarks_2d"]), 3), dtype=np.float32)
        landmarks_3d[:, 0] = photo_data["landmarks_2d"][:, 0] / w
        landmarks_3d[:, 1] = photo_data["landmarks_2d"][:, 1] / h
    print(f"  3D landmarks: {landmarks_3d.shape}")

    # Step 2: Render from multiple angles
    print("\n[2/4] Rendering multiple angles...")
    angles = [
        ("front", 0.0, 0.0),
        ("left15", -0.26, 0.0),
        ("right15", 0.26, 0.0),
        ("left30", -0.52, 0.0),
        ("right30", 0.52, 0.0),
        ("up10", 0.0, 0.17),
        ("down10", 0.0, -0.17),
    ]

    for name, ry, rx in angles:
        img = render_at_angle(texture, uv, faces, landmarks_3d, ry, rx)
        path = os.path.join(args.output, f"angle_{name}.png")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  Saved angle_{name}.png")

    # Step 3: Render with simulated expressions
    print("\n[3/4] Rendering expressions...")
    renderer = FaceRenderer(width=512, height=512)
    renderer.set_texture(texture, uv, faces, landmarks_3d.copy())

    expressions = [
        ("neutral", None, 0.0),
        ("jaw_open", "jawOpen", 1.0),
        ("smile", "mouthSmile", 1.0),
        ("smile_wide", "mouthSmile", 2.0),
        ("blink", "eyeBlink", 1.0),
        ("brow_raise", "browInnerUp", 1.0),
        ("pucker", "mouthPucker", 1.0),
    ]

    for name, bs_name, strength in expressions:
        if bs_name:
            pts = apply_expression(landmarks_3d, bs_name, strength)
        else:
            pts = landmarks_3d
        img = renderer.render(pts)
        path = os.path.join(args.output, f"expr_{name}.png")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  Saved expr_{name}.png")

    try:
        renderer.delete()
    except Exception:
        pass

    # Step 4: Summary
    count = len(os.listdir(args.output))
    print(f"\n[4/4] Done! {count} images saved to: {args.output}/")
    print()
    for f in sorted(os.listdir(args.output)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
