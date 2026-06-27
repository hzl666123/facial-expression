#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Real-time 3D facial expression transfer"
    )
    parser.add_argument(
        "--photo", "-p",
        required=True,
        nargs='+',
        help="Path(s) to face photo(s). First is front view, extras are side views.",
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--cam-width",
        type=int,
        default=640,
        help="Camera width (default: 640)",
    )
    parser.add_argument(
        "--cam-height",
        type=int,
        default=480,
        help="Camera height (default: 480)",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=1200,
        help="Render window width (default: 1200)",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=1200,
        help="Render window height (default: 1200)",
    )
    parser.add_argument(
        "--model", "-m",
        default="models/face_landmarker.task",
        help="Path to MediaPipe face landmarker model",
    )
    parser.add_argument(
        "--subdiv",
        type=int,
        default=2,
        help="Loop subdivision iterations (default: 2, ~7200 faces)",
    )
    parser.add_argument(
        "--backend",
        choices=["mediapipe_subdiv", "flame", "emoca"],
        default="mediapipe_subdiv",
        help="Face reconstruction backend (default: mediapipe_subdiv)",
    )
    parser.add_argument(
        "--flame-path",
        default="/mnt/f/FLAME2020/FLAME2020/generic_model.pkl",
        help="Path to FLAME generic_model.pkl",
    )
    parser.add_argument(
        "--eye-photo",
        default=None,
        help="Path to external eye photo for realistic eyeball texture",
    )

    args = parser.parse_args()

    # Set external eye photo path if provided
    if args.eye_photo:
        import config
        config.EYE_PHOTO_PATH = args.eye_photo

    for p in args.photo:
        if not os.path.exists(p):
            print(f"Error: Photo not found: {p}")
            sys.exit(1)

    photo_paths = args.photo if len(args.photo) > 1 else args.photo[0]

    pipeline = Pipeline(
        photo_path=photo_paths,
        camera_device=args.camera,
        camera_width=args.cam_width,
        camera_height=args.cam_height,
        render_width=args.render_width,
        render_height=args.render_height,
        model_path=args.model,
        subdiv_iterations=args.subdiv,
        backend=args.backend,
        flame_path=args.flame_path,
    )

    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
