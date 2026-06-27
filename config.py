import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_DEVICE_ID = 0

# MediaPipe settings
MEDIAPIPE_MAX_FACES = 1
MEDIAPIPE_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5
MEDIAPIPE_REFINE_LANDMARKS = True  # enables iris + blendshapes

# 3D rendering settings
RENDER_WIDTH = 512
RENDER_HEIGHT = 512
RENDER_BG_COLOR = (0.9, 0.9, 0.9, 1.0)  # light gray

# Display
DISPLAY_CAMERA = True  # show camera feed alongside 3D face

# Photo
UPLOAD_PHOTO_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_PHOTO_DIR, exist_ok=True)

# External eye photo for realistic eyeball texture
EYE_PHOTO_PATH = None  # set via --eye-photo CLI arg
