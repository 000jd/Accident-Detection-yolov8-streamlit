from pathlib import Path
import sys

# For Getting the absolute path of the current file
file_path = Path(__file__).resolve()

# For Getting the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# For Getting the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, RTSP, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'test.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'test_detat.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
#VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4'

#VIDEOS_DICT = {
#    'video_1': VIDEO_1_PATH,
#}

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

DATABASE = ROOT / 'data.json'

# Webcam
#WEBCAM_PATH = 0