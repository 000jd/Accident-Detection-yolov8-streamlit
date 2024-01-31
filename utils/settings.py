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

SOURCES_LIST = [IMAGE, VIDEO, RTSP]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'test.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'test_detat.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
SNAPSHOTS = (ROOT.parent / 'snapshots').resolve()

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

DATABASE = (ROOT.parent / 'database' / 'data.json').resolve()
