import os
import time

# Entry/exit log file
ENTRY_LOG_FILE = "entry_log.json"
COLOR_CAT_FILE = "color_cat.json"
# Directory to save cropped images
SAVE_DIR = "captured_crops"

# Warning threshold in seconds
WARNING_SECONDS = 52.0

# Video configurations for presentation
VIDEO_CONFIGS = [
    {"file": "1.mp4", "description": "1. Normal Situation", "line_y": 600, "flip": False},
    {"file": "2.mp4", "description": "2. Person Entered but No Exit", "line_y": 600, "flip": False},
    #{"file": "3.mp4", "description": "3. Overlapping People", "line_y": 600, "flip": False},
    {"file": "4.mp4", "description": "3. Wearing Coat When Exiting", "line_y": 600, "flip": False},
    {"file": "5.mp4", "description": "4. Dark Environment", "line_y": 600, "flip": False},
    #{"file": "6.mp4", "description": "6. Normal Situation", "line_y": 630, "flip": False},
    {"file": "7.mp4", "description": "5. Normal Situation", "line_y": 630, "flip": False}
]

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)
