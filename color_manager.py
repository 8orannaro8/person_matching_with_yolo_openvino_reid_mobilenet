import os
import json
from config import COLOR_CAT_FILE

def load_color_cat():
    """Load the JSON { id: { image_path, cloth_pred } }"""
    if not os.path.exists(COLOR_CAT_FILE):
        return {}
    with open(COLOR_CAT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_color_cat(data):
    with open(COLOR_CAT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def initialize_color_cat():
    """Clear the color_cat.json on startup"""
    save_color_cat({})
    print("[INIT] color_cat.json cleared")

def add_color_entry(track_id, image_path, cloth_pred):
    """Store the predicted clothes/colors for an entry crop."""
    data = load_color_cat()
    data[str(track_id)] = {
        "image_path": image_path,
        "cloth_pred": cloth_pred
    }
    save_color_cat(data)

def remove_color_entry(track_id):
    """Remove one ID’s record from color_cat.json."""
    data = load_color_cat()
    key = str(track_id)
    if key in data:
        del data[key]
        save_color_cat(data)
