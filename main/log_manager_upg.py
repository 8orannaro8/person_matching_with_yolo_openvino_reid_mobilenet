import os
import json
import time
from datetime import datetime
from config import ENTRY_LOG_FILE, SAVE_DIR
from model_eval_crop_onnx import predict_clothes_colors
from color_manager import add_color_entry, load_color_cat, remove_color_entry, initialize_color_cat

# In-memory cache for entry log
_cached_log = None

# ----------------------------------
# Load entry log (cached)
# ----------------------------------
def load_entry_log():
    global _cached_log
    if _cached_log is None:
        if os.path.exists(ENTRY_LOG_FILE):
            try:
                data = open(ENTRY_LOG_FILE, 'r', encoding='utf-8').read().strip()
                _cached_log = json.loads(data) if data else {"entries": [], "mismatches": []}
            except Exception:
                _cached_log = {"entries": [], "mismatches": []}
        else:
            _cached_log = {"entries": [], "mismatches": []}
    # Ensure both keys exist
    _cached_log.setdefault("entries", [])
    _cached_log.setdefault("mismatches", [])
    return _cached_log

# ----------------------------------
# Save entry log to disk and cache
# ----------------------------------
def save_entry_log(log_data):
    global _cached_log
    # Convert any numpy ints to Python ints in-place
    for lst in (log_data.get("entries", []), log_data.get("mismatches", [])):
        for rec in lst:
            # cast id field if present
            if "id" in rec:
                rec["id"] = int(rec["id"])
    _cached_log = log_data
    with open(ENTRY_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

# ----------------------------------
# Initialize entry log
# ----------------------------------
def initialize_entry_log():
    save_entry_log({"entries": [], "mismatches": []})
     # entry_log.json 초기화
    save_entry_log({"entries": [], "mismatches": []})
     # color_cat.json 도 초기화
    initialize_color_cat()

    for fname in os.listdir(SAVE_DIR):
        path = os.path.join(SAVE_DIR, fname)
        if os.path.isfile(path) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            os.remove(path)
   #print("[INIT] Entry records, mismatches, and images initialized")

# ----------------------------------
# Add entry record
# ----------------------------------
def add_entry(track_id, image_path):
    """
    Record a new entry and immediately perform clothes/color analysis,
    storing results in color_cat.json.
    """
    log = load_entry_log()
    log["entries"].append({
        "id": int(track_id),
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path,
        "status": "entered"
    })
    save_entry_log(log)

    # run clothes/color model and store separately
    cloth_pred = predict_clothes_colors(image_path)
    add_color_entry(int(track_id), image_path, cloth_pred)
   # print(f"[ENTRY] ID {int(track_id)} recorded + color analysis done (keys={list(cloth_pred.keys())})")

# ----------------------------------
# Access pending mismatch exits
# ----------------------------------
def get_pending_exits():
    log = load_entry_log()
    mismatches = [e for e in log['mismatches']]
    return mismatches

def all_out():
    log = load_entry_log()
    inside_count = len(log["entries"])
    mismatch_count = len(log["mismatches"])
    if mismatch_count == 0:
        return 0
    else:
        if inside_count - mismatch_count <= 0:
            print("[FLUSH] All inside exited; resetting log and clearing images")
            initialize_entry_log()
            return 1
        else:
            return 0
# ----------------------------------
# Add mismatch exit record
# ----------------------------------
def add_mismatch_exit(track_id, image_path):
    """
    Record a mismatch exit crop, to be shown on dashboard.
    """
    log = load_entry_log()
    log["mismatches"].append({
        "id": int(track_id),
        "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path
    })
    save_entry_log(log)
    print(f"[MISMATCH] ID {int(track_id)} exit recorded")

# ----------------------------------
# Process exit and Re-ID matching
# ----------------------------------
def process_exit(track_id, exit_image_path, reid_model):
    """
    1) Run color analysis on exit crop.
    2) Filter entries by ReID >= 0.95.
    3) If multiple candidates, break ties by color similarity (20% weight).
    4) On match: remove entry, its color record, and both image files.
       On no match: record as mismatch.
    5) If no one remains inside, reset log & clear images.
    """
    # 1) Color analysis on exit
    pred_exit = predict_clothes_colors(exit_image_path)

    log = load_entry_log()
    entries = log["entries"]
    candidates = []

    # 2) ReID filter
    for e in entries:
        sim_reid = reid_model.compare_final(e["image_path"], exit_image_path)["similarity"]
        if sim_reid >= 0.95:
            candidates.append((e, sim_reid))

    if not candidates:
        add_mismatch_exit(track_id, exit_image_path)
        print(f"[NO CANDIDATE] ID {int(track_id)} exit pending (no ReID>=0.95)")
        return False

    # 3) Tie-break by color similarity if needed
    color_db = load_color_cat()
    best_entry, best_score = None, -1.0

    if len(candidates) == 1:
        best_entry, best_score = candidates[0]
    else:
        for e, sim_reid in candidates:
            entry_pred = color_db.get(str(int(e["id"])), {}).get("cloth_pred", {})
            match_count = 0
            total = 0
            for cloth_type, (_, cols) in entry_pred.items():
                exit_cols = pred_exit.get(cloth_type, ("", []))[1]
                if cols and exit_cols:
                    total += 1
                    if set(cols) & set(exit_cols):
                        match_count += 1
            sim_cloth = (match_count / total) if total > 0 else 0.0
            score = sim_reid * 0.95 + sim_cloth * 0.05
            print(f"   ID {int(e['id'])}: reid={sim_reid:.5f}, cloth={sim_cloth:.2f} → score={score:.5f}")
            if score > best_score:
                best_score = score
                best_entry = e

    # 4) Match handling
    matched = False
    if best_entry:
        log["entries"] = [e for e in entries if e != best_entry]
        save_entry_log(log)
        try: os.remove(best_entry["image_path"])
        except: pass
        try: os.remove(exit_image_path)
        except: pass
        remove_color_entry(int(best_entry["id"]))
        print(f"[MATCHED] ID {int(track_id)} → entry ID {int(best_entry['id'])} removed")
        matched = True
    else:
        add_mismatch_exit(track_id, exit_image_path)

    

    return matched
