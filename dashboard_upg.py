import cv2
import numpy as np
import time
from datetime import datetime
from config import WARNING_SECONDS, SAVE_DIR
from log_manager_upg import load_entry_log, get_pending_exits, all_out
from emergency_manager import try_emergency_call

# 캐시: static overlay 한 번만 생성
_overlay_cache = {}
def _get_overlay(frame_width, frame_height):
    key = (frame_width, frame_height)
    if key not in _overlay_cache:
        ov = np.zeros((frame_height, 400, 3), dtype=np.uint8)
        _overlay_cache[key] = ov
    return _overlay_cache[key].copy()

def resize_img(path, w=120, h=160):
    img = cv2.imread(path)
    return cv2.resize(img, (w, h)) if img is not None else None

def draw_dashboard(frame):
    now = datetime.now()
    fh, fw = frame.shape[:2]
    panel_w = 400
    dx = fw - panel_w

    # 1) static overlay
    overlay = _get_overlay(fw, fh)
    panel = frame[:, dx:dx+panel_w]
    cv2.addWeighted(panel, 0.3, overlay, 0.7, 0, panel)
    frame[:, dx:dx+panel_w] = panel

    # 영역 비율 설정
    entry_region_h = int(fh * 0.7)
    mismatch_region_h = fh - entry_region_h - 40  # 제목/라인 여유 포함

    # 2) Entry 영역 타이틀
    cv2.putText(frame, 'Current People Inside', (dx+10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.line(frame, (dx+10, 40), (fw-10, 40), (255,255,255), 1)

    # Entry 썸네일 그리기
    entries = [e for e in load_entry_log().get('entries', []) if e['status']=='entered']
    cols, margin = 3, 10
    thumb_w, thumb_h = 120, 160
    start_y = 60
    max_rows = max(1, entry_region_h // (thumb_h + margin + 50))
    for i, e in enumerate(entries):
        row, col = divmod(i, cols)
        if row >= max_rows: 
            break
        x = dx + margin + col*(thumb_w + margin)
        y = start_y + row*(thumb_h + margin + 50)
        # thumbnail
        thumb = resize_img(e['image_path'])
        if thumb is not None:
            frame[y:y+thumb_h, x:x+thumb_w] = thumb
        # elapsed
        et = datetime.strptime(e['entry_time'], '%Y-%m-%d %H:%M:%S')
        elapsed = (now - et).total_seconds()
        warn = elapsed >= WARNING_SECONDS
        if warn:
            try_emergency_call(e['id'])
        color = (0,0,255) if warn else (0,255,0)
        # border & text
        cv2.rectangle(frame, (x-2,y-2), (x+thumb_w+2,y+thumb_h+2), color, 2)
        cv2.putText(frame, f"ID:{e['id']}", (x, y+thumb_h+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        ttxt = f"{int(elapsed)}s" if elapsed<60 else f"{int(elapsed//60)}m{int(elapsed%60)}s"
        cv2.putText(frame, ttxt, (x, y+thumb_h+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 3) Mismatch 영역 타이틀
    y_title = entry_region_h + 20
    cv2.putText(frame, 'Mismatch Exits', (dx+10, y_title),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
    cv2.line(frame, (dx+10, y_title+10), (fw-10, y_title+10), (0,165,255), 1)

    # Mismatch 썸네일: 최대 한 줄
    mismatches = [e for e in load_entry_log().get('mismatches', [])]
    y0 = y_title + 30
    for i, e in enumerate(mismatches):
        col = i % cols
        x = dx + margin + col*(thumb_w + margin)
        y = y0
        thumb1 = resize_img(e['image_path'])
        if thumb1 is not None:
            frame[y:y+thumb_h, x:x+thumb_w] = thumb1
        cv2.rectangle(frame, (x-2,y-2), (x+thumb_w+2,y+thumb_h+2), (0,165,255), 2)
    all_out()
    
    return frame
