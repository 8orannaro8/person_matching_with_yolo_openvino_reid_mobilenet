import cv2
import torch
import numpy as np
from sort import Sort
import os
import sys
import time

# YOLOv5 설치 경로
YOLOV5_PATH = r"D:\ESD\yolov5"
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# YOLOv5 모델 로드
model = torch.hub.load(YOLOV5_PATH, 'yolov5s', source='local')
model.conf = 0.25
model.classes = [0]  # 사람만 감지

# SORT tracker 초기화
tracker = Sort()

# 영상 입력 (0: 웹캠)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠 열기 실패")
    exit()

# 기준선 Y 좌표
line_y = 300
id_prev_cy = {}
last_saved_time = {}

# 저장 디렉토리
save_dir = "captured_crops"
os.makedirs(save_dir, exist_ok=True)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    person_dets = [det[:4] for det in detections if int(det[5]) == 0]
    sort_input = np.array(person_dets)
    tracked = tracker.update(sort_input)

    # 기준선 시각화
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

    for track in tracked:
        x1, y1, x2, y2, track_id = track.astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        prev_cy = id_prev_cy.get(track_id, cy)
        id_prev_cy[track_id] = cy

        crossed = False
        if prev_cy < line_y and cy >= line_y:
            crossed = True
        elif prev_cy > line_y and cy <= line_y:
            crossed = True

        if crossed:
            last_time = last_saved_time.get(track_id, 0)
            if current_time - last_time > 1.0:
                person_crop = frame[y1:y2, x1:x2]
                filename = f"{save_dir}/crop_{int(current_time)}_id{track_id}.jpg"
                cv2.imwrite(filename, person_crop)
                print(f"[INFO] 저장: {filename}")
                last_saved_time[track_id] = current_time

        # 바운딩 박스 시각화
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # FPS 표시
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Person Tracking + Line Crossing', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
