import sys
import cv2
import signal
import threading
import queue
import time
import numpy as np

from config import VIDEO_CONFIGS, SAVE_DIR
from model_loader_upg import load_yolo_model, load_tracker, load_reid_model
from log_manager_upg import initialize_entry_log, load_entry_log, add_entry, process_exit
from dashboard_upg import draw_dashboard
from color_manager import initialize_color_cat

SKIP_FRAMES = 3
FRAME_QUEUE_SIZE = 4
RESULT_QUEUE_SIZE = 4

# Handle Ctrl+C
def signal_handler(sig, frame):
    print('\n[EXIT] Exiting...')
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class FrameGrabber(threading.Thread):
    def __init__(self, src, frame_q):
        super().__init__(daemon=True)
        self.src = src
        self.frame_q = frame_q
        self.cap = None

    def run(self): 
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open {self.src}")
            self.frame_q.put(None)
            return
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            try:
                self.frame_q.put(frame, timeout=1)
            except queue.Full:
                continue
        self.cap.release()
        self.frame_q.put(None)

class Worker(threading.Thread):
    def __init__(self, frame_q, result_q, cfg):
        super().__init__(daemon=True)
        self.frame_q = frame_q
        self.result_q = result_q
        self.cfg = cfg
        self.model = load_yolo_model()
        self.tracker = load_tracker()
        self.reid = load_reid_model()
        initialize_entry_log()
        self.id_prev_cy = {}
        self.last_saved_time = {}
        self.frame_count = 0
        self.prev_time = time.time()

    def run(self):
        while True:
            frame = self.frame_q.get()
            if frame is None:
                break
            self.frame_count += 1

            # Flip if needed
            if self.cfg['flip']:
                frame = cv2.flip(frame, 0)

            # Prepare extended frame for dashboard
            h, w = frame.shape[:2]
            ext = np.zeros((h, w + 400, 3), dtype=np.uint8)
            ext[:h, :w] = frame

            # FPS 계산 (실제 처리 시간 기반으로 FPS 조정)
            now = time.time()
            fps = 1.0 / (now - self.prev_time+0.00001) if self.prev_time else 0.0
            self.prev_time = now

            # YOLO 처리: 3프레임마다 YOLO 실행
            orig = ext[:h, :w]
            if self.frame_count % SKIP_FRAMES == 0:
                start_detect_time = time.time()
                results = self.model(orig)
                detection_time = time.time() - start_detect_time
                #print(f"Detection Time: {detection_time:.3f} seconds")

                if results:
                    r = results[0]
                    boxes = r.boxes.xyxy.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy().astype(int)
                    persons = [box for box, c in zip(boxes, classes) if c == 0]
                else:
                    persons = []
            else:
                persons = []

            # Tracker update
            tracks = self.tracker.update(np.array(persons))

            # Draw baseline line
            line_y = self.cfg['line_y']
            cv2.line(ext, (0, line_y), (w, line_y), (0, 0, 255), 2)

            # Process entry/exit and draw boxes
            for x1, y1, x2, y2, tid in tracks.astype(int):
                cy = y2
                prev_cy = self.id_prev_cy.get(tid, cy)
                self.id_prev_cy[tid] = cy

                crossed_in = (prev_cy < line_y <= cy)
                crossed_out = (prev_cy > line_y >= cy)

                if crossed_in or crossed_out:
                    last_ts = self.last_saved_time.get(tid, 0)
                    if now - last_ts > 2.0:
                        direction = 'in' if crossed_in else 'out'
                        crop = orig[y1:y2, x1:x2]
                        filename = f"{SAVE_DIR}/crop_{int(now)}_id{tid}_{direction}.jpg"
                        cv2.imwrite(filename, crop)
                        if direction == 'in':
                            add_entry(tid, filename)
                        else:
                            process_exit(tid, filename, self.reid)
                        self.last_saved_time[tid] = now

                # Draw bounding box and ID
                cv2.rectangle(ext, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(ext, f'ID {tid}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Draw dashboard and stats
            frame_with_dashboard = draw_dashboard(ext)
            cv2.putText(frame_with_dashboard, f'FPS: {fps:.2f}', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            current_count = len([e for e in load_entry_log().get('entries', []) if e.get('status') == 'entered']) - len([e for e in load_entry_log().get('mismatches',[])])
            cv2.putText(frame_with_dashboard, f'Current People: {current_count}', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Push result
            try:
                self.result_q.put(frame_with_dashboard, timeout=1)
            except queue.Full:
                continue

        # signal end to renderer
        self.result_q.put(None)

class Renderer:
    def __init__(self, result_q):
        self.result_q = result_q

    def loop(self):
        while True:
            frame = self.result_q.get()
            if frame is None:
                break
            cv2.imshow('Async Person Re-ID', frame)
            time.sleep(0.075)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                sys.exit(0)
        cv2.destroyAllWindows()


def main():
    video_idx = 0
    while True:
        cfg = VIDEO_CONFIGS[video_idx]
        if video_idx== 2:
            time.sleep(10)
        print(f"[INFO] Starting video: {cfg['description']}")

        frame_q  = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        result_q = queue.Queue(maxsize=RESULT_QUEUE_SIZE)

        fg     = FrameGrabber(cfg['file'], frame_q)
        worker = Worker(frame_q, result_q, cfg)
        renderer = Renderer(result_q)

        fg.start()
        worker.start()
        renderer.loop()

        # Wait threads
        fg.join()
        worker.join()

        video_idx = (video_idx + 1) % len(VIDEO_CONFIGS)

if __name__ == '__main__':

    initialize_entry_log()
    initialize_color_cat()
    main()
