# length_opti.py
import cv2
import mediapipe as mp
import numpy as np

# 회귀 계수
SLOPE = 0.076771
INTERCEPT = 32.938464

def _calc_dist(p1, p2):
    if p1 is None or p2 is None:
        return np.nan
    return np.linalg.norm(np.array(p1) - np.array(p2))

def _get_coords(lm, idx, w, h):
    if lm[idx].visibility < 0.5:
        return None
    return (lm[idx].x * w, lm[idx].y * h)

def _avg_coords(p1, p2):
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def estimate_height(image_path):
    """이미지 path 를 받아서 MediaPipe로 키(cm) 추정"""
    image = cv2.imread(image_path)
    if image is None:
        return float('nan')
    h, w = image.shape[:2]
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
    res = mp_pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mp_pose.close()
    if not res.pose_landmarks:
        return float('nan')
    lm = res.pose_landmarks.landmark

    # 기준점: 코(0) → 발목(27,28) 중간
    nose    = _get_coords(lm, 0, w, h)
    ankle_L = _get_coords(lm,27, w, h)
    ankle_R = _get_coords(lm,28, w, h)
    ankle   = _avg_coords(ankle_L, ankle_R)

    dist_px = _calc_dist(nose, ankle)
    if np.isnan(dist_px):
        return float('nan')
    # 픽셀→cm 선형 변환
    return SLOPE * dist_px + INTERCEPT
