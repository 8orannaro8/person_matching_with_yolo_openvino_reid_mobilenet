import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LinearRegression

# 회귀 모델 학습 (샘플 데이터 기반)
pixel_lengths = np.array([1910, 1840, 1780, 1700]).reshape(-1, 1)
real_heights = np.array([180, 170, 160, 150])
model = LinearRegression()
model.fit(pixel_lengths, real_heights)

# Helper 함수
def calc_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_landmark_coords(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)

# 이미지 및 pose 추정
image_path = 'test2.jpg'
image = cv2.imread(image_path)
h, w, _ = image.shape

mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print("사람을 인식하지 못했습니다.")
        exit()

    lm = results.pose_landmarks.landmark

    # 주요 포인트 계산
    def avg_point(i1, i2): return (
        (lm[i1].x + lm[i2].x) / 2 * w,
        (lm[i1].y + lm[i2].y) / 2 * h
    )

    nose = get_landmark_coords(lm, 0, w, h)
    left_eye = get_landmark_coords(lm, 2, w, h)
    head_top = (nose[0], nose[1] - abs(nose[1] - left_eye[1]) * 1.2)

    ankle = avg_point(27, 28)

    # nose → ankle 거리만 사용 (회귀모델 기반 추정)
    nose_to_ankle_pixel = calc_dist(nose, ankle)

    # 회귀모델 예측
    estimated_height_cm = model.predict(np.array([[nose_to_ankle_pixel]]))[0]

    print(f"추정된 키 (회귀 기반): {estimated_height_cm:.2f} cm")

    # 시각화 (옵션)
    annotated = image.copy()
    points = [head_top, nose, ankle]
    for pt in points:
        cv2.circle(annotated, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
    cv2.line(annotated, (int(nose[0]), int(nose[1])), (int(ankle[0]), int(ankle[1])), (0, 0, 255), 2)

    cv2.imshow("Height Estimation (Regression)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
