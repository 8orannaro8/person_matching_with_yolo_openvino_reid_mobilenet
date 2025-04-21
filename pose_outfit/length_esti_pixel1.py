import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LinearRegression

# 회귀 모델 학습 (nose~ankle 픽셀 → 키(cm))
pixel_lengths = np.array([1930, 1521]).reshape(-1, 1)
real_heights = np.array([180, 115])
model = LinearRegression()
model.fit(pixel_lengths, real_heights)

# 거리 계산 함수
def calc_dist(p1, p2):
    if p1 is None or p2 is None:
        return np.nan
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_coords(lm, idx, w, h):
    if lm[idx].visibility < 0.5:
        return None
    return (lm[idx].x * w, lm[idx].y * h)

def avg_coords(lm, i1, i2, w, h):
    p1 = get_coords(lm, i1, w, h)
    p2 = get_coords(lm, i2, w, h)
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

# 이미지 로드
image_path = 'test3.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"이미지 로드 실패: {image_path}")
    exit()

print("이미지 로드 성공")
h, w, _ = image.shape

# Pose 추정
mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print("사람을 인식하지 못했습니다.")
        exit()

    print("사람 인식 성공!")
    lm = results.pose_landmarks.landmark

    # 주요 좌표
    nose = get_coords(lm, 0, w, h)
    eye = get_coords(lm, 2, w, h)
    head_top = (nose[0], nose[1] - abs(nose[1] - eye[1]) * 1.2) if nose and eye else None
    ankle = avg_coords(lm, 27, 28, w, h)

    # 핵심 기준: nose~ankle 거리 측정
    nose_to_ankle_px = calc_dist(nose, ankle)
    print(f"\n👣 nose_to_ankle_px = {nose_to_ankle_px:.2f} px")

    if np.isnan(nose_to_ankle_px):
        print("❗ nose~ankle 거리 계산 실패")
        exit()

    estimated_height_cm = model.predict(np.array([[nose_to_ankle_px]]))[0]
    print(f"📏 추정된 키: {estimated_height_cm:.2f} cm")
    px_to_cm_ratio = estimated_height_cm / nose_to_ankle_px
    print(f"📐 px_to_cm_ratio = {px_to_cm_ratio:.4f} cm/px")

    # 기타 외형 길이 측정 (px 단위 → cm 변환)
    lengths = {}

    def safe_add_length(name, p1, p2):
        dist = calc_dist(p1, p2)
        if not np.isnan(dist):
            lengths[name] = dist

    # 관절 좌표
    shoulder_L = get_coords(lm, 11, w, h)
    shoulder_R = get_coords(lm, 12, w, h)
    hip_L = get_coords(lm, 23, w, h)
    hip_R = get_coords(lm, 24, w, h)
    elbow_L = get_coords(lm, 13, w, h)
    elbow_R = get_coords(lm, 14, w, h)
    wrist_L = get_coords(lm, 15, w, h)
    wrist_R = get_coords(lm, 16, w, h)
    knee_L = get_coords(lm, 25, w, h)
    knee_R = get_coords(lm, 26, w, h)
    ankle_L = get_coords(lm, 27, w, h)
    ankle_R = get_coords(lm, 28, w, h)
    shoulder_avg = avg_coords(lm, 11, 12, w, h)
    hip_avg = avg_coords(lm, 23, 24, w, h)

    # 길이 계산
    safe_add_length("어깨너비", shoulder_L, shoulder_R)
    safe_add_length("골반너비", hip_L, hip_R)
    safe_add_length("왼팔길이", shoulder_L, elbow_L)
    safe_add_length("왼팔길이", elbow_L, wrist_L)
    safe_add_length("오른팔길이", shoulder_R, elbow_R)
    safe_add_length("오른팔길이", elbow_R, wrist_R)
    safe_add_length("왼다리길이", hip_L, knee_L)
    safe_add_length("왼다리길이", knee_L, ankle_L)
    safe_add_length("오른다리길이", hip_R, knee_R)
    safe_add_length("오른다리길이", knee_R, ankle_R)
    safe_add_length("상체길이", head_top, shoulder_avg)
    safe_add_length("상체길이", shoulder_avg, hip_avg)

    print("\n📊 외형 정보 (실제 cm 기준):")
    if not lengths:
        print("❗ lengths 딕셔너리가 비어 있습니다. 일부 관절이 인식되지 않았을 수 있습니다.")
    else:
        # 누적된 길이 이름이 중복되면 합쳐줌
        from collections import defaultdict
        combined = defaultdict(float)
        for name, dist in lengths.items():
            combined[name] += dist

        for k, v in combined.items():
            print(f"{k}: {v * px_to_cm_ratio:.2f} cm")

    # 시각화
    annotated = image.copy()
    for pt in [head_top, nose, ankle]:
        if pt:
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
    if nose and ankle:
        cv2.line(annotated, (int(nose[0]), int(nose[1])), (int(ankle[0]), int(ankle[1])), (0, 0, 255), 2)
    scale_percent = 30  # 50% 크기로 축소
    width = int(annotated.shape[1] * scale_percent / 100)
    height = int(annotated.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(annotated, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Pose Feature Estimation", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

