import cv2
import mediapipe as mp

# 이미지 로드
image_path = 'test2.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"이미지 로드 실패: {image_path}")
    exit()

print("이미지 로드 성공")
h, w, _ = image.shape

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print("사람을 인식하지 못했습니다.")
        exit()

    print("사람 인식 성공!")

    # 관절에 점과 선 그리기
    annotated = image.copy()
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # 점 스타일
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)                    # 선 스타일
    )

    # 이미지 축소해서 보기 좋게
    scale_percent = 30
    width = int(annotated.shape[1] * scale_percent / 100)
    height = int(annotated.shape[0] * scale_percent / 100)
    resized = cv2.resize(annotated, (width, height), interpolation=cv2.INTER_AREA)

    # 결과 출력
    cv2.imshow("Pose Estimation", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
