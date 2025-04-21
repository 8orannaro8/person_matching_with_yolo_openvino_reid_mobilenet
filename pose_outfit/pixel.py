import cv2
import math

# 원본 이미지 불러오기
img = cv2.imread("test2.jpg")

# 화면에 맞게 축소 (예: 가로 1100픽셀 기준으로 조정)
scale_percent = 30  # 50% 크기로 축소
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# 좌표 저장용 (원본 이미지 기준으로 보정)
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭된 좌표를 원본 기준으로 변환
        orig_x = int(x / (scale_percent / 100))
        orig_y = int(y / (scale_percent / 100))
        points.append((orig_x, orig_y))
        print(f"Clicked (원본 좌표): ({orig_x}, {orig_y})")
        if len(points) == 2:
            dist = math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
            print(f"두 점 사이 거리 (원본 기준): {dist:.2f} pixels")

# 이미지 표시
cv2.imshow("Resized Image", resized)
cv2.setMouseCallback("Resized Image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
