import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 충돌 회피

import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# ✅ GPU 설정 및 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval().to(device)
print(f"[INFO] Model loaded on: {device}")

# ✅ 한글 경로 대응 이미지 로딩
def imread_unicode(path):
    with open(path, "rb") as f:
        byte_array = bytearray(f.read())
    np_array = np.asarray(byte_array, dtype=np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

# ✅ 감지 및 크롭
def detect_and_crop(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tensor = F.to_tensor(image_pil).to(device)

    with torch.no_grad():
        predictions = model([image_tensor])[0]

    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if label.item() == 1 and score.item() > 0.8:
            box = box.int().cpu().numpy()
            x1, y1, x2, y2 = box
            h, w, _ = image_np.shape
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            return image_np[y1:y2, x1:x2]
    return None

# ✅ RAM 절약형 폴더 순회
def process_folder_sequential(folder_path):
    idx = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith(".jpg"):
                continue
            path = os.path.join(root, file)
            image = imread_unicode(path)
            idx += 1
            if image is None:
                print(f"[{idx}] [SKIP] Cannot read: {path}")
                continue
            cropped = detect_and_crop(image)
            if cropped is not None:
                cv2.imwrite(path, cropped)
                print(f"[{idx}] [OK] Cropped and overwritten: {path}")
            else:
                print(f"[{idx}] [SKIP] No person detected: {path}")
    print(f"[DONE] Total processed: {idx}")

# ✅ 메인 실행
if __name__ == "__main__":
    folder = r"C:\Users\kkjoo\Downloads\Fashion\Validation\원천데이터"
    process_folder_sequential(folder)
