import os
import cv2
import numpy as np
import tensorflow as tf
import requests

#한글 파일 부름 에러 해결 코드
def imread_unicode(path):
    stream = open(path, "rb")
    bytes_array = bytearray(stream.read())
    numpy_array = np.asarray(bytes_array, dtype=np.uint8)
    img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    return img



# 모델 다운로드
def download_model():
    url = "https://drive.google.com/uc?id=1Ml260620LIKa-OrWqdzv_z99NJKixt3W"
    output_path = "ssd_mobilenetv2_coco/saved_model.pb"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    response = requests.get(url, stream=True)
    with open(output_path, "wb") as output_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                output_file.write(chunk)

# GPU 메모리 제한 설정
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 이미지에서 사람 감지하고 크롭 반환
def detect_and_crop(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0).astype(np.uint8)
    detections = model.signatures["serving_default"](tf.constant(image_expanded))

    detection_classes = detections['detection_classes'][0].numpy()
    detection_boxes = detections['detection_boxes'][0]
    person_indices = tf.where(tf.equal(detection_classes, 1))[:, 0]
    person_coords = tf.gather(detection_boxes, person_indices).numpy()

    for coords in person_coords:
        ymin, xmin, ymax, xmax = coords
        h, w, _ = image.shape
        ymin = int(max((ymin - 0.1) * h, 0))
        ymax = int(min((ymax + 0.1) * h, h))
        xmin = int(max((xmin - 0.1) * w, 0))
        xmax = int(min((xmax + 0.1) * w, w))
        if ymax > ymin and xmax > xmin:
            return image[ymin:ymax, xmin:xmax]  # 첫 번째 사람만 크롭
    return None

# 폴더 순회하며 처리
def process_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".jpg"):
                path = os.path.join(root, file)
                image = imread_unicode(path)
                if image is None:
                    print(f"[SKIP] Cannot read: {path}")
                    continue
                cropped = detect_and_crop(image)
                if cropped is not None:
                    cv2.imwrite(path, cropped)
                    print(f"[OK] Cropped and overwritten: {path}")
                else:
                    print(f"[SKIP] No person detected: {path}")

# 메인 실행
if __name__ == "__main__":
    target_folder = r"C:\Users\kkjoo\Downloads\Fashion\Training"
    if not os.path.exists("ssd_mobilenetv2_coco/saved_model.pb"):
        print("Downloading model...")
        download_model()
    model = tf.saved_model.load("ssd_mobilenetv2_coco")
    print("Model loaded. Processing...")
    process_folder(target_folder)
    print("All done.")
