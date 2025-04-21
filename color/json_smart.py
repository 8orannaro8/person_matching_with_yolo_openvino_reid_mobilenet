import os
import json
import pickle
from tqdm import tqdm

# -------------------- 설정 --------------------
DATASET_ROOT = r"D:\fashion\Training"  # 절대 경로
LABEL_ROOT = os.path.join(DATASET_ROOT, "label")
SAVE_PATH = os.path.join(DATASET_ROOT, "preprocessed", "train.pkl")

IMAGE_ROOTS = {
    "data1": ["레트로", "로맨틱", "리조트", "매니시", "모던", "밀리터리", "섹시", "소피스트케이티드"],
    "data2": ["스트리트"],
    "data3": ["스포티", "아방가르드", "오리엔탈", "웨스턴", "젠더리스", "컨트리", "클래식", "키치", "톰보이", "펑크", "페미닌", "프레피", "히피", "힙합"]
}

CLOTHES_TYPES = ["상의", "하의", "아우터", "원피스"]
TEST_LIMIT = 100000  # 예: 10000으로 설정하면 1만개만 처리, 전체 처리 시 None

# -------------------- JSON 처리 --------------------
def process_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # 내부 문자열 JSON 처리
        for k, v in raw.items():
            if isinstance(v, str):
                try:
                    raw[k] = json.loads(v)
                except json.JSONDecodeError:
                    continue

        data = raw
        dataset_info = data.get("데이터셋 정보", {})
        details = dataset_info.get("데이터셋 상세설명", {})
        label_info = details.get("라벨링", {})

        style = label_info.get("스타일", [{}])[0].get("스타일", None)
        if style is None:
            return None

        image_id = str(data["이미지 정보"]["이미지 식별자"])
        filename = f"{image_id}.jpg"

        img_path = None
        for base, style_list in IMAGE_ROOTS.items():
            if style in style_list:
                img_path = os.path.join(DATASET_ROOT, base, style, filename)
                break
        if img_path is None:
            return None

        label = {}
        for cloth in CLOTHES_TYPES:
            item = label_info.get(cloth, [])
            if isinstance(item, list) and item and isinstance(item[0], dict):
                cat = item[0].get("카테고리")
                color = item[0].get("색상")
                if cat or color:
                    label[cloth] = {"category": cat, "color": color}
                else:
                    label[cloth] = None
            else:
                label[cloth] = None

        return {
            "img_filename": filename,
            "style": style,
            "img_path": img_path,
            "label": label
        }

    except Exception as e:
        print(f"[ERROR] {json_path}: {e}")
        return None

# -------------------- 데이터 수집 --------------------
def collect_data():
    dataset = []
    for style_folder in os.listdir(LABEL_ROOT):
        if style_folder == "기타":
            continue  # 기타 스타일 제외
        style_path = os.path.join(LABEL_ROOT, style_folder)
        if not os.path.isdir(style_path):
            continue

        json_files = [f for f in os.listdir(style_path) if f.endswith(".json")]
        for file in tqdm(json_files, desc=f"[{style_folder}] 처리 중"):
            full_path = os.path.join(style_path, file)
            result = process_json(full_path)
            if result:
                dataset.append(result)
                if TEST_LIMIT and len(dataset) >= TEST_LIMIT:
                    print(f"[INFO] TEST_LIMIT {TEST_LIMIT}개 도달")
                    return dataset
    return dataset

# -------------------- 저장 --------------------
def save_data(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[INFO] 저장 완료: {save_path} / 샘플 수: {len(data)}")

# -------------------- 실행 --------------------
if __name__ == "__main__":
    result = collect_data()
    save_data(result, SAVE_PATH)
