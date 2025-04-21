import os
import json
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# -------------------- 설정 --------------------
DATASET_ROOT = r"D:\fashion\Training"
LABEL_ROOT = os.path.join(DATASET_ROOT, "label")
SAVE_PATH = os.path.join(DATASET_ROOT, "preprocessed", "train.pkl")

IMAGE_ROOTS = {
    "data1": ["레트로", "로맨틱", "리조트", "매니시", "모던", "밀리터리", "섹시", "소피스트케이티드"],
    "data2": ["스트리트"],
    "data3": ["스포티", "아방가르드", "오리엔탈", "웨스턴", "젠더리스", "컨트리", "클래식", "키치", "톰보이", "펑크", "페미닌", "프레피", "히피", "힙합"]
}

CLOTHES_TYPES = ["상의", "하의", "아우터", "원피스"]
TEST_LIMIT = None
NUM_WORKERS = min(10, cpu_count())

# -------------------- JSON 처리 --------------------
def process_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

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
    files = []

    for style_folder in os.listdir(LABEL_ROOT):
        if style_folder == "기타":
            continue
        style_path = os.path.join(LABEL_ROOT, style_folder)
        if not os.path.isdir(style_path):
            continue
        json_files = [os.path.join(style_path, f) for f in os.listdir(style_path) if f.endswith(".json")]
        files.extend(json_files)

    try:
        with Pool(processes=NUM_WORKERS) as pool:
            for result in tqdm(pool.imap_unordered(process_json, files), total=len(files), desc="[전체 처리 진행 중]"):
                if result:
                    dataset.append(result)
                    if TEST_LIMIT and len(dataset) >= TEST_LIMIT:
                        print(f"[INFO] TEST_LIMIT {TEST_LIMIT}개 도달")
                        break
    except KeyboardInterrupt:
        print("\n[중단됨] 사용자에 의해 Ctrl+C 입력으로 프로세스가 중단되었습니다.")
        os._exit(1)  # 강제종료: 멀티프로세싱 풀도 완전 종료

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
