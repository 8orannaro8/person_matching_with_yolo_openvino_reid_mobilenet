import os
import json
import pickle
from tqdm import tqdm
from multiprocessing import Pool

# 경로 설정
DATASET_ROOT = r"D:\fashion\Training"
LABEL_ROOT = os.path.join(DATASET_ROOT, "label")
SAVE_PATH = os.path.join(DATASET_ROOT, "preprocessed", "train.pkl")

IMAGE_ROOTS = {
    "data1": ["레트로", "로맨틱", "리조트", "매니시", "모던", "밀리터리", "섹시", "소피스트케이티드"],
    "data2": ["스트리트"],
    "data3": ["스포티", "아방가르드", "오리엔탈", "웨스턴", "젠더리스", "컨트리", "클래식", "키치", "톰보이", "펑크", "페미닌", "프레피", "히피", "힙합"]
}

CLOTHES_TYPES = ["상의", "하의", "아우터", "원피스"]
TEST_LIMIT = None  # 테스트 시 숫자로 지정, 전체처리시 None
NUM_WORKERS = 10   # 사용할 멀티프로세싱 코어 수

# 이미지 경로 찾기
def find_image_path(image_id, style):
    filename = f"{image_id}.jpg"
    for data_dir, style_list in IMAGE_ROOTS.items():
        if style in style_list:
            full_path = os.path.join(DATASET_ROOT, data_dir, style, filename)
            if os.path.exists(full_path):
                return full_path
    return None

# JSON 처리
def process_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # 내부 JSON 문자열 처리
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
            raise KeyError("Missing style")

        image_id = str(data["이미지 정보"]["이미지 식별자"])
        image_path = find_image_path(image_id, style)
        if image_path is None:
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
            "img_filename": f"{image_id}.jpg",
            "style": style,
            "img_path": image_path,
            "label": label
        }

    except Exception as e:
        print(f"[ERROR] {json_path}: {e}")
        return None

# 멀티프로세싱 수집
def collect_data():
    dataset = []
    files_to_process = []

    for style_folder in os.listdir(LABEL_ROOT):
        if style_folder == "기타":
            continue  # 기타는 제외
        style_path = os.path.join(LABEL_ROOT, style_folder)
        if not os.path.isdir(style_path):
            continue
        json_files = [os.path.join(style_path, f) for f in os.listdir(style_path) if f.endswith(".json")]
        files_to_process.extend(json_files)

    with Pool(NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_json, files_to_process), total=len(files_to_process)):
            if result:
                dataset.append(result)
                if TEST_LIMIT and len(dataset) >= TEST_LIMIT:
                    break

    return dataset

# 저장
def save_data(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[INFO] 저장 완료: {save_path}")

# 메인 실행
if __name__ == "__main__":
    result = collect_data()
    print(f"[INFO] 최종 저장 샘플 수: {len(result)}")
    save_data(result, SAVE_PATH)
