import pickle
import random

# 저장된 전처리 파일 경로
PKL_PATH = r"D:\fashion\Training\preprocessed\train.pkl"

# 샘플 개수 지정
SAMPLE_NUM = 10

def load_and_sample(path, num_samples=10):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    print(f"[INFO] 전체 샘플 수: {len(data)}")
    samples = random.sample(data, min(len(data), num_samples))

    for i, item in enumerate(samples, 1):
        print(f"\n[SAMPLE {i}]")
        print(f"파일명    : {item['img_filename']}")
        print(f"스타일    : {item['style']}")
        print(f"경로      : {item['img_path']}")
        print("의류 정보 :")
        for part, info in item['label'].items():
            print(f"  - {part}: {info}")

if __name__ == "__main__":
    load_and_sample(PKL_PATH, SAMPLE_NUM)
