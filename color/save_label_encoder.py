import pickle
from sklearn.preprocessing import LabelEncoder

# 설정
PICKLE_PATH = r"D:\fashion\Training\preprocessed\train.pkl"
ENCODER_SAVE_PATH = r"D:\fashion\Training\preprocessed\label_encoders.pkl"
CLOTHES = ['상의', '하의', '아우터', '원피스']

# train.pkl 로드
with open(PICKLE_PATH, 'rb') as f:
    samples = pickle.load(f)

# 라벨 수집
category_encoders = {c: LabelEncoder() for c in CLOTHES}
color_encoders = {c: LabelEncoder() for c in CLOTHES}

cat_labels = {c: [] for c in CLOTHES}
col_labels = {c: [] for c in CLOTHES}

for s in samples:
    for c in CLOTHES:
        label = s['label'].get(c)
        cat_labels[c].append(label['category'] if label else 'None')
        col_labels[c].append(label['color'] if label else 'None')

# 인코더 학습
for c in CLOTHES:
    category_encoders[c].fit(cat_labels[c])
    color_encoders[c].fit(col_labels[c])

# 저장
with open(ENCODER_SAVE_PATH, 'wb') as f:
    pickle.dump((category_encoders, color_encoders), f)

print(f"[INFO] Label encoders saved to {ENCODER_SAVE_PATH}")
