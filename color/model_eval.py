import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# 설정
MODEL_PATH = "fashion_mobilenet.pth"
PICKLE_PATH = r"D:\fashion\Training\preprocessed\train.pkl"
IMAGE_PATH = r"D:\ESD\test.jpg"  # 예시 이미지 경로
IMG_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLOTHES = ['상의', '하의', '아우터', '원피스']

# 라벨 인코더 로딩
with open(PICKLE_PATH, 'rb') as f:
    samples = pickle.load(f)

# 라벨 인코더 학습
category_encoders = {c: LabelEncoder() for c in CLOTHES}
color_encoders = {c: LabelEncoder() for c in CLOTHES}

cat_labels = {c: [] for c in CLOTHES}
col_labels = {c: [] for c in CLOTHES}
for s in samples:
    for c in CLOTHES:
        label = s['label'].get(c)
        cat_labels[c].append(label['category'] if label else 'None')
        col_labels[c].append(label['color'] if label else 'None')
for c in CLOTHES:
    category_encoders[c].fit(cat_labels[c])
    color_encoders[c].fit(col_labels[c])

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

img = Image.open(IMAGE_PATH).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# 모델 정의
class MultiOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = mobilenet_v2(weights='IMAGENET1K_V1').features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleList()
        for c in CLOTHES:
            num_cat = len(category_encoders[c].classes_)
            num_col = len(color_encoders[c].classes_)
            self.heads.append(nn.Linear(1280, num_cat))
            self.heads.append(nn.Linear(1280, num_col))

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return [head(x) for head in self.heads]

# 모델 로드 및 추론
model = MultiOutputModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with torch.no_grad():
    outputs = model(input_tensor)
    print(f"\n[RESULT] {os.path.basename(IMAGE_PATH)} 의 추론 결과:")
for i, c in enumerate(CLOTHES):
    cat_out = outputs[i * 2]
    col_out = outputs[i * 2 + 1]

    # 카테고리
    cat_pred = category_encoders[c].inverse_transform([cat_out.argmax(dim=1).item()])[0]
    # 색상 확률 상위 2개
    col_probs = torch.softmax(col_out, dim=1).cpu().numpy()[0]
    top2_idx = col_probs.argsort()[-2:][::-1]
    top2_labels = color_encoders[c].inverse_transform(top2_idx)
    top2_scores = col_probs[top2_idx]

    print(f"  - {c}:")
    print(f"     ▸ 카테고리: {cat_pred}")
    print(f"     ▸ 색상 TOP2: {top2_labels[0]} ({top2_scores[0]:.2%}), {top2_labels[1]} ({top2_scores[1]:.2%})")