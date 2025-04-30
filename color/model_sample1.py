import os
import pickle
import random
from collections import defaultdict
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from tqdm import tqdm

# -------------------- 설정 --------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 64
NUM_EPOCHS = 5
IMG_SIZE = 224
TOTAL_SAMPLE_SIZE = 50000
DATA_PATH = r"D:\fashion\Training\preprocessed\train.pkl"
ENCODER_PATH = r"D:\fashion\Training\preprocessed\label_encoders.pkl"
CLOTHES = ['상의', '하의', '아우터', '원피스']

# -------------------- 인코더 로딩 --------------------
with open(ENCODER_PATH, 'rb') as f:
    category_encoders, color_encoders = pickle.load(f)

# -------------------- 샘플링 --------------------
def sample_data(data_path, sample_size):
    with open(data_path, 'rb') as f:
        full_data = pickle.load(f)

    style_groups = defaultdict(list)
    for sample in full_data:
        style = sample.get('style')
        if style:
            style_groups[style].append(sample)

    styles = list(style_groups.keys())
    per_style = sample_size // len(styles)
    sampled = []
    for style in styles:
        group = style_groups[style]
        if len(group) >= per_style:
            sampled.extend(random.sample(group, per_style))
        else:
            sampled.extend(group)

    random.shuffle(sampled)
    return sampled

samples = sample_data(DATA_PATH, TOTAL_SAMPLE_SIZE)

# -------------------- Dataset 정의 --------------------
class FashionDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            img = Image.open(sample['img_path']).convert('RGB')
        except Exception as e:
            print(f"[LOAD ERROR] {sample['img_path']}: {e}")
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        labels = []
        for c in CLOTHES:
            label = sample['label'].get(c)
            cat = label['category'] if label else 'None'
            col = label['color'] if label else 'None'
            labels.append(category_encoders[c].transform([cat])[0])
            labels.append(color_encoders[c].transform([col])[0])

        return img, torch.tensor(labels, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset = FashionDataset(samples, transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

EVAL_SAMPLE_SIZE = 1000
eval_samples = samples[:EVAL_SAMPLE_SIZE]
eval_dataset = FashionDataset(eval_samples, transform)
eval_loader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

# -------------------- 모델 정의 --------------------
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

model = MultiOutputModel().to(DEVICE)

# -------------------- 학습 --------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def evaluate(model, loader):
    model.eval()
    correct = [0] * 8
    total = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            outputs = model(imgs)
            for i, out in enumerate(outputs):
                pred = out.argmax(dim=1)
                correct[i] += (pred == targets[:, i]).sum().item()
            total += targets.size(0)

    acc = [round(c / total, 4) for c in correct]
    for i, c in enumerate(CLOTHES):
        print(f"  ▸ {c} 카테고리 정확도: {acc[i*2]:.2%} / 색상 정확도: {acc[i*2+1]:.2%}")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for imgs, targets in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        outputs = model(imgs)
        loss = 0
        for i, out in enumerate(outputs):
            loss += criterion(out, targets[:, i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
    evaluate(model, eval_loader)

# -------------------- 저장 --------------------
MODEL_SAVE_PATH = "fashion_mobilenet.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"[INFO] 모델 저장 완료: {MODEL_SAVE_PATH}")
