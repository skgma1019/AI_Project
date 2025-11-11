import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import json
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =====================================
# ⚙️ 설정
# =====================================
# [입력] 4단계에서 생성된 최종 학습 데이터셋
DATASET_FILE = "final_training_data.json" 

# [출력] 학습된 모델을 저장할 파일
MODEL_SAVE_PATH = "face_shape_classifier.pth"

# 하이퍼 파라미터
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10 # 테스트를 위해 10회 설정, 실제는 30~50회 필요

# CPU/GPU 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

# =====================================
# ✅ 1. 커스텀 데이터셋 클래스 정의
# =====================================

class FaceDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        # 라벨 인코딩 (문자열 라벨을 0, 1, 2, 3, 4 숫자로 변환)
        self.label_map = {
            "둥근형": 0, "긴 타원형": 1, "계란형": 2, 
            "역삼각형": 3, "사각형": 4
        }
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = item['image_path']
        label_name = item['face_shape']
        label_id = self.label_map[label_name]
        
        # [HOTFIX] 한글 경로 문제 해결된 이미지 로드 방식 사용
        try:
            img_array = np.fromfile(img_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # PyTorch는 RGB 순서 사용
        except:
            # 로드 실패 시 대체 이미지 사용 (매우 드물지만 안전장치)
            # 0으로 채워진 검은색 이미지 반환
            image = np.zeros((224, 224, 3), dtype=np.uint8) 

        # PyTorch의 요구사항: 이미지 크기 224x224로 변환 후 텐서화
        if self.transform:
            # OpenCV 이미지를 PIL Image로 변환할 필요 없이 NumPy 배열을 직접 처리하도록 설정
            image = self.transform(image)
        
        return image, label_id

# =====================================
# ✅ 2. 데이터 준비 및 로더 생성
# =====================================

print(f"[1] 데이터 로드 및 분리 중... ({DATASET_FILE})")
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

# 학습(Train) 80%, 검증(Validation) 20%로 분리
train_data, val_data = train_test_split(
    full_data, test_size=0.2, random_state=42, 
    stratify=[item['face_shape'] for item in full_data] # 라벨 비율 유지
)

# 이미지 전처리 정의
data_transforms = transforms.Compose([
    transforms.ToPILImage(), # NumPy 배열 -> PIL Image 변환 (Transforms 호환)
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = FaceDataset(train_data, transform=data_transforms)
val_dataset = FaceDataset(val_data, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"   → 학습 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")

# =====================================
# ✅ 3. 모델 정의 (ResNet18 전이 학습)
# =====================================

print("[2] ResNet18 모델 로드 및 구조 변경 중...")
# 사전 학습된 ResNet18 모델 로드
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 마지막 Fully Connected 레이어만 우리의 5가지 출력 클래스에 맞게 변경
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(FaceDataset(train_data).label_map))

model = model.to(DEVICE)

# 손실 함수 및 최적화 함수 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =====================================
# ✅ 4. 모델 학습 함수
# =====================================

def train_model():
    best_acc = 0.0
    
    print(f"\n[3] 모델 학습 시작... (Epochs: {NUM_EPOCHS})")
    for epoch in range(NUM_EPOCHS):
        # --- Train Phase ---
        model.train()
        running_loss = 0.0
        
        # tqdm으로 학습 진행률 표시
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)", unit="batch")
        for inputs, labels in pbar_train:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pbar_train.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        running_corrects = 0
        
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Valid)", unit="batch")
        with torch.no_grad():
            for inputs, labels in pbar_val:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / len(val_dataset)
        
        # --- 결과 출력 및 저장 ---
        print(f"\n\nEpoch {epoch+1}/{NUM_EPOCHS} 완료 | Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f}\n")

        # 가장 성능이 좋은 모델 저장
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"*** 모델 저장 완료 (정확도: {best_acc:.4f}) ***")
    
    print("\n--- [🎉 모델 학습 완료] ---")
    print(f"최고 검증 정확도: {best_acc:.4f}")
    print(f"최종 모델은 '{MODEL_SAVE_PATH}'에 저장되었습니다.")

# =====================================
# ✅ 5. 실행
# =====================================
if __name__ == '__main__':
    train_model()