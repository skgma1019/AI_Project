import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================
# ⚙️ 설정
# =====================================
# [필수] 1단계에서 다운로드한 모델 파일
MODEL_PATH = "face_shape_classifier.pth" 

# [필수] 테스트할 이미지의 '로컬 PC 경로'
# !! 여기를 테스트할 이미지의 경로로 직접 수정해야 합니다 !!
# 예: C:/Users/qjdd1/Desktop/AI_Project/test_imgs/sample_01.jpg
# (참고: Python에서는 경로 구분자로 / 를 사용해도 됩니다)
TEST_IMAGE_PATH = "test_imgs/eunchan1.jpg" # 👈 (예시 경로) 이 부분을 꼭 수정하세요!

# CPU/GPU 설정 (로컬 PC에 NVIDIA GPU가 없으면 'cpu'로 자동 설정됨)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

# =====================================
# ✅ 1. 모델 아키텍처 정의 (ResNet18)
# =====================================
print(f"[1] 모델 아키텍처 로드 중... (ResNet18)")

# 모델을 불러오기 전에, 모델의 '틀(아키텍처)'을 먼저 생성
model = models.resnet18(weights=None) # 가중치는 불러올 것이므로 None

# 학습 때와 똑같이 마지막 레이어를 5개로 수정
NUM_CLASSES = 5 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# =====================================
# ✅ 2. 학습된 가중치(State Dict) 로드
# =====================================
print(f"[2] 학습된 가중치 로드 중... ({MODEL_PATH})")
try:
    # 로컬에 다운로드한 모델 가중치를 '틀'에 덮어씌움
    # map_location=DEVICE: CPU/GPU 환경에 맞게 로드
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"❌ 오류: '{MODEL_PATH}'에서 모델 파일을 찾을 수 없습니다.")
    print("1단계에서 Google Drive의 .pth 파일을 다운로드했는지 확인하세요.")
    exit()

model = model.to(DEVICE)
model.eval() # [중요] 추론(evaluation) 모드로 설정

# =====================================
# ✅ 3. 이미지 전처리 및 로드 (로컬 PC용)
# =====================================
print(f"[3] 테스트 이미지 로드 및 전처리 중... ({TEST_IMAGE_PATH})")

# [중요] 4단계 학습 때와 '반드시' 동일해야 하는 전처리
data_transforms = transforms.Compose([
    transforms.ToPILImage(), # NumPy 배열 -> PIL Image 변환
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# [HOTFIX] 로컬 PC의 한글 경로/파일명도 처리 가능한 로드 방식
try:
    img_array = np.fromfile(TEST_IMAGE_PATH, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV(BGR) -> PyTorch(RGB)
except Exception as e:
    print(f"❌ 오류: '{TEST_IMAGE_PATH}'에서 테스트 이미지를 로드할 수 없습니다.")
    print(f"오류 상세: {e}")
    exit()

# 전처리 적용
# image(NumPy)를 data_transforms의 ToPILImage()가 PIL로 변환
image_tensor = data_transforms(image).unsqueeze(0) # (Batch, Channel, H, W) 형태로 변환
image_tensor = image_tensor.to(DEVICE)

# =====================================
# ✅ 4. 모델 추론 (예측)
# =====================================
print("[4] 모델 추론 실행...")

# 라벨 맵 (학습 때의 숫자 -> 이름 변환)
class_names = ["둥근형", "긴 타원형", "계란형", "역삼각형", "사각형"]

with torch.no_grad(): # [중요] 추론 시에는 기울기 계산 안 함
    outputs = model(image_tensor)
    
    # Softmax를 통해 5개 클래스에 대한 '확률' 계산
    probabilities = F.softmax(outputs, dim=1)[0] # (Batch 0번)
    
    # 가장 확률이 높은 클래스의 인덱스(0~4)와 확률 값
    top_prob, top_idx = torch.max(probabilities, 0)
    
    predicted_label_idx = top_idx.item()
    predicted_label_name = class_names[predicted_label_idx]
    predicted_probability = top_prob.item()

# =====================================
# ✅ 5. 최종 결과 출력
# =====================================
print("\n--- [🎉 모델 예측 결과] ---")
print(f"➡️ 입력 이미지: {TEST_IMAGE_PATH}")
print(f"➡️ 예측 얼굴형: **{predicted_label_name}**")
print(f"➡️ 신뢰도 (확률): **{predicted_probability * 100:.2f} %**")

print("\n--- (참고) 5개 클래스 전체 확률 ---")
for i, name in enumerate(class_names):
    print(f" {name}: {probabilities[i].item() * 100:.2f} %")