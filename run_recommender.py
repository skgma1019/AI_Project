import torch
import torch.nn as nn
from torchvision import transforms, models
import json
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
# [입력 1] 모델 1의 '뇌' (얼굴형 분류기)
MODEL_PATH = "face_shape_classifier.pth" 

# [입력 2] 모델 2의 '뇌' (추천 통계)
STATS_FILE = "recommendation_stats.json"

# [입력 3] 사용자가 분석할 이미지
# !! 여기를 테스트할 이미지의 로컬 경로로 직접 수정하세요 !!
TEST_IMAGE_PATH = "test_imgs/gam.webp" # 👈 (예시 경로) 이 부분을 꼭 수정하세요!

# [설정] 상위 몇 개까지 추천할지
TOP_K_RECOMMENDATIONS = 3

# CPU/GPU 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

# =====================================
# ✅ 1단계: 모델 1 (얼굴형 분류기) 로드
# =====================================
print(f"[1] 로딩: 얼굴형 분류기 ({MODEL_PATH})")
class_names = ["둥근형", "긴 타원형", "계란형", "역삼각형", "사각형"]
NUM_CLASSES = len(class_names)

# 모델 아키텍처 (틀) 생성
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# 학습된 가중치 로드
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"❌ 오류: '{MODEL_PATH}' 파일을 찾을 수 없습니다. (코랩에서 다운로드 필요)")
    exit()

model = model.to(DEVICE)
model.eval() # 추론 모드로 설정

# =====================================
# ✅ 2단계: 모델 2 (추천 엔진) 로드
# =====================================
print(f"[2] 로딩: 추천 엔진 ({STATS_FILE})")
try:
    with open(STATS_FILE, 'r', encoding='utf-8') as f:
        stats_data = json.load(f)
except FileNotFoundError:
    print(f"❌ 오류: '{STATS_FILE}' 파일을 찾을 수 없습니다. (6단계 스크립트 실행 필요)")
    exit()

# =====================================
# ✅ 3단계: 입력 이미지 처리
# =====================================
print(f"[3] 이미지 처리 중... ({TEST_IMAGE_PATH})")

# 학습 때와 동일한 전처리
data_transforms = transforms.Compose([
    transforms.ToPILImage(), # NumPy -> PIL
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 한글 경로 이미지 로드 (로컬 PC용)
try:
    img_array = np.fromfile(TEST_IMAGE_PATH, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
except Exception as e:
    print(f"❌ 오류: '{TEST_IMAGE_PATH}'에서 이미지를 로드할 수 없습니다.")
    exit()

# 이미지 텐서화
image_tensor = data_transforms(image).unsqueeze(0).to(DEVICE)

# =====================================
# ✅ 4단계: [실행] 얼굴형 예측 (모델 1)
# =====================================
print("[4] 1단계: 얼굴형 분석 실행...")
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = F.softmax(outputs, dim=1)[0]
    top_prob, top_idx = torch.max(probabilities, 0)
    
    predicted_face_shape = class_names[top_idx.item()]
    confidence = top_prob.item()

print(f" ➡️ 분석 결과: {predicted_face_shape} (신뢰도: {confidence*100:.2f}%)")

# =====================================
# ✅ 5단계: [실행] 헤어스타일 추천 (모델 2)
# =====================================
print("[5] 2단계: 헤어스타일 추천 실행...")
if predicted_face_shape in stats_data:
    # 예측된 얼굴형에 맞는 헤어스타일 리스트 가져오기
    recommendations = stats_data[predicted_face_shape]
    
    # 설정한 TOP_K 만큼 상위 N개 추출
    top_k_list = recommendations[:TOP_K_RECOMMENDATIONS]
    
    # --- 최종 결과 출력 ---
    print("\n" + "="*40)
    print("      🎉 K-hairstyle AI 추천 시스템 🎉")
    print("="*40)
    print(f"\n[분석 결과]")
    print(f" ➡️ 고객님의 얼굴형은 **'{predicted_face_shape}'**에 가깝습니다.")
    print(f"    (신뢰도: {confidence*100:.2f}%)")
    
    print("\n[추천 스타일]")
    print(f" ➡️ '{predicted_face_shape}' 얼굴형을 가진 분들이")
    print(f"     가장 많이 선택한 TOP {TOP_K_RECOMMENDATIONS} 헤어스타일입니다.")
    print("-"*40)
    
    for i, item in enumerate(top_k_list, 1):
        print(f"   {i}순위. {item['hairstyle']} (선호도: {item['count']}건)")
    
    print("="*40)

else:
    print(f"❌ 오류: 추천 엔진에 '{predicted_face_shape}' 얼굴형 데이터가 없습니다.")