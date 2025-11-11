import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # 👈 [1.5단계] 정규화
from tqdm import tqdm
import warnings

# =====================================
# ⚙️ 설정
# =====================================
# [입력] 1단계에서 생성된 특징 파일
INPUT_FILE = "face_features.json"

# [출력] 클러스터링(그룹) 결과가 저장될 파일
OUTPUT_FILE = "clustered_labels.json"

# [설정] 얼굴형을 몇 개 그룹으로 나눌 것인가
NUM_CLUSTERS = 5 

# Scikit-learn의 경고 메시지 끄기 (n_init 관련)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# =====================================
# ✅ 1. 데이터 로드
# =====================================
print(f"[1] 특징 데이터 로드 중... ({INPUT_FILE})")
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ '{INPUT_FILE}'을 찾을 수 없습니다. 1단계 스크립트를 먼저 실행하세요.")
    exit()

if not data:
    print(f"❌ '{INPUT_FILE}'에 데이터가 없습니다. 1단계가 성공했는지 확인하세요.")
    exit()

print(f"   → 총 {len(data)}개의 특징 데이터 로드 완료.")

# K-Means가 학습할 '특징' 리스트와, 나중에 매핑할 '이미지 경로' 리스트 분리
features_list = []
image_data_list = []

for item in data:
    features_list.append(item['features'])
    image_data_list.append({
        "image_path": item['image_path']
    })

# Numpy 배열로 변환
X = np.array(features_list)

# =====================================
# ✅ 1.5단계: 데이터 정규화 (Standard Scaling)
# =====================================
print("[1.5] 데이터 정규화 진행 중... (StandardScaler)")
# K-Means는 거리 기반이므로, 모든 특징의 단위를 통일(정규화)해야 함
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   → 정규화 완료.")

# =====================================
# ✅ 2. K-Means 클러스터링 실행
# =====================================
print(f"[2] K-Means 클러스터링 실행 중... (N={NUM_CLUSTERS}개 그룹)")

kmeans = KMeans(
    n_clusters=NUM_CLUSTERS, 
    random_state=42, # 결과를 일정하게 유지하기 위한 값
    n_init=10         # 안정적인 중심점을 찾기 위해 10번 시도
)

# 정규화된 데이터로 학습
kmeans.fit(X_scaled)

# 각 데이터(이미지)가 몇 번 그룹에 속하는지 라벨을 가져옴
labels = kmeans.labels_ # 예: [0, 2, 4, 1, 0, 0, 3, ...]

print("   → 클러스터링 완료.")

# =====================================
# ✅ 3. 결과 취합 및 저장
# =====================================
print(f"[3] 최종 라벨 파일 저장 중... ({OUTPUT_FILE})")
output_data = []

for i in range(len(image_data_list)):
    item = image_data_list[i]
    item['cluster_id'] = int(labels[i]) # NumPy int를 Python int로 변환
    output_data.append(item)

try:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
except Exception as e:
    print(f"❌ '{OUTPUT_FILE}' 파일 저장 중 오류 발생: {e}")
    exit()

print("\n--- [🎉 2단계 완료] ---")
print(f"✅ 클러스터 라벨 파일: {OUTPUT_FILE}")
print(f"   → 총 {len(output_data)}개의 이미지를 {NUM_CLUSTERS}개 그룹으로 분류 완료.")
print("\n👉 이제 '3단계: 결과 분석'을 통해 각 그룹(0~4)이 어떤 얼굴형인지 확인해야 합니다.")