import json
import os
from tqdm import tqdm

# =====================================
# ⚙️ 설정
# =====================================
# [입력] 2단계에서 생성된 클러스터 라벨 파일
INPUT_FILE = "clustered_labels.json"

# [출력] 최종 학습 데이터 (이미지 경로 + 얼굴형 이름)
OUTPUT_FILE = "final_training_data.json"

# 🌟 최종 얼굴형 라벨 맵 (우리가 직접 정의한 정답지) 🌟
FACE_SHAPE_MAP = {
    0: "둥근형",
    1: "긴 타원형",
    2: "계란형",
    3: "역삼각형",
    4: "사각형",
}
# =====================================
# ✅ 1. 데이터 로드
# =====================================
print(f"[1] 클러스터 라벨 로드 중... ({INPUT_FILE})")
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ '{INPUT_FILE}'을 찾을 수 없습니다. 2단계 스크립트를 먼저 실행했는지 확인하세요.")
    exit()

if not data:
    print(f"❌ '{INPUT_FILE}'에 데이터가 없습니다.")
    exit()

print(f"   → 총 {len(data)}개의 라벨 데이터 로드 완료.")

# =====================================
# ✅ 2. 라벨 매핑 및 데이터 구축
# =====================================
print("[2] 클러스터 ID를 얼굴형 이름으로 변환 중...")
final_data = []
error_count = 0

for item in tqdm(data, desc="🏷️ 라벨 변환 중", unit="item"):
    cluster_id = item.get('cluster_id')
    image_path = item.get('image_path')
    
    if cluster_id in FACE_SHAPE_MAP:
        # 클러스터 ID를 정의된 얼굴형 이름으로 변환
        face_shape_name = FACE_SHAPE_MAP[cluster_id]
        
        final_data.append({
            "image_path": image_path,
            "face_shape": face_shape_name
        })
    else:
        # 맵에 없는 ID (발생해서는 안됨)
        error_count += 1

if error_count > 0:
    print(f"⚠️ 경고: {error_count}개의 데이터가 정의되지 않은 클러스터 ID를 가집니다.")
    
print("   → 변환 완료.")

# =====================================
# ✅ 3. 최종 결과 저장
# =====================================
print(f"[3] 최종 학습 데이터셋 저장 중... ({OUTPUT_FILE})")
try:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
except Exception as e:
    print(f"❌ '{OUTPUT_FILE}' 파일 저장 중 오류 발생: {e}")
    exit()

print("\n--- [🎉 4단계 완료: AI 모델 학습 준비] ---")
print(f"✅ 최종 학습 데이터 파일: {OUTPUT_FILE}")
print(f"   → 총 {len(final_data)}개의 이미지에 얼굴형 라벨이 부여되었습니다.")
print("\n👉 이제 이 파일을 사용하여 '얼굴형 분류기' AI 모델을 학습시킬 수 있습니다!")