import os
import cv2
import mediapipe as mp
import numpy as np
import json
from tqdm import tqdm
import math

# =====================================
# ⚙️ 설정
# =====================================
SOURCE_IMAGE_DIR = "./원천데이터"
OUTPUT_FILE = "face_features.json"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# =====================================
# 헬퍼 함수 (이전과 동일)
# =====================================

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_face_features(landmarks):
    lm = landmarks.landmark
    
    face_height = get_distance(lm[10], lm[152])
    face_width = get_distance(lm[234], lm[454])
    if face_width == 0: return None
    feature_1_aspect_ratio = face_height / face_width

    jaw_width = get_distance(lm[137], lm[366]) 
    if jaw_width == 0: return None
    feature_2_jaw_shape = face_width / jaw_width

    lower_face_height = get_distance(lm[164], lm[152])
    feature_3_lower_face = lower_face_height / face_width

    forehead_width = get_distance(lm[103], lm[332])
    feature_4_forehead_jaw = forehead_width / jaw_width

    return [
        feature_1_aspect_ratio, 
        feature_2_jaw_shape, 
        feature_3_lower_face, 
        feature_4_forehead_jaw
    ]

# =====================================
# ✅ 메인 스크립트 실행
# =====================================

print(f"[1] '{SOURCE_IMAGE_DIR}' 폴더에서 이미지 스캔 중...")
image_paths = []
for root, dirs, files in os.walk(SOURCE_IMAGE_DIR):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file).replace("\\", "/")
            image_paths.append(full_path)

if not image_paths:
    print(f"❌ '{SOURCE_IMAGE_DIR}' 폴더에서 이미지를 찾을 수 없습니다.")
    exit()

print(f"   → 총 {len(image_paths)}개의 이미지 파일 발견.")

results_data = []
success_count = 0
fail_count = 0

print(f"[2] MediaPipe 얼굴 특징 추출 시작... (출력 파일: {OUTPUT_FILE})")
pbar = tqdm(image_paths, desc="🤖 얼굴 특징 추출 중", unit="img")
for img_path in pbar:
    try:
        # ▼▼▼▼▼ [HOTFIX] 한글(Unicode) 경로 문제 해결 ▼▼▼▼▼
        # 1. 파일을 바이너리로 읽음 (numpy가 한글 경로 지원)
        img_array = np.fromfile(img_path, np.uint8)
        # 2. 바이너리를 OpenCV 이미지로 디코딩
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # ▲▲▲▲▲ [HOTFIX] ▲▲▲▲▲

        if image is None:
            # (이제 이 오류는 거의 발생하지 않을 것입니다)
            fail_count += 1
            continue

        # MediaPipe는 BGR이 아닌 RGB 이미지를 사용
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            features = calculate_face_features(landmarks)
            
            if features:
                results_data.append({
                    "image_path": img_path,
                    "features": features
                })
                success_count += 1
            else:
                fail_count += 1
        else:
            # 얼굴 인식 실패 (이미지 자체가 흐릿하거나 얼굴이 없는 경우)
            fail_count += 1
            
    except Exception as e:
        fail_count += 1

face_mesh.close()
print("\n[3] 특징 추출 완료. JSON 파일로 저장 중...")

# =====================================
# ✅ 3. 결과 저장 (이전과 동일)
# =====================================
try:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
except Exception as e:
    print(f"❌ '{OUTPUT_FILE}' 파일 저장 중 오류 발생: {e}")
    exit()

print("\n--- [🎉 1단계 완료] ---")
print(f"✅ 특징 데이터 파일: {OUTPUT_FILE}")
print(f"   → 성공 (특징 추출): {success_count} 개")
print(f"   → 실패 (미검출/오류): {fail_count} 개")
print("\n👉 이제 '2단계: K-Means 클러스터링'을 진행할 차례입니다.")