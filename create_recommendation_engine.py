import json
import os
import pandas as pd
from tqdm import tqdm

# =====================================
# ⚙️ 설정
# =====================================
# [입력] 4단계에서 생성된 얼굴형 라벨 파일
INPUT_FILE = "final_training_data.json" 

# [출력] 최종 추천 통계 (우리의 추천 엔진 '뇌')
OUTPUT_FILE = "recommendation_stats.json"

# =====================================
# ✅ 1. Component A (얼굴형) 로드
# =====================================
print(f"[1] Component A (얼굴형 라벨) 로드 중... ({INPUT_FILE})")
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        face_shape_data = json.load(f)
except FileNotFoundError:
    print(f"❌ '{INPUT_FILE}'을 찾을 수 없습니다. 4단계 스크립트를 먼저 실행했는지 확인하세요.")
    exit()

if not face_shape_data:
    print(f"❌ '{INPUT_FILE}'에 데이터가 없습니다.")
    exit()

# Pandas DataFrame으로 변환 (데이터 처리에 매우 용이)
df = pd.DataFrame(face_shape_data)
print(f"   → 총 {len(df)}개의 데이터 로드 완료.")

# =====================================
# ✅ 2. Component B (헤어스타일) 추출
# =====================================
print("[2] Component B (헤어스타일 라벨) 추출 중...")

def get_hairstyle_from_path(path_str):
    """
    이미지 경로에서 헤어스타일 이름을 추출합니다.
    경로 예시: ./원천데이터/0001.hqset/0001.가르마/.../img.jpg
    """
    try:
        # 경로 구분자를 표준 '/'로 통일 (윈도우 '\\' 대비)
        path_str = path_str.replace("\\", "/")
        
        parts = path_str.split('/')
        # parts[0] = .
        # parts[1] = 원천데이터
        # parts[2] = 0001.hqset
        # parts[3] = 0001.가르마 (우리가 필요한 부분)
        
        if len(parts) > 3:
            hairstyle_folder = parts[3] # 예: "0001.가르마"
            
            # "0001." 부분을 제거하고 이름만 반환
            hairstyle_name = hairstyle_folder.split('.', 1)[-1] # "가르마"
            return hairstyle_name
        
        return None
    except Exception:
        return None

# 'hairstyle'이라는 새 열(Column)을 생성
# tqdm.pandas() : Pandas apply 진행률 표시
tqdm.pandas(desc="🏷️ 헤어스타일 라벨 추출 중")
df['hairstyle'] = df['image_path'].progress_apply(get_hairstyle_from_path)

# =====================================
# ✅ 3. 마스터 데이터셋 검증 및 생성
# =====================================
# 혹시라도 헤어스타일 추출에 실패한 데이터(None)가 있다면 제거
original_count = len(df)
df = df.dropna(subset=['face_shape', 'hairstyle'])
new_count = len(df)

print(f"[3] 마스터 데이터셋 생성 완료.")
print(f"   → 유효 데이터: {new_count}개 (제외: {original_count - new_count}개)")

# (선택) 마스터 데이터셋 샘플 출력
print("\n--- [샘플] 마스터 데이터셋 (상위 5개) ---")
print(df.head())
print("--------------------------------------\n")

# =====================================
# ✅ 4. 최종 추천 엔진 (통계) 생성
# =====================================
print("[4] 추천 엔진 통계 생성 중...")

# [핵심] 얼굴형(face_shape)으로 그룹화한 뒤,
# 각 그룹 내의 헤어스타일(hairstyle) 개수를 셉니다.
stats = df.groupby('face_shape')['hairstyle'].value_counts()

# (결과 예시)
# face_shape  hairstyle
# 둥근형         빌드         150
#             보니         120
# 계란형         리프         100 ...

print("   → 통계 집계 완료.")

# =====================================
# ✅ 5. 통계 결과를 JSON 파일로 저장
# =====================================
print(f"[5] 추천 엔진을 '{OUTPUT_FILE}' 파일로 저장 중...")
final_recommendations = {}

# Pandas MultiIndex(stats)를 사용하기 쉬운 JSON 형식으로 변환
for (face_shape, hairstyle), count in stats.items():
    if face_shape not in final_recommendations:
        final_recommendations[face_shape] = []
    
    final_recommendations[face_shape].append({
        "hairstyle": hairstyle,
        "count": int(count) # NumPy int를 Python int로 변환
    })

# (선택) 각 리스트를 count 기준으로 내림차순 정렬 (순위 확인용)
for face_shape in final_recommendations:
    final_recommendations[face_shape] = sorted(
        final_recommendations[face_shape], 
        key=lambda x: x['count'], 
        reverse=True
    )

# 파일로 저장
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_recommendations, f, indent=4, ensure_ascii=False)

print("\n--- [🎉 프로젝트 완료!] ---")
print(f"✅ 최종 추천 엔진('뇌')이 '{OUTPUT_FILE}'에 저장되었습니다.")
print("이 파일을 기반으로 '둥근형' 얼굴에 '빌드' 스타일을 추천하는 앱을 만들 수 있습니다.")