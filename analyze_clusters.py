import os
import json
import shutil
from tqdm import tqdm

# =====================================
# ⚙️ 설정
# =====================================
# [입력] 2단계에서 생성된 클러스터 라벨 파일
INPUT_FILE = "clustered_labels.json"

# [출력] 클러스터별 이미지를 저장할 폴더
OUTPUT_DIR = "analysis_output" 

# =====================================
# ✅ 1. 데이터 로드
# =====================================
print(f"[1] 클러스터 라벨 로드 중... ({INPUT_FILE})")
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ '{INPUT_FILE}'을 찾을 수 없습니다. 2단계 스크립트를 먼저 실행하세요.")
    exit()

if not data:
    print(f"❌ '{INPUT_FILE}'에 데이터가 없습니다.")
    exit()

print(f"   → 총 {len(data)}개의 라벨 데이터 로드 완료.")

# =====================================
# ✅ 2. 출력 폴더 준비
# =====================================
if os.path.exists(OUTPUT_DIR):
    # 기존 폴더가 있다면 삭제하고 새로 만듭니다.
    print(f"[2] 기존 '{OUTPUT_DIR}' 폴더 제거 후 새로 생성 중...")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# 클러스터 ID 목록 확인 (0, 1, 2, 3, 4)
cluster_ids = sorted(list(set(item['cluster_id'] for item in data)))

for cluster_id in cluster_ids:
    cluster_folder = os.path.join(OUTPUT_DIR, f"cluster_{cluster_id}")
    os.makedirs(cluster_folder)

print(f"   → 클러스터별 폴더 ({len(cluster_ids)}개) 생성 완료.")

# =====================================
# ✅ 3. 이미지 복사 및 정렬
# =====================================
print("[3] 이미지 복사 및 정렬 시작...")
success_count = 0
fail_count = 0

pbar = tqdm(data, desc="🖼️ 이미지 복사 중", unit="img")
for item in pbar:
    src_path = item['image_path']
    cluster_id = item['cluster_id']
    
    # 윈도우 환경에서 경로 구분자 통일
    src_path = src_path.replace("/", "\\") 
    
    # 이미지 파일 이름 추출 (대상 파일 이름으로 사용)
    filename = os.path.basename(src_path)
    
    # 복사 대상 경로 설정
    dest_folder = os.path.join(OUTPUT_DIR, f"cluster_{cluster_id}")
    dest_path = os.path.join(dest_folder, filename)
    
    try:
        shutil.copyfile(src_path, dest_path)
        success_count += 1
    except FileNotFoundError:
        # 1단계에서 이미지 경로가 "./원천데이터/..."로 되어 있어
        # 복사가 안 되는 경우를 대비 (경로가 정확한지 확인 필요)
        # tqdm.write(f"⚠️  파일을 찾을 수 없습니다. 경로 확인: {src_path}")
        fail_count += 1
    except Exception as e:
        # tqdm.write(f"⚠️  복사 실패 ({e}): {src_path}")
        fail_count += 1

print("\n--- [🎉 3단계 분석 준비 완료] ---")
print(f"✅ 정렬된 이미지는 '{OUTPUT_DIR}' 폴더에 있습니다.")
print(f"   → 복사 성공: {success_count} 개")
print(f"   → 복사 실패: {fail_count} 개")
print("\n👉 이제 'analysis_output' 폴더를 열어 각 클러스터 폴더의 이미지를 확인하고, 각 그룹(0~4)에 얼굴형 이름(둥근형, 계란형 등)을 붙여주세요!")