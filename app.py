import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import json
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
import warnings

# ★★★ 'url_for'가 'redirect' 뒤에 있었는데, 명시적으로 앞으로 가져왔습니다. ★★★
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore", category=UserWarning)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

MODEL_PATH = "face_shape_classifier.pth" 
STATS_FILE = "recommendation_stats.json"
TOP_K_RECOMMENDATIONS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# 1. 1순위 헤어스타일 예시 이미지 매핑
# -----------------------------------------------------------------
# (key): 'recommendation_stats.json'에 있는 '한글' 헤어스타일 이름
# (value): 'static/hairstyles/' 폴더 안에 저장할 '영어' 이미지 파일명
#
# ※※※ 중요 ※※※
# 8종류의 헤어스타일 이름을 본인의 JSON 파일과 이미지 파일에 맞게
# '정확하게' 수정해야 합니다. (이것은 저의 추측입니다)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
STYLE_IMAGE_MAP = {
    '가르마': '가르마.png',
    '루프': '루프.png',
    '리젠트': '리젠트.png',
    '리프': '리프.png',
    '바디': '바디.png',
    '빌드': '빌드.png',
    '보브': '보브.png',
    '보니': '보니.png'
    # (JSON에 있는 8개 이름을 모두 정확히 입력하세요)
}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 

print(f"[!] 로딩: 얼굴형 분류기 ({MODEL_PATH})")
class_names = ["둥근형", "긴 타원형", "계란형", "역삼각형", "사각형"]
NUM_CLASSES = len(class_names)
try:
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("[!] 로딩: (모델 1) 완료.")
except Exception as e:
    print(f"[!] 치명적 오류: 모델 1(.pth) 로드 실패. {e}")

print(f"[!] 로딩: 추천 엔진 ({STATS_FILE})")
try:
    with open(STATS_FILE, 'r', encoding='utf-8') as f:
        stats_data = json.load(f)
    print("[!] 로딩: (모델 2) 완료.")
except Exception as e:
    print(f"[!] 치명적 오류: 모델 2(.json) 로드 실패. {e}")


data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_and_recommend(image_path):
    try:
        img_array = np.fromfile(image_path, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"이미지 로드 오류: {e}")
        return None, None, None

    image_tensor = data_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probabilities, 0)
        
        predicted_face_shape = class_names[top_idx.item()]
        confidence = top_prob.item()

    recommendations = []
    if predicted_face_shape in stats_data:
        top_k_list = stats_data[predicted_face_shape][:TOP_K_RECOMMENDATIONS]
        recommendations = top_k_list
    
    return predicted_face_shape, confidence, recommendations

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            face_shape, confidence, recs = predict_and_recommend(filepath)
            
            web_image_path = url_for('uploaded_file', filename=filename)
            
            result_data = {
                "face_shape": face_shape,
                "confidence": f"{confidence*100:.2f}%",
                "recommendations": recs,
                "web_image_path": web_image_path
            }
            
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # 2. 1순위 헤어스타일 이미지 경로 추가
            # -----------------------------------------------------------------
            # 'recs' (recommendations) 리스트가 비어있지 않은지 확인
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if recs:
                # 1. 1순위 헤어스타일 이름 가져오기 (예: '빌드')
                top_style_name = recs[0].get('hairstyle', '')
                
                # 2. STYLE_IMAGE_MAP에서 해당 이미지 파일명 찾기
                #    (못 찾을 경우 'default.jpg'를 사용 -> 혹시 모를 오류 방지)
                top_style_image_file = STYLE_IMAGE_MAP.get(top_style_name, 'default.jpg')
                
                # 3. HTML에서 사용할 최종 경로 생성 (예: /static/hairstyles/build.jpg)
                result_data['top_style_image_path'] = url_for('static', filename=f'hairstyles/{top_style_image_file}')
            else:
                # 추천 목록이 없는 경우 (오류 등)
                result_data['top_style_image_path'] = url_for('static', filename='hairstyles/default.jpg')
            
            return render_template('index.html', result=result_data)

    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # debug=True는 개발 중에만 사용하고, 실제 배포 시에는 False로 변경하세요.
    app.run(host='0.0.0.0', port=5000, debug=True)