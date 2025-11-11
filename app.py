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

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore", category=UserWarning)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

MODEL_PATH = "face_shape_classifier.pth" 
STATS_FILE = "recommendation_stats.json"
TOP_K_RECOMMENDATIONS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            
            return render_template('index.html', result=result_data)

    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)