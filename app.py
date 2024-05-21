from flask import Flask, request, jsonify, render_template, url_for
import cv2
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from foam_detection import detect_foam

app = Flask(__name__)

# 学習済みモデルを読み込む
model = load_model('urine_color_model.h5')

# カテゴリ（ラベル）のリスト
categories = ["yellow", "coffee_milky", "light_pink", "red", "transparent_yellow", "white_milky", "brown"]

def preprocess_image(image):
    # 画像を128x128にリサイズ
    image = cv2.resize(image, (128, 128))
    # 画像をnumpy配列に変換し、正規化
    image = img_to_array(image) / 255.0
    # バッチサイズの次元を追加
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    if file:
        # 画像を読み込み
        image = Image.open(io.BytesIO(file.read()))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 色の検出
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        detected_color = categories[np.argmax(prediction)]
        
        # 泡の検出
        foam_detected, result_image = detect_foam(image)
        
        # 結果画像の保存
        output_path = os.path.join('static', 'detected_foam_contours.png')
        if not os.path.exists('static'):
            os.makedirs('static')
        cv2.imwrite(output_path, result_image)
        
        result = {
            'detected_color': detected_color,
            'foam_detected': foam_detected,
            'image_path': url_for('static', filename='detected_foam_contours.png')
        }
        
        return jsonify(result)
    return jsonify({'error': 'No file uploaded'})

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
