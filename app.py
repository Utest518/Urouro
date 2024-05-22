from flask import Flask, request, jsonify, render_template, url_for
import cv2
import numpy as np
import base64
from PIL import Image
import io
import tensorflow as tf
from foam_detection import detect_foam

app = Flask(__name__)

# GPUメモリの使用量を制限
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# モデルの遅延読み込み
model = None
def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model('urine_color_model.h5')

# カテゴリ（ラベル）のリスト
categories = ["yellow", "coffee_milky", "light_pink", "red", "transparent_yellow", "white_milky", "brown"]

def preprocess_image(image):
    # 画像を128x128にリサイズ
    image = cv2.resize(image, (128, 128))
    # 画像をnumpy配列に変換し、正規化
    image = image.astype('float32') / 255.0
    # バッチサイズの次元を追加
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.get_json()
    image_data = data['image']
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # モデルを遅延読み込み
    load_model()
    
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

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
