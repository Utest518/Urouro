from flask import Flask, request, jsonify, render_template, url_for
import cv2
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# 各色のHSV範囲を定義
COLOR_RANGES = {
    "淡黄色": ((20, 50, 50), (30, 255, 255)),
    "水様透明": ((0, 0, 200), (180, 20, 255)),
    "赤紅色": ((0, 50, 50), (10, 255, 255)),
    "赤褐色": ((0, 50, 20), (10, 255, 100)),
    "茶色褐色": ((10, 50, 20), (20, 255, 100)),
    "乳白灰白色": ((0, 0, 200), (180, 50, 255)),
    "乳白色": ((0, 0, 180), (180, 50, 200)),
    "褐色白色混合": ((10, 50, 100), (20, 200, 150)),
    "赤レンガピンク": ((0, 50, 50), (10, 200, 200)),
}

# 複数の色範囲を設定して尿の部分を検出する
URINE_RANGES = [
    ((15, 40, 40), (35, 255, 255)),
    ((20, 50, 50), (30, 255, 255))
]

def analyze_image(image):
    # 画像をOpenCV形式に変換
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 尿の部分を検出するための前処理（複数の色範囲を試す）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    urine_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in URINE_RANGES:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        urine_mask = cv2.bitwise_or(urine_mask, mask)

    # 尿の輪郭を検出
    contours, _ = cv2.findContours(urine_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 最大の輪郭を選択（最も大きい尿の部分と仮定）
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        urine_region = image[y:y+h, x:x+w]

        # 尿の部分に対して色の判定を行う
        hsv_urine = cv2.cvtColor(urine_region, cv2.COLOR_BGR2HSV)
        color_percentages = {}
        for color_name, (lower, upper) in COLOR_RANGES.items():
            mask = cv2.inRange(hsv_urine, np.array(lower), np.array(upper))
            percentage = (cv2.countNonZero(mask) / (urine_region.size / 3)) * 100
            color_percentages[color_name] = percentage

        # 最も割合の高い色を判定
        detected_color = max(color_percentages, key=color_percentages.get)
    else:
        # 尿の部分が検出できなかった場合、画像全体で色を判定
        color_percentages = {}
        for color_name, (lower, upper) in COLOR_RANGES.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            percentage = (cv2.countNonZero(mask) / (image.size / 3)) * 100
            color_percentages[color_name] = percentage

        # 最も割合の高い色を判定
        detected_color = max(color_percentages, key=color_percentages.get)

    # 泡立ちの検出
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHEの適用
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # ガウシアンブラーのカーネルサイズを大きく調整してノイズを減らす
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)  # 二値化の適用（しきい値を調整）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積と周長に基づいて輪郭をフィルタリング
    min_area = 50  # 泡と見なすための最小面積しきい値を設定
    min_perimeter = 50  # 泡と見なすための最小周長しきい値を設定
    foam_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.arcLength(cnt, True) > min_perimeter]

    # 円形度でフィルタリング
    foam_contours = [cnt for cnt in foam_contours if 0.7 < cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2) < 1.3]

    # デバッグ用に検出された輪郭を画像に描画して保存
    result_image = cv2.drawContours(image.copy(), foam_contours, -1, (0, 255, 0), 2)

    # 画像を保存するためのパスを設定
    output_path = os.path.join('static', 'detected_foam_contours.png')
    if not os.path.exists('static'):
        os.makedirs('static')
    cv2.imwrite(output_path, result_image)

    # デバッグ用に検出された輪郭の情報を出力
    debug_info = {
        'contour_areas': [cv2.contourArea(cnt) for cnt in foam_contours],
        'contour_perimeters': [cv2.arcLength(cnt, True) for cnt in foam_contours],
        'total_contours': len(contours),
        'filtered_contours': len(foam_contours)
    }

    # 密度解析：輪郭の数と面積の比率を計算
    foam_density = len(foam_contours) / (image.shape[0] * image.shape[1])

    # 輪郭の数が一定以上で、密度が適切な範囲内であれば泡立ちを検出
    foam_detected = len(foam_contours) > 0

    return {
        'detected_color': detected_color,
        'color_percentages': color_percentages,
        'foam_detected': foam_detected,
        'image_path': url_for('static', filename='detected_foam_contours.png'),
        'debug_info': debug_info  # デバッグ情報を追加
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    if file:
        image = Image.open(io.BytesIO(file.read()))
        result = analyze_image(image)
        return jsonify(result)
    return jsonify({'error': 'No file uploaded'})

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
