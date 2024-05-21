import cv2
import numpy as np

# 各色のHSV範囲を定義
COLOR_RANGES = {
    "淡黄色": ((10, 40, 40), (40, 255, 255)),
    "無色透明": ((0, 0, 200), (180, 30, 255)),
    "赤色": ((0, 50, 50), (10, 255, 255)),
    "茶色": ((10, 50, 50), (20, 255, 255)),
}

def detect_color(image):
    # 画像をHSV色空間に変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 画像の中心部分を切り取る（サイズは任意で調整）
    h, w = hsv.shape[:2]
    center = hsv[h//4:3*h//4, w//4:3*w//4]

    # 切り取った部分に対して色の判定を行う
    color_percentages = {}
    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(center, np.array(lower), np.array(upper))
        percentage = (cv2.countNonZero(mask) / (center.size / 3)) * 100
        color_percentages[color_name] = percentage

    # 最も割合の高い色を判定
    detected_color = max(color_percentages, key=color_percentages.get)

    return detected_color, color_percentages
