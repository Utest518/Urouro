import cv2
import numpy as np

def detect_foam(image):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ヒストグラム平坦化の適用
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # ガウシアンブラーの適用
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # エッジ検出の適用
    edges = cv2.Canny(gray, 50, 150)
    
    # 輪郭の検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 面積と周長に基づいて輪郭をフィルタリング
    min_area = 10  # 最小面積を調整
    min_perimeter = 30  # 最小周長を調整
    foam_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.arcLength(cnt, True) > min_perimeter]
    
    # 円形度でフィルタリング
    foam_contours = [cnt for cnt in foam_contours if 0.5 < cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2) < 2.0]
    
    # 泡の検出結果を描画
    result_image = cv2.drawContours(image.copy(), foam_contours, -1, (0, 255, 0), 2)
    
    # 泡の検出フラグ
    foam_detected = len(foam_contours) > 0
    
    return foam_detected, result_image
