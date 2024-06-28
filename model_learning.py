import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import joblib

# 正常データのディレクトリ
normal_dir = 'dataset/normal'

# データリスト
data = []

# 正常データの読み込み
for filename in os.listdir(normal_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(normal_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        image = image.flatten() / 255.0
        data.append(image)

# NumPy配列に変換
data = np.array(data)

# PCAによる次元削減
n_components = min(len(data), 50)  # 次元数はデータ数以下
pca = PCA(n_components=n_components)
data_pca = pca.fit_transform(data)

# Isolation Forestによる異常検知モデルの学習
iso_forest = IsolationForest(contamination=0.1)
iso_forest.fit(data_pca)

# モデルの保存
joblib.dump(pca, 'pca_model.pkl')
joblib.dump(iso_forest, 'iso_forest_model.pkl')
