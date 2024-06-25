import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# データセットのパス
dataset_path = "dataset"

# カテゴリ（ラベル）のリスト
categories = ["normal"]

# 画像サイズ
img_size = 128

# データセットの準備
data = []
labels = []

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (img_size, img_size))
    return image

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = categories.index(category)
    for img_name in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, img_name)
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(img_path)
                if img is not None:
                    img = preprocess_image(img)
                    data.append(img)
                    labels.append(label)
                else:
                    print(f"Error reading image {img_path}")
            else:
                print(f"Skipping non-image file: {img_name}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# データをnumpy配列に変換
data = np.array(data)
labels = np.array(labels)

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# データを保存
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

