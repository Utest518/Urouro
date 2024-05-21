from PIL import Image

# カラーコードのリスト（それぞれ2つずつ）
colors = {
    "yellow": ["#FFFFE0", "#FFFACD"],  # 淡黄色
    "brown": ["#8B4513", "#A0522D"],  # 茶色
    "red": ["#FF6347", "#FF4500"],  # 赤色
    "transparent_yellow": ["#F5F5DC", "#FFFFE0"]  # 無色透明（薄い黄色）
}

# 各色の画像を生成
for color_name, hex_codes in colors.items():
    for i, hex_code in enumerate(hex_codes):
        # 画像を生成
        img = Image.new("RGB", (128, 128), hex_code)
        # 画像を保存
        img.save(f"{color_name}_{i+1}.png")

print("画像の生成が完了しました。")
