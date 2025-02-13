import pytesseract
from PIL import Image
import cv2
import numpy as np
import os

img_path = "shift.jpeg"

# 画像の存在確認
if not os.path.exists(img_path):
    print(f"Error: File not found -> {img_path}")
else:
    try:
        # PILで画像を開いてNumPy配列に変換
        img_pil = Image.open(img_path).convert("L")  # グレースケール変換
        img = np.array(img_pil)

        # 画像を拡大（解像度向上）
        scale_factor = 3
        img = cv2.resize(img, (img.shape[1] * scale_factor, img.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)

        # ノイズ除去
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # 二値化（しきい値処理）
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OCR実行（日本語 + 英語）
        text = pytesseract.image_to_string(img, lang="jpn+eng")

        print("OCR結果:\n", text)

    except Exception as e:
        print("Error processing image:", e)
