import pytesseract
from PIL import Image
import cv2
import numpy as np

img_path = "shift.jpeg"

# 画像の読み込み（グレースケール変換）
img_pil = Image.open(img_path).convert("L")
img = np.array(img_pil)

# 不要な部分（ステータスバー）をトリミング
height, width = img.shape
cropped_img = img[int(height * 0.1):, :]  # 上部10%をカット

# 画像を拡大（OCR精度向上） → 2倍に変更
scale_factor = 2
resized_img = cv2.resize(cropped_img, (cropped_img.shape[1] * scale_factor, cropped_img.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)

# ノイズ除去
blurred_img = cv2.GaussianBlur(resized_img, (5, 5), 0)

# **適応的二値化で白飛びを防ぐ**
binary_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# ★処理後の画像を保存（確認用）
cv2.imwrite("processed.jpeg", binary_img)

# OCR実行（日本語 + 英語）
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:AMP'
text = pytesseract.image_to_string(binary_img, config=custom_config, lang="jpn+eng")


print("OCR結果:\n", text)
