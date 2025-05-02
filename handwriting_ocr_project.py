import os
import cv2
import pytesseract

# 🔥 Doğru path: '.../tessdata/' olmalı
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata/'

image_path = "images/türkçe_elyazısı_2.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

custom_config = r'--oem 3 --psm 6 -l tur'
text = pytesseract.image_to_string(thresh, config=custom_config)

print("\n📝 OCR ÇIKTISI:")
print(text)
