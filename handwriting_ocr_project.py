import cv2
import numpy as np
import pytesseract
import os
import re
import pandas as pd

# Görsel yolu
image_path = "images/berkay.png"

# Görseli oku
image = cv2.imread(image_path)

# Grayscale dönüşümü
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gürültü azaltma (Gaussian Blur)
blur = cv2.GaussianBlur(gray, (13, 13), 0)

# Binary Thresholding (Adaptive Threshold)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)
# Morphology: Erode ve Dilate ile harfleri netleştir
kernel = np.ones((2, 2), np.uint8)
erode = cv2.erode(thresh, kernel, iterations=1)
morph = cv2.dilate(erode, kernel, iterations=1)

# Tesseract ile metin tanıma (hem Türkçe hem İngilizce)
custom_config = r'--oem 1 --psm 11 -l tur+eng'
text = pytesseract.image_to_string(morph, config=custom_config)

# Post-Processing: Hatalı karakterleri temizle, noktalama işaretlerini koru
text = re.sub(r'[^a-zA-ZğĞıİöÖşŞüÜçÇâî0-9\s.,:\'!?;-]', '', text)
# Yanlış noktaları temizle
text = re.sub(r'\.\s+(?=[a-zğışöüç])', ' ', text)
# Satır atlamalarını koru, her satır içindeki fazla boşlukları temizle
lines = text.split('\n')
lines = [re.sub(r'\s+', ' ', line).strip() for line in lines]
# Boş satırları filtrele
lines = [line for line in lines if line]  # Boş satırları kaldır
# Tire işaretlerini birleştir
lines = [line.replace('- ', '') for line in lines]

# Doğruluk oranını hesapla
data = pytesseract.image_to_data(morph, config=custom_config, output_type=pytesseract.Output.DATAFRAME)
confidences = data[data['conf'] != -1]['conf']  # Güven skoru -1 olmayan kelimeleri al
average_confidence = confidences.mean() if not confidences.empty else 0

# Çıktıyı yazdır
print("\nOCR ÇIKTISI:")
for line in lines:
    print(line)

# Ortalama doğruluk oranını yazdır
print(f"\nOrtalama Doğruluk Oranı: %{average_confidence:.2f}")

# Tanınan metni bir dosyaya kaydet
os.makedirs("output", exist_ok=True)
with open("output/recognized_text.txt", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

# İşlenmiş görseli kaydet
os.makedirs("output", exist_ok=True)
cv2.imwrite("output/final_processed.png", morph)

# (Opsiyonel) Orijinal görüntüde metin kutularını çizdir
boxes = pytesseract.image_to_boxes(morph, config=custom_config)
for b in boxes.splitlines():
    b = b.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(image, (x, image.shape[0] - y), (w, image.shape[0] - h), (0, 255, 0), 2)
cv2.imwrite("output/annotated_image.png", image)