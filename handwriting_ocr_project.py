import cv2
import numpy as np
import pytesseract
import os
import re 
import pandas as pd

# Görsel yolu
image_path = "images/berkay.png"

# Görseli oku
original_image = cv2.imread(image_path)

# Grayscale dönüşümü
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Gürültü azaltma (Gaussian Blur)
blurred_image = cv2.GaussianBlur(grayscale_image, (13, 13), 0)

# Binary Thresholding (Adaptive Threshold)
thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)

# Morphology: Erode ve Dilate ile harfleri netleştir
kernel = np.ones((2, 2), np.uint8)
erode = cv2.erode(thresholded_image, kernel, iterations=1)
morphed_image = cv2.dilate(erode, kernel, iterations=1)

# Tesseract ile metin tanıma (hem Türkçe hem İngilizce)
custom_config = r'--oem 1 --psm 11 -l tur+eng'
extracted_text = pytesseract.image_to_string(morphed_image, config=custom_config)

# Post-Processing: Hatalı karakterleri temizle, noktalama işaretlerini koru
extracted_text = re.sub(r'[^a-zA-ZğĞıİöÖşŞüÜçÇâî0-9\s.,:\'!?;-]', '', extracted_text)
extracted_text = re.sub(r'\.\s+(?=[a-zğışöüç])', ' ', extracted_text)
# Satır atlamalarını koru, her satır içindeki fazla boşlukları temizle ve tireleri birleştir
cleaned_lines = extracted_text.split('\n')
cleaned_lines = [re.sub(r'\s+', ' ', line).strip().replace('- ', '') for line in cleaned_lines]
cleaned_lines = list(filter(None, cleaned_lines))  # Boş satırları kaldır

# Doğruluk oranını hesapla (Pandas ile)
data = pytesseract.image_to_data(morphed_image, config=custom_config, output_type=pytesseract.Output.DATAFRAME)
confidences = data[data['conf'] != -1]['conf']  # Güven skoru -1 olmayan kelimeleri al
average_confidence = confidences.mean() if not confidences.empty else 0

# Çıktıyı yazdır
print("\nOCR ÇIKTISI:")
for line in cleaned_lines:
    print(line)

# Ortalama doğruluk oranını yazdır
print(f"\nOrtalama Doğruluk Oranı: %{average_confidence:.2f}")

# Tanınan metni bir dosyaya kaydet
os.makedirs("output", exist_ok=True)  # Çıktı dizinini bir kez oluştur
with open("output/recognized_text.txt", "w", encoding="utf-8") as f:
    for line in cleaned_lines:
        f.write(line + "\n")

# İşlenmiş görseli kaydet
cv2.imwrite("output/final_processed.png", morphed_image)

# (Opsiyonel) Orijinal görüntüde metin kutularını çizdir
boxes = pytesseract.image_to_boxes(morphed_image, config=custom_config)
for b in boxes.splitlines():
    b = b.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(original_image, (x, original_image.shape[0] - y), (w, original_image.shape[0] - h), (0, 255, 0), 2)
cv2.imwrite("output/annotated_image.png", original_image)