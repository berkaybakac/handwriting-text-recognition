import cv2
import pytesseract

# Görsel dosya yolu — test etmek istediğin görselin adı
image_path = 'images/cemal_sureya.png'

# Görseli yükle
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Gri tonlamaya çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold (eşikleme) uygulayarak yazıyı daha belirgin hale getir
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# OCR işlemi (dil: Türkçe, psm 6 = satır bazlı tanıma)
custom_config = r'--oem 3 --psm 6 -l tur'
extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

# Çıktıyı terminale yazdır
print("\n[Extracted Text from Image]\n")
print(extracted_text)

# OCR çıktısını dosyaya kaydet
output_path = 'ocr_output.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(extracted_text)

print(f"\n✅ OCR result saved to: {output_path}")
