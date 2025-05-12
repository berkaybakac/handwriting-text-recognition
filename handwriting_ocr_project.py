import cv2
import numpy as np
import pytesseract
import os
import re 
import pandas as pd

# Path to the image file
image_path = "images/berkay.png"

# Read the image and convert to grayscale
original_image = cv2.imread(image_path)
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur and Adaptive Threshold
blurred_image = cv2.GaussianBlur(grayscale_image, (13, 13), 0)
thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Morphological operations to sharpen text
kernel = np.ones((2, 2), np.uint8)
morphed_image = cv2.dilate(cv2.erode(thresholded_image, kernel, iterations=1), kernel, iterations=1)

# Extract and clean text using Tesseract
custom_config = r'--oem 1 --psm 11 -l tur+eng'
extracted_text = pytesseract.image_to_string(morphed_image, config=custom_config)
cleaned_lines = [re.sub(r'\s+', ' ', line).strip().replace('- ', '') for line in 
                 re.sub(r'[^a-zA-ZğĞıİöÖşŞüÜçÇâî0-9\s.,:\'!?;-]', '', 
                 re.sub(r'\.\s+(?=[a-zğışöüç])', ' ', extracted_text)).split('\n') if line]

# Calculate accuracy using Pandas
average_confidence = pytesseract.image_to_data(morphed_image, config=custom_config, 
                     output_type=pytesseract.Output.DATAFRAME)['conf'].replace(-1, np.nan).mean() or 0

# Print the extracted text and accuracy
print("\nOCR OUTPUT:")
for line in cleaned_lines:
    print(line)
print(f"\nAverage Accuracy: %{average_confidence:.2f}")

# Save extracted text and processed image
os.makedirs("output", exist_ok=True)
with open("output/recognized_text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines))
cv2.imwrite("output/final_processed.png", morphed_image)

# (Optional) Draw text boxes on original image
boxes = pytesseract.image_to_boxes(morphed_image, config=custom_config)
for b in boxes.splitlines():
    b = b.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(original_image, (x, original_image.shape[0] - y), (w, original_image.shape[0] - h), (0, 255, 0), 2)
cv2.imwrite("output/annotated_image.png", original_image)