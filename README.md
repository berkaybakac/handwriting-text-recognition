# Handwriting Text Recognition using OpenCV and Tesseract (Turkish)

This project focuses on recognizing handwritten Turkish text from images using classical image processing techniques and the Tesseract OCR engine.

## Project Goal

To develop a Python-based application that reads handwritten content from images, processes it using OpenCV, and extracts readable text using Tesseract's Turkish language model.

This work is prepared as a final project for the **Image Processing** course.

## Technologies Used

- **Python 3.11**
- **OpenCV** – image preprocessing (grayscale, thresholding)
- **Tesseract OCR** – text recognition engine
- **Pytesseract** – Python wrapper for Tesseract

## How It Works

1. The image is loaded using `cv2.imread()`.
2. Converted to grayscale with `cv2.cvtColor()`.
3. Thresholding applied using Otsu's method.
4. `pytesseract.image_to_string()` extracts text from the processed image using `-l tur` (Turkish).
5. The output is printed and saved to `ocr_output.txt`.

## Important Notes

- Make sure Tesseract is installed via Homebrew:
brew install tesseract tesseract-lang

📄 Prepared by: Berkay Bakaç
🎓 Course: Image Processing
🏫 University Project – Final Submission







