import easyocr
import cv2
import numpy as np
import re

class OCRProcessor:
    def __init__(self, languages=['en'], model_storage_directory="./models/ocr"):
        import os
        os.makedirs(model_storage_directory, exist_ok=True)
        print(f"Loading EasyOCR models into {model_storage_directory}...")
        self.reader = easyocr.Reader(languages, model_storage_directory=model_storage_directory)

    def preprocess_image(self, image_path):
        """Preprocess image for better OCR results."""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive Thresholding (Otsu's method)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def extract_text(self, image_path, use_preprocessing=True):
        """Extract text from image using EasyOCR."""
        if use_preprocessing:
            processed_img = self.preprocess_image(image_path)
            # EasyOCR can take numpy array
            results = self.reader.readtext(processed_img)
        else:
            results = self.reader.readtext(image_path)

        # Format results: list of (text, bounding_box, confidence)
        extracted_data = []
        for (bbox, text, prob) in results:
            cleaned_text = self.clean_text(text)
            if len(cleaned_text) > 1:  # Filter out very short noise
                extracted_data.append({
                    'text': cleaned_text,
                    'bbox': bbox,
                    'confidence': prob
                })

        return extracted_data

    def clean_text(self, text):
        """Clean and refine OCR output."""
        # Remove non-alphanumeric characters except basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        return text.strip()

if __name__ == "__main__":
    # Test script will be implemented in main.py or separate test
    pass
