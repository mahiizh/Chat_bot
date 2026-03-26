import easyocr
from PIL import Image
import numpy as np

class OCRHandler:
    def __init__(self):
        # Initialize EasyOCR reader (this might take a moment on first run)
        self.reader = easyocr.Reader(['en'], gpu=False)
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR"""
        try:
            # Read image
            image = Image.open(image_path)
            image_np = np.array(image)
            
            # Perform OCR
            results = self.reader.readtext(image_np)
            
            # Combine all detected text
            extracted_text = ' '.join([result[1] for result in results])
            
            return extracted_text
        except Exception as e:
            return f"Error processing image: {str(e)}"