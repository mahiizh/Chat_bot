import pytesseract
from PIL import Image

class OCRHandler:
    def __init__(self):
        pass
    
    def extract_text_from_image(self, image_path):
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            return f"Error processing image: {str(e)}"