import pytesseract
from PIL import Image
import pdf2image
import os
import tempfile
from typing import List, Union, BinaryIO


class OCRService:
    """Service for performing OCR on menu images and PDFs"""
    
    def __init__(self):
        # You can add configuration options here if needed
        pass
        
    def process_image(self, image: Union[str, Image.Image]) -> str:
        """
        Extract text from an image using OCR
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Extracted text as string
        """
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
            
        text = pytesseract.image_to_string(img)
        return text
    
    def process_pdf(self, pdf_file: Union[str, BinaryIO]) -> str:
        """
        Extract text from a PDF using OCR
        
        Args:
            pdf_file: Path to PDF file or file-like object
            
        Returns:
            Extracted text as string
        """
        # Convert PDF to images
        with tempfile.TemporaryDirectory() as temp_dir:
            if isinstance(pdf_file, str):
                images = pdf2image.convert_from_path(pdf_file)
            else:
                images = pdf2image.convert_from_bytes(pdf_file.read())
            
            extracted_text = []
            
            # Process each page
            for img in images:
                text = self.process_image(img)
                extracted_text.append(text)
                
            return "\n\n".join(extracted_text)
    
    def extract_menu_items(self, text: str) -> List[dict]:
        """
        Extract menu items from OCR text
        
        Args:
            text: OCR-extracted text
            
        Returns:
            List of menu items with their details
        """
        # In a real implementation, this would use NLP techniques
        # to identify menu items, descriptions, prices, etc.
        # This is a very simplified placeholder implementation
        
        menu_items = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Very basic detection logic - to be replaced with real NLP
            if any(char.isdigit() for char in line) and '$' in line:
                # Simplistic assumption that lines with digits and $ are menu items
                parts = line.rsplit('$', 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        price = float(parts[1].strip().replace(',', ''))
                        menu_items.append({
                            'name': name,
                            'price': price
                        })
                    except ValueError:
                        # Not a valid price format
                        pass
        
        return menu_items 