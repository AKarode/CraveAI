import pytesseract
from PIL import Image
import pdf2image
import os
import tempfile
import re
from typing import List, Union, BinaryIO

class OCRService:
    """Service for performing OCR on menu images and PDFs."""
    
    def __init__(self):
        # Additional configuration options can be added here if needed.
        pass
        
    def process_image(self, image: Union[str, Image.Image]) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image: Path to an image file or a PIL Image object.
            
        Returns:
            Extracted text as a string.
        """
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
            
        text = pytesseract.image_to_string(img)
        return text
    
    def process_pdf(self, pdf_file: Union[str, BinaryIO]) -> str:
        """
        Extract text from a PDF using OCR.
        
        Args:
            pdf_file: Path to a PDF file or a file-like object.
            
        Returns:
            Extracted text as a string.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            if isinstance(pdf_file, str):
                images = pdf2image.convert_from_path(pdf_file)
            else:
                images = pdf2image.convert_from_bytes(pdf_file.read())
            
            extracted_text = []
            for img in images:
                text = self.process_image(img)
                extracted_text.append(text)
                
            return "\n\n".join(extracted_text)
    
    def extract_menu_items(self, text: str) -> List[dict]:
        """
        Extract menu items from OCR text using heuristics.
        
        This method processes the raw OCR text line by line. It identifies lines
        that contain a price (e.g., "$12.99") and treats them as the start of a new
        menu item. Text before the price is taken as the item name, and any text after
        is an initial description. Subsequent lines without a price are appended to the
        current item's description.
        
        Args:
            text: OCR-extracted text.
            
        Returns:
            List of dictionaries, each containing:
                - 'name': Menu item name.
                - 'price': Menu item price as a float.
                - 'description': A description of the menu item.
        """
        # Split the text into non-empty, stripped lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        items = []
        current_item = None
        
        # Regex pattern to detect prices (e.g., "$12.99")
        price_pattern = re.compile(r'(\$\d+(?:\.\d{2})?)')
        
        for line in lines:
            # Check if the line contains a price
            match = price_pattern.search(line)
            if match:
                # Finalize the previous item if present
                if current_item:
                    items.append(current_item)
                    current_item = None
                
                price_str = match.group(1)
                try:
                    price = float(price_str.replace('$', '').replace(',', ''))
                except ValueError:
                    price = None
                
                # Split the line around the price.
                # Expect parts like: [name text, price, remaining description]
                parts = price_pattern.split(line, maxsplit=1)
                if len(parts) >= 3:
                    name_part = parts[0].strip()
                    description_part = parts[2].strip()
                else:
                    name_part = line.replace(price_str, '').strip()
                    description_part = ""
                
                current_item = {
                    'name': name_part,
                    'price': price,
                    'description': description_part
                }
            else:
                # If the line does not contain a price and we already have an item,
                # append it to the current item's description.
                if current_item:
                    if current_item['description']:
                        current_item['description'] += " " + line
                    else:
                        current_item['description'] = line
                else:
                    # Optionally, handle lines that don't belong to any item.
                    continue
        
        # Append the final item if it exists
        if current_item:
            items.append(current_item)
            
        return items

# Example usage:
if __name__ == "__main__":
    ocr_service = OCRService()
    
    # For an image file:
    # text = ocr_service.process_image("path/to/menu_image.jpg")
    
    # For a PDF file:
    # text = ocr_service.process_pdf("path/to/menu.pdf")
    
    # For demonstration, assume 'text' contains the OCR output.
    text = """
    Caesar Salad $8.99
    Crisp romaine lettuce tossed with Caesar dressing,
    croutons, and parmesan cheese.
    
    Grilled Chicken Sandwich $12.50
    Tender grilled chicken breast served on a toasted bun,
    with lettuce, tomato, and mayo.
    
    Chocolate Cake $6.00
    Rich, moist chocolate cake with a velvety chocolate ganache.
    """
    
    menu_items = ocr_service.extract_menu_items(text)
    for item in menu_items:
        print("Name:", item['name'])
        print("Price:", item['price'])
        print("Description:", item['description'])
        print("-" * 40)

