"""
Unit tests for the OCR service.
"""
import pytest
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from services.ocr import OCRService

@pytest.fixture
def ocr_service():
    """Create an OCR service instance for testing."""
    return OCRService()

@pytest.fixture
def sample_menu_text():
    """Sample menu text for testing menu item extraction."""
    return """
    APPETIZERS
    
    Garlic Bread - $5.99
    Freshly baked bread with garlic butter and herbs.
    
    Mozzarella Sticks - $7.99
    Golden fried mozzarella sticks served with marinara sauce.
    
    MAIN COURSE
    
    Spaghetti Bolognese - $14.99
    Classic pasta with rich meat sauce and parmesan cheese.
    
    Grilled Salmon - $21.99
    Fresh salmon fillet grilled to perfection with lemon herb butter.
    """

def test_extract_menu_items(ocr_service, sample_menu_text):
    """Test extracting menu items from text."""
    menu_items = ocr_service.extract_menu_items(sample_menu_text)
    
    # Check that items were extracted
    assert len(menu_items) >= 4, f"Expected at least 4 menu items, got {len(menu_items)}"
    
    # Check that the menu items have expected fields
    for item in menu_items:
        assert "name" in item, "Menu item missing 'name' field"
        assert "price" in item, "Menu item missing 'price' field"
        assert "description" in item, "Menu item missing 'description' field"
    
    # Check that item names contain the expected dish names
    all_names = ' '.join([item["name"] for item in menu_items])
    assert "Garlic Bread" in all_names, "Failed to extract 'Garlic Bread'"
    assert "Mozzarella Sticks" in all_names, "Failed to extract 'Mozzarella Sticks'"
    assert "Spaghetti Bolognese" in all_names, "Failed to extract 'Spaghetti Bolognese'"
    assert "Grilled Salmon" in all_names, "Failed to extract 'Grilled Salmon'"

@patch('PIL.Image.open')
@patch('pytesseract.image_to_string')
def test_process_image(mock_image_to_string, mock_image_open, ocr_service):
    """Test processing an image with mocked dependencies."""
    # Mock the PIL.Image.open and pytesseract responses
    mock_image = MagicMock()
    mock_image_open.return_value = mock_image
    mock_image_to_string.return_value = "Sample menu text"
    
    # Create a temporary image file path (don't need actual content)
    with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_image:
        # Test the method
        result = ocr_service.process_image(temp_image.name)
        
        # Check the result
        assert result == "Sample menu text"
        mock_image_open.assert_called_once_with(temp_image.name)
        mock_image_to_string.assert_called_once()

@patch('pdf2image.convert_from_bytes')
def test_process_pdf(mock_convert_from_bytes, ocr_service, monkeypatch):
    """Test processing a PDF file with mocked dependencies."""
    # Mock pdf2image
    mock_img1 = MagicMock()
    mock_img2 = MagicMock()
    mock_convert_from_bytes.return_value = [mock_img1, mock_img2]
    
    # Mock OCR process_image method
    monkeypatch.setattr(ocr_service, "process_image", 
                       lambda img: "Page 1 text" if img == mock_img1 else "Page 2 text")
    
    # Create a simple file-like object that responds to read()
    mock_file = MagicMock()
    mock_file.read.return_value = b"PDF content"
    
    # Test with our mock file
    result = ocr_service.process_pdf(mock_file)
    
    # Check that convert_from_bytes was called
    mock_convert_from_bytes.assert_called_once_with(b"PDF content")
    
    # Check the result contains the text
    assert "Page 1 text" in result
    assert "Page 2 text" in result 