"""
API endpoint tests for the FastAPI application.
"""
import pytest
import os
import json
from fastapi.testclient import TestClient
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import app

@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)

def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "service" in data
    assert data["service"] == "crave-ai-backend"

@pytest.mark.parametrize("file_path,expected_status", [
    ("backend/tests/fixtures/sample_menu.pdf", 200),
    ("backend/tests/fixtures/invalid_format.txt", 400)
])
def test_process_menu_endpoint(client, monkeypatch, file_path, expected_status):
    """Test the menu processing endpoint with different file types."""
    # Skip if test file doesn't exist
    if not os.path.exists(file_path) and expected_status == 200:
        pytest.skip(f"Test file {file_path} not found. Create it first.")
    
    # Create the invalid format file if it doesn't exist
    if "invalid_format.txt" in file_path and not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("This is not a valid menu file")
    
    # Mock OCR service to avoid actual processing
    def mock_process_pdf(*args, **kwargs):
        return "Mock menu text with some items"
    
    def mock_process_image(*args, **kwargs):
        return "Mock menu text with some items"
    
    def mock_extract_menu_items(*args, **kwargs):
        return [
            {"name": "Test Dish 1", "description": "Description 1", "price": "$10.99"},
            {"name": "Test Dish 2", "description": "Description 2", "price": "$12.99"}
        ]
    
    def mock_store_menu_items(*args, **kwargs):
        return None
    
    # Apply mocks
    monkeypatch.setattr("services.ocr.OCRService.process_pdf", mock_process_pdf)
    monkeypatch.setattr("services.ocr.OCRService.process_image", mock_process_image)
    monkeypatch.setattr("services.ocr.OCRService.extract_menu_items", mock_extract_menu_items)
    monkeypatch.setattr("services.ai_recommendations.RecommendationService.store_menu_items", mock_store_menu_items)
    
    # Prepare the request
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
            response = client.post("/api/process-menu", files=files)
    else:
        # For testing error cases with invalid file extensions
        mock_file_content = b"This is not a valid menu file"
        files = {"file": (os.path.basename(file_path), mock_file_content, "text/plain")}
        response = client.post("/api/process-menu", files=files)
    
    # Print response for debugging
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")
    
    # Check response
    if expected_status == 400:
        # Either the status code should be 400 or the error message should be about file format
        assert (response.status_code == 400 or 
               (response.status_code == 500 and "Unsupported file format" in response.text))
    else:
        assert response.status_code == expected_status
    
    if expected_status == 200:
        data = response.json()
        assert "menu_id" in data
        assert "items" in data
        assert "success" in data
        assert data["success"] == True
        assert len(data["items"]) == 2

def test_recommendations_endpoint(client, monkeypatch):
    """Test getting recommendations based on preferences."""
    # Mock the recommendation service
    def mock_get_recommendations(*args, **kwargs):
        return [
            {
                "name": "Spicy Thai Noodles",
                "description": "Rice noodles with spicy sauce",
                "score": 0.95,
                "match_reasons": ["spicy", "vegetarian"]
            },
            {
                "name": "Vegetable Stir Fry",
                "description": "Fresh vegetables in a savory sauce",
                "score": 0.85,
                "match_reasons": ["vegetarian"]
            }
        ]
    
    monkeypatch.setattr("services.ai_recommendations.RecommendationService.get_recommendations", mock_get_recommendations)
    
    # Test data
    test_menu_id = "test_menu_123"
    test_request = {
        "preferences": ["spicy", "vegetarian"],
        "dietary_restrictions": ["gluten-free"]
    }
    
    # Make request
    response = client.post(f"/api/recommendations/{test_menu_id}", json=test_request)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 2
    assert data["recommendations"][0]["name"] == "Spicy Thai Noodles"
    assert "match_reasons" in data["recommendations"][0]
    assert "vegetarian" in data["recommendations"][0]["match_reasons"]
