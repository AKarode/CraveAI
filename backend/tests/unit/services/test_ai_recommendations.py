"""
Unit tests for the AI recommendations service.
"""
import pytest
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from services.ai_recommendations import RecommendationService

@pytest.fixture
def recommendation_service(monkeypatch):
    """Create a recommendation service with mocked dependencies."""
    # Mock Pinecone and OpenAI
    class MockPinecone:
        def __init__(self, *args, **kwargs):
            pass
        
        def list_indexes(self):
            class MockIndexes:
                def names(self):
                    return ["test_index"]
            return MockIndexes()
        
        def Index(self, name):
            class MockIndex:
                def upsert(self, *args, **kwargs):
                    return {"upserted_count": len(kwargs.get("vectors", []))}
                
                def query(self, *args, **kwargs):
                    return {
                        "matches": [
                            {
                                "id": "item1",
                                "score": 0.95,
                                "metadata": {
                                    "name": "Spicy Dish",
                                    "description": "A very spicy dish",
                                    "dietary_info": ["Vegetarian", "Gluten Free"]
                                }
                            },
                            {
                                "id": "item2",
                                "score": 0.85,
                                "metadata": {
                                    "name": "Mild Dish",
                                    "description": "A mild dish",
                                    "dietary_info": []
                                }
                            }
                        ]
                    }
            return MockIndex()
    
    class MockOpenAI:
        def __init__(self, *args, **kwargs):
            pass
        
        @property
        def embeddings(self):
            class MockEmbeddings:
                def create(self, *args, **kwargs):
                    class MockResponse:
                        @property
                        def data(self):
                            return [{"embedding": [0.1] * 1536}]
                    return MockResponse()
            return MockEmbeddings()
    
    # Apply mocks
    monkeypatch.setattr("pinecone.Pinecone", MockPinecone)
    monkeypatch.setattr("openai.OpenAI", MockOpenAI)
    
    # Return service instance
    return RecommendationService()

def test_store_menu_items(recommendation_service):
    """Test storing menu items in vector database."""
    menu_items = [
        {
            "name": "Spicy Thai Noodles",
            "description": "Rice noodles with spicy sauce, vegetables, and tofu",
            "price": "$12.99"
        },
        {
            "name": "Classic Cheeseburger",
            "description": "Beef patty with cheddar cheese, lettuce, tomato, and special sauce",
            "price": "$10.99"
        }
    ]
    
    # Call the method
    result = recommendation_service.store_menu_items(menu_items, "test_menu_id")
    
    # Since the actual function doesn't return anything, we just ensure it doesn't error
    assert result is None

def test_get_recommendations(recommendation_service):
    """Test getting recommendations based on preferences."""
    # Call the method
    recommendations = recommendation_service.get_recommendations(
        menu_id="test_menu_id",
        preferences=["spicy", "vegetarian"],
        dietary_restrictions=["gluten-free"]
    )
    
    # Check the results
    assert len(recommendations) > 0
    assert "name" in recommendations[0]
    assert "description" in recommendations[0]
    assert "score" in recommendations[0] or "match_score" in recommendations[0]
    
    # Check sorting - should be sorted by score/match_score in descending order
    # Only check this if we have at least 2 recommendations
    if len(recommendations) >= 2:
        if "score" in recommendations[0]:
            assert recommendations[0]["score"] >= recommendations[1]["score"], "Results should be sorted by score"
        elif "match_score" in recommendations[0]:
            assert recommendations[0]["match_score"] >= recommendations[1]["match_score"], "Results should be sorted by match_score"

# Removed the test_filter_by_dietary_restrictions test since it was testing a private method 