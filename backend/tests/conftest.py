"""
Pytest configuration and shared fixtures for the CraveAI backend tests.
"""
import os
import sys
import pytest
from dotenv import load_dotenv, find_dotenv

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Find and load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

@pytest.fixture(scope="session")
def api_keys():
    """Fixture to provide API keys for tests."""
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_environment": os.getenv("PINECONE_ENVIRONMENT"),
        "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME"),
    }

@pytest.fixture(scope="session")
def openai_client(api_keys):
    """Fixture to provide an OpenAI client."""
    from openai import OpenAI
    return OpenAI(api_key=api_keys["openai_api_key"])

@pytest.fixture(scope="session")
def pinecone_client(api_keys):
    """Fixture to provide a Pinecone client."""
    import pinecone
    pc = pinecone.Pinecone(api_key=api_keys["pinecone_api_key"])
    return pc

@pytest.fixture(scope="function")
def mock_openai_embedding(monkeypatch):
    """Fixture to mock OpenAI embedding API calls."""
    class MockEmbedding:
        def create(self, *args, **kwargs):
            class Response:
                data = [{"embedding": [0.1] * 1536}]
            return Response()
    
    class MockOpenAI:
        embeddings = MockEmbedding()
        
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: MockOpenAI())
    return MockOpenAI()

@pytest.fixture(scope="function")
def sample_menu_items():
    """Fixture to provide sample menu items for tests."""
    return [
        {
            "name": "Spicy Thai Noodles",
            "description": "Rice noodles with spicy sauce, vegetables, and tofu",
            "price": "$12.99",
            "category": "Main Course",
            "dietary_info": ["Vegan", "Spicy", "Gluten-Free"]
        },
        {
            "name": "Classic Cheeseburger",
            "description": "Beef patty with cheddar cheese, lettuce, tomato, and special sauce",
            "price": "$10.99",
            "category": "Burgers",
            "dietary_info": []
        }
    ]
