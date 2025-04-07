"""
Integration tests for OpenAI service.
"""
import pytest
import time
import os

def test_openai_api_key_valid(openai_client):
    """Test if the OpenAI API key is valid and working properly."""
    try:
        # Make a simple API call to test the key
        start_time = time.time()
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'OpenAI API is working!' if you can read this."}
            ]
        )
        end_time = time.time()
        
        # Extract the assistant's message
        assistant_message = response.choices[0].message.content
        
        # Check that we got a valid response
        assert "working" in assistant_message.lower()
        assert end_time - start_time < 10, "API response took too long"
    
    except Exception as e:
        pytest.fail(f"OpenAI API call failed with error: {str(e)}")

def test_openai_embedding(openai_client):
    """Test OpenAI embedding functionality."""
    try:
        # Create a sample embedding
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input="Sample menu item for embedding test"
        )
        
        # Verify embedding response
        assert hasattr(response, "data")
        assert len(response.data) > 0
        assert hasattr(response.data[0], "embedding")
        
        # Check embedding dimensions (should be 1536 for ada-002)
        embedding = response.data[0].embedding
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
        
    except Exception as e:
        pytest.fail(f"OpenAI embedding creation failed with error: {str(e)}")

def test_openai_with_mock(mock_openai_embedding):
    """Test with a mocked OpenAI client to avoid API calls."""
    # Using the mocked client
    response = mock_openai_embedding.embeddings.create(
        model="text-embedding-ada-002",
        input="This is a test"
    )
    
    # Verify the mocked response
    assert hasattr(response, "data")
    assert len(response.data[0]["embedding"]) == 1536
