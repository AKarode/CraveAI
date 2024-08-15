import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from storage.pinecone_manager import PineconeManager
from storage.storage_utils import prepare_user_data_for_storage
from pinecone.core.client.model.fetch_response import FetchResponse

from dotenv import load_dotenv
import numpy as np
load_dotenv()
class TestPineconeManager(unittest.TestCase):
    def setUp(self):
        self.manager = PineconeManager(api_key=os.getenv("PINECONE_API_KEY"), index_name="test-index", dimension=384)

    def test_upsert_vectors(self):
        vector_data = {"user_123": np.random.rand(384).tolist()}
        prepared_data = prepare_user_data_for_storage("user_123", vector_data["user_123"])
        response = self.manager.upsert_vectors([prepared_data])
        
        self.assertIsNotNone(response)

    def test_fetch_vector(self):
        response = self.manager.fetch_vector('user_123')
        
        # Check if the response is an instance of FetchResponse
        self.assertIsInstance(response, FetchResponse)
        
        # Optionally, if you want to check the content of the response
        vectors = response.vectors
        self.assertIsInstance(vectors, dict)
        self.assertIn('user_123', vectors)

if __name__ == '__main__':
    unittest.main()
