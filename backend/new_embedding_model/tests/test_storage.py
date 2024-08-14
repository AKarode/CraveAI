import unittest
from storage.pinecone_manager import PineconeManager
from storage.storage_utils import prepare_user_data_for_storage
import numpy as np

class TestPineconeManager(unittest.TestCase):
    def setUp(self):
        self.manager = PineconeManager(api_key="fake_api_key", index_name="test-index", dimension=384)

    def test_upsert_vectors(self):
        vector_data = {"user_123": np.random.rand(384).tolist()}
        prepared_data = prepare_user_data_for_storage("user_123", vector_data["user_123"])
        response = self.manager.upsert_vectors([prepared_data])
        self.assertIsNotNone(response)

    def test_fetch_vector(self):
        vector_id = "user_123"
        response = self.manager.fetch_vector(vector_id)
        self.assertIsInstance(response, dict)

if __name__ == '__main__':
    unittest.main()
