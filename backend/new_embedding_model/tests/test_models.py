import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'new_embedding_model')))

import unittest
from models.sentence_transformer import SentenceTransformerModel

class TestSentenceTransformerModel(unittest.TestCase):
    def setUp(self):
        self.model = SentenceTransformerModel()

    def test_encode(self):
        sample_text = "This is a test sentence."
        vector = self.model.encode(sample_text)
        self.assertEqual(len(vector), 384)  # Assuming 384 dimensions
        self.assertTrue(isinstance(vector, (list, np.ndarray)))

if __name__ == '__main__':
    unittest.main()
