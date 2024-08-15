import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from preprocessing.feature_extraction import FeatureExtractor
from preprocessing.text_processing import TextProcessor

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_extract_features(self):
        survey_data = {"spicy": 3, "budget": 2}
        vector = self.extractor.extract_features(survey_data)
        self.assertEqual(len(vector), 2)
        self.assertAlmostEqual(sum(vector), 1.0, places=5)  # If scaling is applied

class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor()

    def test_clean_text(self):
        text = "  This is a TEST!!  "
        cleaned_text = self.processor.clean_text(text)
        self.assertEqual(cleaned_text, "this is a test")

if __name__ == '__main__':
    unittest.main()
