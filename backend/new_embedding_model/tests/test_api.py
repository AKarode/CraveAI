import unittest
import api.app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_process_menu(self):
        response = self.app.post('/process_menu', json={
            'user_id': 'user_123',
            'query': 'I want something spicy and cheap',
            'survey_data': {"spicy": 3, "budget": 2}
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("best_matches", response.get_json())

    def test_get_recommendations(self):
        response = self.app.post('/get_recommendations', json={
            'user_id': 'user_123'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("recommendations", response.get_json())

if __name__ == '__main__':
    unittest.main()
