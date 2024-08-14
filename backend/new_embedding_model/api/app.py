from flask import Flask, request, jsonify
from models.sentence_transformer import SentenceTransformerModel
from preprocessing.feature_extraction import FeatureExtractor
from preprocessing.menu_vectorization import vectorize_menu_item
from matching.vector_combination import combine_vectors
from matching.similarity_scoring import rank_menu_items
from storage.pinecone_manager import PineconeManager
from storage.storage_utils import prepare_user_data_for_storage

import os

app = Flask(__name__)

# Initialize models and Pinecone
sentence_model = SentenceTransformerModel()
feature_extractor = FeatureExtractor()
pinecone_manager = PineconeManager(api_key=os.getenv("PINECONE_API_KEY"), index_name="user-preferences", dimension=384 + 5)  # Example dimensions

@app.route('/process_menu', methods=['POST'])
def process_menu():
    try:
        data = request.json
        user_id = data['user_id']
        query_text = data['query']

        # Vectorize the query
        semantic_vector = sentence_model.encode(query_text)
        explicit_features = feature_extractor.extract_features(data['survey_data'])
        combined_vector = combine_vectors(semantic_vector, explicit_features)

        # Store the user vector in Pinecone
        user_vector_data = prepare_user_data_for_storage(user_id, combined_vector)
        pinecone_manager.upsert_vectors([user_vector_data])

        # Query the menu items
        menu_vectors = pinecone_manager.query_vector(combined_vector)
        best_matches = rank_menu_items(combined_vector, menu_vectors)

        return jsonify({"user_id": user_id, "best_matches": best_matches})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        user_id = data['user_id']

        # Fetch the user vector from Pinecone
        user_vector = pinecone_manager.fetch_vector(user_id)['vectors'][user_id]['values']

        # Query menu items to find the best match
        menu_vectors = pinecone_manager.query_vector(user_vector)
        best_matches = rank_menu_items(user_vector, menu_vectors)

        return jsonify({"user_id": user_id, "recommendations": best_matches})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
