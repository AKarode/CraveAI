import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from models.sentence_transformer import SentenceTransformerModel
from preprocessing.feature_extraction import FeatureExtractor
from preprocessing.menu_vectorization import vectorize_menu_item
from matching.vector_combination import combine_vectors
from matching.similarity_scoring import rank_menu_items
from storage.pinecone_manager import PineconeManager
from storage.storage_utils import prepare_user_data_for_storage
from dotenv import load_dotenv
import numpy as np
load_dotenv()
app = Flask(__name__)

# Initialize models and Pinecone
sentence_model = SentenceTransformerModel()
feature_extractor = FeatureExtractor()
pinecone_manager = PineconeManager(api_key=os.getenv("PINECONE_API_KEY"), index_name="user-preferences", dimension=386)  # Adjust dimension as needed

@app.route('/process_menu', methods=['POST'])
def process_menu():
    try:
        data = request.json
        user_id = data['user_id']
        query_text = data['query']

        # Vectorize the query
        semantic_vector = sentence_model.encode(query_text)
        explicit_features = feature_extractor.extract_features(data['survey_data'])

        # Combine the vectors
        combined_vector = combine_vectors(semantic_vector, explicit_features)

        # Convert the combined vector to a list for serialization
        combined_vector_list = combined_vector.tolist()

        # Store the user vector in Pinecone
        user_vector_data = prepare_user_data_for_storage(user_id, combined_vector_list)
        pinecone_manager.upsert_vectors([user_vector_data])
        
        # Query the menu items
        menu_vectors = pinecone_manager.query_vector(combined_vector_list, top_k=1000)
        best_matches = rank_menu_items(combined_vector_list, menu_vectors)
        print("here\n")
        return jsonify({"user_id": user_id, "best_matches": best_matches})

    except Exception as e:
        app.logger.error(f"Error processing menu: {e}")
        return jsonify({"error": str(e)}), 500





@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        user_id = data['user_id']

        # Fetch the user vector from Pinecone
        response = pinecone_manager.fetch_vector(user_id)
        user_vector = response['vectors'].get(user_id)
        if user_vector:
            user_vector_values = user_vector['values']

            # Convert to list if needed
            if isinstance(user_vector_values, np.ndarray):
                user_vector_values = user_vector_values.tolist()

            # Query menu items to find the best match
            menu_vectors = pinecone_manager.query_vector(user_vector_values, top_k=1000)
            best_matches = rank_menu_items(user_vector_values, menu_vectors)

            return jsonify({"user_id": user_id, "recommendations": best_matches})
        else:
            return jsonify({"error": f"User vector for {user_id} not found."}), 404

    except Exception as e:
        app.logger.error(f"Error getting recommendations: {e}")
        return jsonify({"error": str(e)}), 500





if __name__ == '__main__':
    app.run(debug=True)

