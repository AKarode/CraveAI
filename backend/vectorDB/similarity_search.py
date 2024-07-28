import pinecone
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

# Initialize the Sentence-BERT model
text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=api_key)

# Connect to the indexes
palette_index_name = 'palettes'
food_index_name = 'unique-food-recommendations'

palette_index = pc.Index(palette_index_name)
food_index = pc.Index(food_index_name)

def compute_similarity(vector1, vector2):
    """Compute the cosine similarity between two vectors."""
    return 1 - cosine(vector1, vector2)

def find_top_matching_items(user_id, recent_query, top_k=3):
    """Find the top matching restaurant items for a user based on their palette vector and a recent query."""
    # Fetch the original palette vector
    original_vector = palette_index.fetch(ids=[user_id])['vectors'][user_id]
    original_values = np.array(original_vector['values'])

    # Vectorize the recent query
    query_vector_values = text_model.encode(recent_query)

    # Combine the palette vector with the query vector (e.g., averaging)
    combined_vector = (original_values + query_vector_values) / 2

    # Retrieve all vectors from the food index
    query_results = food_index.query(vector=combined_vector.tolist(), top_k=1000, include_values=True, include_metadata=True)
    all_vectors = query_results['matches']

    # Compute similarity scores
    similarities = []
    for match in all_vectors:
        item_id = match['id']
        item_values = np.array(match['values'])
        item_metadata = match['metadata']
        similarity = compute_similarity(combined_vector, item_values)
        similarities.append((item_id, similarity, item_metadata))

    # Sort by similarity and get the top K results
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:top_k]
    return top_matches

# Example usage:
if __name__ == "__main__":
    user_id = "user_124"
    recent_query = "I want sweet"
    top_items = find_top_matching_items(user_id, recent_query, top_k=3)

    for i, item in enumerate(top_items, start=1):
        print(f"Top {i} matching item ID: {item[0]}")
        print(f"Similarity: {item[1]}")
        print(f"Metadata: {item[2]}")
