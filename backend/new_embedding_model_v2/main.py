import os
import numpy as np
from vector_handling import FeatureExtractor, UserVectorManager
from menu_vectorization import MenuVectorizer
from models import cosine_similarity_model, train_neural_network, rule_based_filter
from dotenv import load_dotenv
from time import sleep

# Load environment variables
load_dotenv()

# Initialize necessary components
pinecone_api_key = os.getenv("PINECONE_API_KEY")
feature_extractor = FeatureExtractor()
user_vector_manager = UserVectorManager(pinecone_api_key)
menu_vectorizer = MenuVectorizer(pinecone_api_key)

def initialize_chat(user_text, user_id, menu_data, chat_id=None):
    """
    Initializes the user vector and vectorizes the menu items for a new chat session.
    If `chat_id` is None, a new chat session is started.
    """
    # If no chat_id is provided, generate a new one (new chat session)
    if chat_id is None:
        chat_id = user_vector_manager.create_chat_id(user_id)

        # Extract user constraints from survey responses (assumed to be fetched from a database)
        allergens = ["peanuts"]  # Example
        dietary_restrictions = ["vegan"]  # Example
        dislikes = []  # Example

        # Extract features and constraints from the initial user text
        extracted_features = feature_extractor.extract_features_from_text(user_text)
        semantic_vector = feature_extractor.create_semantic_vector(user_text)
        user_constraints = feature_extractor.extract_user_constraints(
            user_text, allergens, dietary_restrictions, dislikes
        )

        # Initialize the user vector with the "original" item vector ID
        combined_vector, metadata, vector_id = user_vector_manager.create_user_vector(
            user_id, chat_id, extracted_features, semantic_vector, user_constraints
        )
        metadata["item_vector_id"] = "original"

        # Upload the user vector to Pinecone
        user_vector_manager.upload_to_pinecone(combined_vector, metadata, vector_id)

        # Apply rule-based filtering to the menu items
        filtered_menu = rule_based_filter(menu_data, user_constraints)

        # Vectorize the filtered menu items and upload them to Pinecone
        menu_vectorizer.vectorize_menu(filtered_menu, user_id)

    else:  # Ongoing chat session, update the existing user vector
        # Fetch the original user vector based on the chat_id
        original_vector_id = None
        original_vector = None
        for vector in user_vector_manager.index.query(
            vector_query=f"user_id:{user_id} AND chat_id:{chat_id} AND item_vector_id:original", top_k=1
        )['matches']:
            original_vector_id = vector["id"]
            original_vector = np.array(vector['values'])

        if not original_vector_id:
            raise ValueError("No original vector found for this chat_id")

        # Update the user vector with the new user text
        user_vector_manager.update_user_vector(
            original_vector_id, user_text, chat_id
        )

    return chat_id

def get_recommendations(user_id, chat_id, k):
    """
    Fetches top-k recommendations based on similarity scores using cosine similarity or a neural network.
    """
    # Fetch the original user vector
    original_vector_id = None
    original_vector = None
    user_vectors = []

    for vector in user_vector_manager.index.query(
        vector_query=f"user_id:{user_id} AND chat_id:{chat_id} AND item_vector_id:original", top_k=1
    )['matches']:
        original_vector_id = vector["id"]
        original_vector = np.array(vector['values'])
        user_vectors.append(original_vector)

    if not original_vector_id:
        raise ValueError("No original vector found for this chat_id")

    # Fetch item vectors for the same chat_id from Pinecone
    item_vectors = []
    for item_vector in user_vector_manager.item_index.query(
        vector_query=f"chat_id:{chat_id} AND selected:0", top_k=1000
    )['matches']:
        item_vector_values = np.array(item_vector['values'])
        item_vectors.append((item_vector_values, item_vector['metadata']['name'], item_vector['id']))

    # Choose between cosine similarity and neural network based on the number of user vectors
    if len(user_vectors) < k:
        # Use cosine similarity if there aren't enough user vectors
        scores = [
            (item_name, item_id, cosine_similarity_model(original_vector, item_vector_values))
            for item_vector_values, item_name, item_id in item_vectors
        ]
    else:
        # Use the neural network if there are enough user vectors
        item_vector_values = [item_vector_values for item_vector_values, _, _ in item_vectors]
        item_labels = [1] * len(item_vectors)  # Dummy labels; actual labels needed for training

        model = train_neural_network(user_vectors, item_vector_values, item_labels, input_dim=FEATURE_VECTOR_DIM + SEMANTIC_VECTOR_DIM)
        scores = []
        for item_vector_values, item_name, item_id in item_vectors:
            combined_vector = np.concatenate((original_vector, item_vector_values))
            score = model(torch.tensor(combined_vector, dtype=torch.float32)).item()
            scores.append((item_name, item_id, score))

    # Sort by score in descending order and return the top-k
    scores.sort(key=lambda x: x[2], reverse=True)
    top_items = [item_id for _, item_id, _ in scores[:k]]

    return top_items

def create_and_upload_copies(user_id, chat_id, k, top_items):
    """
    Creates copies of the original user vector with references to the top-k recommended item vectors.
    """
    # Fetch the original user vector
    original_vector_id = None
    for vector in user_vector_manager.index.query(
        vector_query=f"user_id:{user_id} AND chat_id:{chat_id} AND item_vector_id:original", top_k=1
    )['matches']:
        original_vector_id = vector["id"]

    if not original_vector_id:
        raise ValueError("No original vector found for this chat_id")

    original_vector = user_vector_manager.index.fetch(ids=[original_vector_id])['vectors'][original_vector_id]

    # Create and upload copies of the original vector with top-k item vectors
    for item_vector_id in top_items[:k]:
        new_vector_id = user_vector_manager.create_unique_vector_id(user_id, chat_id)
        new_metadata = original_vector['metadata'].copy()
        new_metadata['vector_id'] = new_vector_id
        new_metadata['item_vector_id'] = item_vector_id

        # The actual vector values remain the same as the original vector except for the metadata
        user_vector_manager.upload_to_pinecone(
            combined_vector=np.array(original_vector['values']),  # Using the original vector values
            metadata=new_metadata,
            vector_id=new_vector_id
        )

def update_user_vector_with_feedback(chat_id, item_name, selected, liked):
    """
    Updates the user vector with feedback on selected and liked items.
    """
    # Fetch the item vector ID based on the item name and chat ID
    item_vector_id = None
    for vector in user_vector_manager.item_index.query(
        vector_query=f"name:{item_name} AND chat_id:{chat_id}", top_k=1
    )['matches']:
        item_vector_id = vector["id"]

    if not item_vector_id:
        raise ValueError("No item vector found for this chat_id and item_name")

    # Fetch the user vector corresponding to the item vector
    user_vector_id = None
    for vector in user_vector_manager.index.query(
        vector_query=f"chat_id:{chat_id} AND item_vector_id:{item_vector_id}", top_k=1
    )['matches']:
        user_vector_id = vector["id"]

    if not user_vector_id:
        raise ValueError("No user vector found for this chat_id and item_vector_id")

    # Update the user vector with the feedback
    user_vector_manager.update_user_vector(user_vector_id, "", chat_id, selected=selected, liked=liked)

if __name__ == "__main__":
    # Example menu data
    menu_data = [
        {"name": "chicken sandwich", "price": "$20", "description": "yummy chicken fillet with bread"},
        {"name": "veggie burger", "price": "$15", "description": "black bean patty with avocado and tomato"},
        {"name": "pasta", "price": "$25", "description": "creamy alfredo sauce with grilled chicken and broccoli"}
    ]

    # Example: Initialize the chat
    user_id = "user125"
    user_text = "I am allergic to peanuts."
    chat_id = initialize_chat(user_text, user_id, menu_data)
    sleep(1)

    # Example: Get top-k recommendations
    k = 3
    top_items = get_recommendations(user_id, chat_id, k)
    print(f"Top {k} items:", top_items)

    # Example: Create and upload copies of the original vector with top-k item vectors
    create_and_upload_copies(user_id, chat_id, k, top_items)

    # Example: Update user vector with feedback
    update_user_vector_with_feedback(chat_id, "chicken sandwich", selected=1, liked=1)
