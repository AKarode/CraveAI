# Feature vector and semantic vector dimensions
FEATURE_VECTOR_DIM = 5  # Number of features
SEMANTIC_VECTOR_DIM = 384

import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from vector_handling import FeatureExtractor, UserVectorManager
from menu_vectorization import MenuVectorizer
from models import cosine_similarity_model, train_neural_network, rule_based_filter
from dotenv import load_dotenv
from time import sleep
import torch

# Load environment variables
load_dotenv()
firebase_path = os.getenv("FIREBASE_PATH")
# Initialize Firebase
firebase_cred = credentials.Certificate(firebase_path)  # Provide the path to your Firebase credentials
firebase_admin.initialize_app(firebase_cred)
db = firestore.client()

# Initialize necessary components
pinecone_api_key = os.getenv("PINECONE_API_KEY")
feature_extractor = FeatureExtractor()
user_vector_manager = UserVectorManager(pinecone_api_key)
menu_vectorizer = MenuVectorizer(pinecone_api_key)

def fetch_user_preferences(user_id):
    """
    Fetch user preferences such as allergies, dietary restrictions, dislikes, and taste preferences from Firebase.
    """
    #Load survey responses from firebase
    doc_ref = db.collection("surveyResponses").document(user_id)
    doc = doc_ref.get()
    #if the doc is there then load in all the data from the document
    if doc.exists:  # Fixed this line
        data = doc.to_dict()
        allergies = data.get("allergies", [])
        dietary_restrictions = data.get("dietary_restrictions", [])
        dislikes = data.get("dislikes", [])
        spiciness = data.get("spiciness", -1)
        sweetness = data.get("sweetness", -1)
        savory = data.get("savory", -1)
        return allergies, dietary_restrictions, dislikes, spiciness, sweetness, savory
    else:
        return [], [], [], -1, -1, -1  # Return defaults if no data is found

def initialize_chat(user_text, user_id, menu_data):
    """
    Initializes the user vector and vectorizes the menu items for a new chat session.
    """
    # Fetch user preferences from Firebase
    allergies, dietary_restrictions, dislikes, spiciness, sweetness, savory = fetch_user_preferences(user_id)

    # Generate a new chat ID for the session
    chat_id = user_vector_manager.create_chat_id(user_id)

    # Extract features and constraints from the initial user text
    extracted_features = feature_extractor.extract_features_from_text(user_text)
    semantic_vector = feature_extractor.create_semantic_vector(user_text)

    # Combine Firebase preferences with extracted features
    survey_data = {
        "spiciness": spiciness,
        "sweetness": sweetness,
        "savory": savory,
    }
    feature_vector = feature_extractor.create_feature_vector(survey_data, extracted_features)
    #Load user_constraints from text
    user_constraints = feature_extractor.extract_user_constraints(
        user_text, allergies, dietary_restrictions, dislikes
    )

    # Initialize the user vector with the "original" item vector ID
    combined_vector, metadata, vector_id = user_vector_manager.create_user_vector(
        user_id, chat_id, feature_vector, semantic_vector, user_constraints
    )
    metadata["item_vector_id"] = "original"

    # Upload the user vector to Pinecone
    user_vector_manager.upload_to_pinecone(combined_vector, metadata, vector_id)

    # Apply rule-based filtering to the menu items
    filtered_menu = rule_based_filter(menu_data, user_constraints)

    # Vectorize the filtered menu items and upload them to Pinecone
    menu_vectorizer.vectorize_menu(filtered_menu, user_id, chat_id)
    #return chat_id and vector_id for future use
    return chat_id, vector_id

def update_user_vector(user_text, vector_id):
    """
    Updates the user vector in an ongoing chat session.
    """
    # Fetch the original user vector using the provided vector_id
    original_vector = user_vector_manager.index.fetch(ids=[vector_id])['vectors'][vector_id]

    chat_id = original_vector['metadata']['chat_id']

    # Update the user vector with the new user text
    user_vector_manager.update_user_vector(
        vector_id=vector_id, 
        new_query=user_text, 
        chat_id=chat_id
    )


def get_recommendations(user_id, vector_id, chat_id, k):
    # Fetch the original user vector using the provided vector_id
    original_vector = np.array(user_vector_manager.index.fetch(ids=[vector_id])['vectors'][vector_id]['values'])
    
    # Convert the original vector to a list before querying Pinecone
    original_vector_list = original_vector.tolist()
    
    # Fetch all user vectors for the given user_id from Pinecone
    user_vectors = []
    query_response = user_vector_manager.index.query(
        vector=original_vector_list,
        filter={"user_id": {"$eq": user_id}},
        top_k=1000  # Fetch up to 1000 vectors, adjust this if necessary
    )
    
    # Collect all user vectors
    if query_response and 'matches' in query_response:
        for match in query_response['matches']:
            user_vectors.append(np.array(match['values']))
    print(len(user_vectors))
    
    # If there are not enough user vectors, skip neural network and use cosine similarity
    if len(user_vectors) <= k:
        # Use cosine similarity if there aren't enough user vectors
        item_vectors = []
        item_query_response = user_vector_manager.item_index.query(
            vector=original_vector_list,  # Use the original vector to query for similar items
            filter={"chat_id": {"$eq": chat_id}},  # Ensure we only consider items from the current chat session
            top_k=1000,
            include_metadata=True,
            include_values=True
        )
        
        if item_query_response and 'matches' in item_query_response:
            for match in item_query_response['matches']:
                item_vector_values = np.array(match['values'])
                item_vectors.append((item_vector_values, match['metadata']['name'], match['id']))
        print("item_vector number", len(item_vectors))
        scores = [
            (item_name, item_id, cosine_similarity_model(original_vector, item_vector_values))
            for item_vector_values, item_name, item_id in item_vectors
        ]
    else:
        # Use the neural network if there are enough user vectors
        item_vectors = []
        item_query_response = user_vector_manager.item_index.query(
            vector=original_vector_list,  # Use the original vector to query for similar items
            filter={"chat_id": {"$eq": chat_id}},  # Ensure we only consider items from the current chat session
            top_k=1000,
            include_metadata=True,
            include_values=True
        )
        
        if item_query_response and 'matches' in item_query_response:
            for match in item_query_response['matches']:
                print(match)
                item_vector_values = np.array(match['values'])
                item_vectors.append((item_vector_values, match['metadata']['name'], match['id']))
        print("item_vector number", len(item_vectors))
        
        # Ensure that there are item vectors to train the neural network
        if len(item_vectors) > 0:
            item_vector_values = [item_vector_values for item_vector_values, _, _ in item_vectors]
            item_labels = [1] * len(item_vectors)  # Dummy labels; actual labels needed for training

            # Calculate the correct input dimension for the neural network
            combined_vector_dim = original_vector.shape[0] + item_vector_values[0].shape[0]
            print("Combined Vector Dimension:", combined_vector_dim)
            # Pass the correct input dimension to the model
            model = train_neural_network(user_vectors, item_vector_values, item_labels, epochs=10)

            scores = []
            for item_vector_values, item_name, item_id in item_vectors:
                combined_vector = np.concatenate((original_vector, item_vector_values))

                # Create a tensor of shape [778]
                combined_tensor = torch.tensor(combined_vector, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                
                #separate tensor
                # Assuming combined_vector is of size [778] (i.e., concatenation of user_vector and item_vector)
                user_part = combined_tensor[:, :389]   # First part of the tensor
                item_part = combined_tensor[:, 389:]   # Second part of the tensor

                # Stack these parts into a tensor of shape [2, 389]
                stacked_tensor = torch.stack((user_part, item_part)).unsqueeze(0)

                # Get the score from the model
                output = model(stacked_tensor)
                score = output.mean().item()
                scores.append((item_name, item_id, score))
        else:
            scores = []

    # Sort by score in descending order and return the top-k
    scores.sort(key=lambda x: x[2], reverse=True)
    top_items = [item_id for _, item_id, _ in scores[:k]]

    return top_items








def create_and_upload_copies(user_id, vector_id, k, top_items):
    """
    Creates copies of the original user vector with references to the top-k recommended item vectors.
    Returns a list of vector_ids for each copy created.
    """
    # Fetch the original user vector
    original_vector = user_vector_manager.index.fetch(ids=[vector_id])['vectors'][vector_id]

    # List to store the vector_ids of the copies
    created_vector_ids = []

    # Create and upload copies of the original vector with top-k item vectors
    for item_vector_id in top_items[:k]:
        new_vector_id = user_vector_manager.create_unique_vector_id(user_id)
        new_metadata = original_vector['metadata'].copy()
        new_metadata['vector_id'] = new_vector_id
        new_metadata['item_vector_id'] = item_vector_id

        # Upload the new vector to Pinecone
        user_vector_manager.upload_to_pinecone(
            combined_vector=np.array(original_vector['values']),  # Using the original vector values
            metadata=new_metadata,
            vector_id=new_vector_id
        )

        # Append the new vector_id to the list
        created_vector_ids.append(new_vector_id)

    return created_vector_ids


def update_user_vector_with_feedback(vector_ids, item_name, selected, liked):
    """
    Iterates through the list of vector_ids to find the vector pointing to the specified item,
    then updates that vector with feedback on selected and liked items.
    """
    for vector_id in vector_ids:
        # Fetch the vector metadata to get the item_vector_id
        vector_data = user_vector_manager.index.fetch(ids=[vector_id])['vectors'][vector_id]
        item_vector_id = vector_data['metadata']['item_vector_id']

        # Fetch the item vector using item_vector_id
        item_vector_data = user_vector_manager.item_index.fetch(ids=[item_vector_id])['vectors'][item_vector_id]

        # Check if the item vector's name matches the specified item_name
        if item_vector_data['metadata']['name'] == item_name:
            # Update the vector in vector-history with the feedback
            user_vector_manager.update_user_vector(
                vector_id=vector_id,
                new_query="",  # No new query for feedback update
                chat_id=vector_data['metadata']['chat_id'],
                selected=selected,
                liked=liked
            )
            return  # Exit after finding and updating the correct vector

    raise ValueError(f"No vector found pointing to the item '{item_name}'")

def get_item_name_by_vector_id(item_vector_id):
    """
    Retrieves the item name associated with a specific item vector ID.

    Args:
        item_vector_id (str): The ID of the item vector.

    Returns:
        str: The name of the item associated with the vector ID.
    """
    # Query the item vector storage (e.g., Pinecone) to fetch the vector by its ID
    item_vector_data = user_vector_manager.item_index.fetch(ids=[item_vector_id])
    
    # Check if the vector data was successfully retrieved
    if item_vector_data and 'vectors' in item_vector_data:
        # Extract the metadata associated with the vector
        item_metadata = item_vector_data['vectors'][item_vector_id]['metadata']
        
        # Return the item name from the metadata
        return item_metadata.get('name', 'Unknown Item Name')
    
    return 'Item not found'

# Example usage:




# Your provided code with explanations/comments
# This code seems correct and should work as intended.

# Example usage
if __name__ == "__main__":
    user_id = "user125"
    
    # Initialize a new chat session
    chat_id, vector_id = initialize_chat(
        user_text="I want something spicy and vegetarian.",
        user_id=user_id,
        menu_data=[
            {"name": "spicy_pasta", "description": "pasta with spicy tomato sauce", "price": "$15"},
            {"name": "veggie_burger", "description": "burger with black bean patty", "price": "$10"}
        ]
    )
    sleep(1)
    # Later, update the user vector in the ongoing chat session
    update_user_vector(
        user_text="I don't like too much sweetness.",
        vector_id=vector_id
    )

    # Get recommendations
    top_items = get_recommendations(user_id=user_id, vector_id=vector_id, chat_id=chat_id, k=3)
    print("top item :", top_items[0])
    topitem = top_items[0]

    item_name = get_item_name_by_vector_id(topitem)
    print(item_name)
    # Create and upload copies
    new_vector_ids = create_and_upload_copies(user_id=user_id, vector_id=vector_id, k=3, top_items=top_items)
    sleep(2)
    # Example of updating the recommendation with feedback
    update_user_vector_with_feedback(vector_ids=new_vector_ids, item_name="spicy_pasta", selected=1, liked=1)


