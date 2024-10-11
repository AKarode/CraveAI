# Feature vector and semantic vector dimensions
FEATURE_VECTOR_DIM = 5  # Number of features
SEMANTIC_VECTOR_DIM = 384

# Import necessary modules
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from vector_handling import FeatureExtractor, UserVectorManager
from menu_vectorization import MenuVectorizer
from models import cosine_similarity_model, train_neural_network, rule_based_filter
from dotenv import load_dotenv
import torch
from time import sleep

# Load environment variables from the .env file
load_dotenv()
firebase_path = os.getenv("FIREBASE_PATH")

# Initialize Firebase with credentials
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
    # Load survey responses from Firebase
    doc_ref = db.collection("surveyResponses").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        # Extract relevant user preferences from the Firebase document
        allergies = data.get("allergies", [])
        dietary_restrictions = data.get("dietary_restrictions", [])
        dislikes = data.get("dislikes", [])
        spiciness = data.get("spiciness", -1)
        sweetness = data.get("sweetness", -1)
        savory = data.get("savory", -1)
        return allergies, dietary_restrictions, dislikes, spiciness, sweetness, savory
    else:
        # Return default values if the document is not found
        return [], [], [], -1, -1, -1

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

    # Extract user constraints based on user input and preferences
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

    # Return chat_id and vector_id for future use
    return chat_id, vector_id

def update_user_vector(user_text, vector_id):
    """
    Updates the user vector in an ongoing chat session with new user input.
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
    """
    Generate item recommendations based on the user vector and historical interactions.
    """
    # Fetch the original user vector using the provided vector_id
    original_vector = np.array(user_vector_manager.index.fetch(ids=[vector_id])['vectors'][vector_id]['values'])
    original_vector_list = original_vector.tolist()

    # Fetch all user vectors for the given user_id from Pinecone and filter based on feedback
    query_response = user_vector_manager.index.query(
        vector=original_vector_list,
        filter={"user_id": {"$eq": user_id}},
        top_k=1000,
        include_metadata=True,
        include_values=True
    )
    
    # Collect only user vectors with valid feedback (selected != -1)
    valid_user_vectors = [
        np.array(match['values']) for match in query_response.get('matches', [])
        if match['metadata'].get('selected') != -1
    ]

    # If there are not enough valid user vectors, use cosine similarity instead of the neural network
    if len(valid_user_vectors) <= k:
        item_vectors = []
        item_query_response = user_vector_manager.item_index.query(
            vector=original_vector_list,
            filter={"chat_id": {"$eq": chat_id}},
            top_k=1000,
            include_metadata=True,
            include_values=True
        )
        
        if item_query_response and 'matches' in item_query_response:
            for match in item_query_response['matches']:
                item_vector_values = np.array(match['values'])
                item_vectors.append((item_vector_values, match['metadata']['name'], match['id']))

        # Calculate cosine similarity scores for items
        scores = [
            (item_name, item_id, cosine_similarity_model(original_vector, item_vector_values))
            for item_vector_values, item_name, item_id in item_vectors
        ]
    else:
        # Use the neural network if there are enough valid user vectors
        item_vectors = []
        item_query_response = user_vector_manager.item_index.query(
            vector=original_vector_list,
            filter={"chat_id": {"$eq": chat_id}},
            top_k=1000,
            include_metadata=True,
            include_values=True
        )
        
        if item_query_response and 'matches' in item_query_response:
            for match in item_query_response['matches']:
                item_vector_values = np.array(match['values'])
                item_vectors.append((item_vector_values, match['metadata']['name'], match['id']))
        
        if len(item_vectors) > 0:
            # Prepare data for neural network training
            training_user_vectors = []
            training_item_vectors = []
            item_labels = []

            for match in query_response.get('matches', []):
                if match['metadata'].get('selected') != -1:
                    user_vector = np.array(match['values'])
                    training_user_vectors.append(user_vector)
                    
                    item_vector_id = match['metadata'].get('item_vector_id')
                    item_vector_response = user_vector_manager.item_index.fetch(ids=[item_vector_id])
                    item_vector = np.array(item_vector_response['vectors'][item_vector_id]['values'])
                    training_item_vectors.append(item_vector)

                    liked = match['metadata'].get('liked')
                    selected = match['metadata'].get('selected')
                    label = liked if liked != -1 else selected
                    item_labels.append(label)

            # Train the neural network with user vectors and item vectors
            model = train_neural_network(training_user_vectors, training_item_vectors, item_labels, epochs=10)

            # Score each item vector using the neural network
            scores = []
            for item_vector_values, item_name, item_id in item_vectors:
                combined_vector = np.concatenate((original_vector, item_vector_values))
                combined_tensor = torch.tensor(combined_vector, dtype=torch.float32).unsqueeze(0)
                
                # Separate tensor into user and item parts for neural network input
                user_part = combined_tensor[:, :389]
                item_part = combined_tensor[:, 389:]
                stacked_tensor = torch.stack((user_part, item_part)).unsqueeze(0)

                # Get the score from the model
                output = model(combined_tensor)
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
            combined_vector=np.array(original_vector['values']),
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
    user_id = "user130"
    
    # Initialize a new chat session
    chat_id, vector_id = initialize_chat(
        user_text="Reccomend me something that is vegetarian and savory and something spicy.",
        user_id=user_id,
        menu_data=[
    {"name": "Classic Cheeseburger", "description": "A juicy beef patty topped with melted cheddar cheese, lettuce, tomato, pickles, and onions on a toasted sesame seed bun. Served with a side of crispy fries.", "price": "$10.99"},
    {"name": "Grilled Chicken Caesar Salad", "description": "Grilled chicken breast served over a bed of fresh romaine lettuce, tossed with creamy Caesar dressing, Parmesan cheese, and croutons.", "price": "$9.99"},
    {"name": "Margarita Pizza", "description": "A classic pizza topped with fresh mozzarella cheese, tomatoes, basil, and a drizzle of olive oil on a crispy thin crust.", "price": "$12.49"},
    {"name": "Vegan Buddha Bowl", "description": "A nutritious bowl filled with quinoa, roasted chickpeas, avocado, sweet potatoes, kale, and a tahini dressing.", "price": "$11.99"},
    {"name": "Spaghetti Carbonara", "description": "Spaghetti pasta tossed in a creamy sauce made with eggs, Parmesan cheese, pancetta, and black pepper.", "price": "$13.49"},
    {"name": "Buffalo Wings", "description": "Spicy buffalo chicken wings served with celery sticks and a side of blue cheese dipping sauce.", "price": "$8.99"},
    {"name": "Vegetable Stir Fry", "description": "A mix of stir-fried vegetables including broccoli, bell peppers, carrots, and snap peas, tossed in a savory soy sauce. Served with steamed rice.", "price": "$10.49"},
    {"name": "Lamb Gyro", "description": "Sliced lamb, tomatoes, onions, and tzatziki sauce wrapped in a warm pita bread.", "price": "$9.49"},
    {"name": "Seafood Paella", "description": "A traditional Spanish dish with saffron rice, shrimp, mussels, clams, and chorizo, cooked in a rich seafood broth.", "price": "$16.99"},
    {"name": "Beef Tacos", "description": "Soft corn tortillas filled with seasoned ground beef, topped with lettuce, cheese, salsa, and sour cream.", "price": "$7.99"},
    {"name": "Mushroom Risotto", "description": "Creamy risotto with sautéed mushrooms, Parmesan cheese, and a hint of truffle oil.", "price": "$14.99"},
    {"name": "BBQ Ribs", "description": "Slow-cooked pork ribs smothered in a tangy BBQ sauce, served with coleslaw and cornbread.", "price": "$18.99"},
    {"name": "Fish and Chips", "description": "Crispy beer-battered fish fillets served with golden fries and a side of tartar sauce.", "price": "$13.99"},
    {"name": "Eggplant Parmesan", "description": "Layers of breaded eggplant, marinara sauce, and melted mozzarella cheese, baked to perfection. Served with a side of garlic bread.", "price": "$12.99"},
    {"name": "Chicken Alfredo Pasta", "description": "Fettuccine pasta in a rich and creamy Alfredo sauce with grilled chicken breast and fresh parsley.", "price": "$14.49"},
    {"name": "Shrimp Scampi", "description": "Succulent shrimp sautéed in garlic, butter, and white wine, served over a bed of linguine.", "price": "$15.49"},
    {"name": "Pancakes with Maple Syrup", "description": "Fluffy buttermilk pancakes served with a generous drizzle of maple syrup and a side of fresh berries.", "price": "$7.99"},
    {"name": "Pepperoni Pizza", "description": "Classic pizza with a generous topping of pepperoni slices, melted mozzarella, and a rich tomato sauce on a crispy crust.", "price": "$12.49"},
    {"name": "Greek Salad", "description": "A refreshing salad with cucumbers, tomatoes, olives, feta cheese, and red onions, tossed in a lemon-oregano dressing.", "price": "$8.49"},
    {"name": "Chocolate Lava Cake", "description": "A warm, gooey chocolate cake with a molten center, served with a scoop of vanilla ice cream.", "price": "$6.99"}
]
    )
    sleep(1)
    # Later, update the user vector in the ongoing chat session
    
    update_user_vector(
        user_text="I want something spicy",
        vector_id=vector_id
    )
    
    # Get recommendations
    top_items = get_recommendations(user_id=user_id, vector_id=vector_id, chat_id=chat_id, k=3)
    print("top item :", top_items[0])
    topitem = top_items
    itemnames = []
    for item in topitem:
        item_name = get_item_name_by_vector_id(item)
        print(item_name)
        itemnames.append(item_name)
    # Create and upload copies
    new_vector_ids = create_and_upload_copies(user_id=user_id, vector_id=vector_id, k=3, top_items=top_items)
    sleep(2)
    # Example of updating the recommendation with feedback
    update_user_vector_with_feedback(vector_ids=new_vector_ids, item_name=itemnames[0], selected=1, liked=1)


