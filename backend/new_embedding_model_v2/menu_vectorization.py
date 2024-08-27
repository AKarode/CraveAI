import numpy as np
from vector_handling import FeatureExtractor, UserVectorManager
from dotenv import load_dotenv
import os
import uuid

# Feature vector and semantic vector dimensions
FEATURE_VECTOR_DIM = 5  # Number of features
SEMANTIC_VECTOR_DIM = 384

class MenuVectorizer:
    def __init__(self, pinecone_api_key):
        # Initialize the feature extractor and user vector manager
        self.feature_extractor = FeatureExtractor()
        self.user_vector_manager = UserVectorManager(pinecone_api_key)

    def generate_chat_id(self, user_id):
        """Generate a unique chat ID."""
        return f"{user_id}_{uuid.uuid4()}"

    def process_menu_item(self, item):
        """
        Process a single menu item to create its feature and semantic vectors.
        
        Args:
            item (dict): Dictionary containing details of a menu item.
                         Example: {"name": "chicken_sandwich", "description": "spicy chicken fillet", "price": "$20"}
                         
        Returns:
            np.ndarray: Combined feature and semantic vector for the menu item.
        """
        # Extract features from the description and price
        extracted_features = self.feature_extractor.extract_features_from_text(item["description"])
        extracted_features["budget"] = self.feature_extractor.determine_budget_value(item["price"])
        
        # Create the feature vector
        feature_vector = self.feature_extractor.create_feature_vector({}, extracted_features)
        
        # Create the semantic vector from the description and name
        semantic_input = f"{item['name']} {item['description']}"
        semantic_vector = self.feature_extractor.create_semantic_vector(semantic_input)
        
        # Combine the feature vector and semantic vector into a single vector
        combined_vector = np.concatenate((feature_vector, semantic_vector))
        
        return combined_vector

    def vectorize_menu(self, menu_items, user_id, chat_id):
        """
        Vectorize an array of menu items and upload them to Pinecone.
        
        Args:
            menu_items (list): List of dictionaries, where each dictionary contains details of a menu item.
                               Example: [{"name": "chicken_sandwich", "description": "spicy chicken fillet", "price": "$20"}]
            user_id (str): The ID of the user, used for generating a unique chat_id.
            chat_id (str): The unique chat session ID.
        """
        for item in menu_items:
            # Process each menu item to generate its combined vector
            combined_vector = self.process_menu_item(item)
            # Upload the item vector to Pinecone with the name of the item and chat_id
            self.user_vector_manager.upload_item_vector(combined_vector, item_name=item["name"], chat_id=chat_id)

if __name__ == "__main__":
    # Load environment variables from the .env file
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    # Initialize the MenuVectorizer
    menu_vectorizer = MenuVectorizer(pinecone_api_key)

    # Example menu items
    menu_items = [
        {"name": "chicken_sandwich", "description": "spicy chicken fillet with bread and lettuce", "price": "$20"},
        {"name": "veggie_burger", "description": "black bean patty with avocado and tomato", "price": "$15"},
        {"name": "pasta", "description": "creamy alfredo sauce with grilled chicken and broccoli", "price": "$25"}
    ]

    # Example user ID
    user_id = "user125"

    # Example chat ID
    chat_id = menu_vectorizer.generate_chat_id(user_id)

    # Vectorize the menu items and upload them to Pinecone
    menu_vectorizer.vectorize_menu(menu_items, user_id, chat_id)
