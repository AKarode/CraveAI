#Make it so that we don't give each user its own index due to pinecones restrictions
#Look into making user_constraints better

# Feature vector and semantic vector dimensions
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone
import spacy
import uuid
import random
from dotenv import load_dotenv
import os
from time import sleep

# Feature vector and semantic vector dimensions
FEATURE_VECTOR_DIM = 5  # Number of features
SEMANTIC_VECTOR_DIM = 384

# Import necessary modules
import os
import numpy as np
import pinecone
import spacy
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from time import sleep
import random
import uuid

# Constants for feature and semantic vector dimensions
FEATURE_VECTOR_DIM = 5  # Number of features
SEMANTIC_VECTOR_DIM = 384

class FeatureExtractor:
    def __init__(self):
        # Initialize the SentenceTransformer model for semantic analysis
        self.text_model = SentenceTransformer('all-MiniLM-L12-v2')
        # Load spaCy model for Named Entity Recognition (NER) and text processing
        self.nlp = spacy.load("en_core_web_sm")

    def determine_budget_value(self, budget_text):
        """
        Determine the budget value based on the input text.
        
        Args:
            budget_text (str): The budget description (e.g., "under $20").
        
        Returns:
            int: Budget value derived from the text.
        """
        doc = self.nlp(budget_text.lower())
        budget_value = -1

        for ent in doc.ents:
            if ent.label_ == "MONEY":
                amount = float(ent.text.replace("$", "").replace(",", "").strip())
                if "under" in budget_text:
                    budget_value = amount // 10  # Example: under 20 -> 2
                elif "over" in budget_text:
                    budget_value = (amount // 10) + 1  # Example: over 20 -> 3
                elif "around" in budget_text or "about" in budget_text:
                    budget_value = amount // 10  # Example: around 20 -> 2

        return int(budget_value)

    def extract_features_from_text(self, text):
        """
        Extract features such as spiciness, sweetness, savory, budget, and calories from text.
        
        Args:
            text (str): Input text describing preferences.
        
        Returns:
            dict: Extracted features with their corresponding values.
        """
        doc = self.nlp(text)
        
        extracted_features = {
            "spiciness": -1,
            "sweetness": -1,
            "savory": -1,
            "budget": -1,
            "calories": -1,
        }

        for token in doc:
            # Handling negated words (e.g., "not spicy")
            if token.dep_ == 'neg':
                negated_word = token.head.text.lower()
                if negated_word == "spicy":
                    extracted_features["spiciness"] = 0
                elif negated_word == "sweet":
                    extracted_features["sweetness"] = 0
                elif negated_word == "savory":
                    extracted_features["savory"] = 0
            else:
                # Handle non-negated descriptive words
                if token.text.lower() == "spicy":
                    extracted_features["spiciness"] = 2
                elif token.text.lower() == "mild":
                    extracted_features["spiciness"] = 1
                elif token.text.lower() == "sweet":
                    extracted_features["sweetness"] = 2
                elif token.text.lower() == "savory":
                    extracted_features["savory"] = 2

        # Extract additional features using NER
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                extracted_features["budget"] = self.determine_budget_value(text)
            elif ent.label_ == "QUANTITY" and "calories" in ent.text.lower():
                extracted_features["calories"] = float(ent.text.split()[0])

        return extracted_features

    def create_feature_vector(self, survey_data, extracted_features):
        """
        Create a feature vector by combining survey data with extracted features.
        
        Args:
            survey_data (dict): User-provided data from a survey.
            extracted_features (dict): Features extracted from text.
        
        Returns:
            np.ndarray: Feature vector combining both sources of data.
        """
        feature_vector = {
            "spiciness": -1,
            "sweetness": -1,
            "savory": -1,
            "budget": -1,
            "calories": -1,
        }

        for feature in feature_vector.keys():
            if feature in survey_data and feature in extracted_features:
                feature_vector[feature] = (survey_data[feature] + extracted_features[feature]) / 2
            elif feature in survey_data:
                feature_vector[feature] = survey_data[feature]
            elif feature in extracted_features:
                feature_vector[feature] = extracted_features[feature]

        return np.array(list(feature_vector.values()), dtype=float)

    def create_semantic_vector(self, query):
        """
        Create a semantic vector from a text query using a pre-trained model.
        
        Args:
            query (str): The input text.
        
        Returns:
            np.ndarray: The semantic vector representing the text.
        """
        return self.text_model.encode(query)

    def extract_user_constraints(self, query, allergens=[], dietary_restrictions=[], dislikes=[]):
        """
        Extract user constraints such as allergies, dietary restrictions, and dislikes from text.
        
        Args:
            query (str): Input text from the user.
            allergens (list): List of known allergens.
            dietary_restrictions (list): List of dietary restrictions.
            dislikes (list): List of disliked items.
        
        Returns:
            dict: User constraints updated with information from the query.
        """
        constraints = {
            "allergies": allergens,
            "dietary_restrictions": dietary_restrictions,
            "user_dislikes": dislikes,
        }

        doc = self.nlp(query)
        
        # Pattern matching for allergies
        if "allergic to" in query.lower():
            allergies = [token.text for token in doc if token.dep_ == 'pobj']
            constraints["allergies"].extend(allergies)

        # Pattern matching for dislikes
        if "don't like" in query.lower() or "do not like" in query.lower():
            dislikes = [token.text for token in doc if token.dep_ == 'dobj']
            constraints["user_dislikes"].extend(dislikes)

        return constraints

class UserVectorManager:
    def __init__(self, pinecone_api_key):
        # Initialize Pinecone API and connect to vector index
        self.pc = pinecone.Pinecone(api_key=pinecone_api_key)
        self.index_name = "vector-history"

        total_vector_dim = FEATURE_VECTOR_DIM + SEMANTIC_VECTOR_DIM

        # Create vector-history index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=total_vector_dim,
                metric='cosine',
                spec=pinecone.ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = self.pc.Index(self.index_name)

        # Create item-vectors index if it doesn't exist
        self.item_index_name = "item-vectors"
        if self.item_index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.item_index_name,
                dimension=total_vector_dim,
                metric='cosine',
                spec=pinecone.ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.item_index = self.pc.Index(self.item_index_name)

    def create_unique_vector_id(self, user_id, chat_id=None):
        """
        Generate a unique vector ID using user_id and an optional chat_id.
        
        Args:
            user_id (str): The ID of the user.
            chat_id (str): The ID of the chat session (optional).
        
        Returns:
            str: A unique vector ID.
        """
        return f"{chat_id}_{uuid.uuid4()}" if chat_id else f"{user_id}_{uuid.uuid4()}"

    def create_chat_id(self, user_id):
        """
        Generate a unique chat ID based on the user ID and a random number.
        
        Args:
            user_id (str): The ID of the user.
        
        Returns:
            str: A unique chat ID.
        """
        random_number = random.randint(1000, 9999)  # Generate a 4-digit random number
        return f"{user_id}_{random_number}"

    def create_user_vector(self, user_id, chat_id, feature_vector, semantic_vector, constraints):
        """
        Create a user vector by combining feature and semantic vectors and attaching metadata.
        
        Args:
            user_id (str): The ID of the user.
            chat_id (str): The ID of the chat session.
            feature_vector (np.ndarray): The feature vector.
            semantic_vector (np.ndarray): The semantic vector.
            constraints (dict): User constraints.
        
        Returns:
            tuple: Combined vector, metadata, and vector ID.
        """
        combined_vector = np.concatenate((feature_vector, semantic_vector))
        vector_id = self.create_unique_vector_id(user_id, chat_id)
        metadata = {
            "user_id": user_id,
            "chat_id": chat_id,
            "vector_id": vector_id,
            "num_queries": 1,
            "allergies": constraints["allergies"],
            "dietary_restrictions": constraints["dietary_restrictions"],
            "user_dislikes": constraints["user_dislikes"],
            "item_vector_id": "",  # Reference to the item vector ID
            "selected": -1,
            "liked": -1
        }
        return combined_vector, metadata, vector_id

    def upload_to_pinecone(self, combined_vector, metadata, vector_id):
        """
        Upload a vector to the Pinecone index with associated metadata.
        
        Args:
            combined_vector (np.ndarray): The vector to upload.
            metadata (dict): Metadata associated with the vector.
            vector_id (str): The unique ID for the vector.
        """
        self.index.upsert(vectors=[{
            "id": vector_id,
            "values": combined_vector.tolist(),
            "metadata": metadata
        }])

    def upload_item_vector(self, item_vector, item_name, chat_id):
        """
        Upload an item vector to the Pinecone item index.
        
        Args:
            item_vector (np.ndarray): The item vector to upload.
            item_name (str): The name of the item.
            chat_id (str): The chat session ID.
        
        Returns:
            str: The unique ID of the uploaded item vector.
        """
        item_vector_id = f"item_{chat_id}_{uuid.uuid4()}"
        self.item_index.upsert(vectors=[{
            "id": item_vector_id,
            "values": item_vector.tolist(),
            "metadata": {"name": item_name, "chat_id": chat_id}  # Include chat_id in metadata
        }])
        return item_vector_id

    def update_user_vector(self, vector_id, new_query, chat_id, new_item_vector=None, new_item_name=None, selected=None, liked=None):
        """
        Update an existing user vector with new query data, item vector, and feedback.
        
        Args:
            vector_id (str): The ID of the vector to update.
            new_query (str): The new user query to update the vector with.
            chat_id (str): The chat session ID.
            new_item_vector (np.ndarray): The new item vector to associate (optional).
            new_item_name (str): The name of the new item (optional).
            selected (int): Indicator if the item was selected (optional).
            liked (int): Indicator if the item was liked (optional).
        """
        existing_vector = self.index.fetch(ids=[vector_id])['vectors'][vector_id]
        
        feature_extractor = FeatureExtractor()
        new_extracted_features = feature_extractor.extract_features_from_text(new_query)
        new_feature_vector = feature_extractor.create_feature_vector({}, new_extracted_features)
        new_semantic_vector = feature_extractor.create_semantic_vector(new_query)
        new_combined_vector = np.concatenate((new_feature_vector, new_semantic_vector))

        # Update the existing vector by averaging with the new combined vector
        updated_vector = (
            np.array(existing_vector['values']) * existing_vector['metadata']['num_queries'] + new_combined_vector
        ) / (existing_vector['metadata']['num_queries'] + 1)
        
        # Update metadata with new query count and optional item vector info
        existing_vector['metadata']['num_queries'] += 1
        if new_item_vector is not None and new_item_name is not None:
            item_vector_id = self.upload_item_vector(new_item_vector, new_item_name, chat_id)
            existing_vector['metadata']['item_vector_id'] = item_vector_id  # Store the item vector ID reference
        if selected is not None:
            existing_vector['metadata']['selected'] = selected
        if liked is not None:
            existing_vector['metadata']['liked'] = liked

        # Update constraints based on the new query
        constraints = feature_extractor.extract_user_constraints(
            new_query, 
            existing_vector['metadata']['allergies'],
            existing_vector['metadata']['dietary_restrictions'],
            existing_vector['metadata']['user_dislikes']
        )
        existing_vector['metadata']['allergies'] = constraints['allergies']
        existing_vector['metadata']['dietary_restrictions'] = constraints['dietary_restrictions']
        existing_vector['metadata']['user_dislikes'] = constraints['user_dislikes']

        # Upload the updated vector to Pinecone
        self.index.upsert(vectors=[{
            "id": vector_id,
            "values": updated_vector.tolist(),
            "metadata": existing_vector['metadata']
        }])

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    # Initialize feature extractor and user vector manager
    feature_extractor = FeatureExtractor()
    user_vector_manager = UserVectorManager(pinecone_api_key)

    # Example user and chat IDs
    user_id = "user125"
    chat_id = user_vector_manager.create_chat_id(user_id)  # Generate a random chat ID

    # Example survey data and query
    survey_data = {
        "spiciness": 2,
        "sweetness": 1,
        "savory": 1,
        "budget": 2
    }
    query = "I am allergic to peanuts."

    # Extract features and create vectors
    extracted_features = feature_extractor.extract_features_from_text(query)
    feature_vector = feature_extractor.create_feature_vector(survey_data, extracted_features)
    semantic_vector = feature_extractor.create_semantic_vector(query)
    user_constraints = feature_extractor.extract_user_constraints(query)

    # Create and upload the user vector to Pinecone
    combined_vector, metadata, vector_id = user_vector_manager.create_user_vector(
        user_id, chat_id, feature_vector, semantic_vector, user_constraints
    )
    user_vector_manager.upload_to_pinecone(combined_vector, metadata, vector_id)
    sleep(1)

    # Example: Update the user vector with a new item vector
    new_query = "I don't like chicken"
    new_item_vector = np.random.rand(FEATURE_VECTOR_DIM + SEMANTIC_VECTOR_DIM)  # Example item vector
    user_vector_manager.update_user_vector(vector_id, new_query, chat_id, new_item_vector=new_item_vector, new_item_name="fake_item", selected=1, liked=0)
