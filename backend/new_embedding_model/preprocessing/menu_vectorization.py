from preprocessing.text_processing import TextProcessor
from preprocessing.feature_extraction import FeatureExtractor
from models.sentence_transformer import SentenceTransformerModel
from matching.vector_combination import combine_vectors

def vectorize_menu_item(item):
    text_processor = TextProcessor()
    sentence_model = SentenceTransformerModel()
    feature_extractor = FeatureExtractor()

    # Clean and prepare the text
    combined_text = text_processor.clean_text(f"{item['name']} {item['description']}")

    # Generate the semantic vector
    semantic_vector = sentence_model.encode(combined_text)

    # Extract explicit features
    explicit_features = feature_extractor.extract_features({
        "spicy": 4 if 'spicy' in combined_text else 1,
        "budget": 2,  # Example: budget level determined by item price
        "cuisine_italian": 1 if 'pasta' in combined_text else 0,
        "cuisine_indian": 1 if 'curry' in combined_text else 0,
        # Add more features as needed
    })

    # Combine vectors
    combined_vector = combine_vectors(semantic_vector, explicit_features)

    return combined_vector
