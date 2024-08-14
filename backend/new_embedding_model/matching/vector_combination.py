import numpy as np

def combine_vectors(semantic_vector, explicit_features):
    """
    Combines a semantic vector with an explicit feature vector.
    
    Args:
        semantic_vector (numpy.ndarray): The vector representing the semantic meaning of a text (e.g., menu item description).
        explicit_features (dict): A dictionary of explicit features extracted from survey data or structured input.
        
    Returns:
        numpy.ndarray: A combined vector that incorporates both semantic and explicit features.
    """
    explicit_vector = np.array(list(explicit_features.values()), dtype=float)
    combined_vector = np.concatenate((semantic_vector, explicit_vector))
    return combined_vector
