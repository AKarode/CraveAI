from scipy.spatial.distance import cosine

def compute_similarity(vector1, vector2):
    """
    Computes the cosine similarity between two vectors.
    
    Args:
        vector1 (numpy.ndarray): The first vector.
        vector2 (numpy.ndarray): The second vector.
        
    Returns:
        float: The cosine similarity score between the two vectors.
    """
    return 1 - cosine(vector1, vector2)

def rank_menu_items(user_vector, menu_vectors):
    """
    Ranks menu items based on their similarity to the user's preference vector.
    
    Args:
        user_vector (numpy.ndarray): The combined vector representing the user's preferences.
        menu_vectors (dict): A dictionary where keys are menu item IDs and values are their corresponding vectors.
        
    Returns:
        list: A list of menu item IDs sorted by their similarity to the user's preferences.
    """
    similarities = []
    for item_id, menu_vector in menu_vectors.items():
        similarity = compute_similarity(user_vector, menu_vector)
        similarities.append((item_id, similarity))
    
    # Sort items by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in similarities]
