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
    similarities = []
    
    # Iterate over the list of matches
    for match in menu_vectors['matches']:
        item_id = match['id']
        item_values = match['values']  # This should be the vector values
        if not item_values:
            continue  # Skip if there are no values
        
        similarity = compute_similarity(user_vector, item_values)
        similarities.append((item_id, similarity))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the top matches
    return similarities[:3]  # Example: returning top 3 matches

