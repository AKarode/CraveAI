import numpy as np

def format_vector_data(data):
    """
    Formats and prepares data for insertion into Pinecone.
    
    Args:
        data (dict): A dictionary where keys are IDs and values are vectors.
        
    Returns:
        list: A list of tuples where each tuple contains an ID and a formatted vector.
    """
    formatted_data = []
    for key, vector in data.items():
        formatted_vector = np.array(vector).tolist()  # Ensure vector is in the correct format
        formatted_data.append((key, formatted_vector))
    return formatted_data

def normalize_vector(vector):
    """
    Normalizes a vector to have unit norm.
    
    Args:
        vector (numpy.ndarray): The vector to normalize.
        
    Returns:
        numpy.ndarray: The normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def prepare_user_data_for_storage(user_id, combined_vector):
    """
    Prepares user data for storage in Pinecone.
    
    Args:
        user_id (str): The unique identifier for the user.
        combined_vector (numpy.ndarray): The combined vector representing the user's preferences.
        
    Returns:
        tuple: A tuple containing the user ID and the formatted vector.
    """
    formatted_vector = normalize_vector(combined_vector)
    return (user_id, formatted_vector.tolist())
