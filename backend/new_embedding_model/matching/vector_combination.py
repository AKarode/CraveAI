import numpy as np

def combine_vectors(semantic_vector, explicit_features):
    # Ensure that explicit_features is an array
    if isinstance(explicit_features, dict):
        explicit_vector = np.array(list(explicit_features.values()), dtype=float)
    elif isinstance(explicit_features, np.ndarray):
        explicit_vector = explicit_features
    else:
        raise ValueError("explicit_features must be either a dict or a numpy.ndarray")

    # Combine the vectors
    combined_vector = np.concatenate((semantic_vector, explicit_vector))

    # Ensure the combined vector matches the expected dimension
    expected_dimension = 386  # Replace with your actual index dimension
    if combined_vector.shape[0] != expected_dimension:
        raise ValueError(f"Combined vector has incorrect dimension {combined_vector.shape[0]}, expected {expected_dimension}")

    return combined_vector


