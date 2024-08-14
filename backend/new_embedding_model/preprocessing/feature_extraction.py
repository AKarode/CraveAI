import numpy as np

class FeatureExtractor:
    def __init__(self, scaling=True):
        """
        Initializes the FeatureExtractor with optional scaling.
        
        Args:
            scaling (bool): Whether to scale the features to unit length.
        """
        self.scaling = scaling

    def extract_features(self, survey_data):
        """
        Extracts features from survey data and converts them into a feature vector.
        
        Args:
            survey_data (dict): A dictionary containing user preferences from the survey.
                                Example: {"spicy": 2, "budget": 3}
                                
        Returns:
            numpy.ndarray: A feature vector representing the survey data.
        """
        # Convert the survey data into a NumPy array (feature vector)
        features = np.array(list(survey_data.values()), dtype=float)
        
        if self.scaling:
            features = self.scale_features(features)
        
        return features

    def scale_features(self, features):
        """
        Scales the feature vector to have unit norm.
        
        Args:
            features (numpy.ndarray): The feature vector to scale.
            
        Returns:
            numpy.ndarray: The scaled feature vector.
        """
        # Perform simple normalization (scaling) of the feature vector
        norm = np.linalg.norm(features)
        if norm == 0:
            return features  # Avoid division by zero; return original features
        return features / norm
