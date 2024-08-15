import numpy as np

class FeatureExtractor:
    def __init__(self, scaling=True):
        """
        Initializes the FeatureExtractor with optional scaling.
        
        Args:
            scaling (bool): Whether to scale the features to sum to 1.0.
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
        Scales the feature vector so that it sums to 1.0.
        
        Args:
            features (numpy.ndarray): The feature vector to scale.
            
        Returns:
            numpy.ndarray: The scaled feature vector.
        """
        # Normalize the features to sum to 1.0
        sum_features = np.sum(features)
        if sum_features == 0:
            return features  # Avoid division by zero; return original features
        return features / sum_features
