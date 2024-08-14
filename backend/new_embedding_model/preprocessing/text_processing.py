import re
import string

class TextProcessor:
    def __init__(self, lower=True, remove_punctuation=True):
        """
        Initializes the TextProcessor with options for text cleaning.
        
        Args:
            lower (bool): Whether to convert the text to lowercase.
            remove_punctuation (bool): Whether to remove punctuation from the text.
        """
        self.lower = lower
        self.remove_punctuation = remove_punctuation

    def clean_text(self, text):
        """
        Cleans the input text by applying the specified processing steps.
        
        Args:
            text (str): The text to clean.
            
        Returns:
            str: The cleaned text.
        """
        # Convert text to lowercase if specified
        if self.lower:
            text = text.lower()

        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Further cleaning can be added here (e.g., removing extra whitespace)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_text(self, text):
        """
        Tokenizes the input text into a list of words.
        
        Args:
            text (str): The text to tokenize.
            
        Returns:
            list: A list of words (tokens).
        """
        # Tokenize the text by splitting on whitespace
        return text.split()

    def preprocess(self, text):
        """
        Full preprocessing pipeline: clean the text and then tokenize it.
        
        Args:
            text (str): The text to preprocess.
            
        Returns:
            list: A list of cleaned and tokenized words.
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        return tokens
