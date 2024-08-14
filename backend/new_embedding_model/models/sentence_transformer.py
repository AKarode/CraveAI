from sentence_transformers import SentenceTransformer

class SentenceTransformerModel:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        # Load the pre-trained SentenceTransformer model
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        """
        Encodes a given text into a vector using the loaded SentenceTransformer model.
        
        Args:
            text (str): The input text to encode.
            
        Returns:
            numpy.ndarray: The encoded text as a vector.
        """
        return self.model.encode(text)

    def fine_tune(self, train_data, epochs=1, batch_size=8):
        """
        Fine-tunes the model on custom data (optional).
        
        Args:
            train_data (Dataset): The data to fine-tune the model on.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        # Placeholder for fine-tuning logic
        pass
