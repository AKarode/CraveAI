import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from vector_handling import FeatureExtractor

# Cosine Similarity Model
def cosine_similarity_model(vector1, vector2):
    """Calculate the cosine similarity between two vectors."""
    return cosine_similarity([vector1], [vector2])[0][0]

# Custom Neural Network Model
class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_dim=778):
        """
        Initialize the neural network layers.
        
        Args:
            input_dim (int): The dimension of the input vector (default is 778).
        """
        super(CustomNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # Output is a single similarity score
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Define the forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Output between 0 and 1
        return x

# Dataset class to handle user-item vectors
class UserItemDataset(Dataset):
    def __init__(self, user_vectors, item_vectors, labels):
        """
        Initialize the dataset with user vectors, item vectors, and labels.
        
        Args:
            user_vectors (list): List of user vectors.
            item_vectors (list): List of item vectors.
            labels (list): List of labels indicating user preferences.
        """
        self.user_vectors = user_vectors
        self.item_vectors = item_vectors
        self.labels = labels

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a specific user-item pair and its corresponding label.
        
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            torch.Tensor: Combined user-item vector.
            torch.Tensor: Corresponding label.
        """
        user_vector = self.user_vectors[idx]
        item_vector = self.item_vectors[idx]
        combined_vector = np.concatenate((user_vector, item_vector))  # Concatenate to size 778
        label = self.labels[idx]
        return torch.tensor(combined_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Function to train the neural network
def train_neural_network(user_vectors, item_vectors, labels, epochs=10, lr=0.001, batch_size=32):
    """
    Train the neural network model on user-item vectors.
    
    Args:
        user_vectors (list): List of user vectors.
        item_vectors (list): List of item vectors.
        labels (list): List of labels for training.
        epochs (int): Number of training epochs (default is 10).
        lr (float): Learning rate for the optimizer (default is 0.001).
        batch_size (int): Batch size for training (default is 32).
        
    Returns:
        CustomNeuralNetwork: Trained neural network model.
    """
    # Determine the input dimension for the neural network
    input_dim = len(user_vectors[0]) + len(item_vectors[0])  # Expecting the input dimension to be 778
    
    # Create dataset and dataloader for training
    dataset = UserItemDataset(user_vectors, item_vectors, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the neural network model, loss function, and optimizer
    model = CustomNeuralNetwork(input_dim=input_dim)  # Initialize model with correct input_dim
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            combined_vectors, labels = batch
            outputs = model(combined_vectors)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print the loss for each epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

    return model

# Rule-based filtering function
def rule_based_filter(items, user_constraints):
    """
    Filter menu items based on user constraints such as allergies, dietary restrictions, and dislikes.
    
    Args:
        items (list): List of menu items.
        user_constraints (dict): Dictionary of user constraints including allergies, dietary restrictions, and dislikes.
        
    Returns:
        list: List of filtered menu items that do not violate user constraints.
    """
    filtered_items = []
    extractor = FeatureExtractor()

    for item in items:
        item_text = f"{item['name']} {item['description']}"
        
        # Check for allergies in the item description
        if any(allergen in item_text.lower() for allergen in user_constraints.get("allergies", [])):
            continue
        
        # Check for dietary restrictions in the item description
        if any(restriction in item_text.lower() for restriction in user_constraints.get("dietary_restrictions", [])):
            continue
        
        # Check for dislikes in the item description
        if any(dislike in item_text.lower() for dislike in user_constraints.get("user_dislikes", [])):
            continue
        
        # If the item passes all checks, add it to the filtered list
        filtered_items.append(item)

    return filtered_items

# Inference function (getting recommendations)
def get_recommendations(model, user_vector, item_vectors):
    """
    Get recommendations by scoring each item vector using the trained neural network model.
    
    Args:
        model (CustomNeuralNetwork): Trained neural network model.
        user_vector (np.ndarray): User vector.
        item_vectors (list): List of item vectors.
        
    Returns:
        list: List of scores for each item, representing the model's confidence in the recommendation.
    """
    model.eval()  # Set the model to evaluation mode
    scores = []
    
    # Compute scores for each item vector
    with torch.no_grad():
        for item_vector in item_vectors:
            combined_vector = np.concatenate((user_vector, item_vector))  # Ensure size is 778
            combined_tensor = torch.tensor(combined_vector, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            score = model(combined_tensor).item()
            scores.append(score)
    
    return scores
