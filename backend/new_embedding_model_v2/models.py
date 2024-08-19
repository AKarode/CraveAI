import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from vector_handling import FeatureExtractor
import re

# Cosine Similarity Model
def cosine_similarity_model(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

# Custom Neural Network Model
class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CustomNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # Output is a single similarity score
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Output between 0 and 1
        return x

class UserItemDataset(Dataset):
    def __init__(self, user_vectors, item_vectors, labels):
        self.user_vectors = user_vectors
        self.item_vectors = item_vectors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_vector = self.user_vectors[idx]
        item_vector = self.item_vectors[idx]
        label = self.labels[idx]
        combined_vector = np.concatenate((user_vector, item_vector))
        return torch.tensor(combined_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def train_neural_network(user_vectors, item_vectors, labels, input_dim, epochs=10, lr=0.001, batch_size=32):
    dataset = UserItemDataset(user_vectors, item_vectors, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CustomNeuralNetwork(input_dim=input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

    return model

# Rule-based Layer
def rule_based_filter(items, user_constraints):
    filtered_items = []
    extractor = FeatureExtractor()

    for item in items:
        item_text = f"{item['name']} {item['description']}"
        doc = extractor.nlp(item_text)
        
        # Checking for allergies
        if any(allergen in item_text for allergen in user_constraints["allergies"]):
            continue
        
        # Checking for dietary restrictions
        if any(restriction in item_text for restriction in user_constraints["dietary_restrictions"]):
            continue
        
        # Checking for dislikes
        if any(dislike in item_text for dislike in user_constraints["user_dislikes"]):
            continue
        
        filtered_items.append(item)

    return filtered_items
