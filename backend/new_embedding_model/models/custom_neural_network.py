import torch
import torch.nn as nn

class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the custom neural network.
        
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
            output_dim (int): Number of output units (e.g., number of classes or categories).
        """
        super(CustomNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

    def train_model(self, train_loader, epochs=10, learning_rate=0.001):
        """
        Trains the model using the provided training data.
        
        Args:
            train_loader (DataLoader): DataLoader for the training data.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            for data in train_loader:
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return self
