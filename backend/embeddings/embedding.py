import json
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from transformers import BertTokenizer, BertModel
import torch.nn as nn


survey_json = '''
{
    "spiciness": 4,
    "budget": 1500,
    "cuisine": "Italian",
    "exploration_frequency": "Often",
    "cooking_style": "Grilled",
    "dietary_restrictions": ["None"],
    "favorite_dessert": "Chocolate"
}
'''

chat_json = '''
{
    "chat_history": [
        "I am looking for a spicy food option.",
        "I have a budget of $1500 for dining out.",
        "I usually enjoy Italian cuisine."
    ],
    "recent_query": "Can you suggest an Italian restaurant with spicy options?"
}
'''


# Parse JSON inputs
survey_data = json.loads(survey_json)
chat_data = json.loads(chat_json)

# Survey Data Preprocessing
def preprocess_survey_data(survey_data):
    # Define categories
    cuisine_categories = ['Italian', 'Asian', 'Mexican']
    cooking_style_categories = ['Grilled', 'Fried', 'Steamed']
    dietary_restrictions_categories = ['None', 'Vegan', 'Gluten-Free']
    dessert_categories = ['Chocolate', 'Fruit Based', 'Pastries']
    exploration_categories = ['Rrarely', 'Sometimes', 'Often']

    # Initialize encoders and scalers
    cuisine_encoder = OneHotEncoder(categories=[cuisine_categories], sparse_output=False)
    cooking_style_encoder = OneHotEncoder(categories=[cooking_style_categories], sparse_output=False)
    dietary_restrictions_encoder = OneHotEncoder(categories=[dietary_restrictions_categories], sparse_output=False)
    dessert_encoder = OneHotEncoder(categories=[dessert_categories], sparse_output=False)
    exploration_encoder = OneHotEncoder(categories=[exploration_categories], sparse_output=False)
    scaler = MinMaxScaler()

    # Extract and encode features
    spiciness = np.array([[survey_data['spiciness']]])
    budget = np.array([[survey_data['budget']]])
    cuisine = cuisine_encoder.fit_transform(np.array([[survey_data['cuisine']]]))
    exploration_frequency = exploration_encoder.fit_transform(np.array([[survey_data['exploration_frequency']]]))
    cooking_style = cooking_style_encoder.fit_transform(np.array([[survey_data['cooking_style']]]))
    dietary_restrictions = dietary_restrictions_encoder.fit_transform(np.array([survey_data['dietary_restrictions']]))
    favorite_dessert = dessert_encoder.fit_transform(np.array([[survey_data['favorite_dessert']]]))

    # Normalize numerical values
    spiciness_normalized = scaler.fit_transform(spiciness)
    budget_normalized = scaler.fit_transform(budget)

    # Combine all features into a single vector
    survey_vector = np.concatenate([
        spiciness_normalized,
        budget_normalized,
        cuisine,
        exploration_frequency,
        cooking_style,
        dietary_restrictions,
        favorite_dessert
    ], axis=1)

    return survey_vector


# Chat-History Data Preprocessing 

def preprocess_chat_data(chat_data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Combine chat history and recent query
    chat_text = ' '.join(chat_data['chat_history']) + ' ' + chat_data['recent_query']

    # Tokenize and get embeddings
    inputs = tokenizer(chat_text, return_tensors='pt')
    outputs = model(**inputs)

    # Get the mean of the token embeddings to represent the entire text
    chat_embedding = torch.mean(outputs.last_hidden_state, dim=1)

    return chat_embedding, inputs

class FoodPreferenceEmbeddingModel(nn.Module):
    def __init__(self, survey_input_size, chat_embedding_size):
        super(FoodPreferenceEmbeddingModel, self).__init__()
        self.survey_fc = nn.Linear(survey_input_size, 128)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.chat_fc = nn.Linear(768, chat_embedding_size)
        self.final_fc = nn.Linear(128 + chat_embedding_size, 256)

    def forward(self, survey_input, chat_input):
        # Process survey data
        survey_embedding = torch.relu(self.survey_fc(survey_input))

        # Process chat data
        chat_embeddings = self.bert_model(**chat_input).last_hidden_state
        chat_embeddings = torch.mean(chat_embeddings, dim=1)
        chat_embedding = torch.relu(self.chat_fc(chat_embeddings))

        # Combine embeddings
        combined_embedding = torch.cat((survey_embedding, chat_embedding), dim=1)
        final_embedding = torch.relu(self.final_fc(combined_embedding))
        return final_embedding

# Preprocess the data
survey_vector = preprocess_survey_data(survey_data)
survey_tensor = torch.tensor(survey_vector, dtype=torch.float32)
chat_embedding, chat_inputs = preprocess_chat_data(chat_data)

print("Survey Vector:", survey_vector)
print("Chat Embedding:", chat_embedding)

survey_input_size = survey_tensor.shape[1]
chat_embedding_size = 128
model = FoodPreferenceEmbeddingModel(survey_input_size, chat_embedding_size)

# Prepare Example Inputs
survey_input = survey_tensor
chat_input = chat_inputs

# Get the Embedding
embedding = model(survey_input, chat_input)
print("Embedding:", embedding)
