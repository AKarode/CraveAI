#!/usr/bin/env python
"""
Model Comparison Utilities

This module provides utility functions for comparing embedding models,
extracted from the original model_comparison.py
"""

import os
import time
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_bert_model(model_path):
    """Load the BERT model for embeddings
    
    Args:
        model_path: Path to the model file
        
    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=1, problem_type="regression"
        )
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        return None, None, torch.device("cpu")

def get_bert_embedding(text, model, tokenizer, device):
    """Get embedding from the BERT model
    
    Args:
        text: Input text to embed
        model: BERT model
        tokenizer: BERT tokenizer
        device: torch device
        
    Returns:
        tuple: (embedding, elapsed_time)
    """
    if not model or not tokenizer:
        # Return a random embedding if model failed to load
        return np.random.normal(0, 0.1, 768), 0.001
    
    start_time = time.time()
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.bert(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    
    elapsed_time = time.time() - start_time
    return embedding, elapsed_time

def get_openai_embedding(text, client):
    """Get embedding from OpenAI
    
    Args:
        text: Input text to embed
        client: OpenAI client
        
    Returns:
        tuple: (embedding, elapsed_time)
    """
    if not client:
        # Return a random embedding if OpenAI client isn't available
        return np.random.normal(0, 0.1, 1536), 0.001
        
    start_time = time.time()
    
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
    except Exception as e:
        print(f"Error generating OpenAI embedding: {e}")
        embedding = np.random.normal(0, 0.1, 1536)
        
    elapsed_time = time.time() - start_time
    return embedding, elapsed_time

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        float: Cosine similarity score
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def get_test_data():
    """Get test queries and menu items for comparisons
    
    Returns:
        tuple: (test_queries, menu_items)
    """
    test_queries = [
        "I want something spicy with chicken",
        "Looking for a vegetarian pasta dish",
        "I'd like a chocolate dessert"
    ]
    
    menu_items = [
        {
            "name": "Spicy Chicken Curry",
            "description": "Tender chicken pieces in a spicy curry sauce with aromatic spices."
        },
        {
            "name": "Vegetarian Pasta Primavera",
            "description": "Fresh seasonal vegetables with pasta and parmesan cheese."
        },
        {
            "name": "Chocolate Lava Cake",
            "description": "Warm chocolate cake with a molten center and vanilla ice cream."
        },
        {
            "name": "Grilled Salmon",
            "description": "Fresh salmon fillet with seasonal vegetables."
        },
        {
            "name": "Classic Caesar Salad",
            "description": "Crisp romaine lettuce with Caesar dressing and croutons."
        }
    ]
    
    return test_queries, menu_items 