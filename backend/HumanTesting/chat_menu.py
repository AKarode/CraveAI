#!/usr/bin/env python
import os
import sys
import json
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Add the parent directory so we can import from services
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from services.ocr import OCRService

def load_vectorizer_model(model_path, device):
    """
    Loads a BERT-based model for vectorization.
    Assumes the model is a BertForSequenceClassification and uses its encoder.
    """
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=1, problem_type="regression"
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_embedding(model, tokenizer, text, device, max_length=128):
    """
    Computes the embedding for the given text by extracting the [CLS] token output.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.bert(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        embedding = embedding.cpu().numpy().flatten()
    return embedding

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def main():
    # Set up paths (assumes this script is in backend/HumanTesting)
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    menu_image_path = os.path.join(os.path.dirname(__file__), "test_menu.jpg")
    model_path = os.path.join(base_dir, "MLModel", "best_model.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize OCR service and process the menu image.
    ocr_service = OCRService()
    print("Performing OCR on the menu image...")
    ocr_text = ocr_service.process_image(menu_image_path)
    
    print("Extracting menu items...")
    menu_items = ocr_service.extract_menu_items(ocr_text)
    if not menu_items:
        print("No menu items were extracted. Exiting.")
        return
    
    print("\nExtracted Menu Items:")
    for idx, item in enumerate(menu_items, start=1):
        print(f"{idx}. {item['name']} - ${item['price']:.2f}")
        print(f"   Description: {item['description']}")
    
    # Load the vectorization model and tokenizer.
    print("\nLoading vectorization model...")
    vectorizer_model = load_vectorizer_model(model_path, device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Precompute embeddings for each menu item (using name + description).
    print("Computing embeddings for menu items...")
    menu_embeddings = []
    for item in menu_items:
        combined_text = f"{item['name']} {item['description']}"
        embedding = get_embedding(vectorizer_model, tokenizer, combined_text, device)
        menu_embeddings.append(embedding)
    
    # Interactive chat loop.
    print("\nEnter a query to find the best matching menu item (type 'stop' to exit):")
    while True:
        user_query = input("Your query: ").strip()
        if user_query.lower() == "stop":
            print("Exiting.")
            break
        
        query_embedding = get_embedding(vectorizer_model, tokenizer, user_query, device)
        
        best_score = -1
        best_item = None
        for item, emb in zip(menu_items, menu_embeddings):
            score = cosine_similarity(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_item = item
        
        if best_item:
            print("\nBest matching menu item:")
            print(f"Name: {best_item['name']}")
            print(f"Price: ${best_item['price']:.2f}")
            print(f"Description: {best_item['description']}")
            print(f"Similarity Score: {best_score:.4f}\n")
        else:
            print("No matching menu item found.\n")

if __name__ == "__main__":
    main()

