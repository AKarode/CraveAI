import os
import re
import time
import copy
import random
import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error

# For data augmentation using synonym replacement
import nlpaug.augmenter.word as naw
import nltk
nltk.download('averaged_perceptron_tagger_eng')

# ========= Reproducibility ========= #
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ========= Data Parsing ========= #
def parse_pairs(file_path):
    """
    Parses a file containing pairs with the following format:
    
    Pair x:
    Sentence 1: "text"
    Sentence 2: "text"
    Similarity Score: number
    
    Ignores extra white space and random separator lines.
    Returns a list of tuples: (sentence1, sentence2, score)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    pattern = re.compile(
        r'Pair\s*\d+\s*:\s*.*?'
        r'Sentence\s*1\s*:\s*\"([^\"]+)\".*?'
        r'Sentence\s*2\s*:\s*\"([^\"]+)\".*?'
        r'Similarity Score\s*:\s*([0-9]+\.[0-9]+)',
        re.DOTALL | re.IGNORECASE
    )
    
    matches = pattern.findall(text)
    pairs = []
    for match in matches:
        sentence1 = match[0].strip()
        sentence2 = match[1].strip()
        score = float(match[2])
        pairs.append((sentence1, sentence2, score))
    return pairs

# ========= Data Augmentation ========= #
def augment_pair(pair, augmenter):
    """
    Given a data pair (sentence1, sentence2, score) and an augmenter,
    return an augmented pair by applying augmentation to each sentence.
    """
    s1, s2, score = pair
    aug_s1 = augmenter.augment(s1)
    aug_s2 = augmenter.augment(s2)
    return (aug_s1, aug_s2, score)

# ========= PyTorch Dataset ========= #
class FoodPairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=128, augment=False, augmenter=None):
        """
        pairs: list of tuples (sentence1, sentence2, score)
        tokenizer: Hugging Face tokenizer instance
        augment: if True, each sample is augmented and both original and augmented samples are used
        augmenter: an nlpaug augmenter instance (required if augment is True)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augmenter = augmenter
        self.pairs = pairs.copy()
        
        # If augmentation is enabled, double the dataset by adding augmented examples.
        if self.augment and self.augmenter is not None:
            aug_pairs = [augment_pair(pair, self.augmenter) for pair in pairs]
            self.pairs += aug_pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        s1, s2, score = self.pairs[idx]
        # Combine the two sentences with a separator (using [SEP] token)
        encoded = self.tokenizer.encode_plus(
            s1,
            s2,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        # Flatten the tensors
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(score, dtype=torch.float)
        }

# ========= Model Creation ========= #
def create_model(model_name="bert-base-uncased", freeze_layers=6):
    """
    Loads a pretrained BERT model for sequence regression and freezes the early layers.
    """
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression"
    )
    # Freeze embeddings and early encoder layers
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.bert.encoder.layer):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
    return model

# ========= Training & Evaluation ========= #
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            preds.extend(outputs.logits.squeeze().detach().cpu().numpy().tolist())
            labels.extend(batch["labels"].numpy().tolist())
    avg_loss = total_loss / len(dataloader)
    mse = mean_squared_error(labels, preds)
    return avg_loss, mse, preds, labels

def train_model(model, train_loader, val_loader, device, optimizer, scheduler, epochs=4, patience=2, output_dir="final_checkpoint"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_mse, _, _ = evaluate_epoch(model, val_loader, device)
        print(f"  Training loss: {train_loss:.4f} | Validation loss: {val_loss:.4f} | MSE: {val_mse:.4f}")
        
        # Checkpointing & early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            checkpoint_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print("  Best model updated and saved to", checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("  Early stopping triggered.")
                break
    return best_model_state, best_loss

# ========= Main Training Pipeline ========= #
def main():
    # File paths (ensure these files exist from your previous split)
    train_file = "training_data.txt"
    test_file = "testing_data.txt"
    
    # Parse the data pairs
    train_pairs = parse_pairs(train_file)
    test_pairs = parse_pairs(test_file)
    print(f"Loaded {len(train_pairs)} training pairs and {len(test_pairs)} test pairs.")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize a synonym augmenter from nlpaug (requires 'wordnet')
    augmenter = naw.SynonymAug(aug_src='wordnet')
    
    # ===== Fixed Hyperparameters ===== #
    final_lr = 5e-5
    final_batch_size = 16
    final_epochs = 4
    
    # Create full training dataset (with augmentation) and test dataset
    full_train_dataset = FoodPairDataset(train_pairs, tokenizer, augment=True, augmenter=augmenter)
    full_train_loader = DataLoader(full_train_dataset, batch_size=final_batch_size, shuffle=True)
    test_dataset = FoodPairDataset(test_pairs, tokenizer, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=final_batch_size, shuffle=False)
    
    # Create model and move to device
    model = create_model()
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=final_lr)
    total_steps = len(full_train_loader) * final_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    print("\nStarting final training on full training data...")
    best_model_state, _ = train_model(model, full_train_loader, test_loader, device, optimizer, scheduler, epochs=final_epochs, patience=2, output_dir="final_checkpoint")
    model.load_state_dict(best_model_state)
    
    # ===== Evaluation on Test Data ===== #
    test_loss, test_mse, preds, true_labels = evaluate_epoch(model, test_loader, device)
    print(f"\nFinal Test MSE: {test_mse:.4f}")
    
    # Save predictions to file along with actual scores
    predictions_file = "predictions.txt"
    with open(predictions_file, "w", encoding="utf-8") as f:
        for i, (pred, true) in enumerate(zip(preds, true_labels), start=1):
            f.write(f"Pair {i}: Predicted: {pred:.4f} | Actual: {true:.4f}\n")
    print(f"Predictions saved to {predictions_file}")
    
    # ===== Save Configuration Details ===== #
    config = {
        "best_model_checkpoint": os.path.join("final_checkpoint", "best_model.pt"),
        "predictions_file": predictions_file,
        "final_test_mse": test_mse,
        "hyperparameters": {
            "learning_rate": final_lr,
            "batch_size": final_batch_size,
            "epochs": final_epochs
        }
    }
    config_file = "training_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_file}")

if __name__ == "__main__":
    main()

