import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from parse_pairs import load_pairs_from_file  # Import from parse_pairs.py

# Step 1: Select the device (MPS for Apple Silicon, or fallback to CPU)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Step 2: Load the parsed pairs from the file
file_path = 'Synthetic Similarity Scores.txt'  # Change to the actual file path
pairs = load_pairs_from_file(file_path)

# Step 3: Load the pre-trained SentenceTransformer model and move it to the selected device
model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)

# Step 4: Convert the pairs into InputExamples for training
train_examples = []
for pair in pairs:
    example = InputExample(texts=[pair['sentence_1'], pair['sentence_2']], label=pair['score'])
    train_examples.append(example)

# Step 5: Create a DataLoader for the training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1)  # Reduce batch size if necessary

# Step 6: Move the data to the device in the training loop
train_loss = losses.CosineSimilarityLoss(model)

# Step 7: Custom training loop to ensure input tensors are moved to the device
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,  # Adjust epochs as needed
    warmup_steps=100  # You can also adjust warmup steps
)

# Step 8: Save the fine-tuned model
model.save('fine_tuned_model')
