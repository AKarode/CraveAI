import pinecone
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
# Initialize the Sentence-BERT model
text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#Load Environment Variables
load_dotenv()


# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=api_key)

# Create or connect to the index
index_name = 'palettes'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=384,  # Assume 384 is the dimensionality of Sentence-BERT vectors
        metric='cosine',  # You can choose 'cosine', 'euclidean', or other metrics
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Use a supported region
        )
    )

index = pc.Index(index_name)

def initialize_palette_vector(user_id, survey_data):
    """
    Initializes the palette vector with survey data.
    survey_data: dict containing {"sentence": "survey summary"}
    """
    sentence = survey_data.get("sentence")
    if not sentence:
        raise ValueError("Survey data must contain a 'sentence' key with a summary sentence.")

    # Vectorize the survey summary sentence
    vector_values = text_model.encode(sentence).tolist()
    num_times_updated = 0

    # Create the palette vector with initial num
    palette_vector = {
        "id": user_id,
        "values": vector_values,
        "metadata": {"num": num_times_updated}
    }

    # Upsert the vector into Pinecone
    index.upsert(vectors=[palette_vector])

def update_palette_vector(user_id, new_sentence):
    """
    Updates the palette vector with a new sentence.
    new_sentence: string with the new user statement
    """
    # Fetch the original palette vector
    original_vector = index.fetch(ids=[user_id])['vectors'][user_id]
    original_values = np.array(original_vector['values'])
    num_times_updated = original_vector['metadata']['num']

    # Vectorize the new sentence
    new_vector_values = np.array(text_model.encode(new_sentence))

    # Update the palette vector
    updated_values = (original_values * num_times_updated + new_vector_values) / (num_times_updated + 1)
    updated_values = updated_values.tolist()

    # Increment the update count
    num_times_updated += 1

    # Create the updated palette vector
    updated_palette_vector = {
        "id": user_id,
        "values": updated_values,
        "metadata": {"num": num_times_updated}
    }

    # Upsert the updated vector into Pinecone
    index.upsert(vectors=[updated_palette_vector])

# Example usage:
survey_data = {"sentence": "I like food."}
user_id = "user_124"

# Initialize the palette vector
initialize_palette_vector(user_id, survey_data)

# Update the palette vector with a new statement
new_statement = "I like food."
update_palette_vector(user_id, new_statement)

# Fetch and print the updated vector to verify
updated_vector = index.fetch(ids=[user_id])
print(updated_vector)
