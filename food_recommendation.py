import pinecone
from sklearn.feature_extraction.text import TfidfVectorizer

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="5e726cf8-5d0d-456c-addf-cc6b5569ea47")

pc.create_index(
    name="quickstart",
    dimension=8, # Replace with your model dimensions
    metric="euclidean", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

# Initialize Pinecone
pinecone.init(api_key='5e726cf8-5d0d-456c-addf-cc6b5569ea47', environment='us-west1-gcp')
index_name = 'food-recommendations'
pinecone.create_index(index_name, dimension=128)
index = pinecone.Index(index_name)

# Example data
food_items = ['pizza with cheese and pepperoni', 
'vegan salad with avocado', 
'spaghetti with tomato sauce']

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(food_items).toarray()

# Insert vectors
food_vectors = [{'id': f'item{i}', 'values': vector.tolist()} for i, vector in enumerate(vectors)]
index.upsert(vectors=food_vectors)

def recommend_food(query_item):
    query_vector = vectorizer.transform([query_item]).toarray().tolist()[0]
    results = index.query(query_vector, top_k=5)
    return results

# Example query
query_item = 'healthy salad with avocado'
recommendations = recommend_food(query_item)
print(recommendations)
