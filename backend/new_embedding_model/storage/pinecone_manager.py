import pinecone

class PineconeManager:
    def __init__(self, api_key, index_name, dimension):
        """
        Initializes the PineconeManager with the provided API key and index details.
        
        Args:
            api_key (str): The API key for accessing Pinecone.
            index_name (str): The name of the index to create or use in Pinecone.
            dimension (int): The dimension of the vectors to be stored in Pinecone.
        """
        pinecone.init(api_key=api_key)
        self.index_name = index_name
        
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(self.index_name, dimension=dimension, metric='cosine')
        
        self.index = pinecone.Index(self.index_name)
    
    def upsert_vectors(self, vectors):
        """
        Upserts (inserts or updates) vectors into the Pinecone index.
        
        Args:
            vectors (list of tuples): A list of tuples where each tuple contains an ID and a vector.
        """
        self.index.upsert(vectors=vectors)
    
    def query_vector(self, vector, top_k=5):
        """
        Queries the Pinecone index for the top-k most similar vectors.
        
        Args:
            vector (list or numpy.ndarray): The vector to query with.
            top_k (int): The number of top similar vectors to retrieve.
            
        Returns:
            list: A list of IDs corresponding to the most similar vectors.
        """
        return self.index.query(vector=vector, top_k=top_k)
    
    def fetch_vector(self, vector_id):
        """
        Fetches a specific vector from the Pinecone index by its ID.
        
        Args:
            vector_id (str): The ID of the vector to fetch.
            
        Returns:
            dict: The fetched vector data.
        """
        return self.index.fetch(ids=[vector_id])
