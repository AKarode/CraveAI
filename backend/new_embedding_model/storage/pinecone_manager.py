from pinecone import Pinecone, ServerlessSpec

class PineconeManager:
    def __init__(self, api_key, index_name, dimension):
        """
        Initializes the PineconeManager with the provided API key and index details.
        
        Args:
            api_key (str): The API key for accessing Pinecone.
            index_name (str): The name of the index to create or use in Pinecone.
            dimension (int): The dimension of the vectors to be stored in Pinecone.
        """
        # Initialize the Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        # Create the index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',  # or 'euclidean', depending on your needs
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Replace with your preferred region
                )
            )
        
        # Connect to the index
        self.index = self.pc.Index(self.index_name)
    
    def upsert_vectors(self, vectors):
        try:
            # Perform the upsert operation
            response = self.index.upsert(vectors=vectors)
            
            # Print or log the response for debugging
            print(f"Upsert response: {response}")
            
            # Return the response
            return response
        except Exception as e:
            print(f"Error during upsert: {e}")
            return None

    
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
        print("here\n")
        return self.index.fetch(ids=[vector_id])

