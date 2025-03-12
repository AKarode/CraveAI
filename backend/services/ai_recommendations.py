import os
from typing import List, Dict, Any, Optional
import openai
import pinecone


class RecommendationService:
    """Service for providing AI-powered menu recommendations"""
    
    def __init__(self):
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        if openai_api_key and openai_api_key != "your_openai_api_key_here":
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
        
        # Initialize Pinecone for vector search
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        
        self.pc = None
        self.index = None
        
        # Only try to initialize Pinecone if we have valid-looking keys
        if (pinecone_api_key and pinecone_env and self.pinecone_index_name and
            pinecone_api_key != "your_pinecone_api_key_here" and
            pinecone_env != "your_pinecone_environment" and
            self.pinecone_index_name != "your_pinecone_index_name"):
            try:
                self.pc = pinecone.Pinecone(api_key=pinecone_api_key)
                self.index = self.pc.Index(self.pinecone_index_name)
            except Exception as e:
                print(f"Warning: Failed to initialize Pinecone: {e}")
                self.pc = None
                self.index = None
        else:
            print("Warning: Pinecone credentials not found or are using default values")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Create vector embeddings for a text string using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding
        """
        if not self.openai_client:
            # Return a dummy embedding if OpenAI client isn't available
            return [0.0] * 1536  # Standard size for OpenAI embeddings
            
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [0.0] * 1536
    
    def store_menu_items(self, menu_items: List[Dict[str, Any]], menu_id: str) -> None:
        """
        Store menu items in Pinecone for future retrieval
        
        Args:
            menu_items: List of menu items with their details
            menu_id: Identifier for the restaurant menu
        """
        if not self.index:
            print("Pinecone index not initialized, skipping storage")
            return
            
        try:
            vectors = []
            
            for i, item in enumerate(menu_items):
                # Combine item name and description for better embedding
                item_text = f"{item.get('name', '')} - {item.get('description', '')}"
                embedding = self.embed_text(item_text)
                
                # Prepare vector record for Pinecone
                vector = {
                    "id": f"{menu_id}_{i}",
                    "values": embedding,
                    "metadata": {
                        "menu_id": menu_id,
                        "name": item.get('name', ''),
                        "description": item.get('description', ''),
                        "price": item.get('price', 0),
                        "category": item.get('category', '')
                    }
                }
                vectors.append(vector)
            
            # Upsert vectors in batches
            if vectors:
                self.index.upsert(vectors=vectors)
        except Exception as e:
            print(f"Error storing menu items: {e}")
    
    def get_recommendations(self, 
                           menu_id: str, 
                           preferences: List[str], 
                           dietary_restrictions: Optional[List[str]] = None,
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get personalized dish recommendations based on user preferences
        
        Args:
            menu_id: Identifier for the restaurant menu
            preferences: List of user food preferences
            dietary_restrictions: List of dietary restrictions
            top_k: Number of recommendations to return
            
        Returns:
            List of recommended menu items
        """
        # Fallback to dummy recommendations if Pinecone is not available
        if not self.index:
            return [
                {"name": "Example Dish", "description": "Example description", "match_score": 0.95},
                {"name": "Sample Pasta", "description": "A delicious pasta dish", "match_score": 0.92},
                {"name": "Vegan Option", "description": "Plant-based meal", "match_score": 0.90}
            ]
        
        try:
            # Combine preferences into a query
            query_text = f"I like {', '.join(preferences)}"
            if dietary_restrictions:
                query_text += f". I avoid {', '.join(dietary_restrictions)}"
                
            # Get vector embedding for the query
            query_embedding = self.embed_text(query_text)
            
            # Search for similar items in Pinecone
            results = self.index.query(
                vector=query_embedding,
                filter={"menu_id": menu_id},
                top_k=top_k,
                include_metadata=True
            )
            
            # Format recommendations
            recommendations = []
            for match in results.matches:
                recommendations.append({
                    "name": match.metadata.get("name", ""),
                    "description": match.metadata.get("description", ""),
                    "price": match.metadata.get("price", 0),
                    "category": match.metadata.get("category", ""),
                    "match_score": match.score
                })
                
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return [{"name": "Error Dish", "description": "Failed to get recommendations", "match_score": 0.0}] 