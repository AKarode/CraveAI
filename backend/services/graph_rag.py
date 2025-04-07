"""
Graph-based Retrieval-Augmented Generation (GraphRAG) Service

This service enhances food recommendations by using:
1. A knowledge graph built from the Uber Eats dataset
2. Vector embeddings for semantic search
3. LLM-based reasoning to navigate the graph and generate recommendations
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import openai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGService:
    """
    A service that combines knowledge graph navigation with LLM-based 
    retrieval-augmented generation for enhanced food recommendations.
    """
    
    def __init__(self, 
                 knowledge_graph_path: Optional[str] = None,
                 openai_api_key: Optional[str] = None):
        """
        Initialize the GraphRAG service
        
        Args:
            knowledge_graph_path: Path to the knowledge graph JSON file
            openai_api_key: OpenAI API key
        """
        # Load environment variables if not provided
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
        # Initialize OpenAI client
        self.openai_client = None
        if openai_api_key and openai_api_key != "your_openai_api_key_here":
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                
        # Load knowledge graph
        self.knowledge_graph = None
        if knowledge_graph_path:
            self.load_knowledge_graph(knowledge_graph_path)
        else:
            # Try to find knowledge graph in default location
            default_path = Path(__file__).parent.parent / "data" / "processed" / "knowledge_graph.json"
            if default_path.exists():
                self.load_knowledge_graph(str(default_path))
    
    def load_knowledge_graph(self, path: str) -> bool:
        """
        Load the knowledge graph from a JSON file
        
        Args:
            path: Path to the knowledge graph JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading knowledge graph from: {path}")
            with open(path, 'r') as f:
                self.knowledge_graph = json.load(f)
            return True
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        if not self.openai_client:
            # Return a random embedding if OpenAI client isn't available
            logger.warning("OpenAI client not available, using random embedding")
            return list(np.random.normal(0, 0.1, 1536))
            
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return list(np.random.normal(0, 0.1, 1536))
    
    def query_knowledge_graph(self, 
                             query: str, 
                             constraints: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Query the knowledge graph using a combination of semantic search and graph traversal
        
        Args:
            query: User query text
            constraints: Dictionary of constraints (e.g., price range, dietary preferences)
            
        Returns:
            List of relevant menu items
        """
        if not self.knowledge_graph:
            logger.warning("Knowledge graph not loaded")
            return []
            
        try:
            # 1. Parse the query using LLM to extract intents and constraints
            parsed_query = self._parse_query(query, constraints)
            
            # 2. Navigate the knowledge graph based on the parsed query
            graph_results = self._navigate_graph(parsed_query)
            
            # 3. Enhance the results with LLM reasoning
            enhanced_results = self._enhance_results(graph_results, parsed_query)
            
            return enhanced_results
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return []
    
    def _parse_query(self, query: str, constraints: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Parse the user query using an LLM to extract structured information
        
        Args:
            query: User query text
            constraints: Additional constraints
            
        Returns:
            Structured query information
        """
        if not self.openai_client:
            # Return a simple parsed query without LLM
            return {
                "original_query": query,
                "constraints": constraints or {},
                "keywords": query.lower().split()
            }
            
        # Construct a prompt for the LLM to parse the query
        constraints_text = ""
        if constraints:
            constraints_text = "Additional constraints:\n"
            for key, value in constraints.items():
                constraints_text += f"- {key}: {value}\n"
                
        prompt = f"""
        Parse the following food-related query into structured information.
        
        Query: "{query}"
        
        {constraints_text}
        
        Extract the following:
        1. Food type/dish names
        2. Ingredient preferences (likes/dislikes)
        3. Dietary restrictions
        4. Price preferences
        5. Flavor preferences
        6. Other relevant constraints
        
        Return as JSON with the following structure:
        {{
            "food_types": [],
            "ingredients": {{"likes": [], "dislikes": []}},
            "dietary_restrictions": [],
            "price_preference": null,
            "flavor_preferences": [],
            "other_constraints": []
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a food understanding assistant that parses queries into structured data."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # Extract and parse the JSON response
            parsed_json = json.loads(response.choices[0].message.content)
            
            # Add original query and constraints to the result
            parsed_json["original_query"] = query
            parsed_json["constraints"] = constraints or {}
            
            return parsed_json
        except Exception as e:
            logger.error(f"Error parsing query with LLM: {e}")
            # Fallback to simple parsing
            return {
                "original_query": query,
                "constraints": constraints or {},
                "food_types": [],
                "ingredients": {"likes": [], "dislikes": []},
                "dietary_restrictions": [],
                "price_preference": None,
                "flavor_preferences": [],
                "other_constraints": []
            }
    
    def _navigate_graph(self, parsed_query: Dict) -> List[Dict]:
        """
        Navigate the knowledge graph to find relevant nodes based on the parsed query
        
        Args:
            parsed_query: Structured query information
            
        Returns:
            List of relevant menu items
        """
        relevant_items = []
        
        # If knowledge graph isn't loaded, return empty list
        if not self.knowledge_graph:
            return relevant_items
            
        # Extract relevant information from parsed query
        food_types = parsed_query.get("food_types", [])
        liked_ingredients = parsed_query.get("ingredients", {}).get("likes", [])
        disliked_ingredients = parsed_query.get("ingredients", {}).get("dislikes", [])
        dietary_restrictions = parsed_query.get("dietary_restrictions", [])
        price_preference = parsed_query.get("price_preference")
        
        # Step 1: Find menu items matching food types
        food_type_matches = []
        for item_id, item in self.knowledge_graph["nodes"]["menu_items"].items():
            item_name = item["name"].lower()
            item_desc = item.get("description", "").lower()
            
            # Check if item matches any food type
            for food_type in food_types:
                if food_type.lower() in item_name or food_type.lower() in item_desc:
                    food_type_matches.append(item_id)
                    break
        
        # Step 2: Find menu items with liked ingredients
        ingredient_matches = []
        if liked_ingredients:
            for menu_item_id, ingredient_id in self.knowledge_graph["edges"]["item_has_ingredient"]:
                ingredient_name = self.knowledge_graph["nodes"]["ingredients"].get(ingredient_id, {}).get("name", "")
                if any(liked in ingredient_name.lower() for liked in liked_ingredients):
                    ingredient_matches.append(menu_item_id)
        
        # Step 3: Apply price filter if specified
        price_filtered_items = set()
        if price_preference:
            max_price = None
            if isinstance(price_preference, dict) and "max" in price_preference:
                max_price = price_preference["max"]
            elif isinstance(price_preference, (int, float)):
                max_price = price_preference
                
            if max_price:
                for item_id, item in self.knowledge_graph["nodes"]["menu_items"].items():
                    if item.get("price", 0) <= max_price:
                        price_filtered_items.add(item_id)
        
        # Combine results based on matching criteria
        candidate_items = set()
        
        # If we have food type matches, start with those
        if food_type_matches:
            candidate_items.update(food_type_matches)
            
        # If we have ingredient matches, either add to existing candidates or use as candidates
        if ingredient_matches:
            if candidate_items:
                # Prioritize items that match both food type and ingredients
                candidate_items = candidate_items.intersection(set(ingredient_matches))
                if not candidate_items:  # If no overlap, use union instead
                    candidate_items = set(food_type_matches).union(set(ingredient_matches))
            else:
                candidate_items = set(ingredient_matches)
        
        # Apply price filter if specified
        if price_filtered_items:
            if candidate_items:
                candidate_items = candidate_items.intersection(price_filtered_items)
            else:
                candidate_items = price_filtered_items
                
        # If no specific matches found, include all items (up to a limit)
        if not candidate_items:
            candidate_items = set(list(self.knowledge_graph["nodes"]["menu_items"].keys())[:100])
        
        # Convert candidate item IDs to full menu item data
        for item_id in candidate_items:
            if item_id in self.knowledge_graph["nodes"]["menu_items"]:
                item_data = self.knowledge_graph["nodes"]["menu_items"][item_id].copy()
                item_data["id"] = item_id
                
                # Filter out items with disliked ingredients
                has_disliked_ingredient = False
                for menu_item_id, ingredient_id in self.knowledge_graph["edges"]["item_has_ingredient"]:
                    if menu_item_id == item_id:
                        ingredient_name = self.knowledge_graph["nodes"]["ingredients"].get(ingredient_id, {}).get("name", "")
                        if any(disliked in ingredient_name.lower() for disliked in disliked_ingredients):
                            has_disliked_ingredient = True
                            break
                
                if not has_disliked_ingredient:
                    relevant_items.append(item_data)
        
        return relevant_items
    
    def _enhance_results(self, results: List[Dict], parsed_query: Dict) -> List[Dict]:
        """
        Enhance the results with additional information and explanations using an LLM
        
        Args:
            results: List of menu items from graph navigation
            parsed_query: Structured query information
            
        Returns:
            Enhanced results with explanations and relevance scores
        """
        if not results:
            return []
            
        # Calculate relevance scores based on matching criteria
        for item in results:
            score = 0.0
            reasons = []
            
            # Check for food type matches
            item_name = item["name"].lower()
            item_desc = item.get("description", "").lower()
            
            for food_type in parsed_query.get("food_types", []):
                if food_type.lower() in item_name:
                    score += 3.0
                    reasons.append(f"Dish type '{food_type}' directly matches")
                elif food_type.lower() in item_desc:
                    score += 1.5
                    reasons.append(f"Description mentions '{food_type}'")
            
            # Check for ingredient preferences
            for ingredient in parsed_query.get("ingredients", {}).get("likes", []):
                if ingredient.lower() in item_name or ingredient.lower() in item_desc:
                    score += 2.0
                    reasons.append(f"Contains preferred ingredient '{ingredient}'")
            
            # Check price preference match
            price_pref = parsed_query.get("price_preference")
            if price_pref and isinstance(price_pref, dict) and "max" in price_pref:
                if item.get("price", 0) <= price_pref["max"]:
                    score += 1.0
                    reasons.append(f"Price (${item.get('price', 0):.2f}) is within budget")
                    
            # Check flavor preferences
            for flavor in parsed_query.get("flavor_preferences", []):
                if flavor.lower() in item_name or flavor.lower() in item_desc:
                    score += 1.5
                    reasons.append(f"Matches flavor preference '{flavor}'")
            
            # Normalize score between 0 and 1
            max_possible_score = 10.0
            normalized_score = min(score / max_possible_score, 1.0)
            
            # Add score and reasons to item
            item["match_score"] = normalized_score
            item["match_reasons"] = reasons
        
        # Sort results by score
        results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        
        # Use LLM to generate personalized recommendations if available
        if self.openai_client and results:
            top_results = results[:min(5, len(results))]
            
            items_text = ""
            for idx, item in enumerate(top_results, 1):
                items_text += f"{idx}. {item['name']}: ${item.get('price', 0):.2f} - {item.get('description', '')}\n"
            
            prompt = f"""
            Based on the user's query: "{parsed_query.get('original_query')}"
            
            Here are the top matching menu items:
            {items_text}
            
            For each item, please provide:
            1. Why this dish might satisfy the user's preferences
            2. Any dietary notes relevant to the user's restrictions
            3. A personalized explanation of why they might enjoy this item
            
            Return as JSON with the following structure:
            {{
                "recommendations": [
                    {{
                        "item_index": 1,
                        "explanation": "string",
                        "dietary_notes": "string",
                        "personalized_reason": "string"
                    }}
                ]
            }}
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a culinary expert providing personalized food recommendations."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                
                # Extract and parse the JSON response
                llm_enhancements = json.loads(response.choices[0].message.content)
                
                # Add LLM-generated explanations to the results
                for enhancement in llm_enhancements.get("recommendations", []):
                    item_idx = enhancement.get("item_index")
                    if 1 <= item_idx <= len(top_results):
                        top_results[item_idx-1]["explanation"] = enhancement.get("explanation")
                        top_results[item_idx-1]["dietary_notes"] = enhancement.get("dietary_notes")
                        top_results[item_idx-1]["personalized_reason"] = enhancement.get("personalized_reason")
                
                # Update the results list with the enhanced items
                results[:len(top_results)] = top_results
            except Exception as e:
                logger.error(f"Error enhancing results with LLM: {e}")
        
        return results
    
    def generate_recommendations(self, 
                              query: str, 
                              constraints: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Generate personalized food recommendations based on user query
        
        Args:
            query: User's food request/question
            constraints: Additional constraints (dietary, price, etc.)
            
        Returns:
            Dictionary with recommendations and explanations
        """
        # Query the knowledge graph
        results = self.query_knowledge_graph(query, constraints)
        
        if not results:
            return {
                "success": False,
                "message": "No matching items found. Try a different query.",
                "recommendations": []
            }
        
        return {
            "success": True,
            "message": f"Found {len(results)} matching items",
            "recommendations": results[:10]  # Limit to top 10 recommendations
        }
    
    def answer_food_question(self, question: str) -> Dict:
        """
        Answer a food-related question using the knowledge graph and LLM
        
        Args:
            question: User's food-related question
            
        Returns:
            Answer with supporting information from the knowledge graph
        """
        if not self.openai_client or not self.knowledge_graph:
            return {
                "success": False,
                "message": "Service not fully initialized",
                "answer": "Sorry, I cannot answer your question at this time."
            }
            
        # Extract relevant information from knowledge graph
        # First, let's try to understand what the question is about
        prompt = f"""
        Analyze this food-related question: "{question}"
        
        What type of information is the user looking for?
        1. Information about a specific dish
        2. Comparison between dishes
        3. Information about ingredients
        4. Dietary information
        5. Price information
        6. Restaurant information
        7. Other
        
        Return as JSON with the following structure:
        {{
            "question_type": "string",
            "entities": [],
            "keywords": []
        }}
        """
        
        try:
            analysis_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You analyze food-related questions to determine what information to retrieve."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            question_analysis = json.loads(analysis_response.choices[0].message.content)
            
            # Find relevant information in the knowledge graph
            entities = question_analysis.get("entities", [])
            keywords = question_analysis.get("keywords", [])
            
            # Collect relevant information based on the question type
            relevant_info = []
            
            # Search menu items
            for item_id, item in self.knowledge_graph["nodes"]["menu_items"].items():
                item_name = item["name"].lower()
                item_desc = item.get("description", "").lower()
                
                # Check if item matches any entity or keyword
                for term in entities + keywords:
                    if term.lower() in item_name or term.lower() in item_desc:
                        relevant_info.append({
                            "type": "menu_item",
                            "name": item["name"],
                            "description": item.get("description", ""),
                            "price": item.get("price", 0)
                        })
                        break
            
            # Limit the number of items to avoid too long prompts
            relevant_info = relevant_info[:10]
            
            # Generate answer using the relevant information
            if relevant_info:
                info_text = json.dumps(relevant_info, indent=2)
                
                answer_prompt = f"""
                The user asked: "{question}"
                
                I found these relevant items from my knowledge base:
                {info_text}
                
                Based on this information, provide a helpful, informative answer to the user's question.
                If the information doesn't fully answer the question, acknowledge the limitations.
                """
                
                answer_response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a knowledgeable food expert providing accurate information about food and dishes."},
                        {"role": "user", "content": answer_prompt}
                    ],
                    temperature=0.7
                )
                
                return {
                    "success": True,
                    "message": "Found relevant information",
                    "answer": answer_response.choices[0].message.content,
                    "source_items": relevant_info
                }
            else:
                # No relevant information found, generate a generic response
                generic_response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful food expert. When you don't have specific information, you provide general advice and acknowledge limitations."},
                        {"role": "user", "content": f"Answer this food question as best you can, but acknowledge if you don't have specific details: '{question}'"}
                    ],
                    temperature=0.7
                )
                
                return {
                    "success": True,
                    "message": "No specific information found in knowledge base",
                    "answer": generic_response.choices[0].message.content,
                    "source_items": []
                }
                
        except Exception as e:
            logger.error(f"Error answering food question: {e}")
            return {
                "success": False,
                "message": str(e),
                "answer": "Sorry, I wasn't able to process your question."
            } 