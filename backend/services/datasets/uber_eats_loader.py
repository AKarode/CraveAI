"""
Uber Eats Dataset Loader and Processor

This module processes the Uber Eats dataset from Kaggle to create enriched menu data
for use in the CraveAI recommendation system.

The dataset contains:
- restaurants.csv: 63k+ entries with restaurant information
- restaurant-menus.csv: 5M+ entries with menu items, descriptions, and prices

Dataset source: https://www.kaggle.com/datasets/ahmedshahriarsakib/uber-eats-usa-restaurants-menus
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import re
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UberEatsDataset:
    """
    A class to load, process, and provide access to the Uber Eats dataset
    """
    
    def __init__(self, 
                 restaurants_path: Optional[str] = None, 
                 menus_path: Optional[str] = None,
                 cache_dir: str = './data/processed'):
        """
        Initialize the dataset loader.
        
        Args:
            restaurants_path: Path to restaurants.csv file
            menus_path: Path to restaurant-menus.csv file
            cache_dir: Directory to store processed data
        """
        self.restaurants_path = restaurants_path
        self.menus_path = menus_path
        self.cache_dir = Path(cache_dir)
        self.restaurants_df = None
        self.menus_df = None
        self.knowledge_graph = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the Uber Eats dataset from CSV files
        
        Args:
            sample_size: If provided, only load a sample of this size
            
        Returns:
            Tuple of (restaurants_df, menus_df)
        """
        try:
            if self.restaurants_path and os.path.exists(self.restaurants_path):
                logger.info(f"Loading restaurant data from {self.restaurants_path}")
                self.restaurants_df = pd.read_csv(self.restaurants_path)
                if sample_size:
                    self.restaurants_df = self.restaurants_df.sample(min(sample_size, len(self.restaurants_df)))
            else:
                logger.warning("Restaurants data path not provided or doesn't exist")
                self.restaurants_df = pd.DataFrame()
                
            if self.menus_path and os.path.exists(self.menus_path):
                logger.info(f"Loading menu data from {self.menus_path}")
                # Use chunksize for large CSV to avoid memory issues
                chunks = []
                for chunk in pd.read_csv(self.menus_path, chunksize=100000):
                    if sample_size and len(chunks) * 100000 >= sample_size:
                        break
                    chunks.append(chunk)
                self.menus_df = pd.concat(chunks, ignore_index=True)
                if sample_size:
                    self.menus_df = self.menus_df.sample(min(sample_size, len(self.menus_df)))
            else:
                logger.warning("Menu data path not provided or doesn't exist")
                self.menus_df = pd.DataFrame()
                
            return self.restaurants_df, self.menus_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean and preprocess the dataset
        
        Returns:
            Tuple of (cleaned_restaurants_df, cleaned_menus_df)
        """
        if self.restaurants_df is None or self.menus_df is None:
            logger.warning("Data not loaded, calling load_data first")
            self.load_data()
            
        try:
            # Clean restaurant data
            if not self.restaurants_df.empty:
                logger.info("Cleaning restaurant data")
                self.restaurants_df = self.restaurants_df.dropna(subset=['name', 'category'])
                # Convert price range to numeric value
                price_to_numeric = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
                self.restaurants_df['price_numeric'] = self.restaurants_df['price_range'].map(price_to_numeric).fillna(0)
                
            # Clean menu data
            if not self.menus_df.empty:
                logger.info("Cleaning menu data")
                self.menus_df = self.menus_df.dropna(subset=['name', 'price'])
                # Convert price to float - handle currency units like USD, €, etc.
                self.menus_df['price'] = self.menus_df['price'].astype(str).str.replace('$', '')
                self.menus_df['price'] = self.menus_df['price'].str.replace(',', '')
                # Remove any currency unit (like USD, EUR, etc.)
                self.menus_df['price'] = self.menus_df['price'].str.replace(r'\s*[A-Za-z]+\s*$', '', regex=True)
                self.menus_df['price'] = pd.to_numeric(self.menus_df['price'], errors='coerce')
                self.menus_df = self.menus_df.dropna(subset=['price'])  # Drop rows with invalid prices
                
                # Extract ingredients from descriptions using NLP techniques
                self.menus_df['ingredients'] = self.menus_df['description'].apply(self.extract_ingredients)
                
            return self.restaurants_df, self.menus_df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
            
    def extract_ingredients(self, description: Optional[str]) -> List[str]:
        """
        Extract potential ingredients from menu item descriptions
        
        Args:
            description: Menu item description text
            
        Returns:
            List of extracted ingredients
        """
        if not description or not isinstance(description, str):
            return []
            
        # This is a simple regex-based extraction
        # In a production system, you would use a more robust NER model
        ingredients = []
        
        # Common food ingredients to look for
        common_ingredients = [
            'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'shrimp',
            'cheese', 'mozzarella', 'cheddar', 'parmesan',
            'tomato', 'lettuce', 'onion', 'garlic', 'potato', 'mushroom',
            'rice', 'pasta', 'noodle', 'bread', 'sauce', 'oil', 'herb', 'spice'
        ]
        
        # Look for common ingredients in the description
        for ingredient in common_ingredients:
            if re.search(r'\b' + ingredient + r'\b', description.lower()):
                ingredients.append(ingredient)
                
        return ingredients
        
    def create_knowledge_graph(self) -> Dict:
        """
        Create a knowledge graph from the dataset
        
        Returns:
            A dictionary representing the knowledge graph
        """
        if self.restaurants_df is None or self.menus_df is None:
            logger.warning("Data not loaded, calling clean_data first")
            self.clean_data()
            
        logger.info("Creating knowledge graph")
        
        # Initialize graph structure
        graph = {
            'nodes': {
                'restaurants': {},
                'menu_items': {},
                'categories': {},
                'ingredients': {}
            },
            'edges': {
                'restaurant_serves': [],  # (restaurant_id, menu_item_id)
                'item_has_ingredient': [],  # (menu_item_id, ingredient_id)
                'restaurant_has_category': [],  # (restaurant_id, category_id)
                'menu_item_in_category': []  # (menu_item_id, category_id)
            }
        }
        
        # Process restaurants
        for _, row in self.restaurants_df.iterrows():
            restaurant_id = str(row['id'])
            graph['nodes']['restaurants'][restaurant_id] = {
                'name': row['name'],
                'score': row.get('score', 0),
                'ratings': row.get('ratings', 0),  # Adding ratings count
                'price_range': row.get('price_range', ''),
                'address': row.get('full_address', ''),
                'zip_code': row.get('zip_code', ''),  # Adding zip code
                'latitude': row.get('lat', None),     # Adding latitude
                'longitude': row.get('long', None)    # Adding longitude
            }
            
            # Process restaurant categories
            categories = str(row['category']).split(',')
            for category in categories:
                category = category.strip()
                if category:
                    if category not in graph['nodes']['categories']:
                        graph['nodes']['categories'][category] = {
                            'name': category
                        }
                    graph['edges']['restaurant_has_category'].append((restaurant_id, category))
        
        # Process menu items
        menu_item_id_counter = 0
        
        for _, row in self.menus_df.iterrows():
            restaurant_id = str(row['restaurant_id'])
            menu_item_id = f"item_{menu_item_id_counter}"
            menu_item_id_counter += 1
            
            graph['nodes']['menu_items'][menu_item_id] = {
                'name': row['name'],
                'description': row.get('description', ''),
                'price': row.get('price', 0),
                'category': row.get('category', '')
            }
            
            # Link menu item to restaurant
            graph['edges']['restaurant_serves'].append((restaurant_id, menu_item_id))
            
            # Process menu item category
            category = str(row.get('category', '')).strip()
            if category:
                if category not in graph['nodes']['categories']:
                    graph['nodes']['categories'][category] = {
                        'name': category
                    }
                graph['edges']['menu_item_in_category'].append((menu_item_id, category))
            
            # Process ingredients
            for ingredient in row.get('ingredients', []):
                if ingredient not in graph['nodes']['ingredients']:
                    graph['nodes']['ingredients'][ingredient] = {
                        'name': ingredient
                    }
                graph['edges']['item_has_ingredient'].append((menu_item_id, ingredient))
        
        self.knowledge_graph = graph
        return graph
    
    def save_knowledge_graph(self, filepath: Optional[str] = None) -> str:
        """
        Save the knowledge graph to a file
        
        Args:
            filepath: Where to save the knowledge graph
            
        Returns:
            Path to the saved file
        """
        if self.knowledge_graph is None:
            logger.warning("Knowledge graph not created, calling create_knowledge_graph first")
            self.create_knowledge_graph()
            
        if filepath is None:
            filepath = os.path.join(self.cache_dir, 'knowledge_graph.json')
            
        logger.info(f"Saving knowledge graph to {filepath}")
        
        with open(filepath, 'w') as f:
            json.dump(self.knowledge_graph, f)
            
        return filepath
    
    def get_restaurant_menu(self, restaurant_id: str) -> List[Dict]:
        """
        Get all menu items for a specific restaurant
        
        Args:
            restaurant_id: ID of the restaurant
            
        Returns:
            List of menu items for the restaurant
        """
        if self.menus_df is None:
            logger.warning("Data not loaded, calling load_data first")
            self.load_data()
            
        menu_items = self.menus_df[self.menus_df['restaurant_id'] == restaurant_id].to_dict('records')
        return menu_items
    
    def search_menu_items(self, query: str) -> List[Dict]:
        """
        Simple search for menu items containing the query string
        
        Args:
            query: Search string
            
        Returns:
            List of matching menu items
        """
        if self.menus_df is None:
            logger.warning("Data not loaded, calling load_data first")
            self.load_data()
            
        # Simple text match search (in production, use vector search)
        query_lower = query.lower()
        matches = self.menus_df[
            self.menus_df['name'].str.lower().str.contains(query_lower) | 
            self.menus_df['description'].str.lower().str.contains(query_lower)
        ]
        
        return matches.to_dict('records')

# Example usage
if __name__ == "__main__":
    dataset = UberEatsDataset(
        restaurants_path='path/to/restaurants.csv',
        menus_path='path/to/restaurant-menus.csv'
    )
    
    # Load a sample for testing
    dataset.load_data(sample_size=10000)
    
    # Clean the data
    dataset.clean_data()
    
    # Create and save the knowledge graph
    dataset.create_knowledge_graph()
    graph_path = dataset.save_knowledge_graph()
    
    print(f"Knowledge graph saved to {graph_path}") 