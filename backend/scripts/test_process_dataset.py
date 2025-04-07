#!/usr/bin/env python
"""
Standalone script to test the Uber Eats dataset processing
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the UberEatsDataset class
from services.datasets.uber_eats_loader import UberEatsDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(sample_size=1000):
    """Process a sample of the Uber Eats dataset and build knowledge graph."""
    
    # Set paths relative to the project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    restaurants_path = data_dir / "raw" / "restaurants.csv"
    menus_path = data_dir / "raw" / "restaurant-menus.csv"
    processed_dir = data_dir / "processed"
    
    # Check if the files exist
    if not restaurants_path.exists() or not menus_path.exists():
        logger.error(f"Dataset files not found. Make sure to run download_dataset.sh first!")
        logger.error(f"Expected files at: {restaurants_path} and {menus_path}")
        return
    
    logger.info(f"Processing Uber Eats dataset with sample size {sample_size}")
    
    # Initialize the dataset loader
    dataset = UberEatsDataset(
        restaurants_path=str(restaurants_path),
        menus_path=str(menus_path),
        cache_dir=str(processed_dir)
    )
    
    # Load the data (with optional sampling for development)
    logger.info(f"Loading data (sample: {sample_size})")
    dataset.load_data(sample_size=sample_size)
    
    # Clean and preprocess the data
    logger.info("Cleaning and preprocessing data")
    dataset.clean_data()
    
    # Create and save knowledge graph
    logger.info("Building knowledge graph")
    dataset.create_knowledge_graph()
    graph_path = dataset.save_knowledge_graph()
    
    logger.info(f"Processing complete! Knowledge graph saved to: {graph_path}")
    logger.info(f"Processed data available in: {processed_dir}")

if __name__ == "__main__":
    main() 