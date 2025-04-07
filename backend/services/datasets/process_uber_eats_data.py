"""
Process the downloaded Uber Eats dataset 

This script uses the UberEatsDataset class to:
1. Load the raw CSV files from the Kaggle dataset
2. Clean and preprocess the data
3. Build a knowledge graph
4. Save processed data for later use in the recommendation system
"""

import os
import argparse
import logging
from pathlib import Path
from .uber_eats_loader import UberEatsDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(sample_size=None):
    """Process the Uber Eats dataset and build knowledge graph."""
    
    # Set paths relative to the project root
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "backend" / "data"
    
    restaurants_path = data_dir / "raw" / "restaurants.csv"
    menus_path = data_dir / "raw" / "restaurant-menus.csv"
    processed_dir = data_dir / "processed"
    
    # Check if the files exist
    if not restaurants_path.exists() or not menus_path.exists():
        logger.error(f"Dataset files not found. Make sure to run download_dataset.sh first!")
        logger.error(f"Expected files at: {restaurants_path} and {menus_path}")
        return
    
    logger.info(f"Processing Uber Eats dataset with{'out' if sample_size is None else ''} sampling")
    
    # Initialize the dataset loader
    dataset = UberEatsDataset(
        restaurants_path=str(restaurants_path),
        menus_path=str(menus_path),
        cache_dir=str(processed_dir)
    )
    
    # Load the data (with optional sampling for development)
    logger.info(f"Loading data{' (sample)' if sample_size else ''}")
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
    parser = argparse.ArgumentParser(description="Process Uber Eats dataset")
    parser.add_argument("--sample", type=int, help="Number of samples to process (for testing)")
    args = parser.parse_args()
    
    sample_size = args.sample
    main(sample_size) 