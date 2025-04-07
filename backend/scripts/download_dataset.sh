#!/bin/bash
# Script to download the Uber Eats dataset from Kaggle
# Prerequisites: 
# 1. Install Kaggle CLI: pip install kaggle
# 2. Set up Kaggle API credentials in ~/.kaggle/kaggle.json

# Create data directory structure
mkdir -p backend/data/raw
mkdir -p backend/data/processed

echo "Downloading Uber Eats dataset from Kaggle..."
kaggle datasets download ahmedshahriarsakib/uber-eats-usa-restaurants-menus -p backend/data/raw

echo "Extracting dataset..."
unzip -o backend/data/raw/uber-eats-usa-restaurants-menus.zip -d backend/data/raw

echo "Dataset downloaded and extracted to backend/data/raw/"
echo "Files available:"
ls -lh backend/data/raw/

echo "Now you can run the data processing script to load this dataset"
echo "Example usage:"
echo "python -m backend.services.datasets.process_uber_eats_data" 