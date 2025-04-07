# OpenAI vs. BERT Model Comparison

This guide explains how to use the model comparison script to demonstrate the performance benefits of OpenAI embeddings over the local BERT model for food recommendation tasks.

## Overview

The `model_comparison.py` script performs a head-to-head comparison between:
- Our custom-trained BERT model (`MLmodel/best_model.pt`)
- OpenAI's embeddings API (used in production)

The script evaluates and visualizes key performance metrics including:
1. Response time
2. Similarity score distributions
3. Ranking consistency across similar queries

## Requirements

- Make sure your `.env` file contains valid OpenAI API keys
- The script uses the sample menu image in `HumanTesting/test_menu.jpg`
- Required Python packages: matplotlib, scikit-learn, transformers (all included in requirements.txt)

## Running the Comparison

1. Ensure you're in the backend directory
2. Run the comparison script:
   ```
   python model_comparison.py
   ```
3. The script will:
   - Process the sample menu using OCR
   - Compare both models on the extracted menu items
   - Generate visualizations and a detailed report

## Interpreting the Results

The script will create several output files:
- `response_times.png`: Bar chart comparing average response times
- `similarity_distributions.png`: Histograms showing similarity score distributions
- `ranking_consistency.png`: Comparison of consistency across query variations
- `model_comparison_report.json`: Detailed metrics and conclusions

## What to Expect

Typically, you'll see that the OpenAI embeddings demonstrate advantages in:
1. **Response quality**: More consistent rankings for similar queries
2. **Understanding subtlety**: Better grasp of nuanced preferences
3. **Domain knowledge**: Stronger food-specific understanding

The BERT model may perform well on basic queries but will generally show less consistency and semantic understanding compared to OpenAI's embeddings.

## Using This For Presentations

This comparison provides data-driven evidence for why we've chosen to use OpenAI embeddings in production. The visualizations can be directly used in presentations to demonstrate the performance differences.

## Notes

- The script limits API usage by only processing the first 10 menu items
- For a more comprehensive comparison, increase the number of test queries and menu items (but be mindful of API costs) 