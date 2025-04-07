#!/usr/bin/env python
"""
Test script for the GraphRAG service
This script demonstrates how to use the GraphRAG service with the Uber Eats dataset
"""

import os
import json
import sys
from pathlib import Path

# Add the parent directory to the Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.graph_rag import GraphRAGService
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Test the GraphRAG service with sample queries"""
    # Path to knowledge graph
    project_root = Path(__file__).parent.parent
    knowledge_graph_path = project_root / "data" / "processed" / "knowledge_graph.json"
    
    # Check if knowledge graph exists
    if not knowledge_graph_path.exists():
        print(f"Knowledge graph not found at {knowledge_graph_path}")
        print("Please run the dataset processing script first:")
        print("python -m services.datasets.process_uber_eats_data")
        return
    
    # Initialize the GraphRAG service
    rag_service = GraphRAGService(knowledge_graph_path=str(knowledge_graph_path))
    
    # Sample food queries to test
    test_queries = [
        "I want a spicy chicken dish under $15",
        "What vegetarian options are available?",
        "I'm looking for a dessert with chocolate",
        "Can you recommend Italian pasta dishes with seafood?",
        "What's a good breakfast option that's healthy?",
    ]
    
    # Sample food questions to test
    test_questions = [
        "What ingredients are typically in a margherita pizza?",
        "What's the difference between a burrito and a chimichanga?",
        "How spicy is a typical Thai green curry?",
        "What desserts contain chocolate but are still low in calories?",
    ]
    
    # Test recommendation queries
    print("\n===== TESTING FOOD RECOMMENDATIONS =====\n")
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: '{query}'")
        print("-" * 50)
        
        result = rag_service.generate_recommendations(query)
        
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        print(f"Found {len(result['recommendations'])} recommendations")
        
        # Display top 3 recommendations
        for j, rec in enumerate(result['recommendations'][:3], 1):
            print(f"\nRecommendation {j}:")
            print(f"  Name: {rec['name']}")
            print(f"  Price: ${rec.get('price', 0):.2f}")
            print(f"  Match Score: {rec.get('match_score', 0):.2f}")
            if 'match_reasons' in rec:
                print(f"  Match Reasons: {', '.join(rec['match_reasons'])}")
            if 'description' in rec:
                print(f"  Description: {rec['description']}")
            if 'explanation' in rec:
                print(f"  Explanation: {rec['explanation']}")
        
        print("\n" + "=" * 50)
    
    # Test food questions
    print("\n===== TESTING FOOD QUESTIONS =====\n")
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: '{question}'")
        print("-" * 50)
        
        result = rag_service.answer_food_question(question)
        
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        print(f"\nAnswer: {result['answer']}")
        
        if result.get('source_items'):
            print(f"\nBased on {len(result['source_items'])} source items, including:")
            for j, item in enumerate(result['source_items'][:3], 1):
                print(f"  {j}. {item.get('name')} - ${item.get('price', 0):.2f}")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 