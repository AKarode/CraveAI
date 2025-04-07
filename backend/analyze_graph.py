#!/usr/bin/env python
"""
Script to analyze the knowledge graph generated from Uber Eats dataset
"""

import json
import os

# Load the knowledge graph
graph_path = os.path.join('data', 'processed', 'knowledge_graph.json')
with open(graph_path, 'r') as f:
    graph = json.load(f)

# Display node counts
print("=== KNOWLEDGE GRAPH STATISTICS ===")
print("\nNode counts:")
print(f"Restaurants: {len(graph['nodes']['restaurants'])}")
print(f"Menu items: {len(graph['nodes']['menu_items'])}")
print(f"Categories: {len(graph['nodes']['categories'])}")
print(f"Ingredients: {len(graph['nodes']['ingredients'])}")

# Display edge counts
print("\nEdge counts:")
print(f"Restaurant serves: {len(graph['edges']['restaurant_serves'])}")
print(f"Item has ingredient: {len(graph['edges']['item_has_ingredient'])}")
print(f"Restaurant has category: {len(graph['edges']['restaurant_has_category'])}")
print(f"Menu item in category: {len(graph['edges']['menu_item_in_category'])}")

# Sample restaurant data
print("\n=== SAMPLE DATA ===")
print("\nSample Restaurant:")
restaurant_id = list(graph['nodes']['restaurants'].keys())[0]
print(f"ID: {restaurant_id}")
print(json.dumps(graph['nodes']['restaurants'][restaurant_id], indent=2))

# Sample menu item data
print("\nSample Menu Item:")
menu_item_id = list(graph['nodes']['menu_items'].keys())[0]
print(f"ID: {menu_item_id}")
print(json.dumps(graph['nodes']['menu_items'][menu_item_id], indent=2))

# Sample categories
print("\nSample Categories:")
for i, category in enumerate(list(graph['nodes']['categories'].keys())[:5]):
    print(f"{i+1}. {category}")

# Sample ingredients
print("\nSample Ingredients:")
for i, ingredient in enumerate(list(graph['nodes']['ingredients'].keys())[:5]):
    print(f"{i+1}. {ingredient}")

# Sample edges
print("\n=== SAMPLE RELATIONSHIPS ===")
print("\nRestaurant serves Menu Item:")
for i, (restaurant, menu_item) in enumerate(graph['edges']['restaurant_serves'][:3]):
    r_name = graph['nodes']['restaurants'].get(restaurant, {}).get('name', 'Unknown')
    m_name = graph['nodes']['menu_items'].get(menu_item, {}).get('name', 'Unknown')
    print(f"{restaurant} ({r_name}) -> {menu_item} ({m_name})")

print("\nMenu Item has Ingredient:")
for i, (menu_item, ingredient) in enumerate(graph['edges']['item_has_ingredient'][:3]):
    m_name = graph['nodes']['menu_items'].get(menu_item, {}).get('name', 'Unknown')
    print(f"{menu_item} ({m_name}) -> {ingredient}")

print("\nRestaurant has Category:")
for i, (restaurant, category) in enumerate(graph['edges']['restaurant_has_category'][:3]):
    r_name = graph['nodes']['restaurants'].get(restaurant, {}).get('name', 'Unknown')
    print(f"{restaurant} ({r_name}) -> {category}")

print("\nMenu Item in Category:")
for i, (menu_item, category) in enumerate(graph['edges']['menu_item_in_category'][:3]):
    m_name = graph['nodes']['menu_items'].get(menu_item, {}).get('name', 'Unknown')
    print(f"{menu_item} ({m_name}) -> {category}") 