from openai import OpenAI
import os
import json
import re
from main import fetch_user_preferences

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to filter the menu items according to user preferences using CrewAI and GPT-4 API
def filter_menu(user_id, menu_items):
    user_preferences = fetch_user_preferences(user_id)
    # Assuming user_preferences is a tuple, extract the dictionary
    if isinstance(user_preferences, (list, tuple)) and len(user_preferences) > 0:
        user_preferences = user_preferences[0]
    elif isinstance(user_preferences, dict):
        user_preferences = user_preferences
    else:
        raise TypeError("Invalid format for user_preferences. Expected dict or a tuple/list containing a dict.")
    
    # Prepare the message for GPT-4 to filter the menu based on user preferences
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that filters a menu based on user dietary preferences and restrictions."
        },
        {
            "role": "user",
            "content": f"User preferences: {json.dumps(user_preferences)}\n\nMenu items: {json.dumps(menu_items)}\n\nFilter the menu items to only include those that match the user's dietary preferences and restrictions. Return only the filtered menu as a JSON-encoded list of dictionaries in the same format as the menu_items."
        }
    ]
    
    # Make the API call to GPT-4 using the client instance
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=1000,
        temperature=0.5
    )
    
    # Parse the response
    filtered_menu_content = response.choices[0].message.content.strip()
    # Use regex to extract JSON content if present
    json_match = re.search(r'\[.*\]', filtered_menu_content, re.DOTALL)
    if json_match:
        filtered_menu_content = json_match.group(0)
    else:
        raise ValueError("Failed to find JSON in GPT-4 response. The response was: " + filtered_menu_content)
    
    try:
        filtered_menu = json.loads(filtered_menu_content)
    except json.JSONDecodeError:
        raise ValueError("Failed to decode JSON from GPT-4 response. The response was: " + filtered_menu_content)
    
    return filtered_menu

# Example usage
menu_data = [
    {"name": "Classic Cheeseburger", "description": "A juicy beef patty topped with melted cheddar cheese, lettuce, tomato, pickles, and onions on a toasted sesame seed bun. Served with a side of crispy fries.", "price": "$10.99"},
    {"name": "Grilled Chicken Caesar Salad", "description": "Grilled chicken breast served over a bed of fresh romaine lettuce, tossed with creamy Caesar dressing, Parmesan cheese, and croutons.", "price": "$9.99"},
    {"name": "Margarita Pizza", "description": "A classic pizza topped with fresh mozzarella cheese, tomatoes, basil, and a drizzle of olive oil on a crispy thin crust.", "price": "$12.49"},
    {"name": "Vegan Buddha Bowl", "description": "A nutritious bowl filled with quinoa, roasted chickpeas, avocado, sweet potatoes, kale, and a tahini dressing.", "price": "$11.99"},
    {"name": "Spaghetti Carbonara", "description": "Spaghetti pasta tossed in a creamy sauce made with eggs, Parmesan cheese, pancetta, and black pepper.", "price": "$13.49"},
    {"name": "Buffalo Wings", "description": "Spicy buffalo chicken wings served with celery sticks and a side of blue cheese dipping sauce.", "price": "$8.99"}
]

user_id = "user_123"
filtered_menu = filter_menu(user_id, menu_data)
print(filtered_menu)
