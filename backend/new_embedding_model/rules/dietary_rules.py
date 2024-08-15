def apply_dietary_rules(menu_items, dietary_preferences):
    """
    Filters out menu items that don't meet the user's dietary restrictions.
    
    Args:
        menu_items (list): A list of menu item dictionaries.
        dietary_preferences (dict): A dictionary of the user's dietary preferences.
        
    Returns:
        list: A filtered list of menu items that meet the dietary restrictions.
    """
    filtered_items = []
    for item in menu_items:
        if dietary_preferences.get("gluten_free") and "gluten" in item['description'].lower():
            continue
        if dietary_preferences.get("vegetarian") and "meat" in item['description'].lower():
            continue
        filtered_items.append(item)
    return filtered_items
