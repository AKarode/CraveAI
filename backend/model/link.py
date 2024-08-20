from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/endpoint', methods=['POST'])
def receive_url():
    data = request.get_json()
    url = data.get('menuUrl')
    
    if url:
        try:
            # Check if the URL is valid and get the final URL after redirections
            response = requests.head(url, allow_redirects=True)
            final_url = response.url

            if final_url != url:
                print(f"Error: The URL {url} redirects to {final_url}")
                return jsonify({'error': f'The URL {url} redirects to another URL ({final_url}).'}), 400
            else:
                print(f"Received URL: {url}")
                
                # Perform web scraping to get the menu data
                page_response = requests.get(url)
                page_content = page_response.content
                soup = BeautifulSoup(page_content, 'html.parser')
                
                menu_items = []

                # Find all menu items
                menu_item_divs = soup.find_all('div', class_='menu-item')

                for item in menu_item_divs:
                    # Extract item name
                    item_name = item.find('h4')
                    item_name = item_name.get_text(strip=True) if item_name else ''

                    # Extract description
                    description = item.find('p', class_='menu-item-details-description')
                    description = description.get_text(strip=True) if description else ''

                    # Extract price
                    price = item.find('li', class_='menu-item-price-amount')
                    price = price.get_text(strip=True) if price else ''
                    
                    # Add to menu_items list as a list
                    if item_name and description and price:
                        menu_items.append([item_name, description, price])

                # Print the final menu items array
                print(f"Final Menu Items:\n{menu_items}")

                if menu_items:
                    return jsonify({'message': 'Menu data retrieved successfully', 'menu': menu_items}), 200
                else:
                    return jsonify({'message': 'No valid menu items found'}), 200

        except requests.RequestException as e:
            print(f"Error: Failed to process the URL {url}. Exception: {e}")
            return jsonify({'error': 'Failed to process the URL.'}), 500

    else:
        return jsonify({'error': 'No URL provided'}), 400
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
