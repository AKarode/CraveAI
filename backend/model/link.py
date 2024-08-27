from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Route for receiving menu data from a URL
@app.route('/endpoint', methods=['POST'])
def receive_data():
    if request.is_json:
        data = request.get_json()
        url = data.get('menuUrl')
        
        if url:
            try:
                # Check if the URL is valid and get the final URL after redirections
                response = requests.head(url, allow_redirects=True)
                final_url = response.url

                if final_url != url:
                    print(f"Error: The URL {url} redirects to {final_url}")
                    return jsonify({'valid': False, 'error': f'The URL {url} redirects to another URL ({final_url}).'}), 400
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
                        
                        # Add to menu_items list as a dictionary
                        if item_name and description and price:
                            menu_items.append({
                                'name': item_name,
                                'description': description,
                                'price': price
                            })

                    # Print the final menu items array
                    print(f"Final Menu Items:\n{menu_items}")

                    if menu_items:
                        return jsonify({'valid': True, 'message': 'Menu data retrieved successfully', 'menu': menu_items}), 200
                    else:
                        return jsonify({'valid': True, 'message': 'No valid menu items found'}), 200

            except requests.RequestException as e:
                print(f"Error: Failed to process the URL {url}. Exception: {e}")
                return jsonify({'valid': False, 'error': 'Failed to process the URL.'}), 500

        else:
            return jsonify({'valid': False, 'error': 'No URL provided'}), 400
    
    elif request.content_type.startswith('multipart/form-data'):
        # Handle image data
        if 'images[]' in request.files:
            images = request.files.getlist('images[]')  # Handle multiple images
            if images:
                # Process each image (Placeholder logic)
                for image in images:
                    print(f"Received image: {image.filename}")
                    # Add image processing logic here if needed

                return jsonify({'valid': True, 'message': 'Images received successfully'}), 200
            else:
                return jsonify({'valid': False, 'error': 'No images provided'}), 400
        else:
            return jsonify({'valid': False, 'error': 'No images key in form-data'}), 400
    
    else:
        return jsonify({'valid': False, 'error': 'Unsupported Content-Type'}), 400

# Route for processing menu text using OpenAI GPT-4o
@app.route('/process_menu', methods=['POST'])
def process_menu():
    try:
        data = request.json
        print('Received data:', data)  # Log received data for debugging
        query_list = find_top_matching_items('user_124', data['text'], top_k=5)
        input_text = 'this is the most similar items from that query:' + query_list[0][0] + query_list[1][0] + query_list[2][0] + 'pick the best item from in them and make a concise and to the point recommendation to the user'

        messages = [
            {"role": "user", "content": data['text']},
            {"role": "system", "content": input_text},
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=400,
        )


        menu_text = response.choices[0].message.content
        lines = menu_text.split('\n')

        return jsonify({"reply": lines})

    except Exception as exception:
        print('Error processing menu:', exception)
        return jsonify({"error": str(exception)}), 500

# Run the Flask app with both routes under different ports
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
