from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

url = None

@app.route('/endpoint', methods=['POST'])
def receive_url():
    global url
    data = request.get_json()
    url = data.get('menuUrl')
    
    if url:
        
        try:
            response = requests.head(url, allow_redirects=True)
            final_url = response.url

            # Check if the final URL is different from the original URL
            if final_url != url:
                print(f"Error: The URL {url} redirects to {final_url}")
                return jsonify({'error': f'The URL {url} redirects to another URL ({final_url}).'}), 400
            else:
                print(f"Received URL: {url}")
                return jsonify({'message': 'URL received successfully'}), 200

        except requests.RequestException as e:
            print(f"Error: Failed to process the URL {url}. Exception: {e}")
            return jsonify({'error': 'Failed to process the URL.'}), 500
        
    else:
        return jsonify({'error': 'No URL provided'}), 400
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
