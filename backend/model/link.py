from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/endpoint', methods=['POST'])
def receive_url():
    data = request.get_json()
    url = data.get('url')
    if url:
        # Handle the URL (e.g., store it in a database or process it)
        print(f"Received URL: {url}")
        return jsonify({'message': 'URL received successfully'}), 200
    else:
        return jsonify({'error': 'No URL provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
