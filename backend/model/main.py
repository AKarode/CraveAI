from flask import Flask, request, jsonify
import openai

app = Flask(__name__)
openai.api_key = "sk-proj-ju4sZRdIg9raP2wUggocT3BlbkFJkaMG9ovDeW6Bo03U33iB"

@app.route('/process_menu', methods=['POST'])
def process_menu():
    try:
        data = request.json
        print('Received data:', data)  # Log received data for debugging

        messages = [
            {"role": "user", "content": data['text']},
            {"role": "user", "content": data['image_url']},
        ]

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=400,
        )

        menu_text = response.choices[0].message.content
        lines = menu_text.split('\n')

        return jsonify({"reply": lines})
    except Exception as e:
        print('Error processing menu:', e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
