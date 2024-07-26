from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key="your_openai_api_key")

@app.route('/process_menu', methods=['POST'])
def process_menu():
    data = request.json
    messages = [
        {
            "role": "user",
            "content": data['text']
        },
        {
            "type": "image_url",
            "image_url": {
                "url": data['image_url']
            }
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=400,
    )

    menu_text = response.choices[0].message.content
    lines = menu_text.split('\n')

    sorted_menu = [{"item": line.rsplit(' ', 1)[0], "price": line.rsplit(' ', 1)[1]} for line in lines if line]

    return jsonify(sorted_menu)

if __name__ == '__main__':
    app.run(debug=True)
