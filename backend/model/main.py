from flask import Flask, request, jsonify
import openai
import sys,os
##from backend.vectorDB.similaritysearch import findtopmatchingitems
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vectorDB')))

from similarity_search import find_top_matching_items

app = Flask(__name__)

@app.route('/process_menu', methods=['POST'])
def process_menu():
    try:
        data = request.json
        print('Received data:', data)  # Log received data for debugging
        query_list = find_top_matching_items('user_124', data['text'], top_k=5) 
        input_text =  'this is the most similar items from that query:' + query_list[0][0] + query_list[1][0] + query_list[2][0] + 'pick the best item from in them and make a concise and to the point recommendation to the user'

        messages = [
            {"role": "user", "content": data['text']},
            {"role": "system", "content": input_text},
        ]

        response = openai.chat.completions.create(
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

if __name__ == '__main__':
    app.run(debug=True)