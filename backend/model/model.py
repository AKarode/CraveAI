## Give me the text that user will input in the chatbot.
## Just create a variable named `chat_input` and assign the text to it.
# chat_input = request.json.get('text') // This is the user's message

## Give me the command that I can use to respond to that http request.
# output = crew.kickoff()               // Assuming this is the response that you want the model to output
# return jsonify({"reply": output})     // This is the command to respond to the http request

# Required imports for flask server
from flask import Flask, request, jsonify
