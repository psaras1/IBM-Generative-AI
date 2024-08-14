# Responsible for the backend - Flask framework

import base64
import json
from flask import Flask, render_template, request
# Functions from the worker.py file
from worker import speech_to_text, text_to_speech, openai_process_message
from flask_cors import CORS
import os

app = Flask(__name__)
# CORS policies are set to prevent/allow web pages from making requests to different domains
# * option allows any request
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# When a user goes to the / endpoint, they get the index.html (frontend interface)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# /speech-to-text and /process-message are what will be used to process all requests 
# Used to convert the user's STT using the speech_to_text function from worker.py
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    print("processing speech-to-text")
    audio_binary = request.data # Get the user's speech from their request
    text = speech_to_text(audio_binary) # Call speech_to_text function to transcribe the speech

    # Return the response back to the user in JSON format(as expected by the front end)
    response = app.response_class(
        # Actual data to send in the body of the HTTP response(simple dictionary containing single key-value pair)
        response=json.dumps({'text': text}),
        # Status code of HTTP response(200: OK)
        status=200,
        # Format of our response
        mimetype='application/json'
    )
    print(response)
    print(response.data)
    return response


@app.route('/process-message', methods=['POST'])
# Accepts a user's message in text form with their preferred voice
# Then uses helper functions to call the OpenAI API to process this prompt 
# And finally convert the response tot text(using Watson's TTS)
def process_prompt_route():
    user_message = request.json['userMessage'] # Get user's message from their request
    print('user_message', user_message)

    voice = request.json['voice'] # Get user's preferred voice from their request
    print('voice', voice)

    # Call openai_process_message function to process the user's message and get a response back
    openai_response_text = openai_process_message(user_message)
    
    # Clean the response to remove any emptylines
    openai_response_text = os.linesep.join([s for s in openai_response_text.splitlines() if s])
    
    # Call our text_to_speech function to convert OpenAI Api's reponse to speech
    openai_response_speech = text_to_speech(openai_response_text, voice)
    
    # convert openai_response_speech to base64 string so it can be sent back in the JSON response
    openai_response_speech = base64.b64encode(openai_response_speech).decode('utf-8')
    
    # Send a JSON response back to the user containing their message's response both in text and speech formats
    response = app.response_class(
        response=json.dumps({"openaiResponseText": openai_response_text, "openaiResponseSpeech": openai_response_speech}),
        status=200,
        mimetype='application/json'
    )
    print(response)
    return response

# (0.0.0.0 = localhost)
if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0')
