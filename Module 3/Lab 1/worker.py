from openai import OpenAI
import requests

openai_client = OpenAI()



# Takes audio_binary as the only parameter and then sends it in the body of the HTTP request
def speech_to_text(audio_binary):
    # Set up Watson Speech-to-Text HTTP Api url
    base_url = "https://sn-watson-stt.labs.skills.network"
    api_url = base_url+'/speech-to-text/api/v1/recognize'

    # Set up parameters for our HTTP reqeust
    params = {
        'model': 'en-US_Multimedia',
    }

    # Set up the body of our HTTP request
    body = audio_binary

    # Send a HTTP Post request
    response = requests.post(api_url, params=params, data=audio_binary).json()

    # Parse the response to get our transcribed text
    text = 'null'

    # If the response contains any response, extract the text
    while bool(response.get('results')):
        print('speech to text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognised text: ', text)
        return text

# Takes both text and voice as parameters
# Adds the parameter to the api_url and sends it in the body of the HTTP request
def text_to_speech(text, voice=""):
    # Set up Watson Text-to-Speech HTTP API url
    base_url = "https://sn-watson-tts.labs.skills.network"
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'
    
    # Adding voice parameter in api_url if the user has selected a preferred voice
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice
    # Set the headers for our HTTP request
    headers = {
        # Tells Watson that we are sending an audio having wav format
        'Accept': 'audio/wav',
        # Means that the format of the body would be JSON
        'Content-Type': 'application/json',
    }
    # Set the body of our HTTP request
    json_data = {
        'text': text,
    }
    # Send a HTTP Post request to Watson Text-to-Speech Service
    response = requests.post(api_url, headers=headers, json=json_data)
    print('text to speech response:', response)
    return response.content


# Takes in a prompt and passes it to the GPT API to receive a response
def openai_process_message(user_message):
    # Set the prompt for OpenAI API
    prompt = "Act like an asshole, response in a rude manner and improper languae."
    
    # Call the OpenAI API to process our prompt
    openai_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", 
        # Array of objects used to define the conversation flow between the user and the AI
        # Helps tailor the AI's responses to be more relevant and personalized
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ],
        # 30 tokens correspond to roughly 1-2 sentences
        max_tokens=4000
    )
    print("openai response:", openai_response)

    # Parse the response to get the response message for our prompt(initially json format)
    response_text = openai_response.choices[0].message.content
    return response_text



