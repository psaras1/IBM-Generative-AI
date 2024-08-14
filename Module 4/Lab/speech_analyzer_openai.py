#######--------####

import os
import gradio as gr
import openai
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up your OpenAI credentials

openai.api_key = '<API_KEY>'

#######------------- LLM-------------####

# Initialize OpenAI model instance
def openai_llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "List the key points with details from the context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.1
    )
    return response.choices[0].message['content']

#######------------- Prompt Template-------------####
temp = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""

pt = PromptTemplate(
    input_variables=["context"],
    template=temp
)

# Initialize LLMChain
def prompt_to_openai(context):
    prompt = pt.format(context=context)
    return openai_llm(prompt)

#######------------- Speech2text-------------####

def transcript_audio(audio_file):
    try:
        # Initialize the speech recognition pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base.en",
            chunk_length_s=30,
        )
        
        # Transcribe the audio file
        transcript_txt = pipe(audio_file, batch_size=8)["text"]
        
        # Use the OpenAI model to process the transcription
        result = prompt_to_openai(transcript_txt)

        return result
    except Exception as e:
        return f"An error occurred: {e}"

#######------------- Gradio-------------####

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(
    fn=transcript_audio, 
    inputs=audio_input, 
    outputs=output_text, 
    title="Audio Transcription App",
    description="Upload the audio file and get the transcription summarized"
)

iface.launch(server_name="0.0.0.0", server_port=7860)
