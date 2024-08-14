import whisper
import certifi
import ssl

import warnings
warnings.filterwarnings("ignore")

# Set up SSL context to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context

# Load the Whisper model
# Sizes: tiny,base,small,medium,large
# (larger models are more accurate but slower)
model = whisper.load_model("base")

# Transcribe the audio file
result = model.transcribe("resources/ex.mp3")

# Output the transcription
print(result["text"])