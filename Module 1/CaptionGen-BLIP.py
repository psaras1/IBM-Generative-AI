import warnings
warnings.filterwarnings("ignore")

import requests

# Install the transformers library
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image
image_path ="https://media.istockphoto.com/id/1147785920/photo/using-mobile-phone.webp?s=2048x2048&w=is&k=20&c=fUDFhST7xFMrECwTe9gPwOz7yLYY_b1ULQQijLuY7qA="
image = Image.open(requests.get(image_path, stream=True).raw)

# Conditional image captioning
text = "A picture of"
inputs = processor(image, text, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# Unconditional image captioning
inputs = processor(image, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

