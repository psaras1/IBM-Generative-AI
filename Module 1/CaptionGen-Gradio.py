import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration

# BlipProcessor and BlipForConditionalGeneration from the transformers q
# libraryare used  to set up an image captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Takes  a PIL image and returns a string(capiton)
def caption_gen(image):
  try:
    caption = generate_caption(image)
    return caption
  except Exception as e:
    return f"An error occurred: {str(e)}"

interface = gr.Interface(
    fn=caption_gen,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning using BLIP",
    description="Caption images using the BLIP model from huggingface."
)
interface.launch()