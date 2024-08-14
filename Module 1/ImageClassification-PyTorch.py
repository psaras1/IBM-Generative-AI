import torch
import requests
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# Used to bypass macOS SSL certificate issues
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).eval()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Define image preprocessing pipeline
preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Converts the input image into a PIL Image
# and subsequently into a PyTorch tensor
def predict(image,url):
  if url and url.strip():
    image = Image.open(requests.get(url, stream=True).raw)
  elif image:
    image = image
  else:
    return {"error": "No image provided"}
  # Use processed image as input to the model
  inp = preprocess(image).unsqueeze(0)
  with torch.no_grad():
    # Softmax function is used to calculate the probability of each class
    # (which can then be interpreted as confidence levels)
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
# After processing the tensor through the model,
# it returns the predictions in the form of a dictionary named confidences
  return confidences

iface = gr.Interface(
    fn = predict,
    inputs=[
      gr.Image(type="pil", label="Upload an image"),
      gr.Textbox(placeholder="Enter an image URL", label="Or enter an image URL")],
    outputs = gr.Label(num_top_classes=3),
    # The examples component can be used for testing the model
    # examples=["/example/path.jpg", "/example/path2.jpg"]).launch()
)
iface.launch()

