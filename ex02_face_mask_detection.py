"""Ex 2: Face mask detection using a pretrained image classifier + Gradio."""

import gradio as gr
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "DamarJati/Face-Mask-Detection"

model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

id2label = {0: "Mask Found", 1: "Mask Not Found"}


def detect_face_mask(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
    return id2label[torch.argmax(logits, dim=1).item()]


iface = gr.Interface(
    fn=detect_face_mask,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Text(label="Mask Status"),
    title="Face Mask Detection",
    description="Upload an image to check whether a face mask is present.",
)

if __name__ == "__main__":
    iface.launch()
