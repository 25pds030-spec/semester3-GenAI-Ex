"""Ex 5: Multimodal AI assistant (image + question) using Gemini.

Set your key first:  set GEMINI_API_KEY=...   (PowerShell: $env:GEMINI_API_KEY="...")
"""

import os

import google.generativeai as genai
import gradio as gr
from PIL import Image

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")


def multimodal_assistant(image, question):
    response = model.generate_content([question, Image.open(image)])
    return response.text


app = gr.Interface(
    fn=multimodal_assistant,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Textbox(label="Ask a Question"),
    ],
    outputs=gr.Textbox(label="AI Response"),
    title="Multimodal AI Assistant using Gemini",
    description="Upload an image and ask questions about it.",
)

if __name__ == "__main__":
    app.launch()
