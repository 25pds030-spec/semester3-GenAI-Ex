"""Ex 4: Image captioning using BLIP (Bootstrap Language Image Pretraining)."""

import gradio as gr
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_NAME = "Salesforce/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)


def captioner(image):
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)


app = gr.Interface(
    fn=captioner,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning using BLIP",
)

if __name__ == "__main__":
    app.launch()
