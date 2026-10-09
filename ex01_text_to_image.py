"""Ex 1: Text-to-image generation using Stable Diffusion."""

import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"

device = "cuda" if torch.cuda.is_available() else "cpu"
# float16 is only supported on GPU
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)

prompt = input("Enter a text prompt: ")
image = pipe(prompt).images[0]

image.save("generated_image.png")
print("Image saved as generated_image.png")

plt.imshow(image)
plt.axis("off")
plt.show()
