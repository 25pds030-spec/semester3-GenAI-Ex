"""Kitch AI - a recipe generator powered by google/flan-t5-large."""

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-large"

# flan-t5-large is ~3.1 GB in float32. T5 must NOT run in float16 (it was trained
# in bfloat16 and overflows to NaN in fp16), so the GPU needs room for full fp32.
FP32_NEED_BYTES = 4.5 * 1024**3


def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    free, _ = torch.cuda.mem_get_info()
    if free < FP32_NEED_BYTES:
        print(f"[Kitch AI] Only {free/1024**3:.1f} GB VRAM free; using CPU.")
        return "cpu"
    return "cuda"


print(f"[Kitch AI] Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

DEVICE = pick_device()
model = model.to(DEVICE).eval()
print(f"[Kitch AI] Ready on {DEVICE}.")

CUISINES = ["Any", "Indian", "Italian", "Mexican", "Chinese", "Thai",
            "Mediterranean", "American", "Japanese", "French"]
DIETS = ["None", "Vegetarian", "Vegan", "Gluten-Free", "High-Protein", "Low-Carb"]


@torch.no_grad()
def _ask(prompt, max_new, min_new=1):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        min_new_tokens=min_new,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _style(cuisine, diet):
    bits = []
    if cuisine != "Any":
        bits.append(cuisine)
    if diet != "None":
        bits.append(diet.lower())
    return " ".join(bits) + " " if bits else ""


def generate_recipe(ingredients, cuisine, diet, progress=gr.Progress()):
    ingredients = (ingredients or "").strip()
    if not ingredients:
        return "Please enter at least one ingredient."

    style = _style(cuisine, diet)

    progress(0.1, desc="Naming the dish...")
    title = _ask(
        f"Give a short appetizing name for a {style}dish made with {ingredients}. "
        "Answer with only the dish name.",
        max_new=24,
    )

    progress(0.4, desc="Writing the method...")
    steps = _ask(
        f"Explain step by step how to cook a {style}dish using {ingredients}. "
        "Write clear numbered cooking steps.",
        max_new=280,
        min_new=80,
    )

    progress(1.0, desc="Plating up!")
    bullets = "\n".join(
        f"- {item.strip().capitalize()}" for item in ingredients.split(",") if item.strip()
    )
    return (
        f"## {(title or 'Your Recipe').capitalize()}\n\n"
        f"### Ingredients\n{bullets}\n- Salt, oil and water - as needed\n\n"
        f"### Method\n{steps}\n"
    )


with gr.Blocks(title="Kitch AI") as demo:
    gr.Markdown(
        "# Kitch AI\n"
        "Your personal AI chef, powered by FLAN-T5-Large. "
        "Tell it what is in your kitchen and get a full recipe."
    )
    with gr.Row():
        with gr.Column():
            ingredients = gr.Textbox(
                label="Ingredients you have",
                placeholder="e.g. chicken, rice, tomato, onion, garlic",
                lines=4,
            )
            cuisine = gr.Dropdown(CUISINES, value="Any", label="Cuisine")
            diet = gr.Dropdown(DIETS, value="None", label="Diet")
            btn = gr.Button("Cook it up", variant="primary")
        with gr.Column():
            out = gr.Markdown()

    gr.Examples(
        examples=[
            ["chicken, rice, tomato, onion, garlic", "Indian", "None"],
            ["pasta, spinach, cream, parmesan", "Italian", "Vegetarian"],
            ["chickpeas, cumin, lemon, olive oil", "Mediterranean", "Vegan"],
        ],
        inputs=[ingredients, cuisine, diet],
    )

    btn.click(generate_recipe, inputs=[ingredients, cuisine, diet], outputs=out)

if __name__ == "__main__":
    demo.launch()
