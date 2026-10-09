"""Ex 8: Python code generator using a code-specific LLM (Qwen2.5-Coder)."""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("Code AI Model Loaded Successfully!")


def generate_code(prompt):
    instruction = f"""You are a Python code generator.

Generate correct Python code for the following requirement.

Requirement:
{prompt}

Rules:
1. Generate only Python code.
2. Do not include explanations.
3. Do not include markdown code fences.
4. Write simple and readable code.
5. Add comments where useful.

Python code:
"""
    inputs = tokenizer(instruction, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the original instruction from the output
    return generated_text[len(instruction):].strip()


demo = gr.Interface(
    fn=generate_code,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter your Python programming requirement...",
        label="Python Programming Requirement",
    ),
    outputs=gr.Code(language="python", label="Generated Python Code"),
    title="AI Python Code Generator",
    description="Generate Python code using a code-specific AI model.",
    examples=[
        ["Write a Python program to find the factorial of a number."],
        ["Write a Python program to check whether a number is prime."],
        ["Write a Python program to calculate the average of three numbers."],
        ["Write a Python program to sort a list of numbers."],
    ],
)

if __name__ == "__main__":
    demo.launch()
