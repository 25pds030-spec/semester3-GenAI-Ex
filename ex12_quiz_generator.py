"""Ex 12: AI quiz generator using FLAN-T5 (local LLM, prompt engineering).

Paste a passage; the model writes a question for each chunk of it and then answers
its own question from the passage, so the quiz comes with an answer key.
"""

import re

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()


@torch.no_grad()
def ask(prompt, max_new_tokens=48):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4, no_repeat_ngram_size=3)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def split_chunks(text, sentences_per_chunk=2):
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip()) if s]
    return [" ".join(sentences[i:i + sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]


def generate_quiz(passage, num_questions):
    passage = (passage or "").strip()
    if len(passage) < 50:
        return "Please paste a longer passage (at least a couple of sentences)."

    questions, seen = [], set()
    for chunk in split_chunks(passage):
        if len(questions) >= num_questions:
            break
        question = ask(f"Generate a quiz question that can be answered from this text:\n\n{chunk}")
        if not question.endswith("?"):
            question += "?"
        if question.lower() in seen:
            continue
        seen.add(question.lower())
        answer = ask(f"Read the text and answer the question briefly.\n\nText: {passage}\n\nQuestion: {question}")
        questions.append((question, answer))

    if not questions:
        return "Could not generate any questions from this passage."

    quiz = "\n".join(f"**Q{i}.** {q}" for i, (q, _) in enumerate(questions, 1))
    key = "\n".join(f"**Q{i}.** {a}" for i, (_, a) in enumerate(questions, 1))
    return f"## Quiz\n{quiz}\n\n## Answer Key\n{key}"


demo = gr.Interface(
    fn=generate_quiz,
    inputs=[
        gr.Textbox(lines=10, label="Study passage", placeholder="Paste the text to generate a quiz from..."),
        gr.Slider(1, 10, value=5, step=1, label="Number of questions"),
    ],
    outputs=gr.Markdown(),
    title="AI Quiz Generator",
    description="Generates quiz questions with an answer key from any passage, using FLAN-T5.",
    examples=[[
        "Python is a high-level programming language created by Guido van Rossum and first released in 1991. "
        "It emphasizes code readability and supports multiple programming paradigms. "
        "Python is widely used in data science, web development and automation. "
        "Its large standard library and active community make it beginner friendly.",
        3,
    ]],
)

if __name__ == "__main__":
    demo.launch()
