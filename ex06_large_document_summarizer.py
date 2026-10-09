"""Ex 6: Summary generator for large documents (chunk -> summarize -> join) using BART."""

import re

from transformers import BartForConditionalGeneration, BartTokenizer

MODEL_NAME = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")


def preprocess_text(text):
    return re.sub(r"\s+", " ", text).strip()


def split_text(text, chunk_size=500):
    """Transformer models have input limits, so split long documents into word chunks."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def summarize(text, max_length, min_length):
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_large_document(text):
    chunks = split_text(text)
    return " ".join(summarize(chunk, max_length=80, min_length=20) for chunk in chunks)


if __name__ == "__main__":
    document = """
Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in engineering, mathematics and computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals. High-profile applications of AI include advanced web search engines, chatbots, virtual assistants, autonomous vehicles, and play and analysis in strategy games (e.g., chess and Go). Since the 2020s, generative AI has become widely available to generate images, audio, and videos from text prompts.
"""
    document = preprocess_text(document * 20)
    print("Preprocessing completed")

    print("Generated summary (single pass, truncated to 1024 tokens):")
    print(summarize(document, max_length=100, min_length=30))

    print("\nFinal Summary (chunked):\n")
    print(summarize_large_document(document))
