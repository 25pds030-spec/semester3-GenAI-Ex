"""Ex 9: Conversation generator using a fine-tuned GPT-style model (DialoGPT-small)."""

import gradio as gr
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "microsoft/DialoGPT-small"
OUTPUT_DIR = "./conversation_model"

data = {
    "text": [
        "User: Hello\nAssistant: Hello! How can I help you today?",
        "User: What is machine learning?\nAssistant: Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data.",
        "User: What is Python?\nAssistant: Python is a popular programming language used for data science, artificial intelligence, web development and automation.",
        "User: What is deep learning?\nAssistant: Deep learning is a type of machine learning that uses neural networks with multiple layers.",
        "User: What is MongoDB?\nAssistant: MongoDB is a NoSQL database that stores data in flexible document-oriented collections.",
        "User: What is SQL?\nAssistant: SQL is a language used to store, retrieve, update and analyze data in relational databases.",
        "User: What is artificial intelligence?\nAssistant: Artificial intelligence is the field of creating computer systems that can perform tasks that normally require human intelligence.",
        "User: Thank you\nAssistant: You're welcome! I'm happy to help.",
        "User: Bye\nAssistant: Goodbye! Have a great day.",
    ]
}


def fine_tune():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    dataset = Dataset.from_dict(data).map(
        lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=128),
        batched=True,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=5,
            per_device_train_batch_size=2,
            logging_steps=1,
            report_to="none",
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    fine_tune()

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR).eval()

    def generate_response(message, history):
        prompt = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt")
        output = model.generate(
            prompt,
            max_new_tokens=80,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
        )
        return tokenizer.decode(output[0][prompt.shape[-1]:], skip_special_tokens=True)

    gr.ChatInterface(
        fn=generate_response,
        title="Fine-Tuned Conversation Generator",
        description="GPT-style chatbot fine-tuned on a custom dataset",
    ).launch()
