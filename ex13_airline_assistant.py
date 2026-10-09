"""Ex 13: AI airline assistant (prompt-grounded Q&A) using FLAN-T5-small."""

import re

import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

airline_info = """
Airline Name: SkyWays Airlines

Baggage:
Passengers can carry one cabin bag up to 7 kg.
Checked baggage allowance is up to 15 kg for economy passengers.

Check-in:
Online check-in is available 24 hours before departure.
Passengers should arrive at the airport at least 2 hours before domestic flights.

Cancellation:
Tickets can be cancelled before departure.
Cancellation charges may apply depending on the ticket type.

Flight Change:
Passengers can request a flight change before departure.
A change fee may apply depending on the ticket type.

Boarding:
Passengers should reach the boarding gate at least 30 minutes before departure.

Documents:
Passengers should carry a valid government-issued identity document for domestic travel.

Customer Support:
Passengers can contact SkyWays Airlines customer support for booking,
cancellation, refund and other travel-related queries.
"""


def preprocess_text(text):
    return re.sub(r"\s+", " ", text).strip()


def airline_assistant(user_query):
    user_query = preprocess_text(user_query)

    prompt = f"""
You are an AI Airline Assistant.

Answer the user's question using only the airline information provided below.
Give a short, clear and helpful answer.
If the information is not available, say:
"Sorry, I do not have information about that."

Airline Information:
{airline_info}

User Question:
{user_query}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=100,
        min_length=10,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


interface = gr.Interface(
    fn=airline_assistant,
    inputs=gr.Textbox(
        label="Ask your Airline Question",
        placeholder="Example: How much cabin baggage can I carry?",
    ),
    outputs=gr.Textbox(label="AI Airline Assistant"),
    title="AI Airline Assistant",
    description="Ask questions about baggage, check-in, cancellation, boarding and flight changes.",
    examples=[
        ["How much cabin baggage can I carry?"],
        ["How much checked baggage is allowed?"],
        ["When can I do online check-in?"],
        ["When should I arrive at the airport?"],
        ["Can I cancel my ticket?"],
        ["Can I change my flight?"],
        ["When should I reach the boarding gate?"],
        ["What documents do I need for domestic travel?"],
    ],
)

if __name__ == "__main__":
    interface.launch()
