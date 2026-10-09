"""Ex 10: Web page summarization (scrape paragraphs + simple extractive summary).

Try: https://en.wikipedia.org/wiki/Artificial_intelligence
"""

import gradio as gr
import requests
from bs4 import BeautifulSoup


def get_webpage_text(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unnecessary parts
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    return " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))


def summarize_text(text, num_sentences=5):
    """Simple extractive summary: the first few sentences."""
    return ". ".join(text.split(". ")[:num_sentences])


def summarize_webpage(url):
    try:
        if not url:
            return "Please enter a webpage URL."
        if not url.startswith(("http://", "https://")):
            return "Please enter a valid URL."

        text = get_webpage_text(url)
        if len(text) < 100:
            return "Could not extract enough text from this webpage."

        return summarize_text(text, 5)
    except Exception as e:
        return "Error: " + str(e)


interface = gr.Interface(
    fn=summarize_webpage,
    inputs=gr.Textbox(label="Enter Web Page URL", placeholder="https://example.com"),
    outputs=gr.Textbox(label="Generated Summary", lines=10),
    title="Web Page Summarizer",
    description="Extracts text from a webpage and generates a simple summary.",
)

if __name__ == "__main__":
    interface.launch()
