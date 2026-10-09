# Sem3 GenAI Lab

13 exercises, one runnable Python file each. Run any with `python <file>.py`.

| Ex | File | Description | Needs |
|----|------|-------------|-------|
| 1 | `ex01_text_to_image.py` | Text-to-image with Stable Diffusion | GPU recommended |
| 2 | `ex02_face_mask_detection.py` | Face mask detection (Gradio) | - |
| 3 | `ex03_object_detection_yolo.py` | Object detection with YOLOv5 | image path as argument |
| 4 | `ex04_image_captioning.py` | Image captioning with BLIP | - |
| 5 | `ex05_multimodal_assistant.py` | Image + question assistant (Gemini) | `GEMINI_API_KEY` |
| 6 | `ex06_large_document_summarizer.py` | Chunked summarization with BART | - |
| 7 | `ex07_recipe_generator.py` | Recipe generator (FLAN-T5, from [kitch-ai](https://github.com/anandsundaramoorthysa/kitch-ai)) | - |
| 8 | `ex08_code_generator.py` | Python code generator (Qwen2.5-Coder) | - |
| 9 | `ex09_conversation_generator_finetuned.py` | Fine-tuned DialoGPT chatbot | - |
| 10 | `ex10_web_page_summarization.py` | Web page summarizer | - |
| 11 | `ex11_placement_brochure_ai.py` | Placement brochure generator (Gemini) | `GEMINI_API_KEY` |
| 12 | `ex12_quiz_generator.py` | AI quiz generator (FLAN-T5) | - |
| 13 | `ex13_airline_assistant.py` | Airline assistant (FLAN-T5) | - |

## Setup

```bash
pip install -r requirements.txt
# for Ex 5 and 11
set GEMINI_API_KEY=your_key      # PowerShell: $env:GEMINI_API_KEY="your_key"
```
