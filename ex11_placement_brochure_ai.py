"""Ex 11: Mini placement brochure generator using Gemini prompt engineering.

Set GEMINI_API_KEY as an environment variable, or paste a key into the UI.
"""

import os

import gradio as gr
from google import genai
from google.genai import errors as genai_errors

MODEL_NAME = "gemini-2.5-flash"


def build_prompt(college_name, department, course_focus, student_name, student_skills):
    """The prompt-engineering step: each field is labeled and the output shape is spelled out."""
    return f"""You are writing content for a college placement brochure - a short document
colleges give recruiters to introduce a department and its graduating students.

Using the details below, write exactly three short sections in Markdown, in this order:

1. "## About the College" - 2-3 sentences, warm and professional tone.
2. "## About the Course" - 2-3 sentences describing what the course/department focuses on.
3. "## Student Highlight" - a 2-3 sentence third-person profile blurb for the named student,
   naturally mentioning their listed skills (do not just list the skills verbatim).

Details:
- College name: {college_name}
- Department: {department}
- Course focus / specialization: {course_focus}
- Student name: {student_name}
- Student skills: {student_skills}

Keep the whole thing concise - this is a brochure excerpt, not an essay. Do not add any
section other than the three requested.
"""


def generate_brochure_excerpt(college_name, department, course_focus, student_name, student_skills, api_key):
    if not all([college_name, department, course_focus, student_name, student_skills]):
        return "Please fill in every field before generating."

    key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not key:
        return (
            "No Gemini API key found. Set the `GEMINI_API_KEY` environment variable "
            "or paste a key into the API key box.\n\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )

    prompt = build_prompt(college_name, department, course_focus, student_name, student_skills)

    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    except genai_errors.APIError as err:
        return f"Gemini API error: {err}"
    except Exception as err:
        return f"Unexpected error calling Gemini: {err}"

    text = (response.text or "").strip()
    return text or "Gemini returned an empty response - try again or rephrase the inputs."


with gr.Blocks(title="Mini Placement Brochure Generator") as demo:
    gr.Markdown(
        "# Mini Placement Brochure Generator\n"
        "Fill in a few details and Gemini writes a short placement-brochure excerpt "
        "(About the College, About the Course, and a student highlight)."
    )

    api_key_box = gr.Textbox(
        label="Gemini API key (optional if GEMINI_API_KEY is set)",
        type="password",
        placeholder="AIza...",
    )

    with gr.Row():
        with gr.Column():
            college_name = gr.Textbox(label="College name", placeholder="e.g. St. Xavier's College of Engineering")
            department = gr.Textbox(label="Department", placeholder="e.g. Department of Computer Science")
            course_focus = gr.Textbox(label="Course focus / specialization", placeholder="e.g. AI & Data Science")
        with gr.Column():
            student_name = gr.Textbox(label="Student name", placeholder="e.g. Priya Ramesh")
            student_skills = gr.Textbox(label="Student skills (comma-separated)", placeholder="e.g. Python, SQL, Machine Learning")

    generate_btn = gr.Button("Generate brochure excerpt", variant="primary")
    output = gr.Markdown(label="Generated brochure excerpt")

    generate_btn.click(
        fn=generate_brochure_excerpt,
        inputs=[college_name, department, course_focus, student_name, student_skills, api_key_box],
        outputs=output,
    )

    gr.Examples(
        examples=[["Green Valley College of Engineering", "Department of Information Technology", "Cloud Computing", "Arjun Mehta", "Java, AWS, Docker"]],
        inputs=[college_name, department, course_focus, student_name, student_skills],
    )

if __name__ == "__main__":
    demo.launch()
