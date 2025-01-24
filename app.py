from flask import Flask, render_template_string, request
import re
import os
import json
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_PROMPT = "Succinctly organize changes made by the attending to the resident's radiology reports into JSON."

def parse_cases(text):
    cases = re.split(r'(?m)^Case\s+(\d+)', text, flags=re.IGNORECASE)
    parsed_cases = []
    for i in range(1, len(cases), 2):
        case_num = cases[i].strip()
        case_content = cases[i + 1].strip() if (i + 1) < len(cases) else ""
        attending = re.search(r'Attending Report:\s*(.*?)(Resident Report:|$)', case_content, re.DOTALL | re.IGNORECASE)
        resident = re.search(r'Resident Report:\s*(.*)', case_content, re.DOTALL | re.IGNORECASE)
        if attending and resident:
            parsed_cases.append({
                'case_num': case_num,
                'resident': resident.group(1).strip(),
                'attending': attending.group(1).strip()
            })
    return parsed_cases

def get_summary(case_text, custom_prompt, case_number):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs structured JSON summaries of radiology report differences."},
            {"role": "user", "content": f"{custom_prompt}\nCase Number: {case_number}\n{case_text}"}
        ],
        max_tokens=2000,
        temperature=0.5
    )
    return json.loads(response.choices[0].message.content)

@app.route('/', methods=['GET', 'POST'])
def index():
    summaries = []
    if request.method == 'POST':
        cases = parse_cases(request.form.get('report_text', '').strip())
        for case in cases:
            case_text = f"Resident Report: {case['resident']}\nAttending Report: {case['attending']}"
            summaries.append(get_summary(case_text, DEFAULT_PROMPT, case['case_num']))

    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Radiology Report Summarizer</title>
    </head>
    <body>
        <form method="POST">
            <textarea name="report_text" rows="15"></textarea>
            <button type="submit">Submit</button>
        </form>
        {% if summaries %}
            <ul>
                {% for summary in summaries %}
                    <li>Case {{ summary.case_number }}: {{ summary }}</li>
                {% endfor %}
            </ul>
        {% endif %}
    </body>
    </html>
    """
    return render_template_string(template, summaries=summaries)

if __name__ == '__main__':
    app.run()
