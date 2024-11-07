from flask import Flask, render_template_string, request
import difflib
import re
import os
import json
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Default customized prompt
DEFAULT_PROMPT = """Succinctly organize the changes made by the attending to the resident's radiology reports into: 
1) missed major findings (life threatening or treatment altering), 
2) missed minor findings, and 
3) clarified descriptions of findings. 
Assume the attending's version was correct, and anything not included by the attending but was included by the resident should have been left out by the resident. Keep your answers brief and to the point. The reports are: 
Please output your response as structured JSON, with the following keys:
- "case_number": The case number sent in the request.
- "major_findings": A list of any major findings missed by the resident (life-threatening or treatment-altering).
- "minor_findings": A list of minor findings missed by the resident.
- "clarifications": A list of any clarifications the attending made to improve understanding.
- "score": The calculated score, where each major finding is worth 3 points and each minor finding is worth 1 point.
Respond in this JSON format, with no other additional text or pleasantries:
{
  "case_number": <case_number>,
  "major_findings": [<major_findings>],
  "minor_findings": [<minor_findings>],
  "clarifications": [<clarifications>],
  "score": <score>
}"""

# Normalize text: trim spaces but keep returns (newlines) intact
def normalize_text(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

# Remove "attending review" lines for comparison purposes
def remove_attending_review_line(text):
    excluded_lines = [
        "As the attending physician, I have personally reviewed the images, interpreted and/or supervised the study or procedure, and agree with the wording of the above report.",
        "As the Attending radiologist, I have personally reviewed the images, interpreted the study, and agree with the wording of the above report by Sterling M. Jones"
    ]
    return "\n".join([line for line in text.splitlines() if line.strip() not in excluded_lines])

# Extract sections by headers ending with a colon
def extract_sections(text):
    pattern = r'(.*?:)(.*?)(?=(?:\n.*?:)|\Z)'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    sections = [{'header': header.strip(), 'content': content.strip()} for header, content in matches]
    return sections

# Calculate percentage change between two reports
def calculate_change_percentage(resident_text, attending_text):
    matcher = difflib.SequenceMatcher(None, resident_text.split(), attending_text.split())
    return round((1 - matcher.ratio()) * 100, 2)

# Compare reports section by section
def create_diff_by_section(resident_text, attending_text):
    resident_text = normalize_text(resident_text)
    attending_text = normalize_text(remove_attending_review_line(attending_text))
    resident_sections = extract_sections(resident_text)
    attending_sections = extract_sections(attending_text)
    diff_html = ""
    for res_sec, att_sec in zip(resident_sections, attending_sections):
        res_header = res_sec["header"]
        res_content = res_sec["content"]
        att_content = att_sec["content"]
        matcher = difflib.SequenceMatcher(None, res_content.split(), att_content.split())
        section_diff = ""
        for opcode, a1, a2, b1, b2 in matcher.get_opcodes():
            if opcode == 'equal':
                section_diff += " ".join(res_content.split()[a1:a2]) + " "
            elif opcode == 'replace':
                section_diff += f'<span style="color:#ff6b6b;text-decoration:line-through;">{" ".join(res_content.split()[a1:a2])}</span> '
                section_diff += f'<span style="color:lightgreen;">{" ".join(att_content.split()[b1:b2])}</span> '
            elif opcode == 'delete':
                section_diff += f'<span style="color:#ff6b6b;text-decoration:line-through;">{" ".join(res_content.split()[a1:a2])}</span> '
            elif opcode == 'insert':
                section_diff += f'<span style="color:lightgreen;">{" ".join(att_content.split()[b1:b2])}</span> '
        diff_html += f"{res_header}<br>{section_diff}<br><br>"
    return diff_html

# AI function to get a structured JSON summary of report differences
def get_summary(case_text, custom_prompt, case_number):
    try:
        # Use the updated OpenAI ChatCompletion API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs structured JSON summaries of radiology report differences."},
                {"role": "user", "content": f"{custom_prompt}\nCase Number: {case_number}\n{case_text}"}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        # Capture and parse response content as JSON
        response_content = response.choices[0].message.content
        print(f"Response for case {case_number}: {response_content}")  # Log for debugging
        return json.loads(response_content)
    except Exception as e:
        print(f"Error processing case {case_number}: {str(e)}")
        return {"case_number": case_number, "error": "Error processing AI"}

# Process cases for summaries
def process_cases(bulk_text, custom_prompt):
    case_numbers = re.findall(r"Case (\d+)", bulk_text)
    cases = bulk_text.split("Case")
    structured_output = []
    for index, case in enumerate(cases[1:], start=1):
        if "Attending Report" in case and "Resident Report" in case:
            case_number = case_numbers[index - 1] if index - 1 < len(case_numbers) else index
            attending_report = case.split("Attending Report:")[1].split("Resident Report:")[0].strip()
            resident_report = case.split("Resident Report:")[1].strip()
            case_text = f"Resident Report: {resident_report}\nAttending Report: {attending_report}"
            parsed_json = get_summary(case_text, custom_prompt, case_number=case_number) or {}
            parsed_json['score'] = len(parsed_json.get('major_findings', [])) * 3 + len(parsed_json.get('minor_findings', []))
            structured_output.append(parsed_json)
    return structured_output

# Extract cases and add AI summary tab
def extract_cases(text, custom_prompt):
    cases = re.split(r'\bCase\s+(\d+)', text, flags=re.IGNORECASE)
    parsed_cases = []
    for i in range(1, len(cases), 2):
        case_num = cases[i]
        case_content = cases[i + 1].strip()
        reports = re.split(r'\s*(Attending\s+Report\s*:|Resident\s+Report\s*:)\s*', case_content, flags=re.IGNORECASE)
        if len(reports) >= 3:
            attending_report = reports[2].strip()
            resident_report = reports[4].strip() if len(reports) > 4 else ""
            ai_summary = process_cases(f"Case {case_num}\n{case_content}", custom_prompt)
            parsed_cases.append({
                'case_num': case_num,
                'resident_report': resident_report,
                'attending_report': attending_report,
                'percentage_change': calculate_change_percentage(resident_report, remove_attending_review_line(attending_report)),
                'diff': create_diff_by_section(resident_report, attending_report),
                'summary': ai_summary[0] if ai_summary else None
            })
    return parsed_cases

@app.route('/', methods=['GET', 'POST'])
def index():
    sorted_cases = False
    custom_prompt = request.form.get('custom_prompt', DEFAULT_PROMPT)
    case_data = []
    if request.method == 'POST':
        text_block = request.form['report_text']
        case_data = extract_cases(text_block, custom_prompt)
    template = """
    <html>
        <head>
            <title>Radiology Report Diff & Summarizer</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
            <style>
                body {
                    background-color: #1e1e1e;
                    color: #dcdcdc;
                    font-family: Arial, sans-serif;
                }
                textarea, input, button {
                    background-color: #333333;
                    color: #dcdcdc;
                    border: 1px solid #555;
                }
                h2, h3, h4 {
                    color: #f0f0f0;
                    font-weight: normal;
                }
                .diff-output, .summary-output {
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #2e2e2e;
                    border-radius: 8px;
                    border: 1px solid #555;
                    white-space: normal;
                }
                .nav-tabs .nav-link {
                    background-color: #333;
                    border-color: #555;
                    color: #dcdcdc;
                }
                .nav-tabs .nav-link.active {
                    background-color: #007bff;
                    border-color: #007bff #007bff #333;
                    color: white;
                }
                #scrollToTopBtn {
                    position: fixed;
                    right: 30px;
                    bottom: 30px;
                    background-color: #007bff;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 50px;
                    border: none;
                    cursor: pointer;
                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                    font-size: 14px;
                    display: none;
                }
                #scrollToTopBtn:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2 class="mt-4">Compare Revisions & Summarize Reports</h2>
                <form method="POST" class="mb-4">
                    <div class="form-group mb-3">
                        <label for="report_text">Paste your reports block here:</label>
                        <textarea id="report_text" name="report_text" class="form-control" rows="10">{{ request.form.get('report_text', '') }}</textarea>
                    </div>
                    <div class="form-group mb-3">
                        <label for="custom_prompt">Customize your OpenAI API prompt:</label>
                        <textarea id="custom_prompt" name="custom_prompt" class="form-control" rows="5">{{ custom_prompt }}</textarea>
                    </div>
                    <button type="submit" class="btn btn-primary">Compare & Summarize Reports</button>
                </form>

                {% if case_data %}
                    <h3>Case Navigation</h3>
                    <ul>
                        {% for case in case_data %}
                            <li>
                                <a href="#case{{ case.case_num }}">Case {{ case.case_num }}</a> - {{ case.percentage_change }}% change
                            </li>
                        {% endfor %}
                    </ul>
                    <hr>

                    {% for case in case_data %}
                        <div id="case{{ case.case_num }}">
                            <h4>Case {{ case.case_num }} - {{ case.percentage_change }}% change</h4>
                            <ul class="nav nav-tabs" id="myTab{{ case.case_num }}" role="tablist">
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link active" id="combined-tab{{ case.case_num }}" data-bs-toggle="tab" data-bs-target="#combined{{ case.case_num }}" type="button" role="tab" aria-controls="combined{{ case.case_num }}" aria-selected="true">Combined Report</button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="resident-tab{{ case.case_num }}" data-bs-toggle="tab" data-bs-target="#resident{{ case.case_num }}" type="button" role="tab" aria-controls="resident{{ case.case_num }}" aria-selected="false">Resident Report</button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="attending-tab{{ case.case_num }}" data-bs-toggle="tab" data-bs-target="#attending{{ case.case_num }}" type="button" role="tab" aria-controls="attending{{ case.case_num }}" aria-selected="false">Attending Report</button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="summary-tab{{ case.case_num }}" data-bs-toggle="tab" data-bs-target="#summary{{ case.case_num }}" type="button" role="tab" aria-controls="summary{{ case.case_num }}" aria-selected="false">Summary Report</button>
                                </li>
                            </ul>
                            <div class="tab-content" id="myTabContent{{ case.case_num }}">
                                <div class="tab-pane fade show active" id="combined{{ case.case_num }}" role="tabpanel" aria-labelledby="combined-tab{{ case.case_num }}">
                                    <div class="diff-output">
                                        {{ case.diff|safe }}
                                    </div>
                                </div>
                                <div class="tab-pane fade" id="resident{{ case.case_num }}" role="tabpanel" aria-labelledby="resident-tab{{ case.case_num }}">
                                    <div class="diff-output">
                                        <pre>{{ case.resident_report }}</pre>
                                    </div>
                                </div>
                                <div class="tab-pane fade" id="attending{{ case.case_num }}" role="tabpanel" aria-labelledby="attending-tab{{ case.case_num }}">
                                    <div class="diff-output">
                                        <pre>{{ case.attending_report }}</pre>
                                    </div>
                                </div>
                                <div class="tab-pane fade" id="summary{{ case.case_num }}" role="tabpanel" aria-labelledby="summary-tab{{ case.case_num }}">
                                    <div class="summary-output">
                                        {% if case.summary %}
                                            <p><strong>Score:</strong> {{ case.summary.score }}</p>
                                            {% if case.summary.major_findings %}
                                                <p><strong>Major Findings:</strong></p>
                                                <ul>
                                                    {% for finding in case.summary.major_findings %}
                                                        <li>{{ finding }}</li>
                                                    {% endfor %}
                                                </ul>
                                            {% endif %}
                                            {% if case.summary.minor_findings %}
                                                <p><strong>Minor Findings:</strong></p>
                                                <ul>
                                                    {% for finding in case.summary.minor_findings %}
                                                        <li>{{ finding }}</li>
                                                    {% endfor %}
                                                </ul>
                                            {% endif %}
                                            {% if case.summary.clarifications %}
                                                <p><strong>Clarifications:</strong></p>
                                                <ul>
                                                    {% for clarification in case.summary.clarifications %}
                                                        <li>{{ clarification }}</li>
                                                    {% endfor %}
                                                </ul>
                                            {% endif %}
                                        {% else %}
                                            <p><em>No AI Summary available.</em></p>
                                        {% endif %}
                                    </div>
                                </div>
                            </div>
                            <hr>
                        </div>
                    {% endfor %}
                {% endif %}
            </div>

            <button id="scrollToTopBtn">Top</button>

            <script>
                var mybutton = document.getElementById("scrollToTopBtn");
                window.onscroll = function() {
                    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
                        mybutton.style.display = "block";
                    } else {
                        mybutton.style.display = "none";
                    }
                };
                mybutton.onclick = function() {
                    document.body.scrollTop = 0;
                    document.documentElement.scrollTop = 0;
                };
            </script>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
    </html>
    """
    return render_template_string(template, case_data=case_data, custom_prompt=custom_prompt)

if __name__ == '__main__':
    app.run(debug=True)
