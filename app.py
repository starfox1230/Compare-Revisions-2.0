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
1) missed major findings (findings discussed by the attending but not by the resident that also fit under these categories: retained sponge or other clinically significant foreign body, new unexpected clinically significant mass or tumor with potential immediate clinical consequences, malpositioned line or tube of immediate clinical concern, allergic reaction resulting in a code, previously undiagnosed life-threatening hemorrhage or vascular disruption, necrotizing fasciitis, unexpected or previously undiagnosed free air or active leakage from the GI tract, ectopic pregnancy, intestinal ischemia or portomesenteric gas, ovarian torsion, testicular torsion, placental abruption, newly diagnosed absent perfusion in a postoperative transplant, new renal collecting system obstruction with signs of infection, acute cholecystitis (outpatient only), unexpected and clinically significant intracranial hemorrhage, new midline shift, clinically significant herniation, new unexpected cerebral infarction, abscess or meningoencephalitis, acute airway compromise, new clinically significant unexpected abscess or discitis, new clinically significant unexpected hemorrhage, new unexpected clinically significant cord compression or unstable spine fracture or transection, acute cord hemorrhage or infarct, unexpected clinically significant pneumothorax, new large pericardial effusion, findings suggestive of active TB, impending pathologic fracture, new unexpected clinically significant fracture, newly diagnosed absent perfusion in a postoperative kidney, brain death with transplant team waiting for results, new high probability ventilation/perfusion (VQ) lung scan, new clinically significant unexpected arterial dissection or occlusion, previously undiagnosed acute thrombotic or embolic event including DVT and pulmonary thromboembolism, and previously undiagnosed clinically significant aneurysm or vascular disruption), 
2) missed minor findings (this includes all other pathologies not in the above list that were discussed by the attending but not by the resident), and 
3) clarified descriptions of findings (this includes findings removed by the attending, findings re-worded by the attending, etc). 
Assume the attending's version was correct, and anything not included by the attending but was included by the resident should have been left out by the resident. Keep your answers brief and to the point. The reports are: 
Please output your response as structured JSON, with the following keys:
- "case_number": The case number sent in the request.
- "major_findings": A list of any major findings missed by the resident (as described above).
- "minor_findings": A list of minor findings discussed by the attending but not by the resident (as described above).
- "clarifications": A list of any clarifications the attending made (as described above).
- "score": The calculated score, where each major finding is worth 3 points and each minor finding is worth 1 point.
Respond in this JSON format, with no other additional text or pleasantries:
{
  "case_number": <case_number>,
  "major_findings": [<major_findings>],
  "minor_findings": [<minor_findings>],
  "clarifications": [<clarifications>],
  "score": <score>
}"""

def normalize_text(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def remove_attending_review_line(text):
    excluded_lines = [
        "As the attending physician, I have personally reviewed the images, interpreted and/or supervised the study or procedure, and agree with the wording of the above report.",
        "As the Attending radiologist, I have personally reviewed the images, interpreted the study, and agree with the wording of the above report by Sterling M. Jones"
    ]
    return "\n".join([line for line in text.splitlines() if line.strip() not in excluded_lines])

def calculate_change_percentage(resident_text, attending_text):
    matcher = difflib.SequenceMatcher(None, resident_text.split(), attending_text.split())
    return round((1 - matcher.ratio()) * 100, 2)

def create_diff_by_section(resident_text, attending_text):
    resident_text = normalize_text(resident_text)
    attending_text = normalize_text(remove_attending_review_line(attending_text))

    diff_html = ""
    matcher = difflib.SequenceMatcher(None, resident_text.split(), attending_text.split())
    for opcode, a1, a2, b1, b2 in matcher.get_opcodes():
        if opcode == 'equal':
            diff_html += " ".join(resident_text.split()[a1:a2]) + " "
        elif opcode == 'delete':
            diff_html += f'<span style="color:#ff6b6b;text-decoration:line-through;">{" ".join(resident_text.split()[a1:a2])}</span> '
        elif opcode == 'insert':
            diff_html += f'<span style="color:lightgreen;">{" ".join(attending_text.split()[b1:b2])}</span> '
        elif opcode == 'replace':
            diff_html += f'<span style="color:#ff6b6b;text-decoration:line-through;">{" ".join(resident_text.split()[a1:a2])}</span> '
            diff_html += f'<span style="color:lightgreen;">{" ".join(attending_text.split()[b1:b2])}</span> '
    return diff_html

def get_summary(case_text, custom_prompt, case_number):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs structured JSON summaries of radiology report differences."},
                {"role": "user", "content": f"{custom_prompt}\nCase Number: {case_number}\n{case_text}"}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        response_content = response.choices[0].message.content
        return json.loads(response_content)
    except Exception as e:
        return {"case_number": case_number, "error": "Error processing AI"}

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
            ai_summary = get_summary(case_content, custom_prompt, case_num)
            parsed_cases.append({
                'case_num': case_num,
                'resident_report': resident_report,
                'attending_report': attending_report,
                'percentage_change': calculate_change_percentage(resident_report, remove_attending_review_line(attending_report)),
                'diff': create_diff_by_section(resident_report, attending_report),
                'summary': ai_summary if ai_summary else None
            })
    return parsed_cases

@app.route('/', methods=['GET', 'POST'])
def index():
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
            body { background-color: #1e1e1e; color: #dcdcdc; font-family: Arial, sans-serif; }
            textarea, input, button { background-color: #333333; color: #dcdcdc; border: 1px solid #555; }
            h2, h3, h4 { color: #f0f0f0; font-weight: normal; }
            #scrollToTopBtn { position: fixed; right: 30px; bottom: 30px; background-color: #007bff; color: white; padding: 10px 20px; border-radius: 50px; border: none; cursor: pointer; display: none; }
            #scrollToTopBtn:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 class="mt-4">Compare Revisions & Summarize Reports</h2>
            <form method="POST" id="reportForm">
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
                <h3 class="mt-5">Major Findings Missed</h3>
                <ul>
                    {% for case in case_data if case.summary and case.summary.major_findings %}
                        {% for finding in case.summary.major_findings %}
                            <li>- <a href="#case{{ case.case_num }}">Case {{ case.case_num }}</a>: {{ finding }}</li>
                        {% endfor %}
                    {% endfor %}
                </ul>

                <h3>Minor Findings Missed</h3>
                <ul>
                    {% for case in case_data if case.summary and case.summary.minor_findings %}
                        {% for finding in case.summary.minor_findings %}
                            <li>- <a href="#case{{ case.case_num }}">Case {{ case.case_num }}</a>: {{ finding }}</li>
                        {% endfor %}
                    {% endfor %}
                </ul>

                <h3>Case Navigation</h3>
                <ul id="caseNav"></ul>
                <div id="caseContainer"></div>
            {% endif %}
        </div>

        <button id="scrollToTopBtn" onclick="scrollToTop()">⬆ Top</button>

        <script>
            function scrollToTop() {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
            window.onscroll = function() {
                const scrollBtn = document.getElementById('scrollToTopBtn');
                if (document.documentElement.scrollTop > 200) {
                    scrollBtn.style.display = 'block';
                } else {
                    scrollBtn.style.display = 'none';
                }
            };
        </script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
</html>
    """
    return render_template_string(template, case_data=case_data, custom_prompt=custom_prompt)

if __name__ == '__main__':
    app.run(debug=True)
