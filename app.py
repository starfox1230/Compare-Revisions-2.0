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
def split_into_paragraphs(text):
    # Split the text into paragraphs based on double line breaks or single line breaks after punctuation
    paragraphs = re.split(r'\n{2,}|\n(?=\w)', text)
    return [para.strip() for para in paragraphs if para.strip()]

def create_diff_by_section(resident_text, attending_text):
    # Normalize text for comparison
    resident_text = normalize_text(resident_text)
    attending_text = normalize_text(remove_attending_review_line(attending_text))

    # Split text into paragraphs instead of sentences
    resident_paragraphs = split_into_paragraphs(resident_text)
    attending_paragraphs = split_into_paragraphs(attending_text)

    diff_html = ""

    # Use SequenceMatcher on the paragraph level first
    matcher = difflib.SequenceMatcher(None, resident_paragraphs, attending_paragraphs)
    for opcode, a1, a2, b1, b2 in matcher.get_opcodes():
        # Handle matched (equal) paragraphs
        if opcode == 'equal':
            for paragraph in resident_paragraphs[a1:a2]:
                diff_html += paragraph + "<br><br>"

        # Handle inserted paragraphs as a block
        elif opcode == 'insert':
            for paragraph in attending_paragraphs[b1:b2]:
                diff_html += f'<div style="color:lightgreen;">[Inserted: {paragraph}]</div><br><br>'

        # Handle deleted paragraphs as a block
        elif opcode == 'delete':
            for paragraph in resident_paragraphs[a1:a2]:
                diff_html += f'<div style="color:#ff6b6b;text-decoration:line-through;">[Deleted: {paragraph}]</div><br><br>'

        # Handle paragraph replacements by word-by-word comparison within each paragraph
        elif opcode == 'replace':
            res_paragraphs = resident_paragraphs[a1:a2]
            att_paragraphs = attending_paragraphs[b1:b2]
            for res_paragraph, att_paragraph in zip(res_paragraphs, att_paragraphs):
                # Compare the words within the mismatched paragraphs
                word_matcher = difflib.SequenceMatcher(None, res_paragraph.split(), att_paragraph.split())
                for word_opcode, w_a1, w_a2, w_b1, w_b2 in word_matcher.get_opcodes():
                    if word_opcode == 'equal':
                        diff_html += " ".join(res_paragraph.split()[w_a1:w_a2]) + " "
                    elif word_opcode == 'replace':
                        diff_html += (
                            '<span style="color:#ff6b6b;text-decoration:line-through;">' +
                            " ".join(res_paragraph.split()[w_a1:w_a2]) +
                            '</span> <span style="color:lightgreen;">' +
                            " ".join(att_paragraph.split()[w_b1:w_b2]) +
                            '</span> '
                        )
                    elif word_opcode == 'delete':
                        diff_html += (
                            '<span style="color:#ff6b6b;text-decoration:line-through;">' +
                            " ".join(res_paragraph.split()[w_a1:w_a2]) +
                            '</span> '
                        )
                    elif word_opcode == 'insert':
                        diff_html += (
                            '<span style="color:lightgreen;">' +
                            " ".join(att_paragraph.split()[w_b1:w_b2]) +
                            '</span> '
                        )

                diff_html += "<br><br>"  # Separate each replaced paragraph with line breaks

    return diff_html

# AI function to get a structured JSON summary of report differences
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
                textarea { background-color: #333333 !important; color: #dcdcdc !important; border: 1px solid #555 !important; }
                h2, h3, h4 { color: #f0f0f0; font-weight: normal; }
                .diff-output, .summary-output { margin-top: 20px; padding: 15px; background-color: #2e2e2e; border-radius: 8px; border: 1px solid #555; }
                pre { white-space: pre-wrap; word-wrap: break-word; font-family: inherit; }
                .nav-tabs .nav-link { background-color: #333; border-color: #555; color: #dcdcdc; }
                .nav-tabs .nav-link.active { background-color: #007bff; border-color: #007bff #007bff #333; color: white; }

                /* Scroll-to-top button */
                #scrollToTopBtn {
                        position: fixed;
                        right: 20px;
                        bottom: 20px;
                        background-color: #007bff;
                        color: white;
                        padding: 10px 15px;
                        border-radius: 50%;
                        border: none;
                        cursor: pointer;
                        z-index: 1000;
                }
                #scrollToTopBtn:hover {
                        background-color: #0056b3;
                }

                /* Links styling for night mode */
                a {
                        color: #66ccff; /* A softer blue that is easier on the eyes in night mode */
                        text-decoration: none; /* Removes underline */
                }
                a:hover {
                        color: #99e6ff; /* A lighter blue for hover state */
                        text-decoration: none; /* Ensures no underline on hover */
                }
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
                <h3 id="majorFindings">Major Findings Missed</h3>
                <ul>
                    {% for case in case_data %}
                        {% if case.summary and case.summary.major_findings %}
                            {% for finding in case.summary.major_findings %}
                                <li><a href="#case{{ case.case_num }}">Case {{ case.case_num }}</a>: {{ finding }}</li>
                            {% endfor %}
                        {% endif %}
                    {% endfor %}
                </ul>
                <h3>Minor Findings Missed</h3>
                <ul>
                    {% for case in case_data %}
                        {% if case.summary and case.summary.minor_findings %}
                            {% for finding in case.summary.minor_findings %}
                                <li><a href="#case{{ case.case_num }}">Case {{ case.case_num }}</a>: {{ finding }}</li>
                            {% endfor %}
                        {% endif %}
                    {% endfor %}
                </ul>
                <h3>Case Navigation</h3>
                <div class="btn-group" role="group" aria-label="Sort Options">
                    <button type="button" class="btn btn-secondary" onclick="sortCases('case_number')">Sort by Case Number</button>
                    <button type="button" class="btn btn-secondary" onclick="sortCases('percentage_change')">Sort by Percentage Change</button>
                    <button type="button" class="btn btn-secondary" onclick="sortCases('summary_score')">Sort by Summary Score</button>
                </div>
                <ul id="caseNav"></ul>
                <div id="caseContainer"></div>
            {% endif %}
        </div>
        <!-- Added scroll-to-top button -->
        <button id="scrollToTopBtn" onclick="scrollToTop()">Top ⬆</button>
        <script>
            let caseData = {{ case_data | tojson }};
            
            function sortCases(option) {
                if (option === "case_number") {
                    caseData.sort((a, b) => parseInt(a.case_num) - parseInt(b.case_num));
                } else if (option === "percentage_change") {
                    caseData.sort((a, b) => b.percentage_change - a.percentage_change);
                } else if (option === "summary_score") {
                    caseData.sort((a, b) => (b.summary && b.summary.score || 0) - (a.summary && a.summary.score || 0));
                }
                displayCases();
                displayNavigation();
            }
            function displayNavigation() {
                const nav = document.getElementById('caseNav');
                nav.innerHTML = '';
                caseData.forEach(caseObj => {
                    nav.innerHTML += `
                        <li>
                            <a href="#case${caseObj.case_num}">Case ${caseObj.case_num}</a> - ${caseObj.percentage_change}% change - Score: ${(caseObj.summary && caseObj.summary.score) || 'N/A'}
                        </li>
                    `;
                });
            }
            function displayCases() {
                const container = document.getElementById('caseContainer');
                container.innerHTML = '';
                caseData.forEach(caseObj => {
                    container.innerHTML += `
                        <div id="case${caseObj.case_num}">
                            <h4>Case ${caseObj.case_num} - ${caseObj.percentage_change}% change</h4>
                            <ul class="nav nav-tabs" id="myTab${caseObj.case_num}" role="tablist">
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link active" id="summary-tab${caseObj.case_num}" data-bs-toggle="tab" data-bs-target="#summary${caseObj.case_num}" type="button" role="tab">Summary Report</button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="combined-tab${caseObj.case_num}" data-bs-toggle="tab" data-bs-target="#combined${caseObj.case_num}" type="button" role="tab">Combined Report</button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="resident-tab${caseObj.case_num}" data-bs-toggle="tab" data-bs-target="#resident${caseObj.case_num}" type="button" role="tab">Resident Report</button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="attending-tab${caseObj.case_num}" data-bs-toggle="tab" data-bs-target="#attending${caseObj.case_num}" type="button" role="tab">Attending Report</button>
                                </li>
                            </ul>
                            <div class="tab-content" id="myTabContent${caseObj.case_num}">
                                <div class="tab-pane fade show active" id="summary${caseObj.case_num}" role="tabpanel">
                                    <div class="summary-output">
                                        <p><strong>Score:</strong> ${caseObj.summary && caseObj.summary.score || 'N/A'}</p>
                                        ${caseObj.summary && caseObj.summary.major_findings?.length ? `<p><strong>Major Findings:</strong></p><ul>${caseObj.summary.major_findings.map(finding => `<li>${finding}</li>`).join('')}</ul>` : ''}
                                        ${caseObj.summary && caseObj.summary.minor_findings?.length ? `<p><strong>Minor Findings:</strong></p><ul>${caseObj.summary.minor_findings.map(finding => `<li>${finding}</li>`).join('')}</ul>` : ''}
                                        ${caseObj.summary && caseObj.summary.clarifications?.length ? `<p><strong>Clarifications:</strong></p><ul>${caseObj.summary.clarifications.map(clarification => `<li>${clarification}</li>`).join('')}</ul>` : ''}
                                    </div>
                                </div>
                                <div class="tab-pane fade" id="combined${caseObj.case_num}" role="tabpanel">
                                    <div class="diff-output">${caseObj.diff}</div>
                                </div>
                                <div class="tab-pane fade" id="resident${caseObj.case_num}" role="tabpanel">
                                    <div class="diff-output"><pre>${caseObj.resident_report}</pre></div>
                                </div>
                                <div class="tab-pane fade" id="attending${caseObj.case_num}" role="tabpanel">
                                    <div class="diff-output"><pre>${caseObj.attending_report}</pre></div>
                                </div>
                            </div>
                            <hr>
                        </div>
                    `;
                });
            }
            document.addEventListener("DOMContentLoaded", () => {
                displayCases();
                displayNavigation();
            });
            // Added scrollToTop function
            function scrollToTop() {
                const majorFindingsSection = document.getElementById('majorFindings');
                if (majorFindingsSection) {
                    majorFindingsSection.scrollIntoView({ behavior: 'smooth' });
                }
            }
        </script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
</html>
    """
    return render_template_string(template, case_data=case_data, custom_prompt=custom_prompt)

if __name__ == '__main__':
    app.run(debug=True)
