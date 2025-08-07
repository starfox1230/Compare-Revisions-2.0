from flask import Flask, render_template_string, request
import difflib
import re
import os
import json
import logging
from openai import OpenAI  # Ensure correct import based on your OpenAI library version
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

# Configure logging with timestamp and log level
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs during troubleshooting
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Default customized prompt
DEFAULT_PROMPT = """Succinctly organize the changes made by the attending to the resident's radiology reports into: 
1) missed major findings (findings discussed by the attending but not by the resident that also fit under these categories: retained sponge or other clinically significant foreign body, mass or tumor, malpositioned line or tube of immediate clinical concern, life-threatening hemorrhage or vascular disruption, necrotizing fasciitis, free air or active leakage from the GI tract, ectopic pregnancy, intestinal ischemia or portomesenteric gas, ovarian torsion, testicular torsion, placental abruption, absent perfusion in a postoperative transplant, renal collecting system obstruction with signs of infection, acute cholecystitis, intracranial hemorrhage, midline shift, brain herniation, cerebral infarction or abscess or meningoencephalitis, airway compromise, abscess or discitis, hemorrhage, cord compression or unstable spine fracture or transection, acute cord hemorrhage or infarct, pneumothorax, large pericardial effusion, findings suggestive of active TB, impending pathologic fracture, acute fracture, absent perfusion in a postoperative kidney, brain death, high probability ventilation/perfusion (VQ) lung scan, arterial dissection or occlusion, acute thrombotic or embolic event including DVT and pulmonary thromboembolism, and aneurysm or vascular disruption), 
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
}
"""

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
        logger.info(f"Processing case {case_number}")
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs structured JSON summaries of radiology report differences."},
                {"role": "user", "content": f"{custom_prompt}\nCase Number: {case_number}\n{case_text}"}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        response_content = response.choices[0].message.content
        logger.info(f"Received response for case {case_number}: {response_content}")
        return json.loads(response_content)
    except json.JSONDecodeError as jde:
        logger.error(f"JSON decode error for case {case_number}: {jde}")
        return {"case_number": case_number, "error": "Invalid JSON response from AI."}
    except Exception as e:
        logger.error(f"Error processing case {case_number}: {e}")
        return {"case_number": case_number, "error": "Error processing AI"}

# Process cases for summaries with concurrency
def process_cases(cases_data, custom_prompt, max_workers=100):
    structured_output = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all summary tasks to the executor
        future_to_case = {
            executor.submit(get_summary, case_text, custom_prompt, case_num): case_num
            for case_text, case_num in cases_data
        }
        logger.info(f"Submitted {len(cases_data)} cases for concurrent processing.")

        for future in as_completed(future_to_case):
            case_num = future_to_case[future]
            try:
                parsed_json = future.result() or {}
                # Use AI's score if available
                parsed_json['score'] = parsed_json.get('score', len(parsed_json.get('major_findings', [])) * 3 + len(parsed_json.get('minor_findings', [])))
                logger.info(f"Processed summary for case {case_num}: Score {parsed_json['score']}")
            except Exception as e:
                logger.error(f"Error processing case {case_num}: {e}")
                parsed_json = {"case_number": case_num, "error": "Error processing AI"}
            structured_output.append(parsed_json)
    logger.info(f"Completed processing {len(structured_output)} summaries.")
    return structured_output

# Extract cases and add AI summary tab
def extract_cases(text, custom_prompt):
    # Normalize line endings to Unix style
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    logger.debug("Normalized line endings.")
    
    # Split on 'Case <number>' at the start of a line
    cases = re.split(r'(?m)^Case\s+(\d+)', text, flags=re.IGNORECASE)
    logger.debug(f"SPLIT RESULT FOR CASES: {cases}")
    
    logger.info(f"Total elements after split: {len(cases)}")
    for idx, element in enumerate(cases):
        logger.debug(f"Element {idx}: {element[:100]}{'...' if len(element) > 100 else ''}")
    
    cases_data = []
    parsed_cases = []

    # Extract all cases and prepare data for concurrent processing
    for i in range(1, len(cases), 2):
        case_num = cases[i]
        case_content = cases[i + 1].strip() if i + 1 < len(cases) else ""
        logger.debug(f"Processing Case {case_num}:")
        logger.debug(f"Case Content: {case_content[:200]}{'...' if len(case_content) > 200 else ''}")
        
        reports = re.split(r'\s*(Attending\s+Report\s*:|Resident\s+Report\s*:)\s*', case_content, flags=re.IGNORECASE)
        logger.debug(f"Reports split: {reports}")
        
        if len(reports) >= 3:
            attending_report = reports[2].strip()
            resident_report = reports[4].strip() if len(reports) > 4 else ""
            case_text = f"Resident Report: {resident_report}\nAttending Report: {attending_report}"
            cases_data.append((case_text, case_num))
            logger.info(f"Prepared case {case_num} for processing.")
        else:
            logger.warning(f"Case {case_num} does not contain both Attending and Resident Reports.")

    if not cases_data:
        logger.warning("No valid cases found in the submitted text.")
        return parsed_cases

    # Process all summaries concurrently
    ai_summaries = process_cases(cases_data, custom_prompt, max_workers=20)

    # Map each summary to its corresponding case
    for ai_summary in ai_summaries:
        case_num = ai_summary.get('case_number')
        if not case_num:
            logger.warning("Summary missing 'case_number'. Skipping.")
            continue  # Skip if case_num is missing
        
        # **Convert case_num to string for accurate comparison**
        case_num = str(case_num)
        
        # Find the corresponding case content
        case_content = next((ct for ct, num in cases_data if num == case_num), "")
        if case_content:
            try:
                resident_report = case_content.split("\nAttending Report:")[0].replace("Resident Report: ", "").strip()
                attending_report = case_content.split("\nAttending Report:")[1].strip()
            except IndexError:
                logger.error(f"Error parsing reports for case {case_num}.")
                continue
            parsed_cases.append({
                'case_num': case_num,  # Already a string
                'resident_report': resident_report,
                'attending_report': attending_report,
                'percentage_change': calculate_change_percentage(resident_report, remove_attending_review_line(attending_report)),
                'diff': create_diff_by_section(resident_report, attending_report),
                'summary': ai_summary if 'error' not in ai_summary else None
            })
            logger.info(f"Assigned summary to case {case_num}.")
    logger.info(f"Total parsed_cases to return: {len(parsed_cases)}")
    logger.debug(f"Parsed_cases: {json.dumps(parsed_cases, indent=2)}")
    return parsed_cases

@app.route('/', methods=['GET', 'POST'])
def index():
    custom_prompt = request.form.get('custom_prompt', DEFAULT_PROMPT)
    case_data = []

    if request.method == 'POST':
        text_block = request.form['report_text']
        if not text_block.strip():
            logger.warning("No report text provided.")
        else:
            logger.info("Starting case extraction and processing.")
            case_data = extract_cases(text_block, custom_prompt)
            logger.info(f"Completed case extraction and processing. Number of cases extracted: {len(case_data)}")
            logger.debug(f"Extracted case_data: {json.dumps(case_data, indent=2)}")  # Detailed debug log

    # Log the case_data before passing to the template
    logger.info(f"Passing {len(case_data)} cases to the template.")
    logger.debug(f"case_data being passed: {json.dumps(case_data, indent=2)}")  # Detailed debug log

    template = """
<html>
    <head>
        <title>Radiology Report Diff & Summarizer</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
        <!-- Added dotlottie-player script -->
        <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
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
                    border-radius: 15px;
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

            /* Loading animation styling */
            #loadingAnimation {
                display: none; /* Hidden by default */
                margin-top: 20px;
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
                <!-- Added dotlottie-player for loading animation -->
                <dotlottie-player id="loadingAnimation" src="https://lottie.host/817661a8-2608-4435-89a5-daa620a64c36/WtsFI5zdEK.lottie" background="transparent" speed="1" style="width: 300px; height: 300px;" loop autoplay></dotlottie-player>
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
            {% endif %}
            <!-- Always include these elements to prevent JS errors -->
            <div class="btn-group" role="group" aria-label="Sort Options">
                <button type="button" class="btn btn-secondary" onclick="sortCases('case_number')">Sort by Case Number</button>
                <button type="button" class="btn btn-secondary" onclick="sortCases('percentage_change')">Sort by Percentage Change</button>
                <button type="button" class="btn btn-secondary" onclick="sortCases('summary_score')">Sort by Summary Score</button>
            </div>
            <ul id="caseNav"></ul>
            <div id="caseContainer"></div>
        </div>
        <!-- Added scroll-to-top button -->
        <button id="scrollToTopBtn" onclick="scrollToTop()">Top ⬆</button>
        <script>
            let caseData = {{ case_data | tojson }};
            console.log("Received caseData:", caseData); // Debugging

            function sortCases(option) {
                console.log("Sorting cases by:", option); // Debugging
                if (option === "case_number") {
                    caseData.sort((a, b) => parseInt(a.case_num) - parseInt(b.case_num));
                } else if (option === "percentage_change") {
                    caseData.sort((a, b) => b.percentage_change - a.percentage_change);
                } else if (option === "summary_score") {
                    caseData.sort((a, b) => (b.summary && b.summary.score || 0) - (a.summary && a.summary.score || 0));
                }
                console.log("Sorted caseData:", caseData); // Debugging
                displayCases();
                displayNavigation();
            }

            function displayNavigation() {
                const nav = document.getElementById('caseNav');
                if (!nav) {
                    console.error("Element with id 'caseNav' not found.");
                    return; // Prevent errors if element not found
                }
                nav.innerHTML = '';
                if (caseData.length === 0) {
                    nav.innerHTML = '<li>No cases to display.</li>';
                    console.log("No cases to display in navigation.");
                    return;
                }
                console.log("Displaying navigation for cases.");
                caseData.forEach(caseObj => {
                    nav.innerHTML += `
                        <li>
                            <a href="#case${caseObj.case_num}">Case ${caseObj.case_num}</a> - ${caseObj.percentage_change}% change - Score: ${(caseObj.summary && caseObj.summary.score) || 'N/A'}
                        </li>
                    `;
                });
                console.log("Navigation populated.");
            }

            function displayCases() {
                const container = document.getElementById('caseContainer');
                if (!container) {
                    console.error("Element with id 'caseContainer' not found.");
                    return; // Prevent errors if element not found
                }
                container.innerHTML = '';
                if (caseData.length === 0) {
                    container.innerHTML = '<p>No cases to display.</p>';
                    console.log("No cases to display in container.");
                    return;
                }
                console.log("Displaying cases.");
                caseData.forEach(caseObj => {
                    if (!caseObj.summary) {
                        container.innerHTML += `
                            <div id="case${caseObj.case_num}">
                                <h4>Case ${caseObj.case_num} - ${caseObj.percentage_change}% change</h4>
                                <p style="color: red;"><strong>Error:</strong> Unable to generate summary for this case.</p>
                                <hr>
                            </div>
                        `;
                        console.log(`Displayed error for case ${caseObj.case_num}.`);
                        return;
                    }
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
                                        <p><strong>Score:</strong> ${caseObj.summary.score || 'N/A'}</p>
                                        ${caseObj.summary.major_findings && caseObj.summary.major_findings.length > 0 ? `<p><strong>Major Findings:</strong></p><ul>${caseObj.summary.major_findings.map(finding => `<li>${finding}</li>`).join('')}</ul>` : ''}
                                        ${caseObj.summary.minor_findings && caseObj.summary.minor_findings.length > 0 ? `<p><strong>Minor Findings:</strong></p><ul>${caseObj.summary.minor_findings.map(finding => `<li>${finding}</li>`).join('')}</ul>` : ''}
                                        ${caseObj.summary.clarifications && caseObj.summary.clarifications.length > 0 ? `<p><strong>Clarifications:</strong></p><ul>${caseObj.summary.clarifications.map(clarification => `<li>${clarification}</li>`).join('')}</ul>` : ''}
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
                    console.log(`Displayed case ${caseObj.case_num}.`);
                });
                console.log("All cases displayed.");
            }

            document.addEventListener("DOMContentLoaded", () => {
                if (caseData && caseData.length > 0) {
                    console.log("Case data available. Rendering cases and navigation.");
                    displayCases();
                    displayNavigation();
                } else {
                    console.log("No case data available to display.");
                }
            });

            // Scroll-to-top function
            function scrollToTop() {
                const majorFindingsSection = document.getElementById('majorFindings');
                if (majorFindingsSection) {
                    majorFindingsSection.scrollIntoView({ behavior: 'smooth' });
                }
            }

            // Show loading animation on form submission
            document.getElementById('reportForm').addEventListener('submit', function() {
                document.getElementById('loadingAnimation').style.display = 'block';
            });
        </script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
</html>
    """

    return render_template_string(template, case_data=case_data, custom_prompt=custom_prompt)

if __name__ == '__main__':
    app.run(debug=True)
