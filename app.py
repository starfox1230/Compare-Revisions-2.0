from flask import Flask, render_template_string, request
import re
import os
import json
import logging
import openai

app = Flask(__name__)

# Configure logging with timestamp and log level
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for general logs
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

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

def parse_cases(text):
    """
    Parses the input text into individual cases.
    Expected format:
    Case 1
    Resident Report:
    ...
    Attending Report:
    ...
    
    Case 2
    Resident Report:
    ...
    Attending Report:
    ...
    """
    cases = re.split(r'(?m)^Case\s+(\d+)', text, flags=re.IGNORECASE)
    parsed_cases = []
    for i in range(1, len(cases), 2):
        case_num = cases[i].strip()
        case_content = cases[i + 1].strip() if i + 1 < len(cases) else ""
        resident_match = re.search(r'Resident Report:\s*(.*?)\s*Attending Report:', case_content, re.DOTALL | re.IGNORECASE)
        attending_match = re.search(r'Attending Report:\s*(.*)', case_content, re.DOTALL | re.IGNORECASE)
        resident = resident_match.group(1).strip() if resident_match else ""
        attending = attending_match.group(1).strip() if attending_match else ""
        if resident and attending:
            parsed_cases.append({
                'case_num': case_num,
                'resident': resident,
                'attending': attending
            })
        else:
            logger.warning(f"Case {case_num} is missing Resident or Attending Report.")
    return parsed_cases

def get_summary(case_text, custom_prompt, case_number):
    """
    Sends the resident and attending reports to OpenAI and retrieves the summary JSON.
    """
    try:
        logger.info(f"Processing case {case_number}")
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs structured JSON summaries of radiology report differences."},
                {"role": "user", "content": f"{custom_prompt}\nCase Number: {case_number}\n{case_text}"}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        response_content = response.choices[0].message['content']
        logger.debug(f"Received response for case {case_number}: {response_content}")
        return json.loads(response_content)
    except json.JSONDecodeError as jde:
        logger.error(f"JSON decode error for case {case_number}: {jde}")
        return {"case_number": case_number, "error": "Invalid JSON response from AI."}
    except Exception as e:
        logger.error(f"Error processing case {case_number}: {e}")
        return {"case_number": case_number, "error": "Error processing AI"}

@app.route('/', methods=['GET', 'POST'])
def index():
    custom_prompt = request.form.get('custom_prompt', DEFAULT_PROMPT)
    summaries = []
    input_text = ""

    if request.method == 'POST':
        input_text = request.form.get('report_text', '').strip()
        if not input_text:
            logger.warning("No report text provided.")
        else:
            logger.info("Starting case extraction and processing.")
            cases = parse_cases(input_text)
            logger.info(f"Found {len(cases)} cases to process.")
            for case in cases:
                case_text = f"Resident Report: {case['resident']}\nAttending Report: {case['attending']}"
                summary = get_summary(case_text, custom_prompt, case['case_num'])
                summaries.append(summary)
            logger.info("Completed processing all cases.")

    # Simple HTML Template
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Radiology Report Summarizer</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
        <style>
            body { background-color: #f8f9fa; color: #212529; font-family: Arial, sans-serif; }
            textarea { resize: vertical; }
            .summary { margin-top: 20px; }
            .error { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4">Radiology Report Summarizer</h1>
            <form method="POST">
                <div class="mb-3">
                    <label for="report_text" class="form-label">Paste your reports here:</label>
                    <textarea class="form-control" id="report_text" name="report_text" rows="10">{{ input_text }}</textarea>
                </div>
                <div class="mb-3">
                    <label for="custom_prompt" class="form-label">Customize OpenAI Prompt (optional):</label>
                    <textarea class="form-control" id="custom_prompt" name="custom_prompt" rows="5">{{ custom_prompt }}</textarea>
                </div>
                <button type="submit" class="btn btn-primary">Submit</button>
            </form>

            {% if summaries %}
                <div class="summary">
                    <h2>Summaries</h2>
                    <ul class="list-group">
                        {% for summary in summaries %}
                            <li class="list-group-item">
                                <strong>Case {{ summary.case_number }}</strong><br>
                                {% if summary.error %}
                                    <span class="error">Error: {{ summary.error }}</span>
                                {% else %}
                                    <strong>Major Findings:</strong>
                                    {% if summary.major_findings %}
                                        <ul>
                                            {% for mf in summary.major_findings %}
                                                <li>{{ mf }}</li>
                                            {% endfor %}
                                        </ul>
                                    {% else %}
                                        <p>None</p>
                                    {% endif %}
                                    
                                    <strong>Minor Findings:</strong>
                                    {% if summary.minor_findings %}
                                        <ul>
                                            {% for mif in summary.minor_findings %}
                                                <li>{{ mif }}</li>
                                            {% endfor %}
                                        </ul>
                                    {% else %}
                                        <p>None</p>
                                    {% endif %}
                                    
                                    <strong>Clarifications:</strong>
                                    {% if summary.clarifications %}
                                        <ul>
                                            {% for clar in summary.clarifications %}
                                                <li>{{ clar }}</li>
                                            {% endfor %}
                                        </ul>
                                    {% else %}
                                        <p>None</p>
                                    {% endif %}
                                    
                                    <strong>Score:</strong> {{ summary.score }}
                                {% endif %}
                            </li>
                        {% endfor %}
                    </ul>
                </div>
            {% endif %}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return render_template_string(template, summaries=summaries, custom_prompt=custom_prompt, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
