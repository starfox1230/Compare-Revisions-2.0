from flask import Flask, render_template_string, request
import re
import os
import json
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_PROMPT = """Succinctly organize the changes made by the attending to the resident's radiology reports into: 
1) missed major findings (findings discussed by the attending but not by the resident that also fit under these categories: retained sponge or other clinically significant foreign body, mass or tumor, malpositioned line or tube of immediate clinical concern, life-threatening hemorrhage or vascular disruption, necrotizing fasciitis, free air or active leakage from the GI tract, ectopic pregnancy, intestinal ischemia or portomesenteric gas, ovarian torsion, testicular torsion, placental abruption, absent perfusion in a postoperative transplant, renal collecting system obstruction with signs of infection, acute cholecystitis, intracranial hemorrhage, midline shift, brain herniation, cerebral infarction or abscess or meningoencephalitis, airway compromise, abscess or discitis,  hemorrhage, cord compression or unstable spine fracture or transection, acute cord hemorrhage or infarct, pneumothorax, large pericardial effusion, findings suggestive of active TB, impending pathologic fracture, acute fracture, absent perfusion in a postoperative kidney, brain death, high probability ventilation/perfusion (VQ) lung scan, arterial dissection or occlusion, acute thrombotic or embolic event including DVT and pulmonary thromboembolism, and aneurysm or vascular disruption), 
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
