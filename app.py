from flask import Flask, render_template_string, request
import difflib
import re
import os
import json
import openai

app = Flask(__name__)

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs structured JSON summaries of radiology report differences."},
                {"role": "user", "content": f"{custom_prompt}\nCase Number: {case_number}\n{case_text}"}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        return json.loads(response.choices[0].message['content'])
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
            parsed_json = get_summary(case_text, custom_prompt, case_number=case_number)
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
    custom_prompt = request.form.get('custom_prompt', 'Summarize report changes as JSON')
    case_data = []
    if request.method == 'POST':
        text_block = request.form['report_text']
        case_data = extract_cases(text_block, custom_prompt)
    template = """..."""  # Include entire HTML/CSS template here.
    return render_template_string(template, case_data=case_data, custom_prompt=custom_prompt)

if __name__ == '__main__':
    app.run(debug=True)
