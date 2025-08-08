from flask import Flask, render_template_string, request
import difflib, re, os, json, logging
import openai as _openai_pkg
from openai import OpenAI
from openai import (
    APIError, APIConnectionError, BadRequestError,
    AuthenticationError, RateLimitError, NotFoundError
)
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

# --------------------------- Logging ---------------------------------
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for production, DEBUG is too verbose
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

logger.info(f"openai sdk version: {_openai_pkg.__version__}")
logger.info(f"API key present: {bool(os.getenv('OPENAI_API_KEY'))}")

# --------------------------- OpenAI ----------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_ID = os.getenv("MODEL_ID", "gpt-4-turbo") # Updated default model

# ----------------------- Tool Definition -----------------------------
RADIOLOGY_SUMMARY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "summarize_radiology_report",
            "description": "Summarizes the differences between a resident and attending radiology report, categorizing findings and calculating a score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_number": {
                        "type": "string",
                        "description": "The case number being analyzed."
                    },
                    "major_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of major, management-changing findings identified by the attending but missed by the resident."
                    },
                    "minor_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of minor, clinically relevant but not urgent findings."
                    },
                    "clarifications": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of wording changes, formatting corrections, or minor descriptor edits with no management change."
                    },
                    "score": {
                        "type": "integer",
                        "description": "A calculated score based on the findings: 3 points for each major finding and 1 point for each minor finding."
                    }
                },
                "required": ["case_number", "major_findings", "minor_findings", "clarifications", "score"]
            }
        }
    }
]

# ----------------------- Default prompt ------------------------------
DEFAULT_PROMPT = """Developer: Radiology revision differ (GPT-5).

<goal>
Compare resident vs attending radiology reports and output ONLY the attending’s changes, split into:
- major_findings
- minor_findings
- clarifications
Then compute a score.
</goal>

<non_goals>
- Do not restate content already present in the resident report.
- Do not speculate or invent findings.
- Do not output any prose outside the JSON schema.
</non_goals>

<inputs_and_normalization>
You will receive both reports for a case. Assume the attending is correct.
Ignore standard attending attestation boilerplate lines.
If input is malformed or either report is missing → return empty arrays and score 0.
</inputs_and_normalization>

<taxonomy_and_definitions>
We classify each attending change by SEVERITY into one of three output buckets:

MAJOR (management-changing, urgent/critical). Classify as MAJOR when newly added, corrected from benign/absent → present/critical, or critically negated per rules. The canonical MAJOR catalogue includes ALL of the following:

- retained foreign body
- mass / tumor
- malpositioned line/tube of immediate clinical concern (airway, esophagus, pleura, heart, great vessels)
- life-threatening hemorrhage or vascular disruption (including active bleeding/extravasation)
- necrotizing fasciitis
- free air or active GI leak
- ectopic pregnancy
- intestinal ischemia or portomesenteric gas
- ovarian torsion or testicular torsion
- placental abruption
- absent perfusion in a postoperative transplant (e.g., kidney)
- infected renal collecting system obstruction
- acute cholecystitis
- intracranial hemorrhage
- midline shift
- brain herniation
- acute cerebral infarction / abscess / meningoencephalitis
- airway compromise
- abscess / discitis (spinal or deep space infection of similar acuity)
- hemorrhage (non-intracranial acute hemorrhage)
- cord compression / unstable spine fracture / transection
- acute cord hemorrhage / infarct
- pneumothorax
- large pericardial effusion (tamponade concern)
- findings suggestive of active TB
- impending pathologic fracture
- acute fracture
- brain death
- high-probability V/Q scan
- arterial dissection / occlusion
- acute thrombotic/embolic event (DVT, PE)
- aneurysm or vascular disruption

MINOR (clinically relevant but not immediately dangerous):
- New non-urgent pathology or meaningful correction that affects care but is not emergent.
- Examples: new subsegmental PE without instability/right-heart strain; reclassification to a benign diagnosis; size or characteristic updates that cross a follow-up threshold; removal of non-critical diagnoses.

CLARIFICATIONS (descriptor/wording/formatting with no management change):
- Same-entity edits: measurements, exact segment/lobe/level, refined laterality, degree words (trace/small/mild/moderate) without practical impact, count tweaks without category change, chronicity wording that doesn’t alter urgency, certainty hedges, grammar/duplication cleanup.
</taxonomy_and_definitions>

<same_entity_test>
Before assigning severity, perform a SAME-ENTITY match:
(anatomic structure + side/laterality + level/segment + device type).
If an attending item matches a resident item on the same entity, treat differences as descriptor edits unless there is a clear risk upgrade per MAJOR/MINOR rules.
If unclear whether items refer to the same entity, favor the lower-severity bucket (MINOR over MAJOR; CLARIFICATIONS over MINOR) unless there is explicit red-flag language.
</same_entity_test>

<additions_vs_corrections>
An attending change can be an ADDITION (attending adds something) or a CORRECTION (attending negates/downgrades the resident).

A) ADDITIONS:
- Apply MAJOR if the added entity is urgent/critical (red-flags, device malposition, etc.).
- Otherwise MINOR if relevant but not emergent.
- CLARIFICATIONS if purely descriptive.

B) CORRECTIONS (negations/downgrades):
- Critical Negation / Risk-Downshift: If the attending removes or downgrades a resident-reported critical diagnosis in a way that would change immediate management (e.g., “PE” → “no PE”), classify as MAJOR.
- Non-critical corrections (no immediate management change) → MINOR.
- Same-entity descriptor/wording without management change → CLARIFICATIONS.
</additions_vs_corrections>

<certainty_aware_corrections>
For attending negations of resident claims about critical diagnoses (PE, ICH, PTX, bowel perforation, etc.), use the resident’s certainty to calibrate severity:

Resident certainty tiers (parse phrases):
- DEFINITE: “present”, “acute”, “diagnostic of”, “consistent with”
- PROBABLE: “probable”, “likely”, “suspicious for”
- POSSIBLE: “possible”, “questionable”, “cannot exclude”, “equivocal”
- NEUTRAL: “evaluate for…”, “limited for…”, “no convincing evidence of…”

When attending says “no [critical diagnosis]”:
- DEFINITE → NO: MAJOR
- PROBABLE → NO: MINOR
- POSSIBLE/QUESTIONABLE → NO: CLARIFICATIONS
- NEUTRAL/WORKUP → NO: CLARIFICATIONS

Management override: if the resident explicitly recommends urgent therapy/escalation based on their claim (e.g., “start anticoagulation”), the attending’s negation is MAJOR regardless of hedge level.
Do not label as MAJOR solely because the topic is critical; the change itself must be management-changing.
</certainty_aware_corrections>

<descriptor_library_thresholds>
Treat the following as CLARIFICATIONS unless they cross a management threshold:
- measurements/rounding; refined location/lobe/segment; laterality precision; degree words; count; chronicity; certainty hedges; grammar/style.
Upgrade to MINOR if the change crosses a known management/follow-up cutoff or meaningfully alters non-urgent care.
</descriptor_library_thresholds>

<dedupe_and_consolidation>
- Consolidate multi-phrase edits about the same pathology into a single item using the attending’s wording (e.g., “acute MCA infarct with mass effect” = one item).
- De-duplicate semantically identical content.
</dedupe_and_consolidation>

<type_hints_and_modifiers_in_parentheses>
Append short parenthetical annotations to each item to aid learning. This does NOT change the JSON schema; they are part of the strings.

Type (include only when clear):
- perceptual  = attending adds a finding with no resident mention for that location/structure/device.
- interpretive = attending replaces/negates a resident item referring to the same location/structure/device.
- typographic  = spelling/formatting/duplication fix.

Common modifiers (semicolon-separated; pick 0–3 as appropriate; avoid quotes):
- critical correction, risk upgrade, risk downshift
- malpositioned device, artifact resolved
- descriptor: size, descriptor: location, descriptor: count, descriptor: laterality, descriptor: chronicity, descriptor: certainty
- threshold crossed, classification change, laterality correction, duplication removed, follow-up impact

Formatting:
- Use parentheses after the item, e.g., “Right pneumothorax (perceptual; new critical finding)” or
  “No PE (interpretive; critical correction; resident said acute PE)”.
- If uncertain about type, omit the type label; include only safe modifiers.
- Do not use quotation marks inside parentheses; keep them short and plain-text.
</type_hints_and_modifiers_in_parentheses>

<tie_breakers_and_hygiene>
- If ambiguous MAJOR vs MINOR → choose MINOR.
- If ambiguous MINOR vs CLARIFICATIONS → choose CLARIFICATIONS.
- Exclude anything already present (semantically) in the resident report.
- No speculation; no new facts beyond the attending’s text.
</tie_breakers_and_hygiene>

<validation_checklist>
Before output:
- Only attending-introduced or attending-corrected content appears in arrays.
- CLARIFICATIONS contains descriptor/wording/formatting items only.
- Score = (3 × count(major_findings)) + (1 × count(minor_findings)).
- Exact case_number is echoed.
- JSON matches the provided schema; no extra keys; no prose.
</validation_checklist>

<output_contract>
Return your analysis by calling the `summarize_radiology_report` function tool.
Do not include any prose before or after the JSON.
</output_contract>

<examples>

Example 1 — Same-entity descriptor (no management delta):
Resident: “PE in RLL segmental branches.”
Attending: “PE in RLL subsegmental branches; small clot burden.”
→ major: []
→ minor: []
→ clarifications: ["PE location refined to subsegmental; small clot burden (descriptor: location; descriptor: certainty)"]
→ score: 0

Example 2 — New urgent finding:
Resident: “No pneumothorax.”
Attending: “Small right pneumothorax.”
→ major: ["Right pneumothorax (perceptual; new critical finding)"]
→ minor: []
→ clarifications: []
→ score: 3

Example 3 — Critical negation (definite → no):
Resident: “Acute PE; start anticoagulation.”
Attending: “No pulmonary embolism.”
→ major: ["No pulmonary embolism (interpretive; critical correction; resident said acute PE)"]
→ minor: []
→ clarifications: []
→ score: 3

Example 4 — Probable → no (soft negation):
Resident: “Probable segmental PE, RLL.”
Attending: “No PE.”
→ major: []
→ minor: ["No PE (interpretive; correction; resident said probable)"]
→ clarifications: []
→ score: 1

Example 5 — Questionable → no (wording cleanup):
Resident: “Questionable subsegmental PE.”
Attending: “No PE.”
→ major: []
→ minor: []
→ clarifications: ["No PE (interpretive; descriptor: certainty; resident said questionable)"]
→ score: 0

Example 6 — Malpositioned device:
Resident: “NG tube present.”
Attending: “NG tube coiled in esophagus.”
→ major: ["Malpositioned NG tube (esophageal) (perceptual; malpositioned device)"]
→ minor: []
→ clarifications: []
→ score: 3

Example 7 — Free air artifact (critical downshift):
Resident: “Free air under diaphragm.”
Attending: “No free air; prior image was artifact.”
→ major: ["No free air (interpretive; critical correction; artifact resolved)"]
→ minor: []
→ clarifications: []
→ score: 3

Example 8 — Non-urgent correction:
Resident: “Normal abdomen.”
Attending: “Benign hepatic hemangioma; cholelithiasis without cholecystitis.”
→ major: []
→ minor: ["Hepatic hemangioma (perceptual; classification change; follow-up impact)", "Cholelithiasis without cholecystitis (perceptual)"]
→ clarifications: []
→ score: 2

Example 9 — Threshold crossing (upgrade to MINOR):
Resident: “Pulmonary nodule 5 mm.”
Attending: “Pulmonary nodule 8 mm (follow-up recommended).”
→ major: []
- minor: ["Pulmonary nodule 8 mm (descriptor: size; threshold crossed; follow-up impact)"]
→ clarifications: []
→ score: 1

Example 10 — Laterality/location correction:
Resident: “Left lower lobe pneumonia.”
Attending: “Right lower lobe pneumonia.”
→ major: []
→ minor: ["Right lower lobe pneumonia (interpretive; laterality correction)"]
→ clarifications: []
→ score: 1

Example 11 — ICH added:
Resident: “No acute intracranial hemorrhage.”
Attending: “Small acute subarachnoid hemorrhage.”
→ major: ["Acute subarachnoid hemorrhage (perceptual; new critical finding)"]
→ minor: []
→ clarifications: []
→ score: 3

Example 12 — Descriptor only:
Resident: “Sigmoid diverticulitis.”
Attending: “Sigmoid diverticulitis with trace adjacent fluid; no abscess.”
→ major: []
→ minor: []
→ clarifications: ["Adds trace adjacent fluid; no abscess (descriptor: degree; descriptor: negative finding)"]
→ score: 0

</examples>
"""
# --------------------------- Helpers ---------------------------------
def normalize_text(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def remove_attending_review_line(text):
    # This regex is more robust to slight variations in the boilerplate.
    pattern = r"(?i)As the attending.*I have personally reviewed.*and agree with the wording.*"
    return re.sub(pattern, "", text, flags=re.MULTILINE)

def calculate_change_percentage(resident_text, attending_text):
    matcher = difflib.SequenceMatcher(None, resident_text.split(), attending_text.split())
    return round((1 - matcher.ratio()) * 100, 1) # Rounded to 1 decimal place

def split_into_paragraphs(text):
    # Split by one or more newlines, which is more robust for report formatting
    paragraphs = re.split(r'\n+', text)
    return [para.strip() for para in paragraphs if para.strip()]

def create_diff_by_section(resident_text, attending_text):
    """
    MODERNIZED: This function now uses CSS classes instead of inline styles for a cleaner
    separation of concerns and easier styling in the frontend.
    """
    resident_text = normalize_text(resident_text)
    attending_text = normalize_text(remove_attending_review_line(attending_text))
    resident_paragraphs = split_into_paragraphs(resident_text)
    attending_paragraphs = split_into_paragraphs(attending_text)

    diff_html = ""
    matcher = difflib.SequenceMatcher(None, resident_paragraphs, attending_paragraphs)

    for opcode, a1, a2, b1, b2 in matcher.get_opcodes():
        if opcode == 'equal':
            for paragraph in resident_paragraphs[a1:a2]:
                diff_html += f"<p>{paragraph}</p>"
        elif opcode == 'insert':
            for paragraph in attending_paragraphs[b1:b2]:
                diff_html += f'<div class="diff-para-insert"><span class="diff-marker">+</span><p>{paragraph}</p></div>'
        elif opcode == 'delete':
            for paragraph in resident_paragraphs[a1:a2]:
                diff_html += f'<div class="diff-para-delete"><span class="diff-marker">-</span><p>{paragraph}</p></div>'
        elif opcode == 'replace':
            res_paragraphs = " ".join(resident_paragraphs[a1:a2])
            att_paragraphs = " ".join(attending_paragraphs[b1:b2])
            
            # Using a simpler line-by-line diff for replaced paragraphs for clarity
            s = difflib.SequenceMatcher(None, res_paragraphs.split(), att_paragraphs.split())
            replaced_html = ""
            for op, i1, i2, j1, j2 in s.get_opcodes():
                if op == 'equal':
                    replaced_html += " ".join(att_paragraphs.split()[j1:j2]) + " "
                elif op == 'delete':
                    replaced_html += "<del>" + " ".join(res_paragraphs.split()[i1:i2]) + "</del> "
                elif op == 'insert':
                    replaced_html += "<ins>" + " ".join(att_paragraphs.split()[j1:j2]) + "</ins> "
                elif op == 'replace':
                    replaced_html += "<del>" + " ".join(res_paragraphs.split()[i1:i2]) + "</del> "
                    replaced_html += "<ins>" + " ".join(att_paragraphs.split()[j1:j2]) + "</ins> "

            diff_html += f'<div class="diff-para-replace"><p>{replaced_html}</p></div>'
    return diff_html

# ------------------------- OpenAI call --------------------------------
def get_summary(case_text, custom_prompt, case_number):
    try:
        logger.info(f"Processing case {case_number} with model={MODEL_ID} using function calling.")
        
        # Updated to use the chat completions endpoint which is standard now
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": custom_prompt},
                {"role": "user", "content": f"Case Number: {case_number}\n{case_text}"}
            ],
            tools=RADIOLOGY_SUMMARY_TOOL,
            tool_choice={"type": "function", "function": {"name": "summarize_radiology_report"}}
        )

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls and tool_calls[0].function.name == "summarize_radiology_report":
            raw_arguments = tool_calls[0].function.arguments
            parsed_json = json.loads(raw_arguments)
            logger.info(f"Received and parsed function call for case {case_number}: OK")
            return parsed_json
        else:
            logger.error(f"Function call not found in response for case {case_number}")
            return {"case_number": case_number, "error": "AI did not return the expected tool call."}

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failure for case {case_number}: {e!r}")
        logger.debug(f"Raw arguments from model that failed to parse: {raw_arguments}")
        return {"case_number": case_number, "error": "Invalid JSON in tool arguments from AI."}
    except (NotFoundError, BadRequestError, AuthenticationError, RateLimitError, APIConnectionError, APIError) as e:
        logger.error(f"API Error processing case {case_number}: {type(e).__name__} - {e}")
        return {"case_number": case_number, "error": f"API Error: {type(e).__name__}"}
    except Exception as e:
        logger.error(f"Unhandled error for case {case_number}: {repr(e)}")
        return {"case_number": case_number, "error": f"Unhandled: {repr(e)}"}

# ------------------------ Pipeline ------------------------------------
def process_cases(cases_data, custom_prompt, max_workers=8):
    structured_output = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {
            executor.submit(get_summary, case_text, custom_prompt, case_num): case_num
            for case_text, case_num in cases_data
        }
        logger.info(f"Submitted {len(cases_data)} cases for concurrent processing.")

        for future in as_completed(future_to_case):
            case_num = future_to_case[future]
            try:
                parsed_json = future.result() or {}
                # Ensure score exists, calculating it if necessary
                if 'score' not in parsed_json:
                    parsed_json['score'] = (len(parsed_json.get('major_findings', [])) * 3 + 
                                            len(parsed_json.get('minor_findings', [])))
                logger.info(f"Processed summary for case {case_num}: Score {parsed_json.get('score', 'N/A')}")
            except Exception as e:
                logger.error(f"Error processing case {case_num}: {e}")
                parsed_json = {"case_number": case_num, "error": f"Unhandled: {repr(e)}"}
            structured_output.append(parsed_json)
    logger.info(f"Completed processing {len(structured_output)} summaries.")
    return structured_output

def extract_cases(text, custom_prompt):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # More robust regex to find 'Case' followed by a number, even with noise.
    cases = re.split(r'(?i)\bCase\s+([a-zA-Z0-9\-]+)', text)

    cases_data, parsed_cases = [], []
    if len(cases) < 2: # No cases found
        logger.warning("No cases found matching the 'Case [number]' pattern.")
        return []

    # The first element is text before the first "Case", so we skip it.
    for i in range(1, len(cases), 2):
        case_num = cases[i].strip()
        case_content = cases[i + 1].strip() if (i + 1) < len(cases) else ""
        
        # Regex to find resident/attending blocks, ignoring case and looking for a colon
        resident_match = re.search(r'(?i)Resident Report\s*:(.*)', case_content, re.DOTALL)
        attending_match = re.search(r'(?i)Attending Report\s*:(.*)', case_content, re.DOTALL)

        # To ensure we capture the right text, we can split based on the next label
        resident_report = ""
        if resident_match:
            if attending_match and resident_match.start() < attending_match.start():
                 resident_report = case_content[resident_match.end():attending_match.start()].strip()
            else:
                 resident_report = resident_match.group(1).strip()
        
        attending_report = attending_match.group(1).strip() if attending_match else ""

        if resident_report and attending_report:
            case_text = (
                f"Resident Report: {normalize_text(resident_report)}\n\n"
                f"Attending Report: {normalize_text(attending_report)}"
            )
            cases_data.append((case_text, case_num))
            logger.info(f"Prepared case {case_num} for processing.")
        else:
            logger.warning(f"Case {case_num} is missing a clearly labeled Resident or Attending report. Skipping.")

    if not cases_data:
        return []

    ai_summaries = process_cases(cases_data, custom_prompt, max_workers=8)
    summaries_by_case_num = {str(s.get('case_number')): s for s in ai_summaries}

    for case_text, case_num in cases_data:
        try:
            # Extract reports again for diff generation
            resident_report_raw = re.search(r"Resident Report:(.*)Attending Report:", case_text, re.S | re.I).group(1).strip()
            attending_report_raw = re.search(r"Attending Report:(.*)", case_text, re.S | re.I).group(1).strip()

            ai_summary = summaries_by_case_num.get(case_num, {})
            
            parsed_cases.append({
                'case_num': case_num,
                'resident_report': resident_report_raw,
                'attending_report': attending_report_raw,
                'percentage_change': calculate_change_percentage(resident_report_raw, remove_attending_review_line(attending_report_raw)),
                'diff': create_diff_by_section(resident_report_raw, attending_report_raw),
                'summary': ai_summary if 'error' not in ai_summary else None,
                'summary_error': ai_summary.get('error')
            })
        except Exception as e:
            logger.error(f"Error during final parsing for case {case_num}: {e}")
            continue
            
    return parsed_cases
# ---------------------------- Web -------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    custom_prompt = request.form.get('custom_prompt', DEFAULT_PROMPT)
    case_data = []

    if request.method == 'POST':
        text_block = request.form['report_text']
        if text_block.strip():
            case_data = extract_cases(text_block, custom_prompt)

    # MODERNIZED TEMPLATE
    template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Radiology Report Review Hub</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
    <style>
        :root {
            --bg-color: #1a1b26; --surface-color: #24283b; --card-color: #2a2f4a;
            --text-color: #c0caf5; --primary-color: #7aa2f7; --green: #9ece6a;
            --red: #f7768e; --orange: #ff9e64; --border-color: #414868;
        }
        body { background-color: var(--bg-color); color: var(--text-color); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
        .page-wrapper { display: grid; grid-template-columns: 320px 1fr; gap: 2rem; padding: 2rem; max-width: 1800px; margin: auto; }
        .sidebar { background-color: var(--surface-color); padding: 1.5rem; border-radius: 12px; height: calc(100vh - 4rem); position: sticky; top: 2rem; overflow-y: auto; }
        .sidebar h2 { font-size: 1.2rem; font-weight: bold; color: var(--primary-color); border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; margin-top: 1.5rem; }
        .main-content { min-width: 0; } /* Prevents overflow */
        .form-control, .form-select { background-color: var(--card-color); color: var(--text-color); border: 1px solid var(--border-color); }
        .form-control:focus, .form-select:focus { background-color: var(--card-color); color: var(--text-color); border-color: var(--primary-color); box-shadow: 0 0 0 0.25rem rgba(122, 162, 247, 0.25); }
        .btn-primary { background-color: var(--primary-color); border: none; }
        .btn-primary:hover { background-color: #5a7fcf; }
        #loading-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1056; flex-direction: column; justify-content: center; align-items: center; }
        #loading-overlay p { font-size: 1.2rem; margin-top: 1rem; }
        .case-card { background-color: var(--card-color); border: 1px solid var(--border-color); border-radius: 12px; margin-bottom: 1.5rem; overflow: hidden; }
        .case-header { display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 1.25rem; background-color: var(--surface-color); border-bottom: 1px solid var(--border-color); }
        .case-header h4 { margin: 0; font-size: 1.2rem; font-weight: bold; }
        .case-badge { font-size: 0.9rem; padding: 0.3rem 0.6rem; border-radius: 6px; }
        .nav-tabs .nav-link { background: none; border: none; color: var(--text-color); opacity: 0.7; }
        .nav-tabs .nav-link.active { color: var(--primary-color); opacity: 1; border-bottom: 2px solid var(--primary-color); }
        .tab-content { padding: 1.25rem; }
        .report-content, .diff-output { white-space: pre-wrap; word-wrap: break-word; font-family: "SF Mono", "Fira Code", monospace; font-size: 0.9rem; background-color: var(--bg-color); padding: 1rem; border-radius: 8px; }
        .summary-list { list-style-type: none; padding-left: 0; }
        .summary-list li { padding: 0.5rem; border-left: 3px solid var(--border-color); margin-bottom: 0.5rem; background-color: rgba(0,0,0,0.1); }
        .summary-list .major { border-color: var(--red); }
        .summary-list .minor { border-color: var(--orange); }
        .summary-list .clarification { border-color: var(--green); }
        .diff-output del { background-color: rgba(247, 118, 142, 0.2); color: #f7768e; text-decoration: none; }
        .diff-output ins { background-color: rgba(158, 206, 106, 0.2); color: #9ece6a; text-decoration: none; }
        .sidebar-nav-list { list-style: none; padding: 0; }
        .sidebar-nav-list li a { display: block; padding: 0.5rem 0.75rem; border-radius: 6px; text-decoration: none; color: var(--text-color); transition: background-color 0.2s; }
        .sidebar-nav-list li a:hover { background-color: var(--card-color); }
        .sidebar-nav-list li a.has-major { border-left: 3px solid var(--red); }
        .stat-card { background-color: var(--card-color); padding: 0.75rem; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 1.75rem; font-weight: bold; color: var(--primary-color); }
        .stat-label { font-size: 0.8rem; opacity: 0.8; }
        #top-summary-findings { max-height: 250px; overflow-y: auto; background: var(--surface-color); padding: 1rem; border-radius: 8px; }
    </style>
</head>
<body>
    <div id="loading-overlay">
        <dotlottie-player src="https://lottie.host/817661a8-2608-4435-89a5-daa620a64c36/WtsFI5zdEK.lottie" background="transparent" speed="1" style="width: 250px; height: 250px;" loop autoplay></dotlottie-player>
        <p>Analyzing reports... this may take a moment.</p>
    </div>

    <div class="page-wrapper">
        <aside class="sidebar">
            <h1 class="h4 mb-4">Radiology Review Hub</h1>
            
            <h2>Dashboard</h2>
            <div id="dashboard-stats" class="row gx-2 gy-2 mt-2"></div>

            <h2>Controls</h2>
            <div class="mt-2">
                <label for="sort-select" class="form-label">Sort Cases By</label>
                <select id="sort-select" class="form-select">
                    <option value="case_num">Case Number</option>
                    <option value="score">Discrepancy Score</option>
                    <option value="percentage_change">Percent Change</option>
                </select>
                <label for="filter-select" class="form-label mt-3">Filter Cases</label>
                <select id="filter-select" class="form-select">
                    <option value="all">Show All</option>
                    <option value="major">With Major Findings</option>
                    <option value="minor">With Minor Findings</option>
                    <option value="errors">With Errors</option>
                </select>
            </div>
            
            <h2>Navigation</h2>
            <nav id="case-nav-sidebar" class="mt-2">
                <ul id="case-nav-list" class="sidebar-nav-list"></ul>
            </nav>
        </aside>

        <main class="main-content">
            <form method="POST" id="reportForm">
                <div class="form-group mb-3">
                    <label for="report_text" class="form-label">Paste reports block here (each must start with "Case [number]"):</label>
                    <textarea id="report_text" name="report_text" class="form-control" rows="10">{{ request.form.get('report_text', '') }}</textarea>
                </div>
                <div class="form-group mb-3">
                    <label for="custom_prompt" class="form-label">System Prompt (Advanced)</label>
                    <textarea id="custom_prompt" name="custom_prompt" class="form-control" rows="5">{{ custom_prompt }}</textarea>
                </div>
                <button type="submit" class="btn btn-primary">Analyze & Review</button>
            </form>
            
            <div id="results-area" class="mt-4" style="display: none;">
                <hr class="my-4">
                <div id="top-summary-findings"></div>
                <div id="case-container" class="mt-4"></div>
            </div>
        </main>
    </div>

<script>
    let masterCaseData = {{ case_data | tojson }};
    let displayCaseData = [];

    function updateView() {
        applyFilters();
        applySorting();
        renderDashboard();
        renderSidebarNav();
        renderCases();
        renderTopFindings();
    }
    
    function applyFilters() {
        const filterValue = document.getElementById('filter-select').value;
        if (filterValue === 'all') {
            displayCaseData = [...masterCaseData];
        } else if (filterValue === 'major') {
            displayCaseData = masterCaseData.filter(c => c.summary && c.summary.major_findings && c.summary.major_findings.length > 0);
        } else if (filterValue === 'minor') {
            displayCaseData = masterCaseData.filter(c => c.summary && c.summary.minor_findings && c.summary.minor_findings.length > 0);
        } else if (filterValue === 'errors') {
            displayCaseData = masterCaseData.filter(c => !!c.summary_error);
        }
    }

    function applySorting() {
        const sortValue = document.getElementById('sort-select').value;
        displayCaseData.sort((a, b) => {
            if (sortValue === 'case_num') return parseInt(a.case_num.replace(/[^0-9]/g,'') || 0) - parseInt(b.case_num.replace(/[^0-9]/g,'') || 0);
            if (sortValue === 'percentage_change') return (b.percentage_change || 0) - (a.percentage_change || 0);
            if (sortValue === 'score') return (b.summary?.score || 0) - (a.summary?.score || 0);
            return 0;
        });
    }
    
    function renderDashboard() {
        const container = document.getElementById('dashboard-stats');
        if (!container) return;
        
        const totalCases = masterCaseData.length;
        const majorCount = masterCaseData.reduce((acc, c) => acc + (c.summary?.major_findings?.length || 0), 0);
        const minorCount = masterCaseData.reduce((acc, c) => acc + (c.summary?.minor_findings?.length || 0), 0);
        const errorCount = masterCaseData.filter(c => !!c.summary_error).length;

        container.innerHTML = `
            <div class="col-6"><div class="stat-card"><div class="stat-value">${totalCases}</div><div class="stat-label">Total Cases</div></div></div>
            <div class="col-6"><div class="stat-card"><div class="stat-value text-danger">${majorCount}</div><div class="stat-label">Major Findings</div></div></div>
            <div class="col-6"><div class="stat-card"><div class="stat-value text-warning">${minorCount}</div><div class="stat-label">Minor Findings</div></div></div>
            <div class="col-6"><div class="stat-card"><div class="stat-value">${errorCount}</div><div class="stat-label">Errors</div></div></div>
        `;
    }
    
    function renderSidebarNav() {
        const navList = document.getElementById('case-nav-list');
        if (!navList) return;
        navList.innerHTML = displayCaseData.length > 0 ? 
            displayCaseData.map(c => {
                const hasMajor = c.summary?.major_findings?.length > 0;
                const score = c.summary?.score ?? 'N/A';
                return `<li><a href="#case-${c.case_num}" class="${hasMajor ? 'has-major' : ''}">Case ${c.case_num} (Score: ${score})</a></li>`;
            }).join('') : '<li>No cases match filter.</li>';
    }

    function renderCases() {
        const container = document.getElementById('case-container');
        if (!container) return;
        
        if (displayCaseData.length === 0) {
            container.innerHTML = '<div class="alert alert-info">No cases to display. Adjust filters or submit new reports.</div>';
            return;
        }

        container.innerHTML = displayCaseData.map(c => {
            const score = c.summary?.score;
            let badgeClass = 'bg-secondary';
            if (score > 0) badgeClass = 'bg-warning text-dark';
            if (c.summary?.major_findings?.length > 0) badgeClass = 'bg-danger';
            if (c.summary_error) badgeClass = 'bg-danger';

            const summaryHtml = c.summary ? `
                <ul class="summary-list">
                    ${c.summary.major_findings?.map(f => `<li class="major"><strong>Major:</strong> ${f}</li>`).join('') || ''}
                    ${c.summary.minor_findings?.map(f => `<li class="minor"><strong>Minor:</strong> ${f}</li>`).join('') || ''}
                    ${c.summary.clarifications?.map(f => `<li class="clarification"><strong>Clarification:</strong> ${f}</li>`).join('') || ''}
                </ul>
            ` : `<div class="alert alert-danger"><strong>Error:</strong> ${c.summary_error}</div>`;

            return `
            <div class="case-card" id="case-${c.case_num}">
                <div class="case-header">
                    <h4>Case ${c.case_num}</h4>
                    <div>
                        <span class="case-badge ${badgeClass}">Score: ${score ?? 'N/A'}</span>
                        <span class="case-badge bg-info text-dark">${c.percentage_change}% Change</span>
                    </div>
                </div>
                <div class="p-2">
                    <ul class="nav nav-tabs" role="tablist">
                        <li class="nav-item" role="presentation"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#summary-${c.case_num}" type="button">Summary</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#diff-${c.case_num}" type="button">Diff View</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#resident-${c.case_num}" type="button">Resident</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#attending-${c.case_num}" type="button">Attending</button></li>
                    </ul>
                    <div class="tab-content">
                        <div class="tab-pane fade show active" id="summary-${c.case_num}">${summaryHtml}</div>
                        <div class="tab-pane fade" id="diff-${c.case_num}"><div class="diff-output">${c.diff}</div></div>
                        <div class="tab-pane fade" id="resident-${c.case_num}"><pre class="report-content">${c.resident_report}</pre></div>
                        <div class="tab-pane fade" id="attending-${c.case_num}"><pre class="report-content">${c.attending_report}</pre></div>
                    </div>
                </div>
            </div>`;
        }).join('');
    }

    function renderTopFindings() {
        const container = document.getElementById('top-summary-findings');
        if (!container) return;
        const majorFindings = masterCaseData.flatMap(c => 
            c.summary?.major_findings?.map(f => ({ caseNum: c.case_num, finding: f })) || []
        );
        
        if (majorFindings.length === 0) {
            container.innerHTML = '<h5>No Major Findings in this Batch</h5>';
            return;
        }

        container.innerHTML = `
            <h5 class="text-danger">Key Major Findings</h5>
            <ul class="list-group list-group-flush">
                ${majorFindings.map(item => `
                    <li class="list-group-item bg-transparent border-0 px-0 py-1">
                        <a href="#case-${item.caseNum}" class="text-decoration-none"><strong>Case ${item.caseNum}:</strong> ${item.finding}</a>
                    </li>
                `).join('')}
            </ul>
        `;
    }

    document.addEventListener('DOMContentLoaded', () => {
        if (masterCaseData && masterCaseData.length > 0) {
            document.getElementById('results-area').style.display = 'block';
            updateView();
        }
        
        document.getElementById('sort-select').addEventListener('change', updateView);
        document.getElementById('filter-select').addEventListener('change', updateView);

        document.getElementById('reportForm').addEventListener('submit', () => {
            document.getElementById('loading-overlay').style.display = 'flex';
        });
    });
</script>
</body>
</html>
    """
    return render_template_string(template, case_data=case_data, custom_prompt=custom_prompt)

if __name__ == '__main__':
    # Use waitress for a more production-ready local server
    from waitress import serve
    serve(app, host='0.0.0.0', port=5001)
