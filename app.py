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
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

logger.info(f"openai sdk version: {_openai_pkg.__version__}")
logger.info(f"API key present: {bool(os.getenv('OPENAI_API_KEY'))}")

# --------------------------- OpenAI ----------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_ID = os.getenv("MODEL_ID", "gpt-5-mini")  # flip in env as needed

# ----------------------- Tool Definition -----------------------------
RADIOLOGY_SUMMARY_TOOL = [
    {
        "type": "function",
        "name": "summarize_radiology_report",
        "description": "Summarizes the differences between a resident and attending radiology report, categorizing findings and calculating a score.",
        "parameters": {
            "type": "object",
            "properties": {
                "case_number": {"type": "string"},
                "major_findings": {"type": "array", "items": {"type": "string"}},
                "minor_findings": {"type": "array", "items": {"type": "string"}},
                "clarifications": {"type": "array", "items": {"type": "string"}},
                "score": {"type": "integer"}
            },
            "required": ["case_number", "major_findings", "minor_findings", "clarifications", "score"],
            "additionalProperties": False
        },
        "strict": True
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
→ minor: ["Pulmonary nodule 8 mm (descriptor: size; threshold crossed; follow-up impact)"]
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
    excluded_lines = [
        "As the attending physician, I have personally reviewed the images, interpreted and/or supervised the study or procedure, and agree with the wording of the above report.",
        "As the Attending radiologist, I have personally reviewed the images, interpreted the study, and agree with the wording of the above report by Sterling M. Jones"
    ]
    return "\n".join([line for line in text.splitlines() if line.strip() not in excluded_lines])

def calculate_change_percentage(resident_text, attending_text):
    matcher = difflib.SequenceMatcher(None, resident_text.split(), attending_text.split())
    return round((1 - matcher.ratio()) * 100, 2)

def split_into_paragraphs(text):
    paragraphs = re.split(r'\n{2,}|\n(?=\w)', text)
    return [para.strip() for para in paragraphs if para.strip()]

def create_diff_by_section(resident_text, attending_text):
    resident_text = normalize_text(resident_text)
    attending_text = normalize_text(remove_attending_review_line(attending_text))
    resident_paragraphs = split_into_paragraphs(resident_text)
    attending_paragraphs = split_into_paragraphs(attending_text)

    diff_html = ""
    matcher = difflib.SequenceMatcher(None, resident_paragraphs, attending_paragraphs)
    for opcode, a1, a2, b1, b2 in matcher.get_opcodes():
        if opcode == 'equal':
            for paragraph in resident_paragraphs[a1:a2]:
                diff_html += f'<div class="para equal">{paragraph}</div>'
        elif opcode == 'insert':
            for paragraph in attending_paragraphs[b1:b2]:
                diff_html += f'<div class="para ins">[Inserted: {paragraph}]</div>'
        elif opcode == 'delete':
            for paragraph in resident_paragraphs[a1:a2]:
                diff_html += f'<div class="para del">[Deleted: {paragraph}]</div>'
        elif opcode == 'replace':
            res_paragraphs = resident_paragraphs[a1:a2]
            att_paragraphs = attending_paragraphs[b1:b2]
            for res_paragraph, att_paragraph in zip(res_paragraphs, att_paragraphs):
                word_matcher = difflib.SequenceMatcher(None, res_paragraph.split(), att_paragraph.split())
                seg = []
                for word_opcode, w_a1, w_a2, w_b1, w_b2 in word_matcher.get_opcodes():
                    if word_opcode == 'equal':
                        seg.append(" ".join(res_paragraph.split()[w_a1:w_a2]))
                    elif word_opcode == 'replace':
                        seg.append(
                            '<span class="word del">' +
                            " ".join(res_paragraph.split()[w_a1:w_a2]) +
                            '</span> <span class="word ins">' +
                            " ".join(att_paragraph.split()[w_b1:w_b2]) +
                            '</span>'
                        )
                    elif word_opcode == 'delete':
                        seg.append('<span class="word del">' + " ".join(res_paragraph.split()[w_a1:w_a2]) + '</span>')
                    elif word_opcode == 'insert':
                        seg.append('<span class="word ins">' + " ".join(att_paragraph.split()[w_b1:w_b2]) + '</span>')
                diff_html += f'<div class="para rep">{" ".join(seg)}</div>'
    return diff_html

# ------------------------- OpenAI call --------------------------------
def get_summary(case_text, custom_prompt, case_number):
    try:
        logger.info(f"Processing case {case_number} with model={MODEL_ID} using function calling.")
        response = client.responses.create(
            model=MODEL_ID,
            instructions=custom_prompt,
            input=(f"Case Number: {case_number}\n{case_text}"),
            tools=RADIOLOGY_SUMMARY_TOOL,
            tool_choice={"type": "function", "name": "summarize_radiology_report"}
        )

        parsed_json = None
        raw_arguments = ""
        for item in response.output:
            if item.type == "function_call" and item.name == "summarize_radiology_report":
                raw_arguments = item.arguments
                parsed_json = json.loads(raw_arguments)
                break

        if parsed_json:
            logger.info(f"Function call parsed for case {case_number}.")
            return parsed_json
        else:
            logger.error(f"No function call in response for case {case_number}")
            return {"case_number": case_number, "error": "AI did not return the expected tool call."}

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failure for case {case_number}: {e!r}")
        logger.debug(f"Raw arguments: {raw_arguments}")
        return {"case_number": case_number, "error": "Invalid JSON in tool arguments from AI."}
    except NotFoundError as e:
        logger.error(f"[404] NotFound: {e.message}")
        return {"case_number": case_number, "error": f"404 NotFound: {e.message}. Check model id / project key."}
    except BadRequestError as e:
        logger.error(f"[400] BadRequest: {e.message}")
        return {"case_number": case_number, "error": f"400 BadRequest: {e.message}"}
    except AuthenticationError as e:
        logger.error(f"[401] AuthError: {e.message}")
        return {"case_number": case_number, "error": "401 Unauthorized: check OPENAI_API_KEY"}
    except RateLimitError as e:
        logger.error(f"[429] RateLimit: {e.message}")
        return {"case_number": case_number, "error": "429 Rate limited"}
    except APIConnectionError as e:
        logger.error(f"[NET] APIConnectionError: {e.message}")
        return {"case_number": case_number, "error": "Network error"}
    except APIError as e:
        logger.error(f"[5xx] APIError: {e.message}")
        return {"case_number": case_number, "error": "Server error"}
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
                if 'score' not in parsed_json:
                    parsed_json['score'] = len(parsed_json.get('major_findings', [])) * 3 + len(parsed_json.get('minor_findings', []))
                logger.info(f"Summary for case {case_num}: Score {parsed_json.get('score', 'N/A')}")
            except Exception as e:
                logger.error(f"Error processing case {case_num}: {e}")
                parsed_json = {"case_number": case_num, "error": f"Unhandled: {repr(e)}"}
            structured_output.append(parsed_json)
    logger.info(f"Completed {len(structured_output)} summaries.")
    return structured_output

def extract_cases(text, custom_prompt):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    cases = re.split(r'(?m)^Case\s+(\d+)', text, flags=re.IGNORECASE)

    cases_data, parsed_cases = [], []

    for i in range(1, len(cases), 2):
        case_num = cases[i]
        case_content = cases[i + 1].strip() if i + 1 < len(cases) else ""

        regex = r'(?im)^\s*(Attending(?:\s+Report)?\s*:|Resident(?:\s+Report)?\s*:)'
        parts = re.split(regex, case_content)

        label_to_text = {}
        for j in range(1, len(parts), 2):
            label_raw = (parts[j] or "").strip().lower().replace(":", "")
            content = (parts[j+1] if j+1 < len(parts) else "").strip()
            if "attending" in label_raw:
                label_to_text["attending"] = content
            elif "resident" in label_raw:
                label_to_text["resident"] = content

        resident_report = label_to_text.get("resident", "")
        attending_report = label_to_text.get("attending", "")

        if resident_report and attending_report:
            case_text = (
                f"Resident Report: {normalize_text(resident_report)}\n"
                f"Attending Report: {normalize_text(attending_report)}"
            )
            cases_data.append((case_text, case_num))
        else:
            logger.warning(f"Case {case_num} missing resident or attending report.")

    if not cases_data:
        return parsed_cases

    ai_summaries = process_cases(cases_data, custom_prompt, max_workers=8)
    summaries_by_case_num = {str(s.get('case_number')): s for s in ai_summaries}

    for case_text, case_num in cases_data:
        try:
            resident_report = case_text.split("\nAttending Report:")[0].replace("Resident Report: ", "").strip()
            attending_report = case_text.split("\nAttending Report:")[1].strip()
            ai_summary = summaries_by_case_num.get(case_num, {})

            parsed_cases.append({
                'case_num': case_num,
                'resident_report': resident_report,
                'attending_report': attending_report,
                'percentage_change': calculate_change_percentage(resident_report, remove_attending_review_line(attending_report)),
                'diff': create_diff_by_section(resident_report, attending_report),
                'summary': ai_summary if 'error' not in ai_summary else None,
                'summary_error': ai_summary.get('error')
            })
        except IndexError:
            logger.error(f"Error parsing reports for case {case_num}.")
            continue

    return parsed_cases

# ---------------------------- Web -------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    custom_prompt = request.form.get('custom_prompt', DEFAULT_PROMPT)
    case_data = []

    if request.method == 'POST':
        text_block = request.form.get('report_text', '')
        if text_block.strip():
            case_data = extract_cases(text_block, custom_prompt)

    template = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Compare Revisions • Radiology</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <style>
    :root{
      --bg:#0f1115;--panel:#171a21;--panel-2:#1c2028;--text:#e6e6e6;--muted:#aeb4c0;
      --primary:#4da3ff;--primary-2:#2a84f2;--green:#2bd47d;--red:#ff6b6b;--chip-major:#ff375f;--chip-minor:#ffd166;--chip-clar:#6ee7ff;
      --border:#2a2f3a;
    }
    html,body{height:100%}
    body{
      background: radial-gradient(1200px 800px at 10% -30%, #141826 10%, transparent 40%) no-repeat,
                  radial-gradient(1200px 800px at 110% 130%, #121620 5%, transparent 40%) no-repeat,
                  var(--bg);
      color:var(--text);
      font-family: system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
    }
    a{color:var(--primary)} a:hover{color:var(--primary-2)}
    .panel{background:var(--panel);border:1px solid var(--border);border-radius:14px}
    .panel-2{background:var(--panel-2);border:1px solid var(--border);border-radius:12px}
    .form-control,textarea,input{background:#0f131b!important;color:var(--text)!important;border:1px solid var(--border)!important}
    .form-control:focus{box-shadow:0 0 0 .25rem rgba(77,163,255,.15)}
    .badge-score{background:#1e2a3d;color:#b7d3ff;border:1px solid #2e405e}
    .chip{border-radius:999px;padding:.2rem .55rem;font-weight:600;border:1px solid #334155}
    .chip.major{background:color-mix(in srgb,var(--chip-major) 18%,transparent);color:#ffc4cf;border-color:#5c2034}
    .chip.minor{background:color-mix(in srgb,var(--chip-minor) 18%,transparent);color:#fff2c4;border-color:#5c4a20}
    .chip.clar{background:color-mix(in srgb,var(--chip-clar) 18%,transparent);color:#c4f4ff;border-color:#20545c}
    .progress{background:#1b2330;height:8px;border-radius:10px}
    .progress-bar{background:linear-gradient(90deg,var(--primary),#7ab8ff)}
    .para{padding:.45rem .6rem;border-left:3px solid transparent;border-radius:8px;margin-bottom:.5rem;background:#0f131b}
    .para.equal{border-left-color:#2b364a}
    .para.ins{border-left-color:var(--green);background:rgba(43,212,125,.06)}
    .para.del{border-left-color:var(--red);background:rgba(255,107,107,.06);text-decoration:line-through;opacity:.85}
    .para.rep{border-left-color:#7ab8ff;background:rgba(122,184,255,.06)}
    .word.ins{color:var(--green);font-weight:600}
    .word.del{color:var(--red);text-decoration:line-through}
    .loading-bar{position:fixed;top:0;left:0;height:3px;width:0;background:linear-gradient(90deg,var(--primary),#7ab8ff);z-index:1000;transition:width .3s ease}
    .layout{display:grid;grid-template-columns: 280px 1fr; gap:14px}
    .layout.collapsed{grid-template-columns: 0px 1fr}
    .sidebar{position:sticky;top:12px;height:calc(100vh - 24px);overflow:auto;padding:10px;transition:width .25s ease, opacity .25s ease}
    .layout.collapsed .sidebar{width:0;opacity:0;pointer-events:none}
    .sidebar-toggle{position:sticky; top:12px; z-index:10}
    .toggle-btn{border-radius:10px;border:1px solid var(--border);background:#121722;color:#c6d4f2}
    .toggle-btn[aria-pressed="true"]{background:#0e1320;color:#9fb9ff}
    .sort-btn{border-radius:10px;border:1px solid var(--border);background:#121722;color:#c6d4f2}
    .sort-btn.active{outline:2px solid rgba(77,163,255,.35)}
    .sort-btn .mode{font-weight:700;color:#b7d3ff}
    .case-card{scroll-margin-top:90px}
    .tabs{border-bottom:1px solid var(--border)}
    .tabs .nav-link{color:#aeb4c0}
    .tabs .nav-link.active{color:var(--text);background:#1a202c;border-color:var(--border) var(--border) #1a202c}
    .kbd{border:1px solid #3a4252;border-bottom-color:#2e3543;background:#1a1f2b;padding:.15rem .35rem;border-radius:6px;font-size:.8rem;color:var(--muted)}
    .toaster{position:fixed;right:18px;bottom:18px;z-index:50;background:#1c2432;border:1px solid #2a3547;color:#d7e3ff;padding:.65rem .8rem;border-radius:10px;display:none}
  </style>
</head>
<body>
  <div class="loading-bar" id="loadingBar"></div>

  <!-- Top: Paste block at the top -->
  <div class="container-fluid mt-3">
    <div class="panel p-3 mb-3">
      <form method="POST" id="reportForm">
        <div class="row g-3">
          <div class="col-12">
            <label class="form-label fw-semibold">Paste your reports block</label>
            <textarea id="report_text" name="report_text" class="form-control" rows="10" placeholder="Case 1
Resident Report:
...

Attending Report:
...

Case 2
Resident Report:
...

Attending Report:
...">{{ request.form.get('report_text', '') }}</textarea>
          </div>
          <div class="col-12">
            <label class="form-label fw-semibold">Custom prompt (optional)</label>
            <textarea id="custom_prompt" name="custom_prompt" class="form-control" rows="5" placeholder="Paste your system/developer prompt here (optional).">{{ custom_prompt }}</textarea>
          </div>
          <div class="col-12 d-flex gap-2 flex-wrap">
            <button class="btn btn-primary">
              <i class="bi bi-lightning-charge"></i> Compare & Summarize
            </button>
            <button class="btn btn-outline-light" type="button" id="clearBtn">
              <i class="bi bi-eraser"></i> Clear
            </button>
            <button class="btn btn-outline-light" type="button" id="demoBtn">
              <i class="bi bi-journal-text"></i> Load Demo
            </button>
            <button class="btn btn-outline-info" type="button" id="downloadAllBtn">
              <i class="bi bi-download"></i> Download JSON
            </button>
            <!-- Sort mode visual state -->
            <button class="sort-btn ms-auto px-3 py-2" type="button" id="sortCaseBtn" data-mode="number" aria-pressed="true" title="Cycle sort (Number → Change → Score)">
              <i class="bi bi-sort-numeric-down me-1"></i>
              Sort: <span class="mode" id="sortModeLabel">Case #</span>
            </button>
            <!-- Sidebar collapsible toggle -->
            <button class="toggle-btn px-3 py-2" type="button" id="toggleSidebarBtn" aria-pressed="false" title="Show/Hide case list">
              <i class="bi bi-layout-sidebar-inset me-1"></i>
              Case List
            </button>
          </div>
        </div>
      </form>
    </div>
  </div>

  <!-- Beneath: Two-column area with collapsible left sidebar and main content filling width -->
  <div class="container-fluid">
    <div id="gridLayout" class="layout">
      <!-- Left floating/collapsible sidebar -->
      <div class="sidebar panel">
        <div class="d-flex align-items-center gap-2 mb-2">
          <i class="bi bi-list-stars text-primary"></i>
          <strong>Cases</strong>
          <span class="ms-auto badge badge-score" id="caseCountBadge">0</span>
        </div>
        <input id="searchInput" class="form-control form-control-sm mb-2" placeholder="Search text or Case # (press F)"/>
        <div class="d-flex flex-wrap gap-2 mb-2">
          <button class="btn btn-outline-secondary btn-sm" id="filterMajorsBtn"><i class="bi bi-exclamation-octagon"></i> Majors only</button>
          <button class="btn btn-outline-secondary btn-sm" id="filterErrorsBtn"><i class="bi bi-bug"></i> Errors only</button>
          <button class="btn btn-outline-secondary btn-sm" id="expandAllBtn"><i class="bi bi-arrows-expand"></i> Expand all</button>
          <button class="btn btn-outline-secondary btn-sm" id="collapseAllBtn"><i class="bi bi-arrows-collapse"></i> Collapse all</button>
        </div>
        <div id="aggregateBlock" class="panel-2 p-2 mb-2 d-none">
          <div class="d-flex align-items-center gap-2 mb-1">
            <span class="chip major">Major <span id="aggMajor">0</span></span>
            <span class="chip minor">Minor <span id="aggMinor">0</span></span>
            <span class="chip clar">Clar <span id="aggClar">0</span></span>
          </div>
          <div class="progress" title="Percent major across all items">
            <div class="progress-bar" id="aggBar" style="width:0%"></div>
          </div>
        </div>
        <div id="caseNav" class="case-nav"></div>
        <div class="mt-3 small text-secondary">
          <div><span class="kbd">J</span>/<span class="kbd">K</span> next/prev</div>
          <div><span class="kbd">1–4</span> tabs</div>
          <div><span class="kbd">F</span> focus search</div>
          <div><span class="kbd">S</span> cycle sort</div>
          <div><span class="kbd">G</span><span class="mx-1">G</span> top</div>
        </div>
      </div>

      <!-- Main content -->
      <div>
        <div id="resultsPanel" class="panel p-2">
          <div id="emptyState" class="text-center text-secondary p-5">
            <i class="bi bi-arrow-up-right-square text-primary fs-1 d-block mb-2"></i>
            <div class="mb-1">Paste reports above and click <strong>Compare &amp; Summarize</strong>.</div>
            <div class="small">Keyboard: J/K to move, 1–4 to switch tabs.</div>
          </div>
          <div id="caseContainer" class="d-none"></div>
        </div>
      </div>
    </div>
  </div>

  <div class="toaster" id="toaster"></div>

  <script>
    let caseData = {{ case_data | tojson }};
    const navEl = document.getElementById('caseNav');
    const containerEl = document.getElementById('caseContainer');
    const emptyStateEl = document.getElementById('emptyState');
    const caseCountBadge = document.getElementById('caseCountBadge');
    const toaster = document.getElementById('toaster');
    const loadingBar = document.getElementById('loadingBar');
    const aggregateBlock = document.getElementById('aggregateBlock');
    const aggMajor = document.getElementById('aggMajor');
    const aggMinor = document.getElementById('aggMinor');
    const aggClar = document.getElementById('aggClar');
    const aggBar = document.getElementById('aggBar');

    const gridLayout = document.getElementById('gridLayout');
    const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');

    const sortBtn = document.getElementById('sortCaseBtn');
    const sortModeLabel = document.getElementById('sortModeLabel');
    const sortModes = ['number','change','score'];
    const sortLabels = { number: 'Case #', change: 'Δ Change', score: 'Score' };
    const sortIcons = {
      number: 'bi-sort-numeric-down',
      change: 'bi-lightning-charge',
      score: 'bi-trophy'
    };

    // ---------- Loading bar ----------
    const startLoading = () => { loadingBar.style.width = '35%'; };
    const midLoading = () => { loadingBar.style.width = '70%'; };
    const endLoading = () => { loadingBar.style.width = '100%'; setTimeout(()=>{ loadingBar.style.width='0%'; }, 400); };

    // ---------- Toast ----------
    function toast(msg, ms=1600) {
      toaster.textContent = msg;
      toaster.style.display = 'block';
      setTimeout(()=>{ toaster.style.display = 'none'; }, ms);
    }

    // ---------- Sidebar collapse ----------
    toggleSidebarBtn.addEventListener('click', () => {
      const pressed = toggleSidebarBtn.getAttribute('aria-pressed') === 'true';
      toggleSidebarBtn.setAttribute('aria-pressed', String(!pressed));
      gridLayout.classList.toggle('collapsed', !pressed === true);
      toggleSidebarBtn.innerHTML = (pressed)
        ? '<i class="bi bi-layout-sidebar-inset me-1"></i> Case List'
        : '<i class="bi bi-layout-sidebar-inset-reverse me-1"></i> Case List';
    });

    // ---------- Sort button visual state ----------
    function applySortVisual(mode) {
      sortBtn.setAttribute('data-mode', mode);
      sortModeLabel.textContent = sortLabels[mode];
      sortBtn.classList.add('active');
      // swap icon
      sortBtn.innerHTML = '<i class="bi '+sortIcons[mode]+' me-1"></i> Sort: <span class="mode" id="sortModeLabel">'+sortLabels[mode]+'</span>';
    }

    // ---------- Navigation render ----------
    function formatChipCounts(s) {
      const majors = (s?.major_findings?.length)||0;
      const minors = (s?.minor_findings?.length)||0;
      const clar = (s?.clarifications?.length)||0;
      return `
        <span class="chip major">M ${majors}</span>
        <span class="chip minor">m ${minors}</span>
        <span class="chip clar">c ${clar}</span>
      `;
    }

    function renderNav(list) {
      navEl.innerHTML = '';
      list.forEach(c => {
        const score = (c.summary && c.summary.score) || 0;
        const a = document.createElement('a');
        a.href = '#case' + c.case_num;
        a.className = 'd-block rounded mb-1 px-2 py-1';
        a.innerHTML = `
          <div class="d-flex w-100 align-items-center">
            <div class="me-auto">
              <strong>Case ${c.case_num}</strong>
              <div class="small text-secondary">Δ ${c.percentage_change}%</div>
            </div>
            <span class="badge badge-score me-2">Score ${score}</span>
            <div class="d-none d-xl-block">${formatChipCounts(c.summary)}</div>
          </div>
        `;
        navEl.appendChild(a);
      });
      caseCountBadge.textContent = list.length;
    }

    // ---------- Case card ----------
    function caseCardHTML(c) {
      if (!c.summary) {
        return `
          <div id="case${c.case_num}" class="case-card panel-2 p-3 mb-3">
            <div class="d-flex align-items-center gap-2">
              <strong>Case ${c.case_num}</strong>
              <span class="ms-2 text-danger"><i class="bi bi-bug"></i> ${c.summary_error || 'Unable to generate summary.'}</span>
            </div>
            <div class="text-secondary small mt-1">Δ ${c.percentage_change}% change</div>
          </div>
        `;
      }

      const s = c.summary;
      const majors = s.major_findings || [];
      const minors = s.minor_findings || [];
      const clar = s.clarifications || [];
      const total = majors.length + minors.length + clar.length;
      const pctMajor = total ? Math.round((majors.length/total)*100) : 0;

      const majorsList = majors.length ? `<ul class="mb-0">${majors.map(x=>`<li>${x}</li>`).join('')}</ul>` : '<div class="text-secondary small">None</div>';
      const minorsList = minors.length ? `<ul class="mb-0">${minors.map(x=>`<li>${x}</li>`).join('')}</ul>` : '<div class="text-secondary small">None</div>';
      const clarList = clar.length ? `<ul class="mb-0">${clar.map(x=>`<li>${x}</li>`).join('')}</ul>` : '<div class="text-secondary small">None</div>';

      return `
        <div id="case${c.case_num}" class="case-card panel-2 p-3 mb-3">
          <div class="d-flex align-items-center gap-2 flex-wrap">
            <strong class="me-2">Case ${c.case_num}</strong>
            <span class="badge badge-score">Score ${s.score ?? 0}</span>
            <span class="ms-2 text-secondary small">Δ ${c.percentage_change}%</span>
            <span class="ms-auto d-flex gap-2">
              <button class="btn btn-outline-light btn-sm" onclick='copyJSON(${JSON.stringify(JSON.stringify(s))})'><i class="bi bi-clipboard"></i> Copy JSON</button>
              <button class="btn btn-outline-light btn-sm" data-toggle="collapse" data-target="#body${c.case_num}" onclick="toggleCollapse('${c.case_num}')">
                <i class="bi bi-arrows-collapse"></i> Toggle
              </button>
            </span>
          </div>

          <div class="d-flex gap-2 mt-2">
            <span class="chip major">Major ${majors.length}</span>
            <span class="chip minor">Minor ${minors.length}</span>
            <span class="chip clar">Clar ${clar.length}</span>
          </div>

          <div class="progress my-2" title="${pctMajor}% of items are major">
            <div class="progress-bar" style="width: ${pctMajor}%"></div>
          </div>

          <div id="body${c.case_num}">
            <ul class="nav nav-tabs tabs mt-2" role="tablist">
              <li class="nav-item" role="presentation">
                <button class="nav-link active" id="tab-sum-${c.case_num}" data-bs-toggle="tab" data-bs-target="#tab-pane-sum-${c.case_num}" type="button" role="tab">Summary</button>
              </li>
              <li class="nav-item" role="presentation">
                <button class="nav-link" id="tab-combo-${c.case_num}" data-bs-toggle="tab" data-bs-target="#tab-pane-combo-${c.case_num}" type="button" role="tab">Combined Diff</button>
              </li>
              <li class="nav-item" role="presentation">
                <button class="nav-link" id="tab-res-${c.case_num}" data-bs-toggle="tab" data-bs-target="#tab-pane-res-${c.case_num}" type="button" role="tab">Resident</button>
              </li>
              <li class="nav-item" role="presentation">
                <button class="nav-link" id="tab-att-${c.case_num}" data-bs-toggle="tab" data-bs-target="#tab-pane-att-${c.case_num}" type="button" role="tab">Attending</button>
              </li>
            </ul>

            <div class="tab-content p-2">
              <div class="tab-pane fade show active" id="tab-pane-sum-${c.case_num}" role="tabpanel" tabindex="0">
                <div class="row g-2">
                  <div class="col-12 col-lg-4">
                    <div class="panel p-2 h-100">
                      <div class="d-flex align-items-center gap-2 mb-2"><i class="bi bi-exclamation-octagon text-danger"></i><strong>Major</strong></div>
                      ${majorsList}
                    </div>
                  </div>
                  <div class="col-12 col-lg-4">
                    <div class="panel p-2 h-100">
                      <div class="d-flex align-items-center gap-2 mb-2"><i class="bi bi-info-circle text-warning"></i><strong>Minor</strong></div>
                      ${minorsList}
                    </div>
                  </div>
                  <div class="col-12 col-lg-4">
                    <div class="panel p-2 h-100">
                      <div class="d-flex align-items-center gap-2 mb-2"><i class="bi bi-pencil-square text-info"></i><strong>Clarifications</strong></div>
                      ${clarList}
                    </div>
                  </div>
                </div>
              </div>

              <div class="tab-pane fade" id="tab-pane-combo-${c.case_num}" role="tabpanel" tabindex="0">
                <div class="panel p-2">
                  ${c.diff}
                </div>
              </div>

              <div class="tab-pane fade" id="tab-pane-res-${c.case_num}" role="tabpanel" tabindex="0">
                <div class="panel p-2"><pre class="mb-0">${escapeHTML(c.resident_report)}</pre></div>
              </div>
              <div class="tab-pane fade" id="tab-pane-att-${c.case_num}" role="tabpanel" tabindex="0">
                <div class="panel p-2"><pre class="mb-0">${escapeHTML(c.attending_report)}</pre></div>
              </div>
            </div>
          </div>
        </div>
      `;
    }

    // ---------- Rendering ----------
    function renderAll(data) {
      if (!data || data.length === 0) {
        containerEl.classList.add('d-none');
        emptyStateEl.classList.remove('d-none');
        renderNav([]);
        aggregateBlock.classList.add('d-none');
        return;
      }
      midLoading();
      emptyStateEl.classList.add('d-none');
      containerEl.classList.remove('d-none');

      // Aggregate
      let M=0, m=0, c=0;
      data.forEach(d => {
        if (!d.summary) return;
        M += (d.summary.major_findings||[]).length;
        m += (d.summary.minor_findings||[]).length;
        c += (d.summary.clarifications||[]).length;
      });
      const total = M+m+c;
      aggMajor.textContent = M;
      aggMinor.textContent = m;
      aggClar.textContent = c;
      aggBar.style.width = total ? Math.min(100, Math.round((M/Math.max(1,total))*100)) + '%' : '0%';
      aggregateBlock.classList.remove('d-none');

      containerEl.innerHTML = data.map(caseCardHTML).join('');
      renderNav(data);
      endLoading();
    }

    // ---------- Sorting / Filtering / Search ----------
    function sortBy(mode) {
      if (mode === 'number') {
        caseData.sort((a,b)=> parseInt(a.case_num)-parseInt(b.case_num));
      } else if (mode === 'change') {
        caseData.sort((a,b)=> (b.percentage_change||0)-(a.percentage_change||0));
      } else if (mode === 'score') {
        caseData.sort((a,b)=> ((b.summary?.score)||0)-((a.summary?.score)||0));
      }
    }
    function filterMajorsOnly(list) {
      return list.filter(x => (x.summary?.major_findings?.length||0) > 0);
    }
    function filterErrorsOnly(list) {
      return list.filter(x => !x.summary);
    }
    function searchFilter(list, q) {
      if (!q) return list;
      q = q.toLowerCase();
      return list.filter(c => {
        if (String(c.case_num).includes(q)) return true;
        if (c.resident_report?.toLowerCase().includes(q)) return true;
        if (c.attending_report?.toLowerCase().includes(q)) return true;
        const s = c.summary;
        if (s) { return JSON.stringify(s).toLowerCase().includes(q); }
        return false;
      });
    }

    // ---------- Event wiring ----------
    document.addEventListener('DOMContentLoaded', () => {
      if (caseData && caseData.length) {
        renderAll(caseData);
      }
      // initialize sort visual
      applySortVisual('number');
    });

    document.getElementById('reportForm').addEventListener('submit', () => {
      startLoading();
    });

    document.getElementById('clearBtn').addEventListener('click', () => {
      document.getElementById('report_text').value = '';
      document.getElementById('custom_prompt').value = '';
      caseData = [];
      renderAll(caseData);
    });

    document.getElementById('demoBtn').addEventListener('click', () => {
      const demo = `Case 1
Resident Report:
No pneumothorax. Lungs clear.

Attending Report:
Small right apical pneumothorax. Mild bibasilar atelectasis.

Case 2
Resident Report:
Possible segmental PE in RLL.

Attending Report:
No pulmonary embolism.`;
      document.getElementById('report_text').value = demo;
      toast('Demo loaded. Click Compare & Summarize.');
    });

    const searchInput = document.getElementById('searchInput');
    searchInput.addEventListener('input', () => {
      let list = [...caseData];
      list = searchFilter(list, searchInput.value);
      renderAll(list);
    });

    sortBtn.addEventListener('click', (e) => {
      const current = sortBtn.getAttribute('data-mode');
      const idx = sortModes.indexOf(current);
      const next = sortModes[(idx+1)%sortModes.length];
      sortBy(next);
      applySortVisual(next);
      renderAll(caseData);
    });

    document.getElementById('filterMajorsBtn').addEventListener('click', () => {
      const filtered = filterMajorsOnly(caseData);
      renderAll(filtered);
      toast('Showing cases with MAJOR findings');
    });

    document.getElementById('filterErrorsBtn').addEventListener('click', () => {
      const filtered = filterErrorsOnly(caseData);
      renderAll(filtered);
      toast('Showing error cases only');
    });

    document.getElementById('expandAllBtn').addEventListener('click', () => {
      document.querySelectorAll('[id^="body"]').forEach(el => el.style.display = '');
    });
    document.getElementById('collapseAllBtn').addEventListener('click', () => {
      document.querySelectorAll('[id^="body"]').forEach(el => el.style.display = 'none');
    });

    document.getElementById('downloadAllBtn').addEventListener('click', () => {
      const summaries = caseData.map(c => ({ case_number: c.case_num, summary: c.summary, error: c.summary_error || null }));
      const blob = new Blob([JSON.stringify(summaries, null, 2)], {type: 'application/json'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'summaries.json'; a.click();
      URL.revokeObjectURL(url);
    });

    function toggleCollapse(id) {
      const el = document.getElementById('body'+id);
      if (!el) return;
      el.style.display = (el.style.display === 'none') ? '' : 'none';
    }
    window.toggleCollapse = toggleCollapse;

    // ---------- Utils ----------
    function copyJSON(text) {
      navigator.clipboard.writeText(JSON.parse(text))
        .then(()=> toast('JSON copied'))
        .catch(()=> toast('Copy failed', 2000));
    }
    window.copyJSON = copyJSON;

    function escapeHTML(str) {
      if (!str) return '';
      return str.replace(/[&<>"']/g, (m) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#39;'}[m]));
    }
    window.escapeHTML = escapeHTML;

    // ---------- Keyboard shortcuts ----------
    let currentIndex = 0;
    function focusCase(i) {
      const list = document.querySelectorAll('.case-card');
      if (!list.length) return;
      currentIndex = Math.max(0, Math.min(i, list.length-1));
      list[currentIndex].scrollIntoView({behavior:'smooth', block:'start'});
      list[currentIndex].classList.add('ring');
      setTimeout(()=> list[currentIndex].classList.remove('ring'), 600);
    }
    function switchTab(n) {
      const list = document.querySelectorAll('.case-card');
      if (!list.length) return;
      const id = list[currentIndex].id.replace('case','');
      const tabIds = ['sum','combo','res','att'];
      const which = tabIds[n-1] || 'sum';
      const btn = document.getElementById(`tab-${which}-${id}`);
      if (btn) btn.click();
    }

    let gPressedOnce = false;
    document.addEventListener('keydown', (e) => {
      if (['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) {
        if (e.key.toLowerCase()==='escape') document.activeElement.blur();
        return;
      }
      if (e.key==='j' || e.key==='J') {
        e.preventDefault(); focusCase(currentIndex+1);
      } else if (e.key==='k' || e.key==='K') {
        e.preventDefault(); focusCase(currentIndex-1);
      } else if (['1','2','3','4'].includes(e.key)) {
        e.preventDefault(); switchTab(parseInt(e.key,10));
      } else if (e.key.toLowerCase()==='f') {
        e.preventDefault(); document.getElementById('searchInput').focus();
      } else if (e.key.toLowerCase()==='s') {
        e.preventDefault(); sortBtn.click();
      } else if (e.key.toLowerCase()==='g') {
        if (gPressedOnce) {
          window.scrollTo({top:0, behavior:'smooth'}); gPressedOnce=false;
        } else {
          gPressedOnce=true; setTimeout(()=> gPressedOnce=false, 450);
        }
      }
    });
  </script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """
    return render_template_string(template, case_data=case_data, custom_prompt=custom_prompt)

# Local debugging only
if __name__ == '__main__':
    app.run(debug=True, port=5001)
