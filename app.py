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
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

logger.info(f"openai sdk version: {_openai_pkg.__version__}")
logger.info(f"API key present: {bool(os.getenv('OPENAI_API_KEY'))}")

# --------------------------- OpenAI ----------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_ID = os.getenv("MODEL_ID", "gpt-5-mini")  # set MODEL_ID in Render to flip models

# ----------------------- Tool Definition -----------------------------
# Define the tool that represents our desired JSON output structure.
RADIOLOGY_SUMMARY_TOOL = [
    {
        "type": "function",
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
An attending change can be an ADDITION (attending adds something) or a CORRECTION (attending negates/downgrades):

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
    # Keep paragraphs but also split on single newlines that start a word (original heuristics)
    paragraphs = re.split(r'\n{2,}|\n(?=\w)', text)
    return [para.strip() for para in paragraphs if para.strip()]

# --- NEW: produce diff with classes, so we can toggle equal content ---
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
                diff_html += f'<div class="eq">{paragraph}</div><div class="sp"/></div>'
        elif opcode == 'insert':
            for paragraph in attending_paragraphs[b1:b2]:
                diff_html += f'<div class="ins">{paragraph}</div><div class="sp"/></div>'
        elif opcode == 'delete':
            for paragraph in resident_paragraphs[a1:a2]:
                diff_html += f'<div class="del">{paragraph}</div><div class="sp"/></div>'
        elif opcode == 'replace':
            res_paragraphs = resident_paragraphs[a1:a2]
            att_paragraphs = attending_paragraphs[b1:b2]
            # Pairwise word-level diff within replaced segments
            for res_paragraph, att_paragraph in zip(res_paragraphs, att_paragraphs):
                word_matcher = difflib.SequenceMatcher(None, res_paragraph.split(), att_paragraph.split())
                line = []
                for word_opcode, w_a1, w_a2, w_b1, w_b2 in word_matcher.get_opcodes():
                    if word_opcode == 'equal':
                        line.append(" ".join(res_paragraph.split()[w_a1:w_a2]))
                    elif word_opcode == 'replace':
                        old = " ".join(res_paragraph.split()[w_a1:w_a2])
                        new = " ".join(att_paragraph.split()[w_b1:w_b2])
                        line.append(f'<span class="tok-del">{old}</span> <span class="tok-ins">{new}</span>')
                    elif word_opcode == 'delete':
                        old = " ".join(res_paragraph.split()[w_a1:w_a2])
                        line.append(f'<span class="tok-del">{old}</span>')
                    elif word_opcode == 'insert':
                        new = " ".join(att_paragraph.split()[w_b1:w_b2])
                        line.append(f'<span class="tok-ins">{new}</span>')
                diff_html += f'<div class="rep">{" ".join(line)}</div><div class="sp"/></div>'
    return diff_html

# ------------------------- OpenAI call --------------------------------
def get_summary(case_text, custom_prompt, case_number):
    try:
        logger.info(f"Processing case {case_number} with model={MODEL_ID} using function calling.")

        response = client.responses.create(
            model=MODEL_ID,
            instructions=custom_prompt,
            input=(
                f"Case Number: {case_number}\n{case_text}"
            ),
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
            logger.info(f"Received and parsed function call for case {case_number}: OK")
            return parsed_json
        else:
            logger.error(f"Function call not found in response for case {case_number}")
            return {"case_number": case_number, "error": "AI did not return the expected tool call."}

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failure for case {case_number}: {e!r}")
        logger.debug(f"Raw arguments from model that failed to parse: {raw_arguments}")
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
                logger.info(f"Processed summary for case {case_num}: Score {parsed_json.get('score', 'N/A')}")
            except Exception as e:
                logger.error(f"Error processing case {case_num}: {e}")
                parsed_json = {"case_number": case_num, "error": f"Unhandled: {repr(e)}"}
            structured_output.append(parsed_json)
    logger.info(f"Completed processing {len(structured_output)} summaries.")
    return structured_output

def extract_cases(text, custom_prompt):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    logger.debug("Normalized line endings.")

    cases = re.split(r'(?m)^Case\s+(\d+)', text, flags=re.IGNORECASE)
    logger.debug(f"SPLIT RESULT FOR CASES: {cases}")

    logger.info(f"Total elements after split: {len(cases)}")
    for idx, element in enumerate(cases):
        logger.debug(f"Element {idx}: {element[:100]}{'...' if len(element) > 100 else ''}")

    cases_data, parsed_cases = [], []

    for i in range(1, len(cases), 2):
        case_num = cases[i]
        case_content = cases[i + 1].strip() if i + 1 < len(cases) else ""
        logger.debug(f"Processing Case {case_num}:")
        logger.debug(f"Case Content: {case_content[:200]}{'...' if len(case_content) > 200 else ''}")

        regex = r'(?im)^\s*(Attending(?:\s+Report)?\s*:|Resident(?:\s+Report)?\s*:)' \
                r'|(?im)^\s*(Attending)\b|(?im)^\s*(Resident)\b'
        parts = re.split(regex, case_content)
        parts = [p for p in parts if p is not None]  # drop Nones from unmatched groups
        logger.debug(f"Reports split: {parts}")

        label_to_text = {}
        j = 0
        while j < len(parts):
            label_raw = (parts[j] or "").strip().lower().replace(":", "")
            # If label matched in alternate groups, normalize
            if label_raw in ("attending", "attending report"):
                content = (parts[j+1] if j+1 < len(parts) else "").strip()
                label_to_text["attending"] = content
                j += 2
            elif label_raw in ("resident", "resident report"):
                content = (parts[j+1] if j+1 < len(parts) else "").strip()
                label_to_text["resident"] = content
                j += 2
            else:
                j += 1

        resident_report = label_to_text.get("resident", "")
        attending_report = label_to_text.get("attending", "")

        if resident_report and attending_report:
            case_text = (
                f"Resident Report: {normalize_text(resident_report)}\n"
                f"Attending Report: {normalize_text(attending_report)}"
            )
            cases_data.append((case_text, case_num))
            logger.info(f"Prepared case {case_num} (normalized order: Resident → Attending).")
        else:
            logger.warning(f"Case {case_num} does not contain both Attending and Resident Reports.")

    if not cases_data:
        logger.warning("No valid cases found in the submitted text.")
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
            logger.info(f"Assigned summary to case {case_num}.")
        except IndexError:
            logger.error(f"Error parsing reports for case {case_num}.")
            continue

    logger.info(f"Total parsed_cases to return: {len(parsed_cases)}")
    logger.debug(f"Parsed_cases: {json.dumps(parsed_cases, indent=2)}")
    return parsed_cases

# ---------------------------- Web -------------------------------------
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
            logger.debug(f"Extracted case_data: {json.dumps(case_data, indent=2)}")

    logger.info(f"Passing {len(case_data)} cases to the template.")
    logger.debug(f"case_data being passed: {json.dumps(case_data, indent=2)}")

    template = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Radiology Report Diff & Summarizer</title>
  <link rel="preconnect" href="https://cdn.jsdelivr.net"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
  <style>
    :root{
      --bg: #0b0f14;       /* deep slate */
      --panel:#121824;     /* card bg */
      --muted:#8aa0b5;     /* secondary text */
      --text:#e6f0ff;      /* primary text */
      --accent:#5ddcff;    /* cyan */
      --accent-2:#a66bff;  /* purple */
      --danger:#ff6b6b;    /* red */
      --success:#14d39a;   /* green */
      --warning:#ffcf5c;   /* yellow */
      --chip:#1a2332;      /* chip bg */
    }
    html,body{background:var(--bg); color:var(--text);} 
    .app-header{position:sticky; top:0; z-index:1040; backdrop-filter: blur(8px);}    
    .app-header .bar{background:linear-gradient(90deg,var(--panel),#0e1420); border-bottom:1px solid #172033;}
    .brand{font-weight:700; letter-spacing:.5px}
    .brand .dot{color:var(--accent)}
    .panel{background:var(--panel); border:1px solid #1b2a40; border-radius:16px; box-shadow:0 10px 30px rgba(0,0,0,.35)}
    .chip{background:var(--chip); border:1px solid #223149; color:var(--text); border-radius:999px; padding:.25rem .6rem; display:inline-flex; align-items:center; gap:.35rem; font-size:.8rem}
    .chip .dot{width:.5rem; height:.5rem; border-radius:999px; display:inline-block}
    .chip.major .dot{background:var(--danger)}
    .chip.minor .dot{background:var(--warning)}
    .chip.clar .dot{background:var(--accent)}
    .chip.score .dot{background:var(--accent-2)}
    .toolbar{gap:.5rem}
    .btn-ghost{background:transparent; color:var(--text); border:1px solid #223149}
    .btn-ghost:hover{border-color:#2e3f5f; background:#0f1726}
    .btn-accent{background:linear-gradient(90deg,var(--accent-2), var(--accent)); border:0; color:#05101c}
    .sidebar{position:sticky; top:84px}
    .search{background:#0e1523; border:1px solid #21324d; color:var(--text)}

    /* Diff classes */
    .diff-output{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size:.95rem; line-height:1.5; padding:1rem; border-radius:12px; background:#0b1220; border:1px solid #1a2742}
    .diff-controls{display:flex; gap:.5rem; flex-wrap:wrap}
    .eq{opacity:.7; color:var(--muted)}
    .ins{background:rgba(20,211,154,.12); border-left:3px solid var(--success); padding:.35rem .6rem; border-radius:6px; margin:.2rem 0}
    .del{background:rgba(255,107,107,.12); border-left:3px solid var(--danger); padding:.35rem .6rem; border-radius:6px; margin:.2rem 0; text-decoration: line-through; opacity:.9}
    .rep{background:linear-gradient(90deg, rgba(166,107,255,.08), rgba(93,220,255,.06)); border-left:3px solid var(--accent-2); padding:.35rem .6rem; border-radius:6px; margin:.2rem 0}
    .tok-del{background:rgba(255,107,107,.18); border-radius:4px; padding:0 .2rem; text-decoration: line-through}
    .tok-ins{background:rgba(20,211,154,.18); border-radius:4px; padding:0 .2rem}
    .sp{height:.35rem}

    /* Case card */
    .case-card{background:var(--panel); border:1px solid #1b2a40; border-radius:16px; padding:1rem; margin-bottom:1rem}
    .case-head{display:flex; align-items:center; justify-content:space-between; gap:1rem}
    .case-meta{display:flex; align-items:center; gap:.5rem; flex-wrap:wrap}
    .badge-soft{background:#0e1523; border:1px solid #21324d; color:var(--text)}

    .progress{height:.4rem; background:#0d1422}
    .progress-bar{background:linear-gradient(90deg,var(--accent-2), var(--accent))}

    .nav-pills .nav-link{color:var(--text); border:1px solid #223149}
    .nav-pills .nav-link.active{background:linear-gradient(90deg,var(--accent-2),var(--accent)); color:#05101c; border:0}

    a, a:hover{color:var(--accent)}
    .link-muted{color:var(--muted)}

    .kbd{border:1px solid #2b3d5e; background:#0e1523; padding:.15rem .4rem; border-radius:6px; font-size:.8rem}

    .toast-container{z-index: 2000}
  </style>
</head>
<body>
  <!-- Header / Global toolbar -->
  <div class="app-header">
    <div class="bar py-2">
      <div class="container d-flex align-items-center justify-content-between">
        <div class="brand h4 m-0">Compare<span class="dot">•</span>Revisions</div>
        <div class="toolbar d-flex">
          <button class="btn btn-ghost" id="btnExport" title="Download JSON of all summaries">Export JSON</button>
          <button class="btn btn-ghost" id="btnHelp" data-bs-toggle="modal" data-bs-target="#helpModal">Hotkeys</button>
          <a class="btn btn-accent" href="#form">New Analysis</a>
        </div>
      </div>
    </div>
  </div>

  <div class="container mt-4">
    <!-- Input form -->
    <div id="form" class="panel p-3 p-md-4 mb-4">
      <form method="POST" id="reportForm">
        <div class="row g-3 align-items-end">
          <div class="col-12">
            <label for="report_text" class="form-label">Paste your reports block</label>
            <textarea id="report_text" name="report_text" rows="8" class="form-control search" placeholder="Case 123\nResident: ...\nAttending: ...\n\nCase 124\n...">{{ request.form.get('report_text', '') }}</textarea>
          </div>
          <div class="col-12">
            <label for="custom_prompt" class="form-label">Optional: Customize model instructions</label>
            <textarea id="custom_prompt" name="custom_prompt" rows="5" class="form-control search">{{ custom_prompt }}</textarea>
          </div>
          <div class="col-12 d-flex gap-2 align-items-center">
            <button type="submit" class="btn btn-accent">Compare & Summarize</button>
            <dotlottie-player id="loadingAnimation" src="https://lottie.host/817661a8-2608-4435-89a5-daa620a64c36/WtsFI5zdEK.lottie" background="transparent" speed="1" style="width: 56px; height: 56px; display:none" loop autoplay></dotlottie-player>
            <span class="link-muted">Tip: <span class="kbd">J</span>/<span class="kbd">K</span> to jump cases; <span class="kbd">1</span>/<span class="kbd">2</span>/<span class="kbd">3</span>/<span class="kbd">4</span> switch tabs</span>
          </div>
        </div>
      </form>
    </div>

    {% if case_data %}
    <div class="row g-3">
      <!-- Sidebar Filters -->
      <div class="col-lg-3">
        <div class="panel p-3 sidebar">
          <div class="d-flex align-items-center justify-content-between mb-2">
            <div class="h5 m-0">Filters</div>
            <button class="btn btn-sm btn-ghost" id="btnReset">Reset</button>
          </div>
          <input class="form-control search mb-2" id="searchInput" placeholder="Search text or case #"/>

          <div class="mb-2 d-grid gap-2">
            <button class="btn btn-ghost" data-filter="majors">Has Major</button>
            <button class="btn btn-ghost" data-filter="minors">Has Minor</button>
            <button class="btn btn-ghost" data-filter="clar">Has Clarifications</button>
            <button class="btn btn-ghost" data-filter="none">No Findings</button>
          </div>

          <div class="mb-3">
            <label class="form-label">Min Score</label>
            <input type="range" class="form-range" min="0" max="30" step="1" id="minScoreRange"/>
            <div class="d-flex justify-content-between small"><span>0</span><span id="minScoreVal">0</span></div>
          </div>

          <div class="mb-3">
            <label class="form-label">Sort By</label>
            <select id="sortSelect" class="form-select search">
              <option value="case_number">Case Number</option>
              <option value="percentage_change">% Change</option>
              <option value="summary_score">Summary Score</option>
            </select>
          </div>

          <div class="form-check form-switch mb-2">
            <input class="form-check-input" type="checkbox" role="switch" id="toggleOnlyDiff"/>
            <label class="form-check-label" for="toggleOnlyDiff">Show only differences</label>
          </div>

          <div class="form-check form-switch">
            <input class="form-check-input" type="checkbox" role="switch" id="toggleWrap" checked/>
            <label class="form-check-label" for="toggleWrap">Wrap long lines</label>
          </div>

          <hr/>
          <div class="h6">Quick Jump</div>
          <div id="caseNav" class="small"></div>
        </div>
      </div>

      <!-- Main Content -->
      <div class="col-lg-9">
        <div class="d-flex flex-wrap gap-2 mb-2">
          <span class="chip major"><span class="dot"></span><span id="countMajors">0</span> major</span>
          <span class="chip minor"><span class="dot"></span><span id="countMinors">0</span> minor</span>
          <span class="chip clar"><span class="dot"></span><span id="countClars">0</span> clarifications</span>
          <span class="chip score"><span class="dot"></span>avg score: <span id="avgScore">0</span></span>
        </div>
        <div id="caseContainer"></div>
      </div>
    </div>
    {% endif %}
  </div>

  <!-- Help Modal -->
  <div class="modal fade" id="helpModal" tabindex="-1">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content" style="background:var(--panel); color:var(--text); border:1px solid #1b2a40">
        <div class="modal-header">
          <h5 class="modal-title">Hotkeys</h5>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <p><span class="kbd">J</span>/<span class="kbd">K</span> — Next/Prev case (visible set)</p>
          <p><span class="kbd">1</span> Summary • <span class="kbd">2</span> Diff • <span class="kbd">3</span> Resident • <span class="kbd">4</span> Attending</p>
          <p><span class="kbd">/</span> — Focus search • <span class="kbd">s</span> — Sort menu</p>
          <p><span class="kbd">c</span> — Copy summary JSON of focused case</p>
        </div>
      </div>
    </div>
  </div>

  <div class="toast-container position-fixed top-0 end-0 p-3">
    <div id="toast" class="toast align-items-center text-bg-dark border-0" role="alert" aria-live="assertive" aria-atomic="true">
      <div class="d-flex">
        <div class="toast-body">Copied!</div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
      </div>
    </div>
  </div>

  <script>
    let caseData = {{ case_data | tojson }};

    const state = {
      filter: null, // majors|minors|clar|none
      minScore: Number(localStorage.getItem('minScore')||0),
      sort: localStorage.getItem('sort')||'case_number',
      onlyDiff: localStorage.getItem('onlyDiff')==='true',
      wrap: (localStorage.getItem('wrap')||'true')==='true',
      search: ''
    };

    const qs = (s, el=document)=>el.querySelector(s);
    const qsa = (s, el=document)=>[...el.querySelectorAll(s)];

    function applyPrefsUI(){
      if(!caseData || caseData.length===0) return;
      const minRange = qs('#minScoreRange');
      minRange.value = state.minScore; qs('#minScoreVal').textContent = state.minScore;
      qs('#sortSelect').value = state.sort;
      qs('#toggleOnlyDiff').checked = state.onlyDiff;
      qs('#toggleWrap').checked = state.wrap;
    }

    function persistPrefs(){
      localStorage.setItem('minScore', state.minScore);
      localStorage.setItem('sort', state.sort);
      localStorage.setItem('onlyDiff', state.onlyDiff);
      localStorage.setItem('wrap', state.wrap);
    }

    function sortCases(){
      if(state.sort==='case_number'){
        caseData.sort((a,b)=> parseInt(a.case_num) - parseInt(b.case_num));
      } else if(state.sort==='percentage_change'){
        caseData.sort((a,b)=> (b.percentage_change||0) - (a.percentage_change||0));
      } else if(state.sort==='summary_score'){
        caseData.sort((a,b)=> ((b.summary&&b.summary.score)||0) - ((a.summary&&a.summary.score)||0));
      }
    }

    function filterSet(){
      return caseData.filter(c=>{
        // score gate
        const score = (c.summary && c.summary.score) || 0;
        if(score < state.minScore) return false;
        // text / case search
        const s = state.search.trim().toLowerCase();
        if(s){
          const blob = `${c.case_num} ${c.resident_report} ${c.attending_report} ${JSON.stringify(c.summary||{})}`.toLowerCase();
          if(!blob.includes(s)) return false;
        }
        // category filter
        if(!state.filter) return true;
        const hasMaj = c.summary && c.summary.major_findings && c.summary.major_findings.length>0;
        const hasMin = c.summary && c.summary.minor_findings && c.summary.minor_findings.length>0;
        const hasClar = c.summary && c.summary.clarifications && c.summary.clarifications.length>0;
        if(state.filter==='majors') return hasMaj;
        if(state.filter==='minors') return hasMin;
        if(state.filter==='clar') return hasClar;
        if(state.filter==='none') return !hasMaj && !hasMin && !hasClar;
        return true;
      });
    }

    function summarizeTotals(data){
      let majors=0, minors=0, clars=0, scores=[];
      data.forEach(c=>{
        if(c.summary){
          majors += (c.summary.major_findings||[]).length;
          minors += (c.summary.minor_findings||[]).length;
          clars  += (c.summary.clarifications||[]).length;
          scores.push(c.summary.score||0);
        }
      });
      const avg = scores.length? (scores.reduce((a,b)=>a+b,0)/scores.length).toFixed(1) : 0;
      qs('#countMajors').textContent = majors;
      qs('#countMinors').textContent = minors;
      qs('#countClars').textContent = clars;
      qs('#avgScore').textContent = avg;
    }

    function badge(text){
      return `<span class="badge rounded-pill badge-soft">${text}</span>`
    }

    function pill(label, value, cls='chip'){
      return `<span class="${cls}"><span class="dot"></span>${label}: ${value}</span>`
    }

    function tabId(cn, name){ return `${name}${cn}` }

    function caseCard(c){
      const score = (c.summary && c.summary.score) || 0;
      const majors = (c.summary && c.summary.major_findings||[]).length;
      const minors = (c.summary && c.summary.minor_findings||[]).length;
      const clars  = (c.summary && c.summary.clarifications||[]).length;
      const pct = Math.min(100, Math.max(0, c.percentage_change||0));

      const tabs = `
      <ul class="nav nav-pills mb-2" role="tablist" data-case="${c.case_num}">
        <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#${tabId(c.case_num,'summary')}" type="button" role="tab">Summary</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#${tabId(c.case_num,'combined')}" type="button" role="tab">Diff</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#${tabId(c.case_num,'resident')}" type="button" role="tab">Resident</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#${tabId(c.case_num,'attending')}" type="button" role="tab">Attending</button></li>
        <li class="ms-auto"><button class="btn btn-sm btn-ghost" data-copy="${c.case_num}">Copy JSON</button></li>
      </ul>`;

      const diffControls = `
        <div class="diff-controls mb-2">
          <div class="form-check form-check-inline">
            <input class="form-check-input only-eq" type="checkbox" ${state.onlyDiff?'checked':''} data-case="${c.case_num}">
            <label class="form-check-label">Hide unchanged</label>
          </div>
          <div class="form-check form-check-inline">
            <input class="form-check-input wrap" type="checkbox" ${state.wrap?'checked':''} data-case="${c.case_num}">
            <label class="form-check-label">Wrap lines</label>
          </div>
        </div>`;

      const summaryBlock = c.summary ? `
        <div class="p-2">
          <div class="mb-2 d-flex flex-wrap gap-2">
            ${pill('majors', majors, 'chip major')}
            ${pill('minors', minors, 'chip minor')}
            ${pill('clar', clars, 'chip clar')}
            ${pill('score', score, 'chip score')}
          </div>
          ${ majors ? `<div class="mb-2"><strong>Major Findings</strong><ul>${c.summary.major_findings.map(f=>`<li>${f}</li>`).join('')}</ul></div>`:'' }
          ${ minors ? `<div class="mb-2"><strong>Minor Findings</strong><ul>${c.summary.minor_findings.map(f=>`<li>${f}</li>`).join('')}</ul></div>`:'' }
          ${ clars ? `<div class="mb-2"><strong>Clarifications</strong><ul>${c.summary.clarifications.map(f=>`<li>${f}</li>`).join('')}</ul></div>`:'' }
        </div>` : `<div class="text-danger">${c.summary_error || 'Unable to generate summary for this case.'}</div>`;

      return `
      <article class="case-card" id="case${c.case_num}" tabindex="0" data-case="${c.case_num}">
        <div class="case-head">
          <div class="h5 m-0">Case ${c.case_num}</div>
          <div class="case-meta">
            ${badge(pct + '% change')}
            ${badge('score ' + score)}
          </div>
        </div>
        <div class="progress my-2" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}">
          <div class="progress-bar" style="width:${pct}%"></div>
        </div>
        ${tabs}
        <div class="tab-content">
          <div class="tab-pane fade show active" id="${tabId(c.case_num,'summary')}" role="tabpanel">${summaryBlock}</div>
          <div class="tab-pane fade" id="${tabId(c.case_num,'combined')}" role="tabpanel">
            ${diffControls}
            <div class="diff-output ${state.wrap? 'text-wrap':'text-nowrap'}" data-diff="${c.case_num}">${c.diff}</div>
          </div>
          <div class="tab-pane fade" id="${tabId(c.case_num,'resident')}" role="tabpanel"><div class="diff-output ${state.wrap? 'text-wrap':'text-nowrap'}"><pre class="m-0">${escapeHtml(c.resident_report)}</pre></div></div>
          <div class="tab-pane fade" id="${tabId(c.case_num,'attending')}" role="tabpanel"><div class="diff-output ${state.wrap? 'text-wrap':'text-nowrap'}"><pre class="m-0">${escapeHtml(c.attending_report)}</pre></div></div>
        </div>
      </article>`;
    }

    function escapeHtml(str){
      if(!str) return '';
      return str
        .replaceAll('&','&amp;')
        .replaceAll('<','&lt;')
        .replaceAll('>','&gt;');
    }

    function displayNavigation(visible){
      const nav = qs('#caseNav');
      nav.innerHTML = '';
      visible.forEach(c=>{
        const score = (c.summary && c.summary.score) || 0;
        nav.innerHTML += `<div><a href="#case${c.case_num}">Case ${c.case_num}</a> · ${c.percentage_change}% · ${score}</div>`;
      })
    }

    function render(){
      if(!caseData || caseData.length===0) return;
      sortCases();
      const visible = filterSet();
      summarizeTotals(visible);
      const container = qs('#caseContainer');
      container.innerHTML = visible.map(c=>caseCard(c)).join('');
      wirePerCaseControls();
      displayNavigation(visible);
      focusFirst();
    }

    function wirePerCaseControls(){
      // Copy JSON buttons
      qsa('[data-copy]').forEach(btn=>{
        btn.addEventListener('click', ()=>{
          const cn = btn.getAttribute('data-copy');
          const obj = caseData.find(x=>String(x.case_num)===String(cn));
          const jsonStr = JSON.stringify(obj.summary||{}, null, 2);
          navigator.clipboard.writeText(jsonStr).then(()=>showToast('Copied summary JSON'));
        })
      });
      // Local toggles in Diff tab
      qsa('.only-eq').forEach(ch=>{
        ch.addEventListener('change', ()=>{
          const cn = ch.getAttribute('data-case');
          const diff = qs(`[data-diff="${cn}"]`);
          if(ch.checked) qsa('.eq', diff).forEach(el=>el.style.display='none');
          else qsa('.eq', diff).forEach(el=>el.style.display='');
        })
        // Apply initial state
        if(state.onlyDiff){
          const cn = ch.getAttribute('data-case');
          const diff = qs(`[data-diff="${cn}"]`);
          qsa('.eq', diff).forEach(el=>el.style.display='none');
        }
      });
      qsa('.wrap').forEach(ch=>{
        ch.addEventListener('change', ()=>{
          const cn = ch.getAttribute('data-case');
          const diff = qs(`[data-diff="${cn}"]`);
          diff.classList.toggle('text-nowrap');
          diff.classList.toggle('text-wrap');
        })
      });
    }

    // ---------------- Toolbar + Filters wiring ----------------
    document.addEventListener('DOMContentLoaded', ()=>{
      if(!caseData || caseData.length===0) return;

      applyPrefsUI();
      render();

      // Global controls
      qs('#minScoreRange').addEventListener('input', e=>{
        state.minScore = Number(e.target.value); qs('#minScoreVal').textContent = state.minScore; persistPrefs(); render();
      });
      qs('#sortSelect').addEventListener('change', e=>{ state.sort = e.target.value; persistPrefs(); render(); });
      qs('#toggleOnlyDiff').addEventListener('change', e=>{ state.onlyDiff = e.target.checked; persistPrefs(); render(); });
      qs('#toggleWrap').addEventListener('change', e=>{ state.wrap = e.target.checked; persistPrefs(); render(); });
      qs('#btnReset').addEventListener('click', ()=>{ state.filter=null; state.minScore=0; state.search=''; qs('#searchInput').value=''; applyPrefsUI(); persistPrefs(); render(); });
      qs('#searchInput').addEventListener('input', e=>{ state.search = e.target.value; render(); });
      qsa('[data-filter]').forEach(btn=> btn.addEventListener('click', ()=>{ state.filter = btn.getAttribute('data-filter'); render(); }));

      // Export JSON of summaries
      qs('#btnExport').addEventListener('click', ()=>{
        const payload = caseData.map(c=>({case_number:c.case_num, summary:c.summary, error:c.summary_error}));
        const blob = new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'summaries.json'; a.click();
        URL.revokeObjectURL(url);
      });

      // Loading animation
      qs('#reportForm').addEventListener('submit', ()=>{ qs('#loadingAnimation').style.display='block'; });

      // Hotkeys
      document.addEventListener('keydown', (e)=>{
        if(e.key === '/') { e.preventDefault(); qs('#searchInput').focus(); return; }
        if(e.key === 's') { e.preventDefault(); qs('#sortSelect').focus(); return; }
        const focused = document.activeElement.closest('.case-card') || qsa('.case-card')[0];
        if(!focused) return;
        const idx = qsa('.case-card').indexOf? qsa('.case-card').indexOf(focused) : qsa('.case-card').findIndex(n=>n===focused);
        if(e.key.toLowerCase()==='j') moveFocus(idx+1);
        if(e.key.toLowerCase()==='k') moveFocus(idx-1);
        if(['1','2','3','4'].includes(e.key)) switchTab(focused, e.key);
        if(e.key==='c') copyFocusedJSON();
      });
    });

    function focusFirst(){
      const first = qs('.case-card');
      if(first){ first.focus(); }
    }
    function moveFocus(nextIndex){
      const cards = qsa('.case-card');
      if(cards.length===0) return;
      const i = Math.max(0, Math.min(cards.length-1, nextIndex));
      cards[i].scrollIntoView({behavior:'smooth', block:'center'});
      setTimeout(()=>cards[i].focus(), 200);
    }
    function switchTab(card, key){
      const map = { '1':'summary', '2':'combined', '3':'resident', '4':'attending' };
      const tgt = qs(`[data-bs-target="#${map[key]}${card.getAttribute('data-case')}"]`, card);
      if(tgt){ tgt.click(); }
    }
    function copyFocusedJSON(){
      const card = document.activeElement.closest('.case-card');
      if(!card) return;
      const cn = card.getAttribute('data-case');
      const obj = caseData.find(x=>String(x.case_num)===String(cn));
      const jsonStr = JSON.stringify(obj.summary||{}, null, 2);
      navigator.clipboard.writeText(jsonStr).then(()=>showToast('Copied summary JSON'));
    }

    function showToast(msg){
      qs('#toast .toast-body').textContent = msg;
      const t = new bootstrap.Toast(qs('#toast'));
      t.show();
    }

    // Helper to expose to template literal
    window.escapeHtml = escapeHtml;
  </script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """
    return render_template_string(template, case_data=case_data, custom_prompt=custom_prompt)

# Local debugging only (Render uses gunicorn app:app)
if __name__ == '__main__':
    app.run(debug=True, port=5001)
