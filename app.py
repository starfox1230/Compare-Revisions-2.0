
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

    # concurrent summaries
    structured_output = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(get_summary, case_text, custom_prompt, case_num): (case_text, case_num)
            for case_text, case_num in cases_data
        }
        for fut in as_completed(futures):
            case_text, case_num = futures[fut]
            try:
                summary = fut.result() or {}
            except Exception as e:
                summary = {"case_number": case_num, "error": f"Unhandled: {repr(e)}"}

            resident_report = case_text.split("\nAttending Report:")[0].replace("Resident Report: ", "").strip()
            attending_report = case_text.split("\nAttending Report:")[1].strip()

            # % change for quick sort badge
            pct = calculate_change_percentage(resident_report, remove_attending_review_line(attending_report))

            structured_output.append({
                "case_num": case_num,
                "resident_report": resident_report,
                "attending_report": attending_report,
                "percentage_change": pct,
                "summary": summary if "error" not in summary else None,
                "summary_error": summary.get("error")
            })

    # keep stable order by case number
    structured_output.sort(key=lambda x: int(x["case_num"]))
    return structured_output

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
  <!-- jsdiff for best-in-class text diff visuals -->
  <script src="https://cdn.jsdelivr.net/npm/diff@5.2.0/dist/diff.min.js"></script>
  <style>
    :root{
      --bg:#0f1115;--panel:#171a21;--panel-2:#1c2028;--text:#e6e6e6;--muted:#aeb4c0;
      --primary:#4da3ff;--primary-2:#2a84f2;--green:#2bd47d;--red:#ff6b6b;--chip-major:#ff375f;--chip-minor:#ffd166;--chip-clar:#6ee7ff;
      --border:#2a2f3a;--rail:56px;
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

    /* Layout */
    .layout{display:grid;grid-template-columns:320px 1fr;gap:14px;align-items:start}
    .layout.collapsed{grid-template-columns:var(--rail) 1fr}
    .sidebar-col{position:sticky;top:12px;height:calc(100vh - 24px)}
    .sidebar{height:100%;overflow:auto;padding:10px}
    .rail{height:100%;display:none;align-items:flex-start;justify-content:center}
    .layout.collapsed .rail{display:flex}
    .layout.collapsed .sidebar{display:none}
    .rail .rail-inner{display:flex;flex-direction:column;gap:10px;margin-top:8px;width:var(--rail);align-items:center}
    .rail .icon-btn{width:38px;height:38px;border-radius:10px;border:1px solid var(--border);background:#121722;color:#c6d4f2;display:flex;align-items:center;justify-content:center}
    .rail .icon-btn:hover{background:#161c2a;color:#fff}
    .case-nav a{display:flex;align-items:center;gap:.5rem;padding:.35rem .5rem;border-radius:8px;color:var(--text)}
    .case-nav a:hover{background:#1a2030}
    .case-nav .meta{color:#99a3b6;font-size:.8rem}
    .sidebar h6{font-size:.95rem;margin:0}
    .toggle-btn,.sort-btn,.side-btn{border-radius:10px;border:1px solid var(--border);background:#121722;color:#c6d4f2}
    .sort-btn.active{outline:2px solid rgba(77,163,255,.35)}
    .sort-btn .mode{font-weight:700;color:#b7d3ff}

    /* Right panel with visible right border; wrap long text */
    #resultsPanel{padding:8px;box-shadow: inset -1px 0 0 var(--border)}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;margin:0}

    .tabs{border-bottom:1px solid var(--border)}
    .tabs .nav-link{color:#aeb4c0}
    .tabs .nav-link.active{color:var(--text);background:#1a202c;border-color:var(--border) var(--border) #1a202c}

    /* Diff visuals */
    .diff-inline .eq{color:var(--text)}
    .diff-inline .ins{background:rgba(43,212,125,.15);color:#caffdf;border-radius:4px;padding:.05rem .15rem}
    .diff-inline .del{text-decoration:line-through;background:rgba(255,107,107,.12);color:#ffd1d1;border-radius:4px;padding:.05rem .15rem}
    .diff-block{border-left:3px solid transparent;border-radius:8px;margin-bottom:.5rem;padding:.35rem .5rem;background:#0f131b}
    .diff-block.insert{border-left-color:#2bd47d;background:rgba(43,212,125,.06)}
    .diff-block.delete{border-left-color:#ff6b6b;background:rgba(255,107,107,.06)}
    .diff-block.replace{border-left-color:#7ab8ff;background:rgba(122,184,255,.06)}
    .sxs{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .sxs .side{background:#0f131b;border:1px solid var(--border);border-radius:10px;padding:.5rem}
    .sxs .title{font-weight:600;color:#aeb4c0;margin-bottom:.25rem}
    .mode-toggle{border-radius:10px;border:1px solid var(--border);background:#121722;color:#c6d4f2}

    .kbd{border:1px solid #3a4252;border-bottom-color:#2e3543;background:#1a1f2b;padding:.15rem .35rem;border-radius:6px;font-size:.8rem;color:var(--muted)}
    .badge-score{margin-left:.25rem}
  </style>
</head>
<body>
  <!-- Top: paste area -->
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
            <button class="btn btn-primary"><i class="bi bi-lightning-charge"></i> Compare & Summarize</button>
            <button class="btn btn-outline-light" type="button" id="clearBtn"><i class="bi bi-eraser"></i> Clear</button>
            <button class="btn btn-outline-light" type="button" id="demoBtn"><i class="bi bi-journal-text"></i> Load Demo</button>
            <button class="btn btn-outline-info" type="button" id="downloadAllBtn"><i class="bi bi-download"></i> Download JSON</button>
          </div>
        </div>
      </form>
    </div>
  </div>

  <!-- Beneath: collapsible left sidebar + right content -->
  <div class="container-fluid">
    <div id="gridLayout" class="layout">
      <div class="sidebar-col">
        <!-- Collapsed rail -->
        <div class="rail">
          <div class="rail-inner">
            <button class="icon-btn" id="toggleSidebarBtnRail" aria-pressed="true" title="Expand case list"><i class="bi bi-layout-sidebar-inset"></i></button>
            <button class="icon-btn" id="sortCaseBtnRail" data-mode="number" title="Cycle sort"><i class="bi bi-sort-numeric-down"></i></button>
          </div>
        </div>

        <!-- Full sidebar -->
        <div class="sidebar panel">
          <div class="d-flex align-items-center gap-2 mb-2">
            <h6 class="mb-0">Cases</h6>
            <span class="ms-auto badge badge-score" id="caseCountBadge">0</span>
          </div>
          <div class="d-flex gap-2 mb-2">
            <button class="toggle-btn px-2 py-1" type="button" id="toggleSidebarBtn" aria-pressed="false" title="Hide case list">
              <i class="bi bi-layout-sidebar-inset-reverse me-1"></i> Hide
            </button>
            <button class="sort-btn px-2 py-1" type="button" id="sortCaseBtn" data-mode="number" aria-pressed="true" title="Cycle sort (Case # → Δ Change → Score)">
              <i class="bi bi-sort-numeric-down me-1"></i>
              Sort: <span class="mode" id="sortModeLabel">Case #</span>
            </button>
          </div>
          <input id="searchInput" class="form-control form-control-sm mb-2" placeholder="Search text or Case # (F)"/>
          <div class="d-flex flex-wrap gap-2 mb-2">
            <button class="side-btn btn-sm" id="filterMajorsBtn"><i class="bi bi-exclamation-octagon"></i> Majors</button>
            <button class="side-btn btn-sm" id="filterErrorsBtn"><i class="bi bi-bug"></i> Errors</button>
            <button class="side-btn btn-sm" id="expandAllBtn"><i class="bi bi-arrows-expand"></i> Expand</button>
            <button class="side-btn btn-sm" id="collapseAllBtn"><i class="bi bi-arrows-collapse"></i> Collapse</button>
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
          <div id="caseNav" class="case-nav small"></div>
          <div class="mt-3 small text-secondary">
            <div><span class="kbd">J</span>/<span class="kbd">K</span> next/prev</div>
            <div><span class="kbd">1–4</span> tabs</div>
            <div><span class="kbd">F</span> focus search</div>
            <div><span class="kbd">S</span> cycle sort</div>
            <div><span class="kbd">G</span><span class="mx-1">G</span> top</div>
          </div>
        </div>
      </div>

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

  <script>
    // ---------- Data ----------
    let caseData = {{ case_data | tojson }};

    // ---------- DOM ----------
    const gridLayout = document.getElementById('gridLayout');
    const navEl = document.getElementById('caseNav');
    const containerEl = document.getElementById('caseContainer');
    const emptyStateEl = document.getElementById('emptyState');
    const caseCountBadge = document.getElementById('caseCountBadge');
    const aggregateBlock = document.getElementById('aggregateBlock');
    const aggMajor = document.getElementById('aggMajor');
    const aggMinor = document.getElementById('aggMinor');
    const aggClar = document.getElementById('aggClar');
    const aggBar = document.getElementById('aggBar');

    // Sidebar controls (full + rail)
    const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
    const sortBtn = document.getElementById('sortCaseBtn');
    const sortModeLabel = document.getElementById('sortModeLabel');
    const toggleSidebarBtnRail = document.getElementById('toggleSidebarBtnRail');
    const sortBtnRail = document.getElementById('sortCaseBtnRail');

    const sortModes = ['number','change','score'];
    const sortLabels = { number: 'Case #', change: 'Δ Change', score: 'Score' };
    const sortIcons = { number: 'bi-sort-numeric-down', change: 'bi-lightning-charge', score: 'bi-trophy' };

    // ---------- Sidebar collapse/expand ----------
    function setCollapsed(collapsed){
      gridLayout.classList.toggle('collapsed', collapsed);
      toggleSidebarBtn.setAttribute('aria-pressed', String(collapsed));
      toggleSidebarBtnRail.setAttribute('aria-pressed', String(!collapsed));
    }
    toggleSidebarBtn.addEventListener('click', () => setCollapsed(true));
    toggleSidebarBtnRail.addEventListener('click', () => setCollapsed(false));

    // ---------- Sort state ----------
    function applySortVisual(mode) {
      sortBtn.setAttribute('data-mode', mode);
      sortBtn.classList.add('active');
      sortBtn.innerHTML = '<i class="bi '+sortIcons[mode]+' me-1"></i> Sort: <span class="mode" id="sortModeLabel">'+sortLabels[mode]+'</span>';
      sortBtnRail.setAttribute('data-mode', mode);
      sortBtnRail.innerHTML = '<i class="bi '+sortIcons[mode]+'"></i>';
    }
    function sortBy(mode) {
      if (mode === 'number') {
        caseData.sort((a,b)=> parseInt(a.case_num)-parseInt(b.case_num));
      } else if (mode === 'change') {
        caseData.sort((a,b)=> (b.percentage_change||0)-(a.percentage_change||0));
      } else if (mode === 'score') {
        caseData.sort((a,b)=> ((b.summary?.score)||0)-((a.summary?.score)||0));
      }
    }
    function cycleSort(current){
      const next = sortModes[(sortModes.indexOf(current)+1)%sortModes.length];
      sortBy(next); applySortVisual(next); renderAll(caseData);
    }
    sortBtn.addEventListener('click', () => cycleSort(sortBtn.getAttribute('data-mode') || 'number'));
    sortBtnRail.addEventListener('click', () => cycleSort(sortBtnRail.getAttribute('data-mode') || 'number'));

    // ---------- Search / Filters ----------
    const searchInput = document.getElementById('searchInput');
    searchInput.addEventListener('input', () => {
      let list = [...caseData];
      const q = searchInput.value.toLowerCase();
      if (q) {
        list = list.filter(c => {
          if (String(c.case_num).includes(q)) return true;
          if (c.resident_report?.toLowerCase().includes(q)) return true;
          if (c.attending_report?.toLowerCase().includes(q)) return true;
          const s = c.summary;
          return s ? JSON.stringify(s).toLowerCase().includes(q) : false;
        });
      }
      renderAll(list);
    });
    document.getElementById('filterMajorsBtn').addEventListener('click', () => {
      renderAll(caseData.filter(x => (x.summary?.major_findings?.length||0) > 0));
    });
    document.getElementById('filterErrorsBtn').addEventListener('click', () => {
      renderAll(caseData.filter(x => !x.summary));
    });
    document.getElementById('expandAllBtn').addEventListener('click', () => {
      document.querySelectorAll('[id^="body"]').forEach(el => el.style.display = '');
    });
    document.getElementById('collapseAllBtn').addEventListener('click', () => {
      document.querySelectorAll('[id^="body"]').forEach(el => el.style.display = 'none');
    });

    // ---------- Demo / Clear / Download ----------
    document.getElementById('reportForm').addEventListener('submit', () => {});
    document.getElementById('clearBtn').addEventListener('click', () => {
      document.getElementById('report_text').value = '';
      document.getElementById('custom_prompt').value = '';
      caseData = [];
      renderAll(caseData);
    });
    document.getElementById('demoBtn').addEventListener('click', () => {
      const demo = `Case 1
Resident Report:
Heterogeneous masslike lesion in the right lower pole collecting system with severe right hydronephrosis. Mild heterogeneity of the left upper pole renal collecting system with moderate hydronephrosis. No secondary signs of trauma. Findings likely represent bilateral urothelial malignancy, however acute traumatic injury not entirely excluded. Recommend further evaluation with renal protocol CT.

Attending Report:
1. No definite acute traumatic abnormality in the chest, abdomen, or pelvis.
2. Bilateral hydronephrosis and peripelvic cysts. Indeterminate hyperdense tissue in the left peripelvic cysts and right upper pole collecting system is atypical for a posttraumatic finding and concerning for possible underlying neoplasm. Hemorrhagic or infectious etiologies are also considerations. Clinical correlation and multiphase CT/MRI with urological follow-up recommended.
3. Bilateral axillary lymphadenopathy, nonspecific.
4. Indeterminate right middle lobe pulmonary nodule. No follow-up needed if patient is low-risk. Non-contrast chest CT can be considered in 12 months if patient is high-risk (Per Fleischner Society guidelines). Shorter-term follow-up could be considered according to oncological considerations, as indicated.
5. Splenectomy. Additional findings above.`;
      document.getElementById('report_text').value = demo;
    });
    document.getElementById('downloadAllBtn').addEventListener('click', () => {
      const summaries = caseData.map(c => ({ case_number: c.case_num, summary: c.summary, error: c.summary_error || null }));
      const blob = new Blob([JSON.stringify(summaries, null, 2)], {type: 'application/json'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = 'summaries.json'; a.click();
      URL.revokeObjectURL(url);
    });

    // ---------- Nav & Cards ----------
    function renderNav(list) {
      navEl.innerHTML = list.map(c => `
        <a href="#case${c.case_num}">
          <div class="me-auto">
            <div><strong>#${c.case_num}</strong></div>
            <div class="meta">Δ ${c.percentage_change}% • Score ${(c.summary?.score)||0}</div>
          </div>
        </a>
      `).join('');
      caseCountBadge.textContent = list.length;
    }

    function escapeHTML(str) {
      if (!str) return '';
      return str.replace(/[&<>"']/g, (m) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#39;'}[m]));
    }

    function caseCardHTML(c) {
      const s = c.summary;
      if (!s) {
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

      const majors = s.major_findings || [];
      const minors = s.minor_findings || [];
      const clar = s.clarifications || [];
      const total = majors.length + minors.length + clar.length;
      const pctMajor = total ? Math.round((majors.length/total)*100) : 0;

      const listHTML = arr => arr.length ? `<ul class="mb-0">${arr.map(x=>`<li>${x}</li>`).join('')}</ul>` : '<div class="text-secondary small">None</div>';

      return `
        <div id="case${c.case_num}" class="case-card panel-2 p-3 mb-3">
          <div class="d-flex align-items-center gap-2 flex-wrap">
            <strong class="me-2">Case ${c.case_num}</strong>
            <span class="badge badge-score">Score ${s.score ?? 0}</span>
            <span class="ms-2 text-secondary small">Δ ${c.percentage_change}%</span>
            <span class="ms-auto d-flex gap-2">
              <button class="btn btn-outline-light btn-sm" onclick='navigator.clipboard.writeText(${JSON.stringify(JSON.stringify(s))})'><i class="bi bi-clipboard"></i> Copy JSON</button>
              <button class="btn btn-outline-light btn-sm" onclick="toggleCollapse('${c.case_num}')"><i class="bi bi-arrows-collapse"></i> Toggle</button>
            </span>
          </div>

          <div class="d-flex gap-2 mt-2">
            <span class="chip major">Major ${majors.length}</span>
            <span class="chip minor">Minor ${minors.length}</span>
            <span class="chip clar">Clar ${clar.length}</span>
          </div>
          <div class="progress my-2" title="${pctMajor}% of items are major">
            <div class="progress-bar" style="width:${pctMajor}%"></div>
          </div>

          <div id="body${c.case_num}">
            <div class="d-flex justify-content-end mb-2">
              <div class="btn-group btn-group-sm" role="group" aria-label="Diff mode">
                <button class="mode-toggle px-2" onclick="setDiffMode('${c.case_num}','unified')">Unified</button>
                <button class="mode-toggle px-2" onclick="setDiffMode('${c.case_num}','sxs')">Side-by-Side</button>
              </div>
            </div>

            <ul class="nav nav-tabs tabs mt-2" role="tablist">
              <li class="nav-item"><button class="nav-link active" id="tab-sum-${c.case_num}" data-bs-toggle="tab" data-bs-target="#tab-pane-sum-${c.case_num}" type="button">Summary</button></li>
              <li class="nav-item"><button class="nav-link" id="tab-combo-${c.case_num}" data-bs-toggle="tab" data-bs-target="#tab-pane-combo-${c.case_num}" type="button">Combined Diff</button></li>
              <li class="nav-item"><button class="nav-link" id="tab-res-${c.case_num}" data-bs-toggle="tab" data-bs-target="#tab-pane-res-${c.case_num}" type="button">Resident</button></li>
              <li class="nav-item"><button class="nav-link" id="tab-att-${c.case_num}" data-bs-toggle="tab" data-bs-target="#tab-pane-att-${c.case_num}" type="button">Attending</button></li>
            </ul>

            <div class="tab-content p-2">
              <div class="tab-pane fade show active" id="tab-pane-sum-${c.case_num}">
                <div class="row g-2">
                  <div class="col-12 col-lg-4"><div class="panel p-2 h-100"><div class="mb-2"><i class="bi bi-exclamation-octagon text-danger"></i> <strong>Major</strong></div>${listHTML(majors)}</div></div>
                  <div class="col-12 col-lg-4"><div class="panel p-2 h-100"><div class="mb-2"><i class="bi bi-info-circle text-warning"></i> <strong>Minor</strong></div>${listHTML(minors)}</div></div>
                  <div class="col-12 col-lg-4"><div class="panel p-2 h-100"><div class="mb-2"><i class="bi bi-pencil-square text-info"></i> <strong>Clarifications</strong></div>${listHTML(clar)}</div></div>
                </div>
              </div>

              <div class="tab-pane fade" id="tab-pane-combo-${c.case_num}">
                <div class="panel p-2">
                  <div id="diff-${c.case_num}" class="diff-inline"></div>
                  <div id="diff-sxs-${c.case_num}" class="sxs d-none">
                    <div class="side"><div class="title">Resident</div><div id="sxs-left-${c.case_num}"></div></div>
                    <div class="side"><div class="title">Attending</div><div id="sxs-right-${c.case_num}"></div></div>
                  </div>
                </div>
              </div>

              <div class="tab-pane fade" id="tab-pane-res-${c.case_num}">
                <div class="panel p-2"><pre class="mb-0">${escapeHTML(c.resident_report)}</pre></div>
              </div>
              <div class="tab-pane fade" id="tab-pane-att-${c.case_num}">
                <div class="panel p-2"><pre class="mb-0">${escapeHTML(c.attending_report)}</pre></div>
              </div>
            </div>
          </div>
        </div>
      `;
    }

    function renderAll(data) {
      if (!data || !data.length) {
        containerEl.classList.add('d-none'); emptyStateEl.classList.remove('d-none');
        renderNav([]); aggregateBlock.classList.add('d-none'); return;
      }
      containerEl.classList.remove('d-none'); emptyStateEl.classList.add('d-none');

      // Aggregate
      let M=0,m=0,cnt=0;
      data.forEach(d => {
        if (!d.summary) return;
        M += (d.summary.major_findings||[]).length;
        m += (d.summary.minor_findings||[]).length;
        cnt += (d.summary.clarifications||[]).length;
      });
      const total = M+m+cnt;
      aggMajor.textContent=M; aggMinor.textContent=m; aggClar.textContent=cnt;
      aggBar.style.width = total ? Math.round((M/total)*100)+'%' : '0%';
      aggregateBlock.classList.remove('d-none');

      containerEl.innerHTML = data.map(caseCardHTML).join('');
      renderNav(data);

      // After DOM is in, build diffs for each case
      data.forEach(c => buildBestDiff(c.case_num, c.resident_report||'', c.attending_report||''));
    }

    // ---------- Best-possible diff for radiology conclusions ----------
    // 1) Sentence/bullet splitter that respects numbered lists
    function splitSentencesWithBullets(text) {
      if (!text) return [];
      const lines = text.split(/\\r?\\n/).map(s=>s.trim()).filter(Boolean);
      let s = lines.join(' ');
      s = s.replace(/\\s*(\\d+\\.\\s+)/g, '\\n$1');
      s = s.replace(/\\s*(-\\s+|\\*\\s+)/g, '\\n$1');
      const parts = s.split(/(?<=[.!?])\\s+(?=[A-Z0-9])|\\n(?=\\d+\\.\\s+|-\\s+|\\*\\s+)/).map(p=>p.trim()).filter(Boolean);
      return parts;
    }

    // 2) Simple similarity based on jsdiff equal tokens
    function sentenceSimilarity(a, b) {
      const diff = Diff.diffWordsWithSpace(a, b);
      let eq = 0, total = Math.max(a.length, b.length) || 1;
      diff.forEach(p => { if (!p.added && !p.removed) eq += p.value.length; });
      return eq / total;
    }

    // 3) Align sentences greedily with threshold
    function alignBlocks(resSents, attSents, thresh=0.60) {
      const usedA = new Set();
      const pairs = [];
      resSents.forEach((rs, i) => {
        let bestJ = -1, bestScore = 0;
        attSents.forEach((as, j) => {
          if (usedA.has(j)) return;
          const score = sentenceSimilarity(rs, as);
          if (score > bestScore) { bestScore = score; bestJ = j; }
        });
        if (bestJ >= 0 && bestScore >= thresh) {
          usedA.add(bestJ);
          pairs.push({i, j: bestJ});
        }
      });
      const usedR = new Set(pairs.map(p=>p.i));
      const unpairedR = resSents.map((s, i)=> i).filter(i=>!usedR.has(i));
      const unpairedA = attSents.map((s, j)=> j).filter(j=>!usedA.has(j));
      return {pairs: pairs.sort((a,b)=>a.i-b.i), unpairedR, unpairedA};
    }

    // 4) Build both Unified and Side-by-Side diffs
    function buildBestDiff(id, resident, attending) {
      const resSents = splitSentencesWithBullets(resident);
      const attSents = splitSentencesWithBullets(attending);
      const {pairs, unpairedR, unpairedA} = alignBlocks(resSents, attSents, 0.60);

      const unified = [];
      // deleted sentences
      unpairedR.forEach(i => unified.push(`<div class="diff-block delete">[Deleted: ${escapeHTML(resSents[i])}]</div>`));
      // inserted sentences
      unpairedA.forEach(j => unified.push(`<div class="diff-block insert">[Inserted: ${escapeHTML(attSents[j])}]</div>`));
      // replacements with word-level highlights
      pairs.forEach(({i,j}) => {
        const chunks = Diff.diffWordsWithSpace(resSents[i], attSents[j]);
        const line = chunks.map(p => {
          if (p.added) return `<span class="ins">${escapeHTML(p.value)}</span>`;
          if (p.removed) return `<span class="del">${escapeHTML(p.value)}</span>`;
          return `<span class="eq">${escapeHTML(p.value)}</span>`;
        }).join('');
        unified.push(`<div class="diff-block replace"><div class="diff-inline">${line}</div></div>`);
      });

      const mount = document.getElementById('diff-'+id);
      if (mount) mount.innerHTML = unified.join('');

      // Side-by-side: render independent word highlights on each side
      const left = document.getElementById('sxs-left-'+id);
      const right = document.getElementById('sxs-right-'+id);
      if (left && right) {
        const leftHTML = [];
        const rightHTML = [];
        // Show deletions in left
        unpairedR.forEach(i => leftHTML.push(`<div class="diff-block delete">${escapeHTML(resSents[i])}</div>`));
        // Show insertions in right
        unpairedA.forEach(j => rightHTML.push(`<div class="diff-block insert">${escapeHTML(attSents[j])}</div>`));
        // Paired replacements with inline highlights on each side independently
        pairs.forEach(({i,j}) => {
          const lw = Diff.diffWordsWithSpace(resSents[i], ''); // mark all as baseline
          const rw = Diff.diffWordsWithSpace('', attSents[j]);

          // but better: build both sides using the same diff so corresponding tokens line up visually
          const chunks = Diff.diffWordsWithSpace(resSents[i], attSents[j]);
          const L = chunks.map(p=>{
            if (p.removed) return `<span class="del">${escapeHTML(p.value)}</span>`;
            if (!p.added) return `<span class="eq">${escapeHTML(p.value)}</span>`;
            return '';
          }).join('');
          const R = chunks.map(p=>{
            if (p.added) return `<span class="ins">${escapeHTML(p.value)}</span>`;
            if (!p.removed) return `<span class="eq">${escapeHTML(p.value)}</span>`;
            return '';
          }).join('');

          leftHTML.push(`<div class="diff-block replace"><div class="diff-inline">${L}</div></div>`);
          rightHTML.push(`<div class="diff-block replace"><div class="diff-inline">${R}</div></div>`);
        });

        left.innerHTML = leftHTML.join('');
        right.innerHTML = rightHTML.join('');
      }
    }

    // Toggle unified vs sxs
    window.setDiffMode = function(id, mode) {
      const u = document.getElementById('diff-'+id);
      const sxs = document.getElementById('diff-sxs-'+id);
      if (!u || !sxs) return;
      if (mode === 'sxs') { u.classList.add('d-none'); sxs.classList.remove('d-none'); }
      else { sxs.classList.add('d-none'); u.classList.remove('d-none'); }
    }

    window.toggleCollapse = function(id){
      const el = document.getElementById('body'+id);
      if (!el) return;
      el.style.display = (el.style.display === 'none') ? '' : 'none';
    }

    // ---------- Init ----------
    document.addEventListener('DOMContentLoaded', () => {
      if (caseData && caseData.length) renderAll(caseData);
      applySortVisual('number'); setCollapsed(false);
    });

    // Keyboard helpers
    let currentIndex = 0;
    function focusCase(i) {
      const list = document.querySelectorAll('.case-card');
      if (!list.length) return;
      currentIndex = Math.max(0, Math.min(i, list.length-1));
      list[currentIndex].scrollIntoView({behavior:'smooth', block:'start'});
    }
    function switchTab(n) {
      const list = document.querySelectorAll('.case-card');
      if (!list.length) return;
      const id = list[currentIndex].id.replace('case','');
      const tabIds = ['sum','combo','res','att'];
      const btn = document.getElementById(`tab-${tabIds[n-1]}-${id}`);
      if (btn) btn.click();
    }
    document.addEventListener('keydown', (e) => {
      if (['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) {
        if (e.key.toLowerCase()==='escape') document.activeElement.blur();
        return;
      }
      if (e.key.toLowerCase()==='j') { e.preventDefault(); focusCase(currentIndex+1); }
      if (e.key.toLowerCase()==='k') { e.preventDefault(); focusCase(currentIndex-1); }
      if (['1','2','3','4'].includes(e.key)) { e.preventDefault(); switchTab(parseInt(e.key,10)); }
      if (e.key.toLowerCase()==='f') { e.preventDefault(); document.getElementById('searchInput').focus(); }
      if (e.key.toLowerCase()==='s') { e.preventDefault(); (gridLayout.classList.contains('collapsed') ? sortBtnRail : sortBtn).click(); }
      if (e.key.toLowerCase()==='g') { window.scrollTo({top:0, behavior:'smooth'}); }
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
