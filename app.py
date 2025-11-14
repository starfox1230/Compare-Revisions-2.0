from flask import Flask, render_template_string, request
import difflib, re, os, json, logging, html
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
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "minimal")  # minimal|low|medium|high
VERBOSITY = os.getenv("VERBOSITY", "low")                    # low|medium|high

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
                "major_findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary_text": {"type": "string"},
                            "resident_snippet": {"type": "string"},
                            "attending_snippet": {"type": "string"},
                            "attending_highlights": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "resident_highlights": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": [
                            "summary_text",
                            "attending_snippet",
                            "attending_highlights"
                        ],
                        "additionalProperties": False
                    }
                },
                "minor_findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary_text": {"type": "string"},
                            "resident_snippet": {"type": "string"},
                            "attending_snippet": {"type": "string"},
                            "attending_highlights": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "resident_highlights": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": [
                            "summary_text",
                            "attending_snippet",
                            "attending_highlights"
                        ],
                        "additionalProperties": False
                    }
                },
                "clarifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary_text": {"type": "string"},
                            "resident_snippet": {"type": "string"},
                            "attending_snippet": {"type": "string"},
                            "attending_highlights": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "resident_highlights": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": [
                            "summary_text",
                            "attending_snippet",
                            "attending_highlights"
                        ],
                        "additionalProperties": False
                    }
                },
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

<output_schema_details>
Each of the three arrays (major_findings, minor_findings, clarifications)
must contain OBJECTS, not plain strings.

For each finding object:

- summary_text:
  - A concise sentence or phrase summarizing the change, exactly as you would have written each list item previously.
  - You may include your parenthetical type/modifiers here (e.g., "(perceptual; new critical finding)").
  - This is what will be shown to the learner in the summary column.

- resident_snippet:
  - A short excerpt copied directly from the RESIDENT report that corresponds to this change.
  - If the resident did not mention this entity at all (purely perceptual addition), use an empty string "".

- attending_snippet:
  - A short excerpt copied directly from the ATTENDING report that corresponds to this change.
  - This MUST be a literal substring of the attending report, not paraphrased.

- attending_highlights:
  - An array of 1–3 short words or phrases taken VERBATIM from attending_snippet
    that represent the key new or corrected concept.
  - These are the phrases whose text you want the learner’s eye to go to first
    (e.g., "active hemorrhage", "right pneumothorax", "no PE").

- resident_highlights:
  - Optional. An array of up to 1–3 short words or phrases taken VERBATIM from resident_snippet
    that represent the incorrect or missing idea (e.g., "no pneumothorax", "probable PE").
  - If not applicable, return an empty array [].
</output_schema_details>

<validation_checklist>
Before output:
- Only attending-introduced or attending-corrected content appears as finding objects.
- Each item in major_findings, minor_findings, clarifications is an OBJECT with:
  summary_text, attending_snippet, attending_highlights
  (and optional resident_snippet, resident_highlights).
- attending_highlights phrases are exact substrings of attending_snippet.
- resident_highlights phrases (if present) are exact substrings of resident_snippet.
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
→ major_findings: []
→ minor_findings: []
→ clarifications: [
  {
    "summary_text": "PE location refined to subsegmental; small clot burden (descriptor: location; descriptor: certainty)",
    "resident_snippet": "PE in RLL segmental branches.",
    "attending_snippet": "PE in RLL subsegmental branches; small clot burden.",
    "attending_highlights": ["subsegmental branches", "small clot burden"],
    "resident_highlights": ["segmental branches"]
  }
]
→ score: 0

Example 2 — New urgent finding:
Resident: “No pneumothorax.”
Attending: “Small right pneumothorax.”
→ major_findings: [
  {
    "summary_text": "Right pneumothorax (perceptual; new critical finding)",
    "resident_snippet": "No pneumothorax.",
    "attending_snippet": "Small right pneumothorax.",
    "attending_highlights": ["right pneumothorax"],
    "resident_highlights": ["No pneumothorax"]
  }
]
→ minor_findings: []
→ clarifications: []
→ score: 3

Example 3 — Critical negation (definite → no):
Resident: “Acute PE; start anticoagulation.”
Attending: “No pulmonary embolism.”
→ major_findings: [
  {
    "summary_text": "No pulmonary embolism (interpretive; critical correction; resident said acute PE)",
    "resident_snippet": "Acute PE; start anticoagulation.",
    "attending_snippet": "No pulmonary embolism.",
    "attending_highlights": ["No pulmonary embolism"],
    "resident_highlights": ["Acute PE"]
  }
]
→ minor_findings: []
→ clarifications: []
→ score: 3

Example 4 — Probable → no (soft negation):
Resident: “Probable segmental PE, RLL.”
Attending: “No PE.”
→ major_findings: []
→ minor_findings: [
  {
    "summary_text": "No PE (interpretive; correction; resident said probable)",
    "resident_snippet": "Probable segmental PE, RLL.",
    "attending_snippet": "No PE.",
    "attending_highlights": ["No PE"],
    "resident_highlights": ["Probable segmental PE"]
  }
]
→ clarifications: []
→ score: 1

Example 5 — Questionable → no (wording cleanup):
Resident: “Questionable subsegmental PE.”
Attending: “No PE.”
→ major_findings: []
→ minor_findings: []
→ clarifications: [
  {
    "summary_text": "No PE (interpretive; descriptor: certainty; resident said questionable)",
    "resident_snippet": "Questionable subsegmental PE.",
    "attending_snippet": "No PE.",
    "attending_highlights": ["No PE"],
    "resident_highlights": ["Questionable subsegmental PE"]
  }
]
→ score: 0

Example 6 — Malpositioned device:
Resident: “NG tube present.”
Attending: “NG tube coiled in esophagus.”
→ major_findings: [
  {
    "summary_text": "Malpositioned NG tube (esophageal) (perceptual; malpositioned device)",
    "resident_snippet": "NG tube present.",
    "attending_snippet": "NG tube coiled in esophagus.",
    "attending_highlights": ["NG tube coiled in esophagus"],
    "resident_highlights": ["NG tube present"]
  }
]
→ minor_findings: []
→ clarifications: []
→ score: 3

Example 7 — Free air artifact (critical downshift):
Resident: “Free air under diaphragm.”
Attending: “No free air; prior image was artifact.”
→ major_findings: [
  {
    "summary_text": "No free air (interpretive; critical correction; artifact resolved)",
    "resident_snippet": "Free air under diaphragm.",
    "attending_snippet": "No free air; prior image was artifact.",
    "attending_highlights": ["No free air"],
    "resident_highlights": ["Free air under diaphragm"]
  }
]
→ minor_findings: []
→ clarifications: []
→ score: 3

Example 8 — Non-urgent correction:
Resident: “Normal abdomen.”
Attending: “Benign hepatic hemangioma; cholelithiasis without cholecystitis.”
→ major_findings: []
→ minor_findings: [
  {
    "summary_text": "Hepatic hemangioma (perceptual; classification change; follow-up impact)",
    "resident_snippet": "Normal abdomen.",
    "attending_snippet": "Benign hepatic hemangioma; cholelithiasis without cholecystitis.",
    "attending_highlights": ["hepatic hemangioma"],
    "resident_highlights": ["Normal abdomen"]
  },
  {
    "summary_text": "Cholelithiasis without cholecystitis (perceptual)",
    "resident_snippet": "Normal abdomen.",
    "attending_snippet": "Benign hepatic hemangioma; cholelithiasis without cholecystitis.",
    "attending_highlights": ["cholelithiasis", "without cholecystitis"],
    "resident_highlights": ["Normal abdomen"]
  }
]
→ clarifications: []
→ score: 2

Example 9 — Threshold crossing (upgrade to MINOR):
Resident: “Pulmonary nodule 5 mm.”
Attending: “Pulmonary nodule 8 mm (follow-up recommended).”
→ major_findings: []
→ minor_findings: [
  {
    "summary_text": "Pulmonary nodule 8 mm (descriptor: size; threshold crossed; follow-up impact)",
    "resident_snippet": "Pulmonary nodule 5 mm.",
    "attending_snippet": "Pulmonary nodule 8 mm (follow-up recommended).",
    "attending_highlights": ["8 mm", "follow-up recommended"],
    "resident_highlights": ["5 mm"]
  }
]
→ clarifications: []
→ score: 1

Example 10 — Laterality/location correction:
Resident: “Left lower lobe pneumonia.”
Attending: “Right lower lobe pneumonia.”
→ major_findings: []
→ minor_findings: [
  {
    "summary_text": "Right lower lobe pneumonia (interpretive; laterality correction)",
    "resident_snippet": "Left lower lobe pneumonia.",
    "attending_snippet": "Right lower lobe pneumonia.",
    "attending_highlights": ["Right lower lobe pneumonia"],
    "resident_highlights": ["Left lower lobe pneumonia"]
  }
]
→ clarifications: []
→ score: 1

Example 11 — ICH added:
Resident: “No acute intracranial hemorrhage.”
Attending: “Small acute subarachnoid hemorrhage.”
→ major_findings: [
  {
    "summary_text": "Acute subarachnoid hemorrhage (perceptual; new critical finding)",
    "resident_snippet": "No acute intracranial hemorrhage.",
    "attending_snippet": "Small acute subarachnoid hemorrhage.",
    "attending_highlights": ["acute subarachnoid hemorrhage"],
    "resident_highlights": ["No acute intracranial hemorrhage"]
  }
]
→ minor_findings: []
→ clarifications: []
→ score: 3

Example 12 — Descriptor only:
Resident: “Sigmoid diverticulitis.”
Attending: “Sigmoid diverticulitis with trace adjacent fluid; no abscess.”
→ major_findings: []
→ minor_findings: []
→ clarifications: [
  {
    "summary_text": "Adds trace adjacent fluid; no abscess (descriptor: degree; descriptor: negative finding)",
    "resident_snippet": "Sigmoid diverticulitis.",
    "attending_snippet": "Sigmoid diverticulitis with trace adjacent fluid; no abscess.",
    "attending_highlights": ["trace adjacent fluid", "no abscess"],
    "resident_highlights": []
  }
]
→ score: 0

</examples>
"""

# --------------------------- Helpers ---------------------------------
def normalize_text(text):
    if not text:
        return ""

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.rstrip() for line in text.split('\n')]
    # Preserve intentional blank lines so paragraph boundaries remain intact
    normalized = "\n".join(lines)
    return normalized.strip("\n")

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
    paragraphs = re.split(r'\n\s*\n+', text)
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
                diff_html += f'<div class="para equal">{html.escape(paragraph)}</div>'
        elif opcode == 'insert':
            for paragraph in attending_paragraphs[b1:b2]:
                diff_html += f'<div class="para ins">{html.escape(paragraph)}</div>'
        elif opcode == 'delete':
            for paragraph in resident_paragraphs[a1:a2]:
                diff_html += f'<div class="para del">{html.escape(paragraph)}</div>'
        elif opcode == 'replace':
            res_paragraphs = resident_paragraphs[a1:a2]
            att_paragraphs = attending_paragraphs[b1:b2]
            max_len = max(len(res_paragraphs), len(att_paragraphs))

            for idx in range(max_len):
                res_paragraph = res_paragraphs[idx] if idx < len(res_paragraphs) else None
                att_paragraph = att_paragraphs[idx] if idx < len(att_paragraphs) else None

                if res_paragraph and att_paragraph:
                    similarity = difflib.SequenceMatcher(None, res_paragraph, att_paragraph).ratio()
                    if similarity < 0.4:
                        diff_html += f'<div class="para del">{html.escape(res_paragraph)}</div>'
                        diff_html += f'<div class="para ins">{html.escape(att_paragraph)}</div>'
                        continue

                    res_words = res_paragraph.split()
                    att_words = att_paragraph.split()
                    word_matcher = difflib.SequenceMatcher(None, res_words, att_words)
                    seg = []
                    for word_opcode, w_a1, w_a2, w_b1, w_b2 in word_matcher.get_opcodes():
                        if word_opcode == 'equal':
                            seg.append(html.escape(" ".join(res_words[w_a1:w_a2])))
                        elif word_opcode == 'replace':
                            deleted = html.escape(" ".join(res_words[w_a1:w_a2]))
                            inserted = html.escape(" ".join(att_words[w_b1:w_b2]))
                            seg.append(
                                f'<span class="word del">{deleted}</span> '
                                f'<span class="word ins">{inserted}</span>'
                            )
                        elif word_opcode == 'delete':
                            seg.append(f'<span class="word del">{html.escape(" ".join(res_words[w_a1:w_a2]))}</span>')
                        elif word_opcode == 'insert':
                            seg.append(f'<span class="word ins">{html.escape(" ".join(att_words[w_b1:w_b2]))}</span>')
                    diff_html += f'<div class="para rep">{" ".join(seg)}</div>'
                elif res_paragraph:
                    diff_html += f'<div class="para del">{html.escape(res_paragraph)}</div>'
                elif att_paragraph:
                    diff_html += f'<div class="para ins">{html.escape(att_paragraph)}</div>'
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
            tool_choice={"type": "function", "name": "summarize_radiology_report"},
            reasoning={"effort": REASONING_EFFORT},   # <-- uses env variable
            text={"verbosity": VERBOSITY},            # <-- uses env variable
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
def process_cases(cases_data, custom_prompt, max_workers=100):
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

    ai_summaries = process_cases(cases_data, custom_prompt, max_workers=100)
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
  <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
  <style>
    :root{
      --bg:#0f1115;--panel:#171a21;--panel-2:#1c2028;--text:#e6e6e6;--muted:#aeb4c0;
      --primary:#4da3ff;--primary-2:#2a84f2;--green:#2bd47d;--red:#ff6b6b;--chip-major:#ff375f;--chip-minor:#ffd166;--chip-clar:#6ee7ff;
      --border:#2a2f3a;
      --rail:56px; /* minimalist collapsed sidebar width */
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
    .chip{border-radius:999px;padding:.2rem .55rem;font-weight:600;border:1px solid #334155;display:inline-flex;align-items:center;gap:.35rem}
    .chip.major{background:color-mix(in srgb,var(--chip-major) 18%,transparent);color:#ffc4cf;border-color:#5c2034}
    .chip.minor{background:color-mix(in srgb,var(--chip-minor) 18%,transparent);color:#fff2c4;border-color:#5c4a20}
    .chip.clar{background:color-mix(in srgb,var(--chip-clar) 18%,transparent);color:#c4f4ff;border-color:#20545c}

    .key-change{color:var(--chip-major);font-weight:700}

    .toggle-chip{cursor:pointer;border-color:#2a3344;background:#101623;color:#c6d4f2;transition:background .2s ease,border-color .2s ease,color .2s ease}
    .toggle-chip[data-active="true"].chip-major{background:color-mix(in srgb,var(--chip-major) 22%,transparent);color:#ffc4cf;border-color:#5c2034}
    .toggle-chip[data-active="true"].chip-minor{background:color-mix(in srgb,var(--chip-minor) 22%,transparent);color:#fff2c4;border-color:#5c4a20}
    .toggle-chip[data-active="true"].chip-clar{background:color-mix(in srgb,var(--chip-clar) 22%,transparent);color:#c4f4ff;border-color:#20545c}
    .toggle-chip:focus-visible{outline:2px solid rgba(77,163,255,.45);outline-offset:2px}

    .severity-block{transition:background .2s ease,border-color .2s ease}
    .severity-block.highlight-major{background:color-mix(in srgb,var(--chip-major) 16%,transparent);border-color:color-mix(in srgb,var(--chip-major) 35%,#1a2233)}
    .severity-block.highlight-minor{background:color-mix(in srgb,var(--chip-minor) 16%,transparent);border-color:color-mix(in srgb,var(--chip-minor) 35%,#1a2233)}
    .severity-block.highlight-clar{background:color-mix(in srgb,var(--chip-clar) 16%,transparent);border-color:color-mix(in srgb,var(--chip-clar) 35%,#1a2233)}
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

    /* Layout: collapsible left, content right with right border */
    .layout{
      display:grid;
      grid-template-columns: 320px 1fr;
      gap:14px;
      align-items:start;
    }
    .layout.collapsed{
      grid-template-columns: var(--rail) 1fr;
    }

    .sidebar-col{position:sticky;top:12px;height:calc(100vh - 24px)}
    .sidebar{
      height:100%;overflow:auto;padding:10px;
    }
    /* minimalist rail shown when collapsed */
    .rail{
      height:100%;display:none;align-items:flex-start;justify-content:center;
    }
    .layout.collapsed .rail{display:flex}
    .layout.collapsed .sidebar{display:none}

    .rail .rail-inner{
      display:flex;flex-direction:column;gap:10px;margin-top:8px;
      width:var(--rail);align-items:center;
    }
    .rail .icon-btn{
      width:38px;height:38px;border-radius:10px;border:1px solid var(--border);
      background:#121722;color:#c6d4f2;display:flex;align-items:center;justify-content:center;
    }
    .rail .icon-btn:hover{background:#161c2a;color:#fff}

    /* Sidebar minimalist list */
    .case-nav a{
      display:flex;align-items:center;gap:.5rem;
      padding:.4rem .5rem;border-radius:8px;color:var(--text);
    }
    .case-nav a:hover{background:#1a2030}
    .case-nav .meta{color:#99a3b6;font-size:.8rem}
    .sidebar h6{font-size:.95rem;margin:0}

    /* Right container with visible right border and wrapping */
    #resultsPanel{
      padding:8px;
      box-shadow: inset -1px 0 0 var(--border); /* subtle right border “fit” */
    }
    pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;margin:0}

    .tabs{border-bottom:1px solid var(--border)}
    .tabs .nav-link{color:#aeb4c0}
    .tabs .nav-link.active{color:var(--text);background:#1a202c;border-color:var(--border) var(--border) #1a202c}

    /* Buttons in sidebar */
    .toggle-btn,.sort-btn,.side-btn{
      border-radius:10px;border:1px solid var(--border);background:#121722;color:#c6d4f2
    }
    .toggle-btn[aria-pressed="true"]{background:#0e1320;color:#9fb9ff}
    .sort-btn.active{outline:2px solid rgba(77,163,255,.35)}
    .sort-btn .mode{font-weight:700;color:#b7d3ff}

    .case-card{scroll-margin-top:90px}
    .case-card.selected{border-color:rgba(43,212,125,.6);box-shadow:0 0 0 2px rgba(43,212,125,.35)}
    .case-card.single-hidden{display:none}
    .kbd{border:1px solid #3a4252;border-bottom-color:#2e3543;background:#1a1f2b;padding:.15rem .35rem;border-radius:6px;font-size:.8rem;color:var(--muted)}
    .toaster{position:fixed;right:18px;bottom:18px;z-index:50;background:#1c2432;border:1px solid #2a3547;color:#d7e3ff;padding:.65rem .8rem;border-radius:10px;display:none}
  </style>
  <style>
    #loading-overlay {
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,.72); z-index: 2000;
      display: none; /* toggled to flex in JS */
      align-items: center; justify-content: center; flex-direction: column;
    }
    #loading-overlay p { color: #e6e6e6; margin-top: 1rem; font-size: 1.05rem; }
  </style>
</head>
<body>
  <div id="loading-overlay" aria-hidden="true">
    <dotlottie-player
      src="https://lottie.host/817661a8-2608-4435-89a5-daa620a64c36/WtsFI5zdEK.lottie"
      background="transparent" speed="1"
      style="width: 240px; height: 240px;" loop autoplay>
    </dotlottie-player>
    <p>Analyzing reports… this may take a moment.</p>
    <p id="waitTimerText" aria-live="polite" class="small text-secondary" style="margin-top:.25rem;">
      Waiting: <span id="waitTimer">00:00</span>
</p>
  </div>
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
            <!-- NEW: Upload button + hidden input (added) -->
            <button class="btn btn-outline-info" type="button" id="uploadBtn">
              <i class="bi bi-upload"></i> Upload JSON
            </button>
            <input type="file" id="uploadInput" accept="application/json" style="display:none" />
            <!-- END NEW -->
          </div>
        </div>
      </form>
    </div>
  </div>

  <!-- Beneath: Collapsible left sidebar + right content -->
  <div class="container-fluid">
    <div id="gridLayout" class="layout">
      <!-- Left column: contains full sidebar AND minimalist rail -->
      <div class="sidebar-col">
        <!-- Minimalist rail (shown when collapsed) -->
        <div class="rail">
          <div class="rail-inner">
            <button class="icon-btn" id="toggleSidebarBtnRail" aria-pressed="true" title="Expand case list">
              <i class="bi bi-layout-sidebar-inset"></i>
            </button>
            <button class="icon-btn" id="sortCaseBtnRail" data-mode="number" title="Cycle sort">
              <i class="bi bi-sort-numeric-down"></i>
            </button>
          </div>
        </div>

        <!-- Full sidebar (hidden when collapsed) -->
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
            <button class="side-btn btn-sm" id="expandAllBtn"><i class="bi bi-arrows-expand"></i> Expand</button>
            <button class="side-btn btn-sm" id="collapseAllBtn"><i class="bi bi-arrows-collapse"></i> Collapse</button>
            <button class="side-btn btn-sm" type="button" id="singleViewToggle" aria-pressed="false">
              <i class="bi bi-view-stacked"></i> Show One
            </button>
          </div>

          <div id="aggregateBlock" class="panel-2 p-2 mb-2 d-none">
            <div class="d-flex align-items-center gap-2 mb-1">
              <span class="chip toggle-chip chip-major" role="button" tabindex="0" aria-pressed="false" data-severity="major" data-active="false">Major <span id="aggMajor">0</span></span>
              <span class="chip toggle-chip chip-minor" role="button" tabindex="0" aria-pressed="false" data-severity="minor" data-active="false">Minor <span id="aggMinor">0</span></span>
              <span class="chip toggle-chip chip-clar" role="button" tabindex="0" aria-pressed="false" data-severity="clar" data-active="false">Clar <span id="aggClar">0</span></span>
            </div>
            <div class="progress" title="Percent major across all items">
              <div class="progress-bar" id="aggBar" style="width:0%"></div>
            </div>
          </div>

          <div id="caseNav" class="case-nav small"></div>

          <div class="mt-3 small text-secondary">
            <div><span class="kbd">X</span> prev • <span class="kbd">C</span> next</div>
            <div><span class="kbd">Shift+Tab</span> prev • <span class="kbd">Tab</span> next</div>
            <div><span class="kbd">1–4</span> tabs</div>
            <div><span class="kbd">F</span> focus search</div>
            <div><span class="kbd">S</span> cycle sort</div>
            <div><span class="kbd">G</span><span class="mx-1">G</span> top</div>
          </div>
        </div>
      </div>

      <!-- Main content with right border & wrapping -->
      <div>
        <div id="resultsPanel" class="panel p-2">
          <div id="emptyState" class="text-center text-secondary p-5">
            <i class="bi bi-arrow-up-right-square text-primary fs-1 d-block mb-2"></i>
            <div class="mb-1">Paste reports above and click <strong>Compare &amp; Summarize</strong>.</div>
            <div class="small">Keyboard: X/C or Tab/Shift+Tab to move, 1–4 to switch tabs.</div>
          </div>
          <div id="caseContainer" class="d-none"></div>
        </div>
      </div>
    </div>
  </div>

  <div class="toaster" id="toaster"></div>

  <script>
    let caseData = {{ case_data | tojson }};
    if (!Array.isArray(caseData)) caseData = [];
    const navEl = document.getElementById('caseNav');
    const containerEl = document.getElementById('caseContainer');
    const emptyStateEl = document.getElementById('emptyState');
    const caseCountBadge = document.getElementById('caseCountBadge');
    const searchInput = document.getElementById('searchInput');
    const toaster = document.getElementById('toaster');
    const loadingBar = document.getElementById('loadingBar');
    const overlayEl = document.getElementById('loading-overlay');
    let waitTimerInterval = null;
    let waitStart = null;
    
    function formatDuration(ms){
      const s = Math.floor(ms / 1000);
      const mm = String(Math.floor(s / 60)).padStart(2, '0');
      const ss = String(s % 60).padStart(2, '0');
      return `${mm}:${ss}`;
    }

    function showOverlay(msg) {
      const msgEl = document.getElementById('loadingMsg');
      if (msgEl && msg) msgEl.textContent = msg;
    
      const timerEl = document.getElementById('waitTimer');
      if (timerEl) timerEl.textContent = '00:00';
    
      waitStart = Date.now();
      if (waitTimerInterval) clearInterval(waitTimerInterval);
      waitTimerInterval = setInterval(() => {
        const timerEl = document.getElementById('waitTimer');
        if (timerEl) timerEl.textContent = formatDuration(Date.now() - waitStart);
      }, 1000);
    
      overlayEl.style.display = 'flex';
      overlayEl.setAttribute('aria-hidden', 'false');
    }
    
    function hideOverlay() {
      overlayEl.style.display = 'none';
      overlayEl.setAttribute('aria-hidden', 'true');
      if (waitTimerInterval) {
        clearInterval(waitTimerInterval);
        waitTimerInterval = null;
      }
    }

    const aggregateBlock = document.getElementById('aggregateBlock');
    const aggMajor = document.getElementById('aggMajor');
    const aggMinor = document.getElementById('aggMinor');
    const aggClar = document.getElementById('aggClar');
    const aggBar = document.getElementById('aggBar');
    const singleViewToggle = document.getElementById('singleViewToggle');
    const severityChips = document.querySelectorAll('.toggle-chip');
    const activeSeverities = new Set();
    let currentSearchTerm = '';
    let selectedCaseId = null;
    let showSingleMode = false;
    let currentIndex = 0;
    let lastActiveTabKey = 'sum';

    const gridLayout = document.getElementById('gridLayout');

    // Full sidebar controls
    const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
    const sortBtn = document.getElementById('sortCaseBtn');
    const sortModeLabel = document.getElementById('sortModeLabel');

    // Rail controls (collapsed)
    const toggleSidebarBtnRail = document.getElementById('toggleSidebarBtnRail');
    const sortBtnRail = document.getElementById('sortCaseBtnRail');

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

    function updateSeverityChipStyles() {
      severityChips.forEach(chip => {
        const sev = chip.dataset.severity;
        const active = activeSeverities.has(sev);
        chip.setAttribute('data-active', active ? 'true' : 'false');
        chip.setAttribute('aria-pressed', active ? 'true' : 'false');
      });
    }

    function updateAggregateTotals() {
      if (!caseData || !caseData.length) {
        aggMajor.textContent = '0';
        aggMinor.textContent = '0';
        aggClar.textContent = '0';
        aggBar.style.width = '0%';
        aggregateBlock.classList.add('d-none');
        return;
      }

      let M = 0, m = 0, c = 0;
      caseData.forEach(d => {
        if (!d.summary) return;
        M += (d.summary.major_findings || []).length;
        m += (d.summary.minor_findings || []).length;
        c += (d.summary.clarifications || []).length;
      });
      const total = M + m + c;
      aggMajor.textContent = M;
      aggMinor.textContent = m;
      aggClar.textContent = c;
      aggBar.style.width = total ? Math.min(100, Math.round((M / Math.max(1, total)) * 100)) + '%' : '0%';
      aggregateBlock.classList.remove('d-none');
    }

    // ---------- Sidebar collapse/expand ----------
    function setCollapsed(collapsed){
      gridLayout.classList.toggle('collapsed', collapsed);
      toggleSidebarBtn.setAttribute('aria-pressed', String(collapsed));
      toggleSidebarBtnRail.setAttribute('aria-pressed', String(!collapsed));
    }
    toggleSidebarBtn.addEventListener('click', () => setCollapsed(true));
    toggleSidebarBtnRail.addEventListener('click', () => setCollapsed(false));

    // ---------- Sort button visual state (sync full + rail) ----------
    function applySortVisual(mode) {
      // full button
      sortBtn.setAttribute('data-mode', mode);
      sortBtn.classList.add('active');
      sortBtn.innerHTML = '<i class="bi '+sortIcons[mode]+' me-1"></i> Sort: <span class="mode" id="sortModeLabel">'+sortLabels[mode]+'</span>';
      // rail button
      sortBtnRail.setAttribute('data-mode', mode);
      sortBtnRail.innerHTML = '<i class="bi '+sortIcons[mode]+'"></i>';
    }

    function cycleSort(currentMode){
      const idx = sortModes.indexOf(currentMode);
      const next = sortModes[(idx+1)%sortModes.length];
      sortBy(next);
      applySortVisual(next);
      rerender();
    }

    sortBtn.addEventListener('click', () => {
      const current = sortBtn.getAttribute('data-mode') || 'number';
      cycleSort(current);
    });
    sortBtnRail.addEventListener('click', () => {
      const current = sortBtnRail.getAttribute('data-mode') || 'number';
      cycleSort(current);
    });

    // ---------- Navigation render ----------
    function formatNavRow(c) {
      const score = (c.summary && c.summary.score) || 0;
      return `
        <a href="#case${c.case_num}">
          <div class="me-auto">
            <div><strong>#${c.case_num}</strong></div>
            <div class="meta">Δ ${c.percentage_change}% • Score ${score}</div>
          </div>
        </a>
      `;
    }

    function renderNav(list) {
      navEl.innerHTML = list.map(formatNavRow).join('');
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

      const majorsList = majors.length
        ? `<ul class="mb-0">${majors.map(x => renderFindingItem(x, c.case_num, 'major')).join('')}</ul>`
        : '<div class="text-secondary small">None</div>';
      const minorsList = minors.length
        ? `<ul class="mb-0">${minors.map(x => renderFindingItem(x, c.case_num, 'minor')).join('')}</ul>`
        : '<div class="text-secondary small">None</div>';
      const clarList = clar.length
        ? `<ul class="mb-0">${clar.map(x => renderFindingItem(x, c.case_num, 'clar')).join('')}</ul>`
        : '<div class="text-secondary small">None</div>';

      return `
        <div id="case${c.case_num}" class="case-card panel-2 p-3 mb-3" data-major-count="${majors.length}" data-minor-count="${minors.length}" data-clar-count="${clar.length}">
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
                <button class="nav-link active" id="tab-sum-${c.case_num}" data-tab-key="sum" data-bs-toggle="tab" data-bs-target="#tab-pane-sum-${c.case_num}" type="button" role="tab">Summary</button>
              </li>
              <li class="nav-item" role="presentation">
                <button class="nav-link" id="tab-combo-${c.case_num}" data-tab-key="combo" data-bs-toggle="tab" data-bs-target="#tab-pane-combo-${c.case_num}" type="button" role="tab">Combined Diff</button>
              </li>
              <li class="nav-item" role="presentation">
                <button class="nav-link" id="tab-res-${c.case_num}" data-tab-key="res" data-bs-toggle="tab" data-bs-target="#tab-pane-res-${c.case_num}" type="button" role="tab">Resident</button>
              </li>
              <li class="nav-item" role="presentation">
                <button class="nav-link" id="tab-att-${c.case_num}" data-tab-key="att" data-bs-toggle="tab" data-bs-target="#tab-pane-att-${c.case_num}" type="button" role="tab">Attending</button>
              </li>
            </ul>

            <div class="tab-content p-2">
              <div class="tab-pane fade show active" id="tab-pane-sum-${c.case_num}" role="tabpanel" tabindex="0">
                <div class="row g-2">
                  <div class="col-12 col-lg-4">
                    <div class="panel p-2 h-100 severity-block severity-major">
                      <div class="d-flex align-items-center gap-2 mb-2"><i class="bi bi-exclamation-octagon text-danger"></i><strong>Major</strong></div>
                      ${majorsList}
                    </div>
                  </div>
                  <div class="col-12 col-lg-4">
                    <div class="panel p-2 h-100 severity-block severity-minor">
                      <div class="d-flex align-items-center gap-2 mb-2"><i class="bi bi-info-circle text-warning"></i><strong>Minor</strong></div>
                      ${minorsList}
                    </div>
                  </div>
                  <div class="col-12 col-lg-4">
                    <div class="panel p-2 h-100 severity-block severity-clar">
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

    function updateSingleToggleButton() {
      if (!singleViewToggle) return;
      if (showSingleMode) {
        singleViewToggle.innerHTML = '<i class="bi bi-collection me-1"></i> Show All';
      } else {
        singleViewToggle.innerHTML = '<i class="bi bi-view-stacked me-1"></i> Show One';
      }
      singleViewToggle.setAttribute('aria-pressed', showSingleMode ? 'true' : 'false');
    }

    function updateSelectedCardVisual() {
      const targetId = selectedCaseId ? `case${selectedCaseId}` : null;
      document.querySelectorAll('.case-card').forEach(card => {
        const isSelected = targetId && card.id === targetId;
        card.classList.toggle('selected', Boolean(isSelected));
        card.classList.toggle('single-hidden', showSingleMode && !isSelected);
      });
    }

    function setSelectedCard(card, { scroll = false } = {}) {
      if (!card) return;
      const cards = Array.from(document.querySelectorAll('.case-card'));
      const idx = cards.indexOf(card);
      if (idx === -1) return;

      selectedCaseId = card.id.replace('case', '');
      currentIndex = idx;
      updateSelectedCardVisual();
      activateTabForCase(selectedCaseId, lastActiveTabKey);

      if (scroll) {
        card.scrollIntoView({ behavior: 'smooth', block: showSingleMode ? 'center' : 'start' });
      }
    }

    function ensureSelectedCard() {
      const cards = Array.from(document.querySelectorAll('.case-card'));
      if (!cards.length) {
        selectedCaseId = null;
        currentIndex = 0;
        updateSelectedCardVisual();
        return;
      }

      const target = selectedCaseId ? cards.find(card => card.id === `case${selectedCaseId}`) : cards[0];
      setSelectedCard(target || cards[0], { scroll: false });
    }

    function attachCaseCardListeners() {
      document.querySelectorAll('.case-card').forEach(card => {
        card.addEventListener('click', () => setSelectedCard(card));
      });
    }

    // ---------- Rendering ----------
    function renderAll(data) {
      updateAggregateTotals();

      if (!caseData || !caseData.length) {
        containerEl.classList.add('d-none');
        emptyStateEl.classList.remove('d-none');
        renderNav([]);
        ensureSelectedCard();
        updateSingleToggleButton();
        hideOverlay();
        return;
      }

      emptyStateEl.classList.add('d-none');
      containerEl.classList.remove('d-none');

      if (!data || data.length === 0) {
        containerEl.innerHTML = '<div class="panel-2 p-3 mb-3 text-secondary">No cases match your current filters.</div>';
        renderNav([]);
        ensureSelectedCard();
        updateSingleToggleButton();
        endLoading();
        hideOverlay();
        return;
      }

      midLoading();
      containerEl.innerHTML = data.map(caseCardHTML).join('');
      renderNav(data);
      attachCaseCardListeners();
      ensureSelectedCard();
      updateSingleToggleButton();
      endLoading();
      hideOverlay();
    }

    function applySeverityHighlights() {
      const highlightMajor = activeSeverities.has('major');
      const highlightMinor = activeSeverities.has('minor');
      const highlightClar = activeSeverities.has('clar');

      document.querySelectorAll('.case-card').forEach(card => {
        const majorCount = parseInt(card.getAttribute('data-major-count') || '0', 10);
        const minorCount = parseInt(card.getAttribute('data-minor-count') || '0', 10);
        const clarCount = parseInt(card.getAttribute('data-clar-count') || '0', 10);

        const majorBlock = card.querySelector('.severity-block.severity-major');
        const minorBlock = card.querySelector('.severity-block.severity-minor');
        const clarBlock = card.querySelector('.severity-block.severity-clar');

        if (majorBlock) majorBlock.classList.toggle('highlight-major', highlightMajor && majorCount > 0);
        if (minorBlock) minorBlock.classList.toggle('highlight-minor', highlightMinor && minorCount > 0);
        if (clarBlock) clarBlock.classList.toggle('highlight-clar', highlightClar && clarCount > 0);
      });
    }

    function computeVisibleData() {
      let list = [...caseData];
      if (currentSearchTerm) {
        list = searchFilter(list, currentSearchTerm);
      }
      if (activeSeverities.size > 0) {
        list = list.filter(c => {
          if (!c.summary) return false;
          const s = c.summary;
          const hasMajor = (s.major_findings || []).length > 0;
          const hasMinor = (s.minor_findings || []).length > 0;
          const hasClar = (s.clarifications || []).length > 0;
          return (
            (activeSeverities.has('major') && hasMajor) ||
            (activeSeverities.has('minor') && hasMinor) ||
            (activeSeverities.has('clar') && hasClar)
          );
        });
      }
      return list;
    }

    function rerender() {
      const visible = computeVisibleData();
      renderAll(visible);
      updateSeverityChipStyles();
      applySeverityHighlights();
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
    document.addEventListener('shown.bs.tab', (event) => {
      const target = event.target;
      if (target && target.dataset && target.dataset.tabKey) {
        lastActiveTabKey = target.dataset.tabKey;
      }
    });

    document.addEventListener('DOMContentLoaded', () => {
      updateSingleToggleButton();
      rerender();
      applySortVisual('number'); // initialize sort visual
      setCollapsed(false);       // start expanded
    });

    document.getElementById('reportForm').addEventListener('submit', () => {
      startLoading();
      showOverlay('Analyzing reports… this may take a moment.');
    });

    document.getElementById('clearBtn').addEventListener('click', () => {
      document.getElementById('report_text').value = '';
      document.getElementById('custom_prompt').value = '';
      caseData = [];
      currentSearchTerm = '';
      searchInput.value = '';
      activeSeverities.clear();
      rerender();
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

    searchInput.addEventListener('input', () => {
      currentSearchTerm = searchInput.value;
      rerender();
    });

    function toggleSeverityFilter(sev) {
      if (!sev) return;
      if (activeSeverities.has(sev)) {
        activeSeverities.delete(sev);
      } else {
        activeSeverities.add(sev);
      }
      rerender();
    }

    severityChips.forEach(chip => {
      chip.addEventListener('click', () => toggleSeverityFilter(chip.dataset.severity));
      chip.addEventListener('keydown', (evt) => {
        if (evt.key === 'Enter' || evt.key === ' ') {
          evt.preventDefault();
          toggleSeverityFilter(chip.dataset.severity);
        }
      });
    });

    document.getElementById('expandAllBtn').addEventListener('click', () => {
      document.querySelectorAll('[id^="body"]').forEach(el => el.style.display = '');
    });
    document.getElementById('collapseAllBtn').addEventListener('click', () => {
      document.querySelectorAll('[id^="body"]').forEach(el => el.style.display = 'none');
    });

    if (singleViewToggle) {
      singleViewToggle.addEventListener('click', () => {
        showSingleMode = !showSingleMode;
        ensureSelectedCard();
        updateSingleToggleButton();
      });
    }

    navEl.addEventListener('click', (evt) => {
      const link = evt.target.closest('a[href^="#case"]');
      if (!link) return;
      const id = link.getAttribute('href').replace('#case', '');
      const card = document.getElementById(`case${id}`);
      if (card) {
        setSelectedCard(card, { scroll: true });
      }
    });

    document.getElementById('downloadAllBtn').addEventListener('click', () => {
      if (!caseData || caseData.length === 0) {
        toast('Nothing to download yet', 2000);
        return;
      }

      const promptField = document.getElementById('custom_prompt');
      const exportPayload = {
        version: 'compare-revisions.v1',
        exported_at: new Date().toISOString(),
        custom_prompt: promptField ? promptField.value : null,
        cases: caseData.map(c => ({
          case_num: c.case_num,
          case_number: c.case_num,
          resident_report: c.resident_report,
          attending_report: c.attending_report,
          percentage_change: c.percentage_change,
          diff: c.diff,
          combined_diff_html: c.diff,
          summary: c.summary,
          summary_error: c.summary_error ?? null
        }))
      };

      const blob = new Blob([JSON.stringify(exportPayload, null, 2)], {type: 'application/json'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'compare-revisions-session.json'; a.click();
      URL.revokeObjectURL(url);
    });

    // ---------- NEW: Upload JSON handlers (added) ----------
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadInput = document.getElementById('uploadInput');

    uploadBtn.addEventListener('click', () => uploadInput.click());

    uploadInput.addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      try {
        startLoading();
        showOverlay('Loading JSON…');

        const text = await file.text();
        const parsed = JSON.parse(text);

        // Accept either the original summaries-only export (array) or a richer object with "cases"
        function normalizeFromSummariesOnly(arr) {
          if (!Array.isArray(arr)) return null;
          return arr.map(x => ({
            case_num: String(x.case_number ?? ''),
            resident_report: '',
            attending_report: '',
            percentage_change: 0,
            diff: '',
            summary: x.summary || null,
            summary_error: x.error || null
          }));
        }
        function normalizeFromFullState(obj) {
          if (!obj || !Array.isArray(obj.cases)) return null;
          return obj.cases.map(c => ({
            case_num: String(c.case_num ?? c.case_number ?? ''),
            resident_report: c.resident_report || '',
            attending_report: c.attending_report || '',
            percentage_change: typeof c.percentage_change === 'number' ? c.percentage_change : 0,
            diff: c.diff || c.combined_diff_html || '',
            summary: c.summary || null,
            summary_error: c.summary_error || null
          }));
        }

        let restored = null;
        if (Array.isArray(parsed)) {
          restored = normalizeFromSummariesOnly(parsed);
        } else if (parsed && typeof parsed === 'object') {
          if (Array.isArray(parsed.cases)) {
            restored = normalizeFromFullState(parsed);
            if (parsed.custom_prompt != null) {
              document.getElementById('custom_prompt').value = parsed.custom_prompt;
            }
          } else if (Array.isArray(parsed.summariesOnly)) {
            restored = normalizeFromSummariesOnly(parsed.summariesOnly);
          }
        }

        if (!restored) {
          toast('Unrecognized JSON format', 2200);
          hideOverlay(); endLoading();
          e.target.value = '';
          return;
        }

        caseData = restored;
        sortBy('number');
        applySortVisual('number');
        currentSearchTerm = '';
        searchInput.value = '';
        activeSeverities.clear();
        rerender();
        toast('Upload complete');
      } catch (err) {
        console.error(err);
        toast('Failed to load JSON', 2200);
      } finally {
        hideOverlay();
        endLoading();
        e.target.value = '';
      }
    });
    // ---------- END NEW ----------

    // ---------- Utils ----------
    function toggleCollapse(id) {
      const el = document.getElementById('body'+id);
      if (!el) return;
      el.style.display = (el.style.display === 'none') ? '' : 'none';
    }
    window.toggleCollapse = toggleCollapse;

    function copyJSON(text) {
      navigator.clipboard.writeText(text)
        .then(()=> toast('JSON copied'))
        .catch(()=> toast('Copy failed', 2000));
    }
    window.copyJSON = copyJSON;

    function escapeHTML(str) {
      if (!str) return '';
      return str.replace(/[&<>"']/g, (m) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#39;'}[m]));
    }
    window.escapeHTML = escapeHTML;

    function applyExplicitHighlights(text, highlightPhrases) {
      if (!text) return '';
      if (!Array.isArray(highlightPhrases) || !highlightPhrases.length) {
        return escapeHTML(text);
      }

      let result = escapeHTML(text);

      const sorted = [...highlightPhrases]
        .filter(Boolean)
        .map(p => p.trim())
        .filter(p => p.length > 0)
        .sort((a, b) => b.length - a.length);

      for (const phrase of sorted) {
        const pattern = phrase.replace(/[.*+?^${}()|[\]\\]/g, '\$&');
        const re = new RegExp(pattern, 'gi');

        result = result.replace(re, (match) => {
          if (match.includes('key-change')) return match;
          return `<span class="key-change">${escapeHTML(match)}</span>`;
        });
      }

      return result;
    }

    function renderFindingItem(finding, caseNum, severity) {
      if (typeof finding === 'string') {
        return `
      <li
        class="finding-item"
        data-case-num="${caseNum}"
        data-severity="${severity}"
        data-mode="legacy"
      >
        ${escapeHTML(finding)}
      </li>
    `;
      }

      const text = finding.summary_text || finding.attending_snippet || '';
      const highlights = Array.isArray(finding.attending_highlights)
        ? finding.attending_highlights
        : [];

      const rendered = applyExplicitHighlights(text, highlights);

      const resSnippet = finding.resident_snippet || '';
      const attSnippet = finding.attending_snippet || '';

      return `
    <li
      class="finding-item"
      data-case-num="${caseNum}"
      data-severity="${severity}"
      data-mode="structured"
      data-res="${escapeHTML(resSnippet)}"
      data-att="${escapeHTML(attSnippet)}"
    >
      ${rendered}
    </li>
  `;
    }

    // ---------- Keyboard shortcuts ----------
    function focusCase(i) {
      const cards = Array.from(document.querySelectorAll('.case-card'));
      if (!cards.length) return;
      const bounded = Math.max(0, Math.min(i, cards.length - 1));
      const card = cards[bounded];
      setSelectedCard(card, { scroll: true });
      card.classList.add('ring');
      setTimeout(()=> card.classList.remove('ring'), 600);
    }
    function switchTab(n) {
      const list = document.querySelectorAll('.case-card');
      if (!list.length) return;
      const id = list[currentIndex].id.replace('case','');
      const tabIds = ['sum','combo','res','att'];
      const which = tabIds[n-1] || 'sum';
      const btn = document.getElementById(`tab-${which}-${id}`);
      if (btn) {
        lastActiveTabKey = which;
        btn.click();
      }
    }

    function activateTabForCase(caseId, tabKey) {
      if (!caseId || !tabKey) return;
      const btn = document.getElementById(`tab-${tabKey}-${caseId}`);
      if (!btn) return;
      lastActiveTabKey = tabKey;
      if (!btn.classList.contains('active')) {
        btn.click();
      }
    }

    let gPressedOnce = false;
    document.addEventListener('keydown', (e) => {
      if (['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) {
        if (e.key.toLowerCase()==='escape') document.activeElement.blur();
        return;
      }
      const key = e.key.toLowerCase();

      if (key === 'tab') {
        if (!e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          if (e.shiftKey) {
            focusCase(currentIndex-1);
          } else {
            focusCase(currentIndex+1);
          }
          return;
        }
      }

      const hasModifier = e.ctrlKey || e.metaKey || e.altKey;
      if (!hasModifier && key==='c') {
        e.preventDefault(); focusCase(currentIndex+1);
      } else if (!hasModifier && key==='x') {
        e.preventDefault(); focusCase(currentIndex-1);
      } else if (['1','2','3','4'].includes(e.key)) {
        e.preventDefault(); switchTab(parseInt(e.key,10));
      } else if (!hasModifier && key==='f') {
        e.preventDefault(); document.getElementById('searchInput').focus();
      } else if (!hasModifier && key==='s') {
        e.preventDefault();
        // trigger whichever sort control is visible
        if (gridLayout.classList.contains('collapsed')) sortBtnRail.click(); else sortBtn.click();
      } else if (!hasModifier && key==='g') {
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
