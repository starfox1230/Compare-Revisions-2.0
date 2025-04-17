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
    level=logging.INFO,  # Changed to INFO for production, DEBUG for troubleshooting
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI API key
# Ensure the OPENAI_API_KEY environment variable is set
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set or empty.")
    logger.info("OpenAI client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    # Depending on your application's needs, you might want to exit or handle this differently
    # For now, we'll let it proceed but log the error. API calls will fail later.
    client = None # Set client to None to indicate failure


# Default customized prompt
DEFAULT_PROMPT = """Succinctly organize the changes made by the attending to the resident's radiology reports into:
1) missed major findings (findings discussed by the attending but not by the resident that also fit under these categories: retained sponge or other clinically significant foreign body, mass or tumor, malpositioned line or tube of immediate clinical concern, life-threatening hemorrhage or vascular disruption, necrotizing fasciitis, free air or active leakage from the GI tract, ectopic pregnancy, intestinal ischemia or portomesenteric gas, ovarian torsion, testicular torsion, placental abruption, absent perfusion in a postoperative transplant, renal collecting system obstruction with signs of infection, acute cholecystitis, intracranial hemorrhage, midline shift, brain herniation, cerebral infarction or abscess or meningoencephalitis, airway compromise, abscess or discitis, hemorrhage, cord compression or unstable spine fracture or transection, acute cord hemorrhage or infarct, pneumothorax, large pericardial effusion, findings suggestive of active TB, impending pathologic fracture, acute fracture, absent perfusion in a postoperative kidney, brain death, high probability ventilation/perfusion (VQ) lung scan, arterial dissection or occlusion, acute thrombotic or embolic event including DVT and pulmonary thromboembolism, and aneurysm or vascular disruption),
2) missed minor findings (this includes all other pathologies not in the above list that were discussed by the attending but not by the resident), and
3) clarified descriptions of findings (this includes findings removed by the attending, findings re-worded by the attending, etc).
Assume the attending's version was correct, and anything not included by the attending but was included by the resident should have been left out by the resident. Keep your answers brief and to the point. The reports are:
Please output your response as structured JSON, with the following keys:
- "case_number": The case number sent in the request (as a string).
- "major_findings": A list of any major findings missed by the resident (as described above).
- "minor_findings": A list of minor findings discussed by the attending but not by the resident (as described above).
- "clarifications": A list of any clarifications the attending made (as described above).
- "score": The calculated score, where each major finding is worth 3 points and each minor finding is worth 1 point.
Respond ONLY in this JSON format, with no other additional text, explanations, or pleasantries:
{
  "case_number": "<case_number>",
  "major_findings": ["<major_finding_1>", "<major_finding_2>", ...],
  "minor_findings": ["<minor_finding_1>", "<minor_finding_2>", ...],
  "clarifications": ["<clarification_1>", "<clarification_2>", ...],
  "score": <score>
}
"""

# Default model constant
DEFAULT_MODEL = "gpt-4o-mini" # As requested

# --- Text Processing Functions ---

def normalize_text(text):
    """Trim spaces but keep returns (newlines) intact."""
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def remove_attending_review_line(text):
    """Remove standard attending review sign-off lines."""
    excluded_lines = [
        "As the attending physician, I have personally reviewed the images, interpreted and/or supervised the study or procedure, and agree with the wording of the above report.",
        "As the Attending radiologist, I have personally reviewed the images, interpreted the study, and agree with the wording of the above report by Sterling M. Jones",
        # Add any other variations of the sign-off line here
    ]
    # Case-insensitive and whitespace-flexible comparison
    return "\n".join([line for line in text.splitlines() if line.strip().lower() not in [ex.lower() for ex in excluded_lines]])

def calculate_change_percentage(resident_text, attending_text):
    """Calculate percentage change between two reports based on word sequences."""
    # Ensure inputs are strings, handle None gracefully
    res_text = resident_text or ""
    att_text = attending_text or ""
    matcher = difflib.SequenceMatcher(None, res_text.split(), att_text.split())
    return round((1 - matcher.ratio()) * 100, 2)

def split_into_paragraphs(text):
    """Split text into paragraphs based on significant line breaks."""
    # Split on one or more blank lines (more robust than just double newlines)
    paragraphs = re.split(r'\n\s*\n+', text)
    return [para.strip() for para in paragraphs if para.strip()]

def create_diff_by_section(resident_text, attending_text):
    """Compare reports section by section (paragraph level first, then word level)."""
    # Normalize text for comparison
    norm_resident_text = normalize_text(resident_text or "") # Handle None
    # Remove sign-off before normalizing for diffing
    norm_attending_text = normalize_text(remove_attending_review_line(attending_text or "")) # Handle None

    resident_paragraphs = split_into_paragraphs(norm_resident_text)
    attending_paragraphs = split_into_paragraphs(norm_attending_text)

    diff_html = ""
    matcher = difflib.SequenceMatcher(None, resident_paragraphs, attending_paragraphs)

    for opcode, a1, a2, b1, b2 in matcher.get_opcodes():
        if opcode == 'equal':
            for paragraph in resident_paragraphs[a1:a2]:
                diff_html += paragraph.replace('\n', '<br>') + "<br><br>" # Preserve line breaks within paragraphs
        elif opcode == 'insert':
            for paragraph in attending_paragraphs[b1:b2]:
                diff_html += f'<div style="color:lightgreen; background-color: #284028; padding: 5px; border-radius: 4px;">[Inserted Paragraph:]<br>{paragraph.replace("\n", "<br>")}</div><br><br>'
        elif opcode == 'delete':
            for paragraph in resident_paragraphs[a1:a2]:
                diff_html += f'<div style="color:#ff6b6b; background-color: #4a2a2a; text-decoration:line-through; padding: 5px; border-radius: 4px;">[Deleted Paragraph:]<br>{paragraph.replace("\n", "<br>")}</div><br><br>'
        elif opcode == 'replace':
            # More robust handling for unequal number of paragraphs in replace block
            max_len = max(a2 - a1, b2 - b1)
            res_pars = resident_paragraphs[a1:a2]
            att_pars = attending_paragraphs[b1:b2]

            for i in range(max_len):
                res_paragraph = res_pars[i] if i < len(res_pars) else ""
                att_paragraph = att_pars[i] if i < len(att_pars) else ""

                if not res_paragraph: # Only insertion happened at the end of the block
                     diff_html += f'<div style="color:lightgreen; background-color: #284028; padding: 5px; border-radius: 4px;">[Inserted Paragraph:]<br>{att_paragraph.replace("\n", "<br>")}</div><br><br>'
                     continue
                if not att_paragraph: # Only deletion happened at the end of the block
                     diff_html += f'<div style="color:#ff6b6b; background-color: #4a2a2a; text-decoration:line-through; padding: 5px; border-radius: 4px;">[Deleted Paragraph:]<br>{res_paragraph.replace("\n", "<br>")}</div><br><br>'
                     continue

                # Both paragraphs exist, perform word-level diff
                paragraph_diff_html = ""
                # Ensure paragraphs are strings before splitting
                res_words = res_paragraph.split() if res_paragraph else []
                att_words = att_paragraph.split() if att_paragraph else []
                word_matcher = difflib.SequenceMatcher(None, res_words, att_words)

                for word_opcode, w_a1, w_a2, w_b1, w_b2 in word_matcher.get_opcodes():
                    if word_opcode == 'equal':
                        paragraph_diff_html += " ".join(res_words[w_a1:w_a2]) + " "
                    elif word_opcode == 'replace':
                        paragraph_diff_html += (
                            '<span style="color:#ff6b6b; background-color: #4a2a2a; text-decoration:line-through;">' +
                            " ".join(res_words[w_a1:w_a2]) +
                            '</span> <span style="color:lightgreen; background-color: #284028;">' +
                            " ".join(att_words[w_b1:w_b2]) +
                            '</span> '
                        )
                    elif word_opcode == 'delete':
                         paragraph_diff_html += (
                            '<span style="color:#ff6b6b; background-color: #4a2a2a; text-decoration:line-through;">' +
                            " ".join(res_words[w_a1:w_a2]) +
                            '</span> '
                        )
                    elif word_opcode == 'insert':
                         paragraph_diff_html += (
                            '<span style="color:lightgreen; background-color: #284028;">' +
                            " ".join(att_words[w_b1:w_b2]) +
                            '</span> '
                        )

                # Add the diff for this paragraph pair, preserving internal line breaks
                diff_html += paragraph_diff_html.replace('\n', '<br>') + "<br><br>"

    return diff_html


# --- OpenAI Interaction Functions ---

def get_summary(case_text, custom_prompt, case_number, model): # Accepts model
    """Get a structured JSON summary of report differences using the specified model."""
    if not client:
        logger.error(f"OpenAI client not initialized. Cannot process case {case_number}.")
        return {"case_number": case_number, "error": "OpenAI client not initialized."}

    try:
        logger.info(f"Processing case {case_number} with model {model}") # Logs the model used
        response = client.chat.completions.create(
            model=model, # Use the passed-in model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs structured JSON summaries of radiology report differences, following the requested format exactly."},
                {"role": "user", "content": f"{custom_prompt}\nCase Number: {case_number}\n{case_text}"}
            ],
            max_tokens=2000,
            temperature=0.5,
            response_format={ "type": "json_object" } # Enforce JSON output if model supports it
        )
        response_content = response.choices[0].message.content
        logger.debug(f"Raw response for case {case_number} from model {model}: {response_content}")

        # Attempt to parse the JSON
        parsed_json = json.loads(response_content)

        # Basic validation of the parsed JSON structure
        if not isinstance(parsed_json, dict):
             raise json.JSONDecodeError("Response is not a JSON object.", response_content, 0)
        if "case_number" not in parsed_json:
             logger.warning(f"AI response for case {case_number} missing 'case_number'. Adding it back.")
             parsed_json["case_number"] = case_number # Ensure case_number is present
        else:
             # Ensure case_number from AI matches request, log if different
             ai_case_num = str(parsed_json.get("case_number"))
             if ai_case_num != case_number:
                 logger.warning(f"Case number mismatch for request {case_number}. AI returned {ai_case_num}. Using requested number.")
                 parsed_json["case_number"] = case_number # Standardize to requested number

        # Ensure score calculation if 'score' key is missing or invalid
        if 'score' not in parsed_json or not isinstance(parsed_json.get('score'), (int, float)):
            major = len(parsed_json.get('major_findings', []))
            minor = len(parsed_json.get('minor_findings', []))
            parsed_json['score'] = (major * 3) + minor
            logger.info(f"Calculated score for case {case_number} as {parsed_json['score']} (Major: {major}, Minor: {minor}).")

        logger.info(f"Successfully parsed summary for case {case_number} using model {model}.")
        return parsed_json

    except json.JSONDecodeError as jde:
        logger.error(f"JSON decode error for case {case_number} on model {model}: {jde}. Response: {response_content}")
        # Try to extract useful info even from broken JSON if possible, or return error
        return {"case_number": case_number, "error": f"Invalid JSON response from AI: {jde}. Response was: {response_content}", "raw_response": response_content}
    except Exception as e:
        # Catch potential API errors or other issues
        logger.error(f"Error processing case {case_number} with model {model}: {e}") # Logs the model used
        # Check if the error is specific (e.g., API connection error, rate limit)
        error_message = f"Error processing case {case_number}: {type(e).__name__} - {e}"
        # Log the full traceback for detailed debugging if needed
        # logger.exception(f"Full traceback for error on case {case_number}")
        return {"case_number": case_number, "error": error_message}

def process_cases(cases_data, custom_prompt, model, max_workers=10): # Accepts model
    """Process multiple cases concurrently using the specified model."""
    structured_output = []
    if not cases_data:
        logger.warning("No cases data provided to process_cases.")
        return []

    # Reduce default max_workers to avoid overwhelming API limits unless specifically needed
    logger.info(f"Starting concurrent processing for {len(cases_data)} cases with model {model} (max_workers={max_workers}).") # Logs the model used

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all summary tasks to the executor, passing the model
        future_to_case = {
            executor.submit(get_summary, case_text, custom_prompt, case_num, model): case_num # Passes model
            for case_text, case_num in cases_data
        }
        logger.debug(f"Submitted {len(future_to_case)} tasks to ThreadPoolExecutor.")

        # Process results as they complete
        for future in as_completed(future_to_case):
            case_num = future_to_case[future]
            try:
                result_json = future.result()
                if not result_json:
                    # Handle cases where get_summary might return None or empty dict unexpectedly
                    logger.error(f"Received empty result for case {case_num}. Creating error entry.")
                    result_json = {"case_number": case_num, "error": "Received empty result from processing."}
                elif 'error' in result_json:
                     logger.warning(f"Processed case {case_num} resulted in an error: {result_json['error']}")
                else:
                     # Ensure score exists, calculate if needed (double-check)
                     if 'score' not in result_json or not isinstance(result_json.get('score'), (int, float)):
                        major = len(result_json.get('major_findings', []))
                        minor = len(result_json.get('minor_findings', []))
                        result_json['score'] = (major * 3) + minor
                        logger.info(f"Re-calculated score for case {case_num} as {result_json['score']}.")
                     logger.info(f"Successfully processed summary for case {case_num}. Score: {result_json.get('score', 'N/A')}")

                # Ensure the case_number in the result is a string for consistency
                result_json['case_number'] = str(result_json.get('case_number', case_num))
                structured_output.append(result_json)

            except Exception as e:
                # Catch errors from future.result() itself (though get_summary should catch most)
                logger.error(f"Critical error retrieving result for case {case_num}: {e}", exc_info=True)
                structured_output.append({"case_number": str(case_num), "error": f"Failed to retrieve result: {e}"})

    logger.info(f"Completed processing {len(structured_output)} summaries using model {model}.") # Logs the model used
    # Sort results by case number numerically before returning for consistent order
    structured_output.sort(key=lambda x: int(x.get('case_number', 0)))
    return structured_output

# --- Case Extraction and Orchestration ---

def extract_cases(text, custom_prompt, model): # Accepts model
    """Extract individual cases, generate diffs, and orchestrate AI summaries using the specified model."""
    # Normalize line endings early
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    logger.debug("Normalized line endings for the entire input text.")

    # Split on 'Case <number>' that must be at the start of a line, case-insensitive
    parts = re.split(r'(?m)^(Case\s+\d+)', text, flags=re.IGNORECASE)
    logger.debug(f"Text split into {len(parts)} parts by '^Case \\d+'.")

    cases_data_for_processing = []
    all_extracted_case_info = {} # Store raw reports mapped by case number

    if parts[0].strip():
        logger.warning(f"Ignoring text before the first 'Case' marker: {parts[0][:100]}...")

    for i in range(1, len(parts), 2):
        marker = parts[i].strip()
        case_num_match = re.search(r'\d+', marker)
        if not case_num_match:
             logger.warning(f"Could not extract number from marker: {marker}. Skipping.")
             continue
        case_num = case_num_match.group(0)

        case_content = parts[i + 1].strip() if (i + 1) < len(parts) else ""
        logger.info(f"Processing content associated with marker '{marker}' (Case Num: {case_num})")

        report_parts = re.split(r'\s*(Attending\s+Report\s*:?|Resident\s+Report\s*:?)\s*', case_content, flags=re.IGNORECASE)

        resident_report = ""
        attending_report = ""
        current_marker = None
        for part in report_parts:
             if not part or not part.strip(): continue # Skip empty/whitespace strings
             part_lower_strip = part.strip().lower()

             # Use startswith for more flexible matching (e.g., "Resident Report:")
             if part_lower_strip.startswith("resident report"):
                 current_marker = "resident"
                 # Content might start immediately after the marker on the same line
                 content_after_marker = part[len("Resident Report"):].lstrip(': \t\n\r')
                 if content_after_marker:
                     resident_report += content_after_marker.strip() + "\n"
                 logger.debug(f"Case {case_num}: Found Resident marker.")
             elif part_lower_strip.startswith("attending report"):
                 current_marker = "attending"
                 content_after_marker = part[len("Attending Report"):].lstrip(': \t\n\r')
                 if content_after_marker:
                     attending_report += content_after_marker.strip() + "\n"
                 logger.debug(f"Case {case_num}: Found Attending marker.")
             elif current_marker == "resident":
                 resident_report += part.strip() + "\n"
             elif current_marker == "attending":
                 attending_report += part.strip() + "\n"

        resident_report = resident_report.strip()
        attending_report = attending_report.strip()

        if resident_report and attending_report:
            logger.info(f"Successfully extracted Resident and Attending reports for Case {case_num}.")
            case_text_for_ai = f"Resident Report:\n{resident_report}\n\nAttending Report:\n{attending_report}"
            cases_data_for_processing.append((case_text_for_ai, case_num))
            all_extracted_case_info[case_num] = {
                'resident_report': resident_report,
                'attending_report': attending_report
            }
        else:
            logger.warning(f"Case {case_num}: Could not find both Resident and Attending reports. Resident found: {bool(resident_report)}, Attending found: {bool(attending_report)}.")
            all_extracted_case_info[case_num] = {
                 'resident_report': resident_report or "[Not Found]",
                 'attending_report': attending_report or "[Not Found]",
                 'error': 'Missing one or both reports.'
            }


    if not cases_data_for_processing:
        logger.warning("No valid cases with both reports found to send for AI processing.")
        parsed_cases_results = []
        for case_num, info in all_extracted_case_info.items():
             parsed_cases_results.append({
                'case_num': case_num,
                'resident_report': info['resident_report'],
                'attending_report': info['attending_report'],
                'percentage_change': 0,
                'diff': '<p class="error-message">Error: Could not extract both reports.</p>',
                'summary': {'case_number': case_num, 'error': info.get('error', 'Extraction failed.')}
            })
        return parsed_cases_results


    # Process all valid cases concurrently using the selected model
    logger.info(f"Sending {len(cases_data_for_processing)} cases for AI summary processing using model: {model}") # Logs model
    # Pass the selected model down
    ai_summaries = process_cases(cases_data_for_processing, custom_prompt, model, max_workers=10) # Passes model parameter


    # Combine AI summaries with original reports and diffs
    parsed_cases_results = []
    ai_summary_map = {str(summary.get('case_number')): summary for summary in ai_summaries if summary.get('case_number')}

    for case_num, info in all_extracted_case_info.items():
         case_num_str = str(case_num)

         if 'error' in info:
             parsed_cases_results.append({
                'case_num': case_num_str,
                'resident_report': info['resident_report'],
                'attending_report': info['attending_report'],
                'percentage_change': 0,
                'diff': '<p class="error-message">Error: Could not extract both reports.</p>',
                'summary': {'case_number': case_num_str, 'error': info.get('error', 'Extraction failed.')}
             })
             continue


         resident_report = info['resident_report']
         attending_report = info['attending_report']

         # Calculate diff and percentage change
         attending_for_calc = remove_attending_review_line(attending_report)
         percentage_change = calculate_change_percentage(resident_report, attending_for_calc)
         diff_html = create_diff_by_section(resident_report, attending_report)

         ai_summary = ai_summary_map.get(case_num_str)

         if ai_summary and 'error' not in ai_summary:
             logger.info(f"Successfully combined data for case {case_num_str}.")
             parsed_cases_results.append({
                'case_num': case_num_str,
                'resident_report': resident_report,
                'attending_report': attending_report,
                'percentage_change': percentage_change,
                'diff': diff_html,
                'summary': ai_summary
             })
         else:
             error_message = "AI summary processing failed."
             if ai_summary and 'error' in ai_summary:
                 error_message = f"AI Error: {ai_summary['error']}"
             elif not ai_summary:
                  error_message = "AI summary not found for this case."

             logger.warning(f"Case {case_num_str}: Combining failed. {error_message}")
             parsed_cases_results.append({
                 'case_num': case_num_str,
                 'resident_report': resident_report,
                 'attending_report': attending_report,
                 'percentage_change': percentage_change,
                 'diff': diff_html,
                 'summary': {'case_number': case_num_str, 'error': error_message, 'score': 0}
             })

    logger.info(f"Finished extracting and processing. Returning {len(parsed_cases_results)} case results.")
    parsed_cases_results.sort(key=lambda x: int(x.get('case_num', 0)))
    return parsed_cases_results


# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    custom_prompt = request.form.get('custom_prompt', DEFAULT_PROMPT)
    # Get the selected model from the form, falling back to the default
    selected_model = request.form.get('model', DEFAULT_MODEL) # Reads model from form
    case_data = []
    report_text_input = ""

    if request.method == 'POST':
        report_text_input = request.form.get('report_text', '')
        if not report_text_input.strip():
            logger.warning("Form submitted but no report text was provided.")
        else:
            logger.info(f"Starting case extraction and processing using model: {selected_model}.") # Logs model
            # Pass the selected model to extract_cases
            case_data = extract_cases(report_text_input, custom_prompt, selected_model) # Passes model
            logger.info(f"Completed case extraction and processing. Found {len(case_data)} cases.")

    logger.info(f"Rendering template with {len(case_data)} cases. Selected model: {selected_model}.") # Logs model

    # --- HTML Template String ---
    template = """
<!DOCTYPE html>
<html>
    <head>
        <title>Radiology Report Diff & Summarizer</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
        <style>
            body { background-color: #1e1e1e; color: #dcdcdc; font-family: Arial, sans-serif; }
            .container { max-width: 1200px; }
            textarea, input, button, select { background-color: #333333; color: #dcdcdc; border: 1px solid #555; }
            textarea.form-control, select.form-control { background-color: #333333 !important; color: #dcdcdc !important; border: 1px solid #555 !important; }
            textarea:focus, select:focus { background-color: #444 !important; color: #eeeeee !important; border-color: #777; box-shadow: none; }
            h2, h3, h4 { color: #f0f0f0; font-weight: normal; margin-top: 1.5rem; }
            .diff-output, .summary-output { margin-top: 15px; padding: 15px; background-color: #2e2e2e; border-radius: 8px; border: 1px solid #555; font-size: 0.9rem; }
            pre { white-space: pre-wrap; word-wrap: break-word; font-family: Consolas, 'Courier New', monospace; font-size: 0.85rem; }
            .nav-tabs { border-bottom: 1px solid #555; }
            .nav-tabs .nav-link { background-color: #333; border: 1px solid #555; border-bottom-color: transparent; color: #dcdcdc; margin-right: 2px; }
            .nav-tabs .nav-link.active { background-color: #007bff; border-color: #007bff #007bff #2e2e2e; color: white; }
            .nav-tabs .nav-link:hover { border-color: #666; background-color: #444; }
            .tab-content { border: 1px solid #555; border-top: none; padding: 15px; background-color: #2e2e2e; border-radius: 0 0 8px 8px; }
            #scrollToTopBtn {
                position: fixed; right: 20px; bottom: 20px; background-color: #007bff; color: white;
                padding: 10px 15px; border-radius: 50%; border: none; cursor: pointer; z-index: 1000;
                opacity: 0.8; transition: opacity 0.3s; font-size: 1.2rem; box-shadow: 0 2px 5px rgba(0,0,0,0.3); display: none; /* Hidden initially */
            }
            #scrollToTopBtn:hover { opacity: 1; background-color: #0056b3; }
            a { color: #66ccff; text-decoration: none; }
            a:hover { color: #99e6ff; text-decoration: underline; }
            #loadingAnimationContainer { display: none; text-align: center; margin-top: 20px; }
            #loadingMessage { font-size: 1.1rem; color: #aaa; margin-top: -20px;}
            ul { padding-left: 20px; }
            li { margin-bottom: 5px; }
            .summary-output p strong { color: #87cefa; }
            .summary-output ul { margin-top: 5px; margin-bottom: 15px; }
            span[style*="color:#ff6b6b"] { background-color: #502020; padding: 1px 3px; border-radius: 3px; }
            span[style*="color:lightgreen"] { background-color: #205020; padding: 1px 3px; border-radius: 3px; }
            div[style*="color:#ff6b6b"] { background-color: #502020; border-left: 3px solid #ff6b6b; padding: 10px; margin-bottom: 10px; }
            div[style*="color:lightgreen"] { background-color: #205020; border-left: 3px solid lightgreen; padding: 10px; margin-bottom: 10px; }
            .btn-group .btn { margin-right: 5px;}
            .error-message { color: #ff6b6b; font-weight: bold; background-color: #4a2a2a; padding: 10px; border-radius: 5px; border: 1px solid #ff6b6b; margin-top: 10px;}
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <h2>Radiology Report Diff & Summarizer</h2>
            <form method="POST" id="reportForm">
                <div class="mb-3">
                    <label for="report_text" class="form-label">Paste your reports block here (ensure format 'Case NUMBER', 'Resident Report:', 'Attending Report:'):</label>
                    <textarea id="report_text" name="report_text" class="form-control" rows="12" required>{{ report_text_input }}</textarea>
                </div>

                <!-- Model Selection Dropdown - CORRECTED AS PER INSTRUCTIONS -->
                <div class="form-group mb-3">
                  <label for="model_select">Choose OpenAI Model:</label>
                  <select id="model_select" name="model" class="form-control">
                    <option value="gpt-4o-mini" {{ 'selected' if chosen_model=='gpt-4o-mini' else '' }}>gpt-4o-mini</option>
                    <option value="gpt-4o" {{ 'selected' if chosen_model=='gpt-4o' else '' }}>gpt-4o</option>
                    <option value="gpt-4o-mini-tts" {{ 'selected' if chosen_model=='gpt-4o-mini-tts' else '' }}>gpt-4o-mini-tts</option>
                    <option value="gpt-4o-tts" {{ 'selected' if chosen_model=='gpt-4o-tts' else '' }}>gpt-4o-tts</option>
                    <option value="gpt-4.1" {{ 'selected' if chosen_model=='gpt-4.1' else '' }}>gpt-4.1</option> {# Note: gpt-4.1 might not be a standard public model name, adjust if needed #}
                    <option value="gpt-4.1-turbo" {{ 'selected' if chosen_model=='gpt-4.1-turbo' else '' }}>gpt-4.1-turbo</option> {# Note: gpt-4.1-turbo might not be a standard public model name, adjust if needed #}
                    <option value="gpt-4-turbo" {{ 'selected' if chosen_model=='gpt-4-turbo' else '' }}>gpt-4-turbo</option> {# Added common alternative #}
                    <!-- Add any other variants you need -->
                  </select>
                </div>

                <div class="mb-3">
                    <label for="custom_prompt" class="form-label">Customize OpenAI Prompt (Edit with care):</label>
                    <textarea id="custom_prompt" name="custom_prompt" class="form-control" rows="6">{{ custom_prompt }}</textarea>
                </div>

                <button type="submit" class="btn btn-primary">Compare & Summarize Reports</button>

                <!-- Loading Animation Container -->
                <div id="loadingAnimationContainer">
                    <dotlottie-player id="loadingAnimation" src="https://lottie.host/817661a8-2608-4435-89a5-daa620a64c36/WtsFI5zdEK.lottie" background="transparent" speed="1" style="width: 200px; height: 200px; margin: auto;" loop autoplay></dotlottie-player>
                    <p id="loadingMessage">Processing reports...</p>
                </div>
            </form>

            {% if case_data is defined and case_data %}
                <hr style="border-color: #555; margin-top: 2rem; margin-bottom: 2rem;">
                <h3 id="resultsSummary">Results Summary</h3>

                <!-- Aggregated Findings -->
                <div class="row">
                    <div class="col-md-6">
                        <h4 id="majorFindings">Major Findings Missed (Aggregated)</h4>
                        <ul>
                            {% set found_major = [] %}
                            {% for case in case_data %}
                                {% if case.summary and case.summary.major_findings %}
                                    {% for finding in case.summary.major_findings %}
                                        <li><a href="#case{{ case.case_num }}">Case {{ case.case_num }}</a>: {{ finding }}</li>
                                        {% set _ = found_major.append(1) %}
                                    {% endfor %}
                                {% endif %}
                            {% endfor %}
                            {% if not found_major %}<li>None</li>{% endif %}
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h4>Minor Findings Missed (Aggregated)</h4>
                        <ul>
                            {% set found_minor = [] %}
                            {% for case in case_data %}
                                {% if case.summary and case.summary.minor_findings %}
                                    {% for finding in case.summary.minor_findings %}
                                        <li><a href="#case{{ case.case_num }}">Case {{ case.case_num }}</a>: {{ finding }}</li>
                                        {% set _ = found_minor.append(1) %}
                                    {% endfor %}
                                {% endif %}
                            {% endfor %}
                             {% if not found_minor %}<li>None</li>{% endif %}
                        </ul>
                    </div>
                </div>


                <h3>Case Details & Navigation</h3>
                 <!-- Sort Options -->
                <div class="btn-group mb-3" role="group" aria-label="Sort Options">
                    <button type="button" class="btn btn-secondary btn-sm" onclick="sortCases('case_number')">Sort by Case Number</button>
                    <button type="button" class="btn btn-secondary btn-sm" onclick="sortCases('percentage_change')">Sort by % Change</button>
                    <button type="button" class="btn btn-secondary btn-sm" onclick="sortCases('summary_score')">Sort by Summary Score</button>
                </div>

                <!-- Case Navigation Links -->
                <ul id="caseNavList" class="list-group list-group-flush mb-3"></ul>

                 <!-- Container for Individual Case Details -->
                <div id="caseContainer">
                    <!-- Cases will be dynamically inserted here by JavaScript -->
                </div>

            {% elif request.method == 'POST' %}
                 <div class="alert alert-warning mt-3" role="alert">
                    No cases could be extracted. Please check the format of your input text. It should include lines starting with 'Case [Number]', 'Resident Report:', and 'Attending Report:'.
                 </div>
            {% endif %}

        </div> <!-- /container -->

        <!-- Scroll-to-top button -->
        <button id="scrollToTopBtn" onclick="scrollToTop()" title="Scroll to top">⬆</button>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let caseData = {{ case_data | default([]) | tojson }};
            const initialCaseData = JSON.parse(JSON.stringify(caseData));

            console.log("Initial caseData received from Flask:", caseData);

            function sortCases(option) {
                console.log("Sorting cases by:", option);
                if (!Array.isArray(caseData)) {
                    console.error("caseData is not an array, cannot sort.");
                    return;
                }
                let sortedData = [...caseData];
                try {
                     if (option === "case_number") {
                        sortedData.sort((a, b) => parseInt(a.case_num || '0') - parseInt(b.case_num || '0'));
                    } else if (option === "percentage_change") {
                        sortedData.sort((a, b) => (b.percentage_change || 0) - (a.percentage_change || 0));
                    } else if (option === "summary_score") {
                        sortedData.sort((a, b) => {
                            const scoreA = (a.summary && typeof a.summary.score === 'number') ? a.summary.score : -Infinity; // Sort errors/missing lowest
                            const scoreB = (b.summary && typeof b.summary.score === 'number') ? b.summary.score : -Infinity;
                            return scoreB - scoreA;
                        });
                    } else {
                         console.warn("Unknown sort option:", option);
                         return;
                    }
                    caseData = sortedData;
                    console.log("Sorted caseData:", caseData);
                    displayCases();
                    displayNavigation();
                } catch (error) {
                     console.error("Error during sorting:", error);
                }
            }

            function displayNavigation() {
                const navList = document.getElementById('caseNavList');
                if (!navList) { console.error("Element with id 'caseNavList' not found."); return; }
                navList.innerHTML = '';
                if (!Array.isArray(caseData) || caseData.length === 0) {
                    navList.innerHTML = '<li class="list-group-item" style="background-color: #2e2e2e; color: #aaa;">No cases to navigate.</li>';
                    return;
                }
                caseData.forEach(caseObj => {
                    const score = (caseObj.summary && typeof caseObj.summary.score === 'number') ? caseObj.summary.score : 'N/A';
                    const percentage = typeof caseObj.percentage_change === 'number' ? `${caseObj.percentage_change}%` : 'N/A';
                    const errorClass = (caseObj.summary && caseObj.summary.error) ? 'list-group-item-danger' : '';
                    navList.innerHTML += `
                        <li class="list-group-item ${errorClass}" style="background-color: #333; border-color: #555;">
                            <a href="#case${caseObj.case_num}">Case ${caseObj.case_num}</a> - Change: ${percentage} - Score: ${score}
                            ${(caseObj.summary && caseObj.summary.error) ? ' <small style="color: #ff6b6b;">(Error)</small>' : ''}
                        </li>
                    `;
                });
            }

            function displayCases() {
                const container = document.getElementById('caseContainer');
                if (!container) { console.error("Element with id 'caseContainer' not found."); return; }
                container.innerHTML = '';
                if (!Array.isArray(caseData) || caseData.length === 0) { return; }

                caseData.forEach((caseObj, index) => {
                    const caseNum = caseObj.case_num || `unknown_${index}`;
                    const percentageChange = typeof caseObj.percentage_change === 'number' ? `${caseObj.percentage_change}% change` : 'Change N/A';
                    const residentReport = caseObj.resident_report || "[Resident Report Not Available]";
                    const attendingReport = caseObj.attending_report || "[Attending Report Not Available]";
                    const diffContent = caseObj.diff || '<p class="error-message">Diff not available.</p>';
                    const summary = caseObj.summary;

                    let summaryHtml = '<div class="summary-output"><p class="error-message">Summary data is missing or invalid.</p></div>';
                    if (summary) {
                        if (summary.error) {
                             summaryHtml = `<div class="summary-output"><p class="error-message"><strong>Error generating summary:</strong> ${summary.error}</p>${summary.raw_response ? `<hr><p><strong>Raw AI Response (may contain useful details):</strong></p><pre style="font-size: 0.75rem; color: #ccc; max-height: 150px; overflow-y: auto;">${summary.raw_response}</pre>` : ''}</div>`;
                        } else {
                             const score = typeof summary.score === 'number' ? summary.score : 'N/A';
                             const majorFindingsHtml = (summary.major_findings && summary.major_findings.length > 0) ? `<p><strong>Major Findings:</strong></p><ul>${summary.major_findings.map(finding => `<li>${finding}</li>`).join('')}</ul>` : '<p><strong>Major Findings:</strong> None</p>';
                             const minorFindingsHtml = (summary.minor_findings && summary.minor_findings.length > 0) ? `<p><strong>Minor Findings:</strong></p><ul>${summary.minor_findings.map(finding => `<li>${finding}</li>`).join('')}</ul>` : '<p><strong>Minor Findings:</strong> None</p>';
                             const clarificationsHtml = (summary.clarifications && summary.clarifications.length > 0) ? `<p><strong>Clarifications:</strong></p><ul>${summary.clarifications.map(clarification => `<li>${clarification}</li>`).join('')}</ul>` : '<p><strong>Clarifications:</strong> None</p>';
                             summaryHtml = `<div class="summary-output"><p><strong>Score:</strong> ${score}</p>${majorFindingsHtml}${minorFindingsHtml}${clarificationsHtml}</div>`;
                        }
                    }

                    container.innerHTML += `
                        <div id="case${caseNum}" class="case-entry mb-4">
                            <h4 style="border-bottom: 1px solid #555; padding-bottom: 5px;">Case ${caseNum} - ${percentageChange}</h4>
                            <ul class="nav nav-tabs" id="myTab${caseNum}" role="tablist">
                                <li class="nav-item" role="presentation"><button class="nav-link active" id="summary-tab${caseNum}" data-bs-toggle="tab" data-bs-target="#summary${caseNum}" type="button" role="tab" aria-controls="summary${caseNum}" aria-selected="true">Summary Report</button></li>
                                <li class="nav-item" role="presentation"><button class="nav-link" id="combined-tab${caseNum}" data-bs-toggle="tab" data-bs-target="#combined${caseNum}" type="button" role="tab" aria-controls="combined${caseNum}" aria-selected="false">Diff Report</button></li>
                                <li class="nav-item" role="presentation"><button class="nav-link" id="resident-tab${caseNum}" data-bs-toggle="tab" data-bs-target="#resident${caseNum}" type="button" role="tab" aria-controls="resident${caseNum}" aria-selected="false">Resident Report</button></li>
                                <li class="nav-item" role="presentation"><button class="nav-link" id="attending-tab${caseNum}" data-bs-toggle="tab" data-bs-target="#attending${caseNum}" type="button" role="tab" aria-controls="attending${caseNum}" aria-selected="false">Attending Report</button></li>
                            </ul>
                            <div class="tab-content" id="myTabContent${caseNum}">
                                <div class="tab-pane fade show active" id="summary${caseNum}" role="tabpanel" aria-labelledby="summary-tab${caseNum}">${summaryHtml}</div>
                                <div class="tab-pane fade" id="combined${caseNum}" role="tabpanel" aria-labelledby="combined-tab${caseNum}"><div class="diff-output">${diffContent}</div></div>
                                <div class="tab-pane fade" id="resident${caseNum}" role="tabpanel" aria-labelledby="resident-tab${caseNum}"><div class="diff-output"><pre>${residentReport}</pre></div></div>
                                <div class="tab-pane fade" id="attending${caseNum}" role="tabpanel" aria-labelledby="attending-tab${caseNum}"><div class="diff-output"><pre>${attendingReport}</pre></div></div>
                            </div>
                        </div>
                    `;
                });
            }

            document.addEventListener("DOMContentLoaded", () => {
                console.log("DOM fully loaded.");
                if (caseData && caseData.length > 0) {
                    console.log("Case data exists on load. Rendering initial view sorted by case number.");
                    sortCases('case_number'); // Initial sort and display
                } else {
                    console.log("No case data available on initial load.");
                     displayNavigation(); // Display empty state
                     displayCases();
                }

                const reportForm = document.getElementById('reportForm');
                const loadingContainer = document.getElementById('loadingAnimationContainer');
                if(reportForm && loadingContainer) {
                    reportForm.addEventListener('submit', function(event) {
                        const reportText = document.getElementById('report_text').value;
                        if (!reportText.trim()) {
                           alert("Please paste report text before submitting.");
                           event.preventDefault();
                           return;
                        }
                        console.log("Form submitted, showing loading animation.");
                        loadingContainer.style.display = 'block';
                    });
                } else {
                     console.error("Could not find reportForm or loadingAnimationContainer elements.")
                }

                const scrollToTopButton = document.getElementById('scrollToTopBtn');
                 window.onscroll = function() {
                    if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) { // Show after scrolling down a bit
                        scrollToTopButton.style.display = "block";
                    } else {
                        scrollToTopButton.style.display = "none";
                    }
                };
            });

            function scrollToTop() {
                 window.scrollTo({ top: 0, behavior: 'smooth' });
            }

        </script>
    </body>
</html>
"""
    # Pass the selected model to the template context as 'chosen_model'
    # Pass the original input text back to the textarea
    return render_template_string(template,
                                  case_data=case_data,
                                  custom_prompt=custom_prompt,
                                  chosen_model=selected_model, # Passes selection back to template
                                  report_text_input=report_text_input)

if __name__ == '__main__':
    if not client:
         print("\n !!! WARNING: OpenAI client failed to initialize. API calls will fail. Ensure OPENAI_API_KEY is set correctly. !!! \n")
    # Set debug=False for production/general use
    app.run(debug=False, host='0.0.0.0', port=5001)
