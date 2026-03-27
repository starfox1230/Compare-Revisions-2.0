import json
import logging
import os
import re
from functools import lru_cache

import openai as _openai_pkg
from openai import OpenAI
from openai import (
    APIError,
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

logger.info("openai sdk version: %s", _openai_pkg.__version__)
logger.info("OPENAI_API_KEY present: %s", bool(os.getenv("OPENAI_API_KEY")))
logger.info("GEMINI_API_KEY present: %s", bool(os.getenv("GEMINI_API_KEY")))

AI_PROVIDER_CHATGPT = "chatgpt"
AI_PROVIDER_GEMINI = "gemini"

AI_MODEL_OPTIONS = {
    AI_PROVIDER_CHATGPT: "ChatGPT",
    AI_PROVIDER_GEMINI: "Gemini 3.1 Pro Preview",
}

DEFAULT_AI_PROVIDER = (os.getenv("DEFAULT_AI_PROVIDER") or AI_PROVIDER_CHATGPT).strip().lower()
if DEFAULT_AI_PROVIDER not in AI_MODEL_OPTIONS:
    logger.warning(
        "Invalid DEFAULT_AI_PROVIDER=%r; falling back to %s.",
        DEFAULT_AI_PROVIDER,
        AI_PROVIDER_CHATGPT,
    )
    DEFAULT_AI_PROVIDER = AI_PROVIDER_CHATGPT

OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", os.getenv("MODEL_ID", "gpt-5-mini"))
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.1-pro-preview")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "minimal")
VERBOSITY = os.getenv("VERBOSITY", "low")

TOOL_NAME = "summarize_radiology_report"

RADIOLOGY_SUMMARY_PARAMETERS = {
    "type": "object",
    "properties": {
        "case_number": {"type": "string"},
        "major_findings": {"type": "array", "items": {"type": "string"}},
        "minor_findings": {"type": "array", "items": {"type": "string"}},
        "clarifications": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "integer"},
        "major_key_phrases": {"type": "array", "items": {"type": "string"}},
        "minor_key_phrases": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "case_number",
        "major_findings",
        "minor_findings",
        "clarifications",
        "score",
        "major_key_phrases",
        "minor_key_phrases",
    ],
    "additionalProperties": False,
}

OPENAI_RESPONSES_TOOL = [
    {
        "type": "function",
        "name": TOOL_NAME,
        "description": (
            "Summarizes the differences between a resident and attending radiology "
            "report, categorizing findings and calculating a score."
        ),
        "parameters": RADIOLOGY_SUMMARY_PARAMETERS,
        "strict": True,
    }
]

GEMINI_CHAT_COMPLETIONS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Summarizes the differences between a resident and attending radiology "
                "report, categorizing findings and calculating a score."
            ),
            "parameters": RADIOLOGY_SUMMARY_PARAMETERS,
        },
    }
]

_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_SMART_QUOTES_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


@lru_cache(maxsize=1)
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# BEGIN GEMINI SECTION
@lru_cache(maxsize=1)
def get_gemini_client():
    return OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url=GEMINI_BASE_URL,
    )


def call_gemini_compare(case_text, custom_prompt, case_number):
    if not os.getenv("GEMINI_API_KEY"):
        return {
            "case_number": case_number,
            "error": "GEMINI_API_KEY is not set on the server.",
        }

    try:
        logger.info(
            "Processing case %s with provider=%s model=%s using OpenAI-compatible chat.completions.",
            case_number,
            AI_PROVIDER_GEMINI,
            GEMINI_MODEL_ID,
        )
        response = get_gemini_client().chat.completions.create(
            model=GEMINI_MODEL_ID,
            messages=[
                {"role": "system", "content": custom_prompt},
                {"role": "user", "content": build_case_input(case_text, case_number)},
            ],
            tools=GEMINI_CHAT_COMPLETIONS_TOOLS,
            tool_choice="auto",
            reasoning_effort=REASONING_EFFORT,
        )

        raw_arguments = extract_gemini_arguments(response)
        if raw_arguments is None:
            raw_arguments = extract_gemini_message_content(response)

        if raw_arguments is None:
            logger.error("No tool call or JSON content in Gemini response for case %s", case_number)
            return {
                "case_number": case_number,
                "error": "Gemini did not return the expected tool call or JSON payload.",
            }

        parsed_json = parse_json_payload(
            raw_arguments,
            case_number=case_number,
            provider=AI_PROVIDER_GEMINI,
            allow_repair=True,
        )
        if parsed_json is None:
            return {
                "case_number": case_number,
                "error": "Gemini returned malformed JSON that could not be repaired.",
            }

        return normalize_compare_result(parsed_json, case_number, AI_PROVIDER_GEMINI)

    except NotFoundError as e:
        logger.error("[gemini][404] NotFound for case %s: %s", case_number, e.message)
        return {
            "case_number": case_number,
            "error": f"Gemini 404 NotFound: {e.message}. Check GEMINI_MODEL_ID / API key.",
        }
    except BadRequestError as e:
        logger.error("[gemini][400] BadRequest for case %s: %s", case_number, e.message)
        return {"case_number": case_number, "error": f"Gemini 400 BadRequest: {e.message}"}
    except AuthenticationError as e:
        logger.error("[gemini][401] AuthError for case %s: %s", case_number, e.message)
        return {
            "case_number": case_number,
            "error": "Gemini 401 Unauthorized: check GEMINI_API_KEY",
        }
    except RateLimitError as e:
        logger.error("[gemini][429] RateLimit for case %s: %s", case_number, e.message)
        return {"case_number": case_number, "error": "Gemini 429 Rate limited"}
    except APIConnectionError as e:
        logger.error("[gemini][net] APIConnectionError for case %s: %s", case_number, e.message)
        return {"case_number": case_number, "error": "Gemini network error"}
    except APIError as e:
        logger.error("[gemini][5xx] APIError for case %s: %s", case_number, e.message)
        return {"case_number": case_number, "error": "Gemini server error"}
    except Exception as e:
        logger.error("Unhandled Gemini error for case %s: %r", case_number, e)
        return {"case_number": case_number, "error": f"Gemini unhandled error: {repr(e)}"}


def extract_gemini_arguments(response):
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None

    message = getattr(choices[0], "message", None)
    if message is None:
        return None

    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        if getattr(function, "name", None) == TOOL_NAME:
            return getattr(function, "arguments", None)
    return None


def extract_gemini_message_content(response):
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None

    message = getattr(choices[0], "message", None)
    if message is None:
        return None

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif getattr(item, "type", None) == "text":
                text_parts.append(getattr(item, "text", ""))
        combined = "\n".join(part for part in text_parts if part).strip()
        return combined or None

    return None
# END GEMINI SECTION


def call_openai_compare(case_text, custom_prompt, case_number):
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "case_number": case_number,
            "error": "OPENAI_API_KEY is not set on the server.",
        }

    try:
        logger.info(
            "Processing case %s with provider=%s model=%s using function calling.",
            case_number,
            AI_PROVIDER_CHATGPT,
            OPENAI_MODEL_ID,
        )
        response = get_openai_client().responses.create(
            model=OPENAI_MODEL_ID,
            instructions=custom_prompt,
            input=build_case_input(case_text, case_number),
            tools=OPENAI_RESPONSES_TOOL,
            tool_choice={"type": "function", "name": TOOL_NAME},
            reasoning={"effort": REASONING_EFFORT},
            text={"verbosity": VERBOSITY},
        )

        raw_arguments = extract_openai_arguments(response)
        if raw_arguments is None:
            logger.error("No function call in OpenAI response for case %s", case_number)
            return {
                "case_number": case_number,
                "error": "AI did not return the expected tool call.",
            }

        parsed_json = parse_json_payload(
            raw_arguments,
            case_number=case_number,
            provider=AI_PROVIDER_CHATGPT,
            allow_repair=False,
        )
        if parsed_json is None:
            return {
                "case_number": case_number,
                "error": "Invalid JSON in tool arguments from AI.",
            }

        return normalize_compare_result(parsed_json, case_number, AI_PROVIDER_CHATGPT)

    except NotFoundError as e:
        logger.error("[openai][404] NotFound for case %s: %s", case_number, e.message)
        return {
            "case_number": case_number,
            "error": f"404 NotFound: {e.message}. Check model id / project key.",
        }
    except BadRequestError as e:
        logger.error("[openai][400] BadRequest for case %s: %s", case_number, e.message)
        return {"case_number": case_number, "error": f"400 BadRequest: {e.message}"}
    except AuthenticationError as e:
        logger.error("[openai][401] AuthError for case %s: %s", case_number, e.message)
        return {
            "case_number": case_number,
            "error": "401 Unauthorized: check OPENAI_API_KEY",
        }
    except RateLimitError as e:
        logger.error("[openai][429] RateLimit for case %s: %s", case_number, e.message)
        return {"case_number": case_number, "error": "429 Rate limited"}
    except APIConnectionError as e:
        logger.error("[openai][net] APIConnectionError for case %s: %s", case_number, e.message)
        return {"case_number": case_number, "error": "Network error"}
    except APIError as e:
        logger.error("[openai][5xx] APIError for case %s: %s", case_number, e.message)
        return {"case_number": case_number, "error": "Server error"}
    except Exception as e:
        logger.error("Unhandled OpenAI error for case %s: %r", case_number, e)
        return {"case_number": case_number, "error": f"Unhandled: {repr(e)}"}


def compare_case_summary(case_text, custom_prompt, case_number, provider):
    # Single dispatch point: provider-specific request/response handling stays isolated
    # above, while callers below the route layer always get the same normalized payload.
    provider = normalize_provider(provider)
    if provider == AI_PROVIDER_GEMINI:
        return call_gemini_compare(case_text, custom_prompt, case_number)
    return call_openai_compare(case_text, custom_prompt, case_number)


def normalize_provider(provider):
    normalized = (provider or DEFAULT_AI_PROVIDER).strip().lower()
    if normalized not in AI_MODEL_OPTIONS:
        raise ValueError(f"Unsupported AI model selection: {provider}")
    return normalized


def get_provider_config_error(provider):
    provider = normalize_provider(provider)
    if provider == AI_PROVIDER_GEMINI and not os.getenv("GEMINI_API_KEY"):
        return "GEMINI_API_KEY is not set on the server."
    if provider == AI_PROVIDER_CHATGPT and not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not set on the server."
    return None


def build_case_input(case_text, case_number):
    return f"Case Number: {case_number}\n{case_text}"


def extract_openai_arguments(response):
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call" and getattr(item, "name", None) == TOOL_NAME:
            return getattr(item, "arguments", None)
    return None


def parse_json_payload(raw_payload, case_number, provider, allow_repair):
    if isinstance(raw_payload, dict):
        return raw_payload

    if not isinstance(raw_payload, str):
        logger.error("[%s] Non-string JSON payload for case %s: %r", provider, case_number, raw_payload)
        return None

    try:
        return json.loads(raw_payload)
    except json.JSONDecodeError as first_error:
        logger.warning(
            "[%s] JSON parse failure for case %s: %s",
            provider,
            case_number,
            first_error,
        )
        if not allow_repair:
            logger.debug("[%s] Raw arguments for case %s: %s", provider, case_number, raw_payload)
            return None

        repaired_payload = repair_json_payload(raw_payload)
        if not repaired_payload or repaired_payload == raw_payload:
            logger.debug("[%s] Raw Gemini arguments for case %s: %s", provider, case_number, raw_payload)
            return None

        try:
            logger.info("[%s] Repaired malformed JSON for case %s with one cleanup pass.", provider, case_number)
            return json.loads(repaired_payload)
        except json.JSONDecodeError as repair_error:
            logger.error(
                "[%s] JSON repair failed for case %s: %s",
                provider,
                case_number,
                repair_error,
            )
            logger.debug("[%s] Repaired payload for case %s: %s", provider, case_number, repaired_payload)
            return None


def repair_json_payload(raw_payload):
    candidate = raw_payload.strip().translate(_SMART_QUOTES_TRANSLATION)
    candidate = _FENCE_RE.sub("", candidate).strip()

    first_brace = candidate.find("{")
    last_brace = candidate.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = candidate[first_brace:last_brace + 1]

    candidate = candidate.replace("\ufeff", "").strip()
    candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)
    return candidate


def normalize_compare_result(parsed_json, case_number, provider):
    if not isinstance(parsed_json, dict):
        return {"case_number": case_number, "error": f"{provider} returned a non-object JSON payload."}

    major_findings = coerce_string_list(parsed_json.get("major_findings"))
    minor_findings = coerce_string_list(parsed_json.get("minor_findings"))
    clarifications = coerce_string_list(parsed_json.get("clarifications"))

    normalized = {
        "case_number": str(case_number),
        "major_findings": major_findings,
        "minor_findings": minor_findings,
        "clarifications": clarifications,
        "major_key_phrases": align_key_phrases(parsed_json.get("major_key_phrases"), len(major_findings)),
        "minor_key_phrases": align_key_phrases(parsed_json.get("minor_key_phrases"), len(minor_findings)),
    }

    expected_score = len(major_findings) * 3 + len(minor_findings)
    raw_score = parsed_json.get("score")
    if not isinstance(raw_score, int) or raw_score != expected_score:
        logger.info(
            "[%s] Normalized score for case %s from %r to %s.",
            provider,
            case_number,
            raw_score,
            expected_score,
        )
    normalized["score"] = expected_score
    return normalized


def coerce_string_list(value):
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if item is not None and str(item).strip()]


def align_key_phrases(value, target_length):
    if not isinstance(value, list):
        return [""] * target_length

    cleaned = [str(item).strip() if item is not None else "" for item in value[:target_length]]
    if len(cleaned) < target_length:
        cleaned.extend([""] * (target_length - len(cleaned)))
    return cleaned
